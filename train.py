import gc
import os
import shutil
import sys
import time
import warnings
import pdb
from functools import partial
import math
import numpy as np
import multiprocessing
import datetime
import json

# Fix CUDA multiprocessing issue - must be set before CUDA initialization
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass  # Already set

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


# build models
from torch.nn.parallel import DistributedDataParallel as DDP
from models import (
    VAR,
    VQVAE,
    build_vae_var,
    build_vae_var_with_ray_adaptation,
)
from trainer import VARTrainer
from utils.amp_sc import AmpOptimizer
from utils.lr_control import filter_params
from utils.erp_grouping import make_view_groups, validate_groups

import dist
from utils import arg_util, misc
from utils.wandb_logger import init_wandb_logger
from dataset.data import (
    build_dataset,
    build_dataset_webtar,
    build_erp_dataset,
)
from models.text_encoder import build_text
import torch.distributed as tdist
from utils.data_sampler import (
    DistInfiniteBatchSampler,
    EvalDistributedSampler,
    SceneViewBatchSampler,
    DistSceneViewBatchSampler,
)
from utils.misc import auto_resume
from utils.weights import zero_out_cross_attention_weights
import torchvision
from trainer import VARTrainer
from utils.lr_control import lr_wd_annealing


def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None  # Handle empty batch case
    return torch.utils.data.dataloader.default_collate(batch)


def save_training_batch_images(
    inp_B3HW,
    obj,
    ep,
    it,
    g_it,
    save_dir,
    suffix=None,
    view_ids=None,
    gather=True,
    update_dir=None,
    filename=None,
):
    """Save a batch of training input images with camera direction annotations.

    In multi-GPU (DDP) mode, gathers batches from all ranks to visualize the complete global batch.

    Args:
        inp_B3HW: Input image tensor (B, 3, H, W) in [-1, 1] range (local batch per rank).
        obj: Data batch dict containing 'cam_dir', 'prompt', 'fov_deg', etc.
        ep: Current epoch.
        it: Current iteration within epoch.
        g_it: Global iteration.
        save_dir: Root output directory.
    """
    import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
    import PIL.ImageFont as PImageFont

    batch_dir = (
        os.path.join(save_dir, "train_batch_vis") if update_dir is None else update_dir
    )
    os.makedirs(batch_dir, exist_ok=True)

    # Extract data from obj dict
    cam_dir_batch = obj.get("cam_dir", None)
    prompts = obj.get("prompt", [""] * inp_B3HW.shape[0])
    fov_deg = obj.get("fov_deg", None)
    view_ids_list = view_ids

    # Gather batches from all ranks in DDP mode (optional)
    if gather and dist.initialized() and dist.get_world_size() > 1:
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # Gather image tensors from all ranks
        local_B = inp_B3HW.shape[0]
        gathered_imgs = [torch.zeros_like(inp_B3HW) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_imgs, inp_B3HW.contiguous())

        # Gather camera directions (lon, lat tensors)
        gathered_lon_lat = None
        if (
            cam_dir_batch is not None
            and isinstance(cam_dir_batch, (list, tuple))
            and len(cam_dir_batch) == 2
        ):
            lon_tensor, lat_tensor = cam_dir_batch
            if isinstance(lon_tensor, torch.Tensor) and isinstance(
                lat_tensor, torch.Tensor
            ):
                # Move tensors to same device as inp_B3HW (GPU) for NCCL all_gather
                device = inp_B3HW.device
                lon_tensor = lon_tensor.to(device)
                lat_tensor = lat_tensor.to(device)

                gathered_lons = [
                    torch.zeros_like(lon_tensor) for _ in range(world_size)
                ]
                gathered_lats = [
                    torch.zeros_like(lat_tensor) for _ in range(world_size)
                ]
                torch.distributed.all_gather(gathered_lons, lon_tensor.contiguous())
                torch.distributed.all_gather(gathered_lats, lat_tensor.contiguous())
                gathered_lon_lat = (torch.cat(gathered_lons), torch.cat(gathered_lats))

        # Gather prompts (string lists - need to use object list)
        gathered_prompts = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(gathered_prompts, prompts)
        all_prompts = []
        for prompt_list in gathered_prompts:
            if prompt_list is not None:
                all_prompts.extend(prompt_list)

        # Gather fov_deg
        gathered_fov = None
        if fov_deg is not None:
            if isinstance(fov_deg, torch.Tensor):
                # Move tensor to same device as inp_B3HW (GPU) for NCCL all_gather
                device = inp_B3HW.device
                fov_deg = fov_deg.to(device)

                gathered_fov_list = [
                    torch.zeros_like(fov_deg) for _ in range(world_size)
                ]
                torch.distributed.all_gather(gathered_fov_list, fov_deg.contiguous())
                gathered_fov = torch.cat(gathered_fov_list)
            else:
                # fov_deg is a scalar, replicate for all gathered images
                total_B = local_B * world_size
                gathered_fov = [fov_deg] * total_B

        # Update with gathered tensors
        inp_B3HW = torch.cat(gathered_imgs, dim=0)
        cam_dir_batch = gathered_lon_lat
        prompts = all_prompts
        fov_deg = gathered_fov

        # Only rank 0 needs to process and save images
        # Other ranks return after participating in all_gather
        if rank != 0:
            return
    elif not gather and dist.initialized() and dist.get_world_size() > 1:
        # For non-gathered saves, only rank 0 proceeds
        if dist.get_rank() != 0:
            return

    # Denormalize from [-1, 1] to [0, 255]
    imgs = inp_B3HW.detach().cpu().float()
    imgs = (imgs + 1.0) * 0.5  # [-1,1] -> [0,1]
    imgs = imgs.clamp(0, 1)

    B = imgs.shape[0]
    H, W = imgs.shape[2], imgs.shape[3]

    # Parse (lon, lat) from collated cam_dir
    lon_lat_list = []
    if (
        cam_dir_batch is not None
        and isinstance(cam_dir_batch, (list, tuple))
        and len(cam_dir_batch) == 2
    ):
        lon_tensor, lat_tensor = cam_dir_batch
        if isinstance(lon_tensor, torch.Tensor):
            for i in range(lon_tensor.shape[0]):
                lon_deg = math.degrees(lon_tensor[i].item())
                lat_deg = math.degrees(lat_tensor[i].item())
                lon_lat_list.append((lon_deg, lat_deg))

    # Build annotated images
    annotated_imgs = []
    for i in range(B):
        img_np = (imgs[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pil_img = PImage.fromarray(img_np)
        draw = PImageDraw.Draw(pil_img)

        # Build annotation text
        lines = []
        if i < len(lon_lat_list):
            lon_d, lat_d = lon_lat_list[i]
            lines.append(f"lon={lon_d:.1f} lat={lat_d:.1f}")
        if view_ids_list is not None and i < len(view_ids_list):
            lines.append(f"view={view_ids_list[i]}")
        if fov_deg is not None:
            if isinstance(fov_deg, torch.Tensor):
                fov_val = fov_deg[i].item()
            elif isinstance(fov_deg, list):
                fov_val = fov_deg[i]
            else:
                fov_val = fov_deg
            lines.append(f"fov={fov_val:.0f}")
        if i < len(prompts):
            prompt_short = prompts[i][:60] + ("..." if len(prompts[i]) > 60 else "")
            lines.append(prompt_short)

        # Draw text with black background for readability
        y_offset = 5
        for line in lines:
            bbox = draw.textbbox((5, y_offset), line)
            draw.rectangle(
                [bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2], fill="black"
            )
            draw.text((5, y_offset), line, fill="white")
            y_offset = bbox[3] + 5

        annotated_imgs.append(
            torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).float() / 255.0
        )

    # Make grid with 2x2 layout for better visualization
    # nrow=2 creates 2 columns, so with 4 images we get a 2x2 grid
    nrow = 2 if B == 4 else min(B, 4)
    grid = torchvision.utils.make_grid(
        annotated_imgs, nrow=nrow, padding=4, pad_value=1.0
    )
    grid_np = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    if filename is None:
        if suffix:
            filename = f"ep{ep}_iter{it}_git{g_it}_{suffix}.png"
        else:
            filename = f"ep{ep}_iter{it}_git{g_it}.png"
    PImage.fromarray(grid_np).save(os.path.join(batch_dir, filename))

    # Include world size info in log message for multi-GPU
    # gpu_info = ""
    # if dist.initialized() and dist.get_world_size() > 1:
    #     gpu_info = f" [gathered from {dist.get_world_size()} GPUs]"
    # print(f"     [train_batch_vis] Saved {filename} ({B} images{gpu_info})")


def build_everything(args: arg_util.Args):
    # resume
    auto_resume_info, start_ep, start_it, trainer_state = auto_resume(
        args, "ar-ckpt*.pth"
    )
    start_ep = 0 if args.from_0 else start_ep

    # Save config as YAML in output directory
    if dist.is_master():
        import yaml

        config_dict = vars(args)
        # Filter out non-serializable objects
        config_filtered = {}
        for key, value in config_dict.items():
            try:
                if isinstance(
                    value, (str, int, float, bool, list, dict, tuple, type(None))
                ):
                    config_filtered[key] = value
            except:
                pass

        config_save_path = os.path.join(args.local_out_dir_path, "config.yaml")
        os.makedirs(args.local_out_dir_path, exist_ok=True)
        with open(config_save_path, "w") as f:
            yaml.dump(config_filtered, f, default_flow_style=False, sort_keys=False)
        print(f"[Config] Saved configuration to: {config_save_path}")

    # create tensorboard logger
    tb_lg: misc.TensorboardLogger
    with_tb_lg = dist.is_master()
    if with_tb_lg:
        os.makedirs(args.tb_log_dir_path, exist_ok=True)
        # Initialize WandB logger (replaces TensorboardLogger)
        # noinspection PyTypeChecker
        tb_lg = misc.DistLogger(
            init_wandb_logger(args),
            verbose=True,
        )
        tb_lg.flush()
    else:
        # noinspection PyTypeChecker
        tb_lg = misc.DistLogger(None, verbose=False)
    dist.barrier()

    # log args
    print(f"global bs={args.glb_batch_size}, local bs={args.batch_size}")

    # build data
    views_per_pano_train = None
    if not args.local_debug:
        print(f"[build PT data] ...\n")

        dataset_type = getattr(args, "dataset_type", None)

        if dataset_type == "erp":
            print("[using ERP dataset for ray adaptation] ...\n")
            dataset_train, dataset_val = build_erp_dataset(args)
            types = str((type(dataset_train).__name__, type(dataset_val).__name__))
            views_per_pano_train = dataset_train.views_per_pano
            scenes_per_batch = 1
            dist_inited = tdist.is_available() and tdist.is_initialized()
            world_size = tdist.get_world_size() if dist_inited else 1
            if getattr(args, "erp_same_scene_accum", False):
                local_batch_size = math.ceil(args.erp_views_per_scene / world_size)
            else:
                local_batch_size = args.batch_size
            if local_batch_size > dataset_train.views_per_pano:
                raise ValueError(
                    "ERP batching requires batch_size/views_per_scene <= views_per_pano"
                )
            sampler_cls = (
                DistSceneViewBatchSampler
                if dist_inited and world_size > 1
                else SceneViewBatchSampler
            )
            sampler_kwargs = dict(
                num_panos=len(dataset_train.data),
                views_per_pano=dataset_train.views_per_pano,
                batch_size=local_batch_size,
                shuffle=True,
                seed=args.seed if hasattr(args, "seed") else 0,
                views_per_scene=args.erp_views_per_scene
                if getattr(args, "erp_same_scene_accum", False)
                else None,
            )
            if sampler_cls is DistSceneViewBatchSampler:
                sampler_kwargs["scenes_per_global_batch"] = scenes_per_batch
            else:
                sampler_kwargs["scenes_per_batch"] = scenes_per_batch
            ld_train = DataLoader(
                dataset=dataset_train,
                num_workers=args.workers,
                pin_memory=True,
                generator=args.get_different_generator_for_each_rank(),
                batch_sampler=sampler_cls(**sampler_kwargs),
            )

            ld_val = DataLoader(
                dataset_val,
                num_workers=0,
                pin_memory=True,
                batch_size=round(args.batch_size * 1.5),
                sampler=EvalDistributedSampler(
                    dataset_val,
                    num_replicas=dist.get_world_size(),
                    rank=dist.get_rank(),
                ),
                shuffle=False,
                drop_last=False,
            )

            del dataset_train, dataset_val

        elif dataset_type == "cubemap":
            print("[using Cubemap dataset for anchor-weighted training] ...\n")
            from dataset.data import build_cubemap_dataset_v2
            from utils.data_sampler import CubemapSceneSampler

            dataset_train, dataset_val = build_cubemap_dataset_v2(args)
            types = str((type(dataset_train).__name__, type(dataset_val).__name__))
            # Each scene is one complete 6-face unit; sampler yields single scene indices
            cubemap_sampler = CubemapSceneSampler(
                num_scenes=len(dataset_train),
                shuffle=True,
                seed=args.seed
                if hasattr(args, "seed") and args.seed is not None
                else 0,
            )
            ld_train = DataLoader(
                dataset=dataset_train,
                num_workers=args.workers,
                pin_memory=True,
                generator=args.get_different_generator_for_each_rank(),
                batch_sampler=cubemap_sampler,
            )
            ld_val = DataLoader(
                dataset_val,
                num_workers=0,
                pin_memory=True,
                batch_size=1,
                sampler=EvalDistributedSampler(
                    dataset_val,
                    num_replicas=dist.get_world_size(),
                    rank=dist.get_rank(),
                ),
                shuffle=False,
                drop_last=False,
            )
            del dataset_train, dataset_val

        elif args.using_webtar:
            print("[using webtar] ...\n")
            dataset_train, dataset_val = build_dataset_webtar(args)
            types = str((type(dataset_train).__name__, type(dataset_val).__name__))
            ld_train = DataLoader(
                dataset=dataset_train,
                num_workers=args.workers,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=custom_collate_fn,
            )
            ld_val = DataLoader(
                dataset_val,
                num_workers=0,
                batch_size=round(args.batch_size * 1.5),
                shuffle=False,
                drop_last=False,
                collate_fn=custom_collate_fn,
            )
        else:
            print("[using lmdb] ...\n")
            dataset_train, dataset_val = build_dataset(args)
            types = str((type(dataset_train).__name__, type(dataset_val).__name__))

            ld_val = DataLoader(
                dataset_val,
                num_workers=0,
                pin_memory=True,
                batch_size=round(args.batch_size * 1.5),
                sampler=EvalDistributedSampler(
                    dataset_val,
                    num_replicas=dist.get_world_size(),
                    rank=dist.get_rank(),
                ),
                shuffle=False,
                drop_last=False,
            )
            del dataset_val

            ld_train = DataLoader(
                dataset=dataset_train,
                num_workers=args.workers,
                pin_memory=True,
                generator=args.get_different_generator_for_each_rank(),
                batch_sampler=DistInfiniteBatchSampler(
                    dataset_len=len(dataset_train),
                    glb_batch_size=args.glb_batch_size,
                    same_seed_for_all_ranks=args.same_seed_for_all_ranks,
                    shuffle=True,
                    fill_last=True,
                    rank=dist.get_rank(),
                    world_size=dist.get_world_size(),
                    start_ep=start_ep,
                    start_it=start_it,
                ),
            )
            del dataset_train

        [print(line) for line in auto_resume_info]
        print(f"[dataloader multi processing] ...", end="", flush=True)
        try:
            stt = time.time()
            if dataset_type == "erp":
                iters_train = len(ld_train)
                ld_train = iter(ld_train)
            elif dataset_type == "cubemap":
                iters_train = len(ld_train)
                ld_train = iter(ld_train)
            elif args.using_webtar:
                iters_train = args.web_tar_len // args.glb_batch_size
            else:
                iters_train = len(ld_train)
                ld_train = iter(ld_train)
            # noinspection PyArgumentList
            print(
                f"     [dataloader multi processing](*) finished! ({time.time() - stt:.2f}s)",
                flush=True,
                clean=True,
            )
            # print(
            #     f"[dataloader] gbs={args.glb_batch_size}, lbs={args.batch_size}, iters_train={iters_train}, types(tr, va)={types}"
            # )
        except:
            print(sys.exc_info())

    else:
        ld_val = ld_train = None
        iters_train = 10

    text_encoder, in_dim_cross = build_text(
        pretrained_path=args.text_enc_path,
        device=dist.get_device(),
        text_encoder=args.text_enc,
    )

    # Check if ray adaptation is enabled
    enable_ray_adaptation = getattr(args, "enable_ray_adaptation", False)

    if enable_ray_adaptation:
        print("\n" + "=" * 80)
        print("[RAY ADAPTATION] Enabled - building ModifiedVAR with ray adaptation")
        print("=" * 80 + "\n")

        # Use specialized patch_nums for ray adaptation (L=2240)
        ray_patch_nums = getattr(
            args, "ray_patch_nums", (1, 2, 3, 4, 6, 9, 13, 18, 24, 32)
        )
        adapter_dim = getattr(args, "adapter_dim", 128)
        num_memory_tokens = getattr(args, "num_memory_tokens", 32)
        ray_adapter_num_heads = getattr(args, "ray_adapter_num_heads", 4)
        ray_adapter_head_dim = getattr(args, "ray_adapter_head_dim", 32)
        adapter_active_scale_indices = getattr(
            args, "adapter_active_scale_indices", None
        )
        theta_gain_value = getattr(args, "theta_gain_value", 8.0)
        temp_gain_value = getattr(args, "temp_gain_value", 8.0)
        warm_start_steps = getattr(args, "warm_start_steps", 0)
        warm_theta_gain_value = getattr(args, "warm_theta_gain_value", 12.0)
        warm_temp_gain_value = getattr(args, "warm_temp_gain_value", 12.0)
        warm_unfreeze = getattr(args, "warm_unfreeze", True)
        gate_floor = getattr(args, "gate_floor", 0.0)
        gate_max = getattr(args, "gate_max", 0.1)
        gate_init = getattr(args, "gate_init", 0.03)
        gate_temperature = getattr(args, "gate_temperature", 1.0)

        print(
            f"[RAY ADAPTATION] patch_nums: {ray_patch_nums} (L={sum(pn**2 for pn in ray_patch_nums)})"
        )
        # print(f"[RAY ADAPTATION] adapter_dim: {adapter_dim}")
        # print(f"[RAY ADAPTATION] num_memory_tokens: {num_memory_tokens}")
        # print(f"[RAY ADAPTATION] ray_adapter_num_heads: {ray_adapter_num_heads}")
        # print(f"[RAY ADAPTATION] ray_adapter_head_dim: {ray_adapter_head_dim}")
        print(
            f"[RAY ADAPTATION] adapter_active_scale_indices: {adapter_active_scale_indices}"
        )
        print(
            f"[RAY ADAPTATION] gate_floor: {gate_floor} (deprecated; Option B gate used)"
        )
        print(
            f"[RAY ADAPTATION] gate_max={gate_max}, gate_init={gate_init}, gate_temperature={gate_temperature}\n"
        )
        print(
            f"[RAY ADAPTATION] theta_gain_value={theta_gain_value}, temp_gain_value={temp_gain_value}"
        )
        print(
            f"[RAY ADAPTATION] warm_start_steps={warm_start_steps}, warm_theta_gain_value={warm_theta_gain_value}, warm_temp_gain_value={warm_temp_gain_value}, warm_unfreeze={warm_unfreeze}"
        )

        vae_local, var_wo_ddp = build_vae_var_with_ray_adaptation(
            V=4096,
            Cvae=32,
            ch=160,
            share_quant_resi=4,
            device=dist.get_device(),
            patch_nums=ray_patch_nums,
            depth=args.depth,
            shared_aln=args.saln,
            attn_l2_norm=args.anorm,
            enable_cross=args.enable_cross,
            in_dim_cross=in_dim_cross,
            flash_if_available=args.fuse,
            fused_if_available=args.fuse,
            init_adaln=args.aln,
            init_adaln_gamma=args.alng,
            init_head=args.hd,
            init_std=args.ini,
            rope_emb=args.rope_emb,
            lvl_emb=args.lvl_emb,
            rope_norm=args.rope_norm,
            drop_scale_length=args.drop_scale_length,
            enable_logit_norm=args.logit_norm,
            enable_adaptive_norm=False,
            train_mode="all",
            rope_theta=args.rope_theta,
            vae_ada=False,
            # Ray adaptation parameters
            adapter_dim=adapter_dim,
            num_memory_tokens=num_memory_tokens,
            ray_adapter_num_heads=ray_adapter_num_heads,
            ray_adapter_head_dim=ray_adapter_head_dim,
            adapter_active_scale_indices=adapter_active_scale_indices,
            warm_start_steps=warm_start_steps,
            warm_theta_gain_value=warm_theta_gain_value,
            warm_temp_gain_value=warm_temp_gain_value,
            warm_unfreeze=warm_unfreeze,
            theta_gain_value=theta_gain_value,
            temp_gain_value=temp_gain_value,
            gate_floor=gate_floor,
            gate_max=gate_max,
            gate_init=gate_init,
            gate_temperature=gate_temperature,
        )
    else:
        vae_local, var_wo_ddp = build_vae_var(
            V=4096,
            Cvae=32,
            ch=160,
            share_quant_resi=4,  # hard-coded VQVAE hyperparameters
            device=dist.get_device(),
            patch_nums=args.patch_nums,
            depth=args.depth,
            shared_aln=args.saln,
            attn_l2_norm=args.anorm,
            enable_cross=args.enable_cross,
            in_dim_cross=in_dim_cross,  # TODO:换成从text enc得到的参数
            flash_if_available=args.fuse,
            fused_if_available=args.fuse,
            init_adaln=args.aln,
            init_adaln_gamma=args.alng,
            init_head=args.hd,
            init_std=args.ini,
            rope_emb=args.rope_emb,
            lvl_emb=args.lvl_emb,
            rope_norm=args.rope_norm,
            drop_scale_length=args.drop_scale_length,
            enable_logit_norm=args.logit_norm,
            enable_adaptive_norm=False,
            train_mode="all",
            rope_theta=args.rope_theta,
            vae_ada=False,
        )

    dist.barrier()
    vae_local.load_state_dict(
        torch.load(args.vae_ckpt, map_location="cpu"), strict=True
    )
    # if trainer_state is not None and len(trainer_state):
    #     print("unsing strict=False in loading...")
    #     new_state_dict = apply_lvl_emb_and_pos_1LC(
    #         args, state_dict=trainer_state["var_wo_ddp"], patch_nums=args.patch_nums
    #     )
    #     missing, unexpected = var_wo_ddp.load_state_dict(new_state_dict, strict=False)
    #     print("checkpoints incompatible: ", missing, unexpected)

    if not args.from_scratch:
        zero_out_cross_attention_weights(var_wo_ddp)

    # Freeze pretrained parameters if ray adaptation is enabled
    if enable_ray_adaptation and hasattr(var_wo_ddp, "freeze_pretrained_parameters"):
        print("\n[RAY ADAPTATION] Freezing pretrained VAR parameters...")
        var_wo_ddp.freeze_pretrained_parameters()
        print("[RAY ADAPTATION] Parameter freezing complete\n")

    vae_local: VQVAE = args.compile_model(vae_local, args.vfast)
    var_wo_ddp: VAR = args.compile_model(var_wo_ddp, args.tfast)
    var: DDP = (DDP if dist.initialized() else NullDDP)(
        var_wo_ddp,
        device_ids=[dist.get_local_rank()],
        find_unused_parameters=True,
        broadcast_buffers=False,
    )
    # var: FSDP = (FSDP if dist.initialized() else NullDDP)(var_wo_ddp, device_id=dist.get_local_rank(),
    #                                                       sharding_strategy=ShardingStrategy.FULL_SHARD)

    # print(f"[INIT] VAR model = {var_wo_ddp}\n\n")
    count_p = lambda m: f"{sum(p.numel() for p in m.parameters()) / 1e6:.2f}"
    print(
        f"[INIT][#para] "
        + ", ".join(
            [
                f"{k}={count_p(m)}"
                for k, m in (
                    ("VAE", vae_local),
                    ("VAE.enc", vae_local.encoder),
                    ("VAE.dec", vae_local.decoder),
                    ("VAE.quant", vae_local.quantize),
                )
            ]
        )
    )
    print(
        f"[INIT][#para] "
        + ", ".join([f"{k}={count_p(m)}" for k, m in (("VAR", var_wo_ddp),)])
        + "\n\n"
    )

    # build optimizer
    # fsdp我暂时不会写混合adam和adamw的
    names, paras, para_groups = filter_params(
        var_wo_ddp,
        nowd_keys={  # nowd_keys:没有权重衰减的参数名
            "cls_token",
            "start_token",
            "task_token",
            "cfg_uncond",
            "pos_embed",
            "pos_1LC",
            "pos_start",
            "start_pos",
            "lvl_embed",
            "gamma",
            "beta",
            "ada_gss",
            "moe_bias",
            "scale_mul",
        },
    )
    opt_clz = {
        "adam": partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=args.afuse),
        "adamw": partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=args.afuse),
    }[args.opt.lower().strip()]
    opt_kw = dict(lr=args.tlr, weight_decay=0)
    print(f"[INIT] optim={opt_clz}, opt_kw={opt_kw}\n")

    var_optim = AmpOptimizer(
        mixed_precision=args.fp16,
        optimizer=opt_clz(params=para_groups, **opt_kw),
        names=names,
        paras=paras,
        grad_clip=args.tclip,
        n_gradient_accumulation=args.ac,
    )
    del names, paras, para_groups

    # build trainer
    trainer = VARTrainer(
        device=args.device,
        patch_nums=args.patch_nums,
        resos=args.resos,
        vae_local=vae_local,
        var_wo_ddp=var_wo_ddp,
        var=var,
        var_opt=var_optim,
        label_smooth=args.ls,
        tclip=args.tclip,
        views_per_pano=views_per_pano_train,
    )
    trainer.adapter_metrics_every = getattr(args, "adapter_metrics_every", 0)
    if trainer_state is not None and len(trainer_state):
        missing, unexpected = trainer.load_state_dict(
            trainer_state, strict=False, skip_vae=True
        )  # don't load vae again
        print(
            f"[Checkpoint partial load] missing={len(missing)}, unexpected={len(unexpected)}"
        )
        # print('Checkpoints Weights Lists: ',missing,unexpected)

    del vae_local, var_wo_ddp, var, var_optim

    if args.local_debug:
        rng = torch.Generator("cpu")
        rng.manual_seed(0)
        B = 4
        inp = torch.rand(B, 3, args.data_load_reso, args.data_load_reso).to(args.device)
        label = torch.ones(B, dtype=torch.long)

        me = misc.MetricLogger(delimiter="  ")
        prompt_embeds_ = [
            torch.zeros([B, 77, 1024], device=inp.device),
            torch.zeros([B, 77], device=inp.device),
            torch.zeros([B, 1024], device=inp.device),
        ]
        trainer.train_step(
            it=0,
            g_it=0,
            stepping=True,
            metric_lg=me,
            tb_lg=tb_lg,
            inp_B3HW=inp,
            label_B=label,
            prompt_embeds=prompt_embeds_,
            prog_si=-1,
            prog_wp_it=20,
            args=args,
        )
        trainer.train_step(
            it=99,
            g_it=599,
            stepping=True,
            metric_lg=me,
            tb_lg=tb_lg,
            inp_B3HW=inp,
            label_B=label,
            prompt_embeds=prompt_embeds_,
            prog_si=-1,
            prog_wp_it=20,
            args=args,
        )
        print({k: meter.global_avg for k, meter in me.meters.items()})

        args.dump_log()
        tb_lg.flush()
        tb_lg.close()
        if isinstance(sys.stdout, misc.SyncPrint) and isinstance(
            sys.stderr, misc.SyncPrint
        ):
            sys.stdout.close(), sys.stderr.close()
        exit(0)

    # for name, param in text_encoder.named_parameters():
    #     print(f"Parameter: {name}, Requires Grad: {param.requires_grad}")
    # for name, param in trainer.vae_local.named_parameters():
    #     print(f"Parameter: {name}, Requires Grad: {param.requires_grad}")
    # for name, param in trainer.var_wo_ddp.named_parameters():
    #     print(f"Parameter: {name}, Requires Grad: {param.requires_grad}")

    dist.barrier()
    return (
        tb_lg,
        trainer,
        start_ep,
        start_it,
        iters_train,
        ld_train,
        ld_val,
        text_encoder,
    )


def main_training():
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    if args.local_debug:
        torch.autograd.set_detect_anomaly(True)

    (
        tb_lg,
        trainer,
        start_ep,
        start_it,
        iters_train,
        ld_train,
        ld_val,
        text_encoder,
    ) = build_everything(args)

    # train
    start_time = time.time()
    best_L_mean, best_L_tail, best_acc_mean, best_acc_tail = 999.0, 999.0, -1.0, -1.0
    best_val_loss_mean, best_val_loss_tail, best_val_acc_mean, best_val_acc_tail = (
        999,
        999,
        -1,
        -1,
    )

    L_mean, L_tail = -1, -1

    # Early stopping setup
    early_stop_patience = getattr(args, "early_stop_patience", 5)
    early_stop_counter = 0
    best_val_loss_for_early_stop = float("inf")
    print(f"[EARLY STOPPING] Enabled with patience={early_stop_patience}")

    # Default validation metrics (used when we skip eval until final epoch)
    val_loss_mean = val_loss_tail = val_acc_mean = val_acc_tail = -1

    # Checkpoint saving setup
    ckpt_save_it = getattr(args, "ckpt_save_it", 150000)
    print(f"[CHECKPOINT SAVING] Saving checkpoints every {ckpt_save_it} iterations")

    for ep in range(start_ep, args.ep):
        if hasattr(ld_train, "sampler") and hasattr(ld_train.sampler, "set_epoch"):
            ld_train.sampler.set_epoch(ep)
        # Skip pre-epoch eval to reduce overhead; run eval only at final epoch
        if dist.is_local_master():
            cur_start_it = start_it if ep == start_ep else 0
            print(
                f"[Epoch {ep + 1}/{args.ep}] training started "
                f"(start_it={cur_start_it}, total_iters={iters_train})",
                flush=True,
            )

        # def train_one_ep(ep: int, is_first_ep: bool, start_it: int, args: arg_util.Args, tb_lg: misc.TensorboardLogger, ld_or_itrt, iters_train: int, trainer):
        print(f"[Debug {ep}] training started ", flush=True)
        stats, (sec, remain_time, finish_time) = train_one_ep(
            ep=ep,
            is_first_ep=(ep == start_ep),
            start_it=start_it if ep == start_ep else 0,
            args=args,
            tb_lg=tb_lg,
            ld_or_itrt=ld_train,
            text_enc=text_encoder,
            iters_train=iters_train,
            trainer=trainer,
            ckpt_save_it=ckpt_save_it,
        )
        print(f"[Debug {ep}] training one ep finished ", flush=True)

        L_mean, L_tail, acc_mean, acc_tail, grad_norm = (
            stats["Lm"],
            stats["Lt"],
            stats["Accm"],
            stats["Acct"],
            stats["tnm"],
        )
        best_L_mean, best_acc_mean = (
            min(best_L_mean, L_mean),
            max(best_acc_mean, acc_mean),
        )
        if L_tail != -1:
            best_L_tail, best_acc_tail = (
                min(best_L_tail, L_tail),
                max(best_acc_tail, acc_tail),
            )
        args.L_mean, args.L_tail, args.acc_mean, args.acc_tail, args.grad_norm = (
            L_mean,
            L_tail,
            acc_mean,
            acc_tail,
            grad_norm,
        )
        args.cur_ep = f"{ep + 1}/{args.ep}"
        args.remain_time, args.finish_time = remain_time, finish_time

        AR_ep_loss = dict(
            L_mean=L_mean, L_tail=L_tail, acc_mean=acc_mean, acc_tail=acc_tail
        )
        if (ep + 1) % 5 == 0 or (ep + 1) == args.ep:  # 每10 epoch验证一次
            # if is_val_and_also_saving:
            if dist.is_local_master():
                local_out_ckpt = os.path.join(
                    args.local_out_dir_path, "ar-ckpt-last.pth"
                )
                local_out_ckpt_best = os.path.join(
                    args.local_out_dir_path, "ar-ckpt-best.pth"
                )
                print(f"[saving ckpt] ...", end="", flush=True)
                torch.save(
                    {
                        "epoch": ep + 1,
                        "iter": 0,
                        "trainer": trainer.state_dict(),
                        # 'text_enc': text_encoder.state_dict()
                        # 'args':     args.state_dict(),
                    },
                    local_out_ckpt,
                )
                # if best_updated:
                #     shutil.copy(local_out_ckpt, local_out_ckpt_best)
                trainer.inference_pic(
                    args,
                    text_encoder,
                    g_it=(ep + 1) * iters_train,  # end-of-epoch step
                    top_k=600,
                    top_p=0.8,
                    w_mask=False,
                    tb_lg=tb_lg,
                )
                torch.cuda.empty_cache()
                print(
                    f"     [saving ckpt](*) finished!  @ {local_out_ckpt}",
                    flush=True,
                    clean=True,
                )

            if ep == args.ep - 1:
                val_loss_mean, val_loss_tail, val_acc_mean, val_acc_tail, tot, cost = (
                    trainer.eval_ep(args, ld_val, text_encoder)
                )
                best_updated = best_val_loss_tail > val_loss_tail
                best_val_loss_mean, best_val_loss_tail = (
                    min(best_val_loss_mean, val_loss_mean),
                    min(best_val_loss_tail, val_loss_tail),
                )
                best_val_acc_mean, best_val_acc_tail = (
                    max(best_val_acc_mean, val_acc_mean),
                    max(best_val_acc_tail, val_acc_tail),
                )

                # Save best checkpoint if validation improved
                if best_updated and dist.is_local_master():
                    local_out_ckpt_best = os.path.join(
                        args.local_out_dir_path, "ar-ckpt-best.pth"
                    )
                    shutil.copy(local_out_ckpt, local_out_ckpt_best)
                    print(
                        f"     [✅ NEW BEST] Saved best checkpoint @ {local_out_ckpt_best} (val_loss_tail={val_loss_tail:.8f})"
                    )

                # Early stopping check
                if val_loss_tail < best_val_loss_for_early_stop:
                    best_val_loss_for_early_stop = val_loss_tail
                    early_stop_counter = 0
                    print(
                        f"     [✅ IMPROVEMENT] Validation loss improved: {val_loss_tail:.8f}"
                    )
                else:
                    early_stop_counter += 1
                    print(
                        f"     [⚠️  NO IMPROVEMENT] Counter: {early_stop_counter}/{early_stop_patience} epochs"
                    )

                if early_stop_counter >= early_stop_patience:
                    print(f"\n{'=' * 80}")
                    print(
                        f"[EARLY STOPPING] No improvement for {early_stop_patience} epochs"
                    )
                    print(f"Best validation loss: {best_val_loss_for_early_stop:.4f}")
                    print(f"Stopping training at epoch {ep + 1}/{args.ep}")
                    print(f"{'=' * 80}\n")
                    # Save final checkpoint before stopping
                    if dist.is_local_master():
                        local_out_ckpt_final = os.path.join(
                            args.local_out_dir_path, "ar-ckpt-final-earlystop.pth"
                        )
                        torch.save(
                            {
                                "epoch": ep + 1,
                                "iter": 0,
                                "trainer": trainer.state_dict(),
                                "early_stopped": True,
                                "best_val_loss": best_val_loss_for_early_stop,
                            },
                            local_out_ckpt_final,
                        )
                        print(f"     [FINAL CKPT] Saved @ {local_out_ckpt_final}")
                    break  # Exit training loop

            AR_ep_loss.update(
                vL_mean=val_loss_mean,
                vL_tail=val_loss_tail,
                vacc_mean=val_acc_mean,
                vacc_tail=val_acc_tail,
            )
            args.vL_mean, args.vL_tail, args.vacc_mean, args.vacc_tail = (
                val_loss_mean,
                val_loss_tail,
                val_acc_mean,
                val_acc_tail,
            )
            print(
                f" [*] [ep{ep}]  (val {tot})  Lm: {L_mean:.4f}, Lt: {L_tail:.4f}, Acc m&t: {acc_mean:.2f} {acc_tail:.2f},  Val cost: {cost:.2f}s"
            )

            dist.barrier()

        print(
            f"     [ep{ep}]  (training )  Lm: {best_L_mean:.3f} ({L_mean:.3f}), Lt: {best_L_tail:.3f} ({L_tail:.3f}),  Acc m&t: {best_acc_mean:.2f} {best_acc_tail:.2f},  Remain: {remain_time},  Finish: {finish_time}",
            flush=True,
        )
        args.dump_log()
        tb_lg.flush()

    # Log view histogram once at training end

    if trainer.view_hist is not None:
        print(f"Logging view histogram...", flush=True)
        hist = trainer.view_hist.clone()
        if tdist.is_available() and tdist.is_initialized():
            tdist.all_reduce(hist)
        if dist.is_master():
            hist_cpu = hist.cpu().tolist()
            tb_lg.update(
                head="view_hist",
                **{f"view_{i}": v for i, v in enumerate(hist_cpu)},
                step=args.ep * iters_train,
            )
        trainer.view_hist.zero_()

    total_time = f"{(time.time() - start_time) / 60 / 60:.1f}h"
    print("\n\n")
    print(
        f"  [*] [PT finished]  Total cost: {total_time},   Lm: {best_L_mean:.3f} ({L_mean}),   Lt: {best_L_tail:.3f} ({L_tail})"
    )
    print("\n\n")

    # Clean up variables if they exist
    print(f"Starting Clean up...", flush=True)
    try:
        del stats
    except NameError:
        pass
    del iters_train, ld_train
    time.sleep(3), gc.collect(), torch.cuda.empty_cache(), time.sleep(3)
    print(f"Ending Clean up...", flush=True)

    args.remain_time, args.finish_time = (
        "-",
        time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() - 60)),
    )
    print(f"final args:\n\n{str(args)}")
    args.dump_log()
    tb_lg.flush()
    tb_lg.close()
    dist.barrier()


def train_one_ep(
    ep: int,
    is_first_ep: bool,
    start_it: int,
    args: arg_util.Args,
    tb_lg: misc.TensorboardLogger,
    ld_or_itrt,
    text_enc,
    iters_train: int,
    trainer,
    ckpt_save_it: int = 150000,
):
    # import heavy packages after Dataloader object creation
    # ld_or_itrt:dataloader
    print(f"[Debug {ep}] Entered train_one_ep ", flush=True)

    trainer: VARTrainer

    step_cnt = 0
    me = misc.MetricLogger(delimiter="  ")
    me.add_meter("tlr", misc.SmoothedValue(window_size=1, fmt="{value:.2g}"))
    me.add_meter("tnm", misc.SmoothedValue(window_size=1, fmt="{value:.2f}"))
    [
        me.add_meter(x, misc.SmoothedValue(fmt="{median:.3f} ({global_avg:.3f})"))
        for x in ["Lm", "Lt"]
    ]
    [
        me.add_meter(x, misc.SmoothedValue(fmt="{median:.2f} ({global_avg:.2f})"))
        for x in ["Accm", "Acct"]
    ]
    header = f"[Ep]: [{ep:4d}/{args.ep}]"

    if is_first_ep:
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
    g_it, max_it = ep * iters_train, args.ep * iters_train

    # --- optimizer-step aligned counters ---
    use_erp_accum = getattr(args, "dataset_type", "") == "erp" and getattr(
        args, "erp_same_scene_accum", False
    )
    use_cubemap_accum = getattr(args, "dataset_type", "") == "cubemap"
    micro_per_update = (
        2  # cubemap always has exactly 2 groups (set1 + set2) per effective update
        if use_cubemap_accum
        else max(
            1,
            # Derive num_groups from the dynamic grouping parameters
            math.ceil(
                getattr(args, "erp_views_per_scene", 12)
                / max(1, getattr(args, "erp_group_size", 4))
            ),
        )
        if use_erp_accum
        else max(1, args.ac)
    )
    updates_per_epoch = math.ceil(iters_train / micro_per_update)
    max_updates = args.ep * updates_per_epoch
    update_step = ep * updates_per_epoch + (start_it // micro_per_update)
    # Align logger step to optimizer-step index to keep WandB/TB monotonic
    tb_lg.set_step(step=update_step)

    if args.using_webtar:
        all_file_keys = []

    # Setup timing trackers
    start_time = time.time()
    iter_end_t = time.time()
    iter_time = misc.SmoothedValue(fmt="{avg:.4f}")
    data_time = misc.SmoothedValue(fmt="{avg:.4f}")
    log_every_it = getattr(args, "log_every_it", 100)  # M iteration마다 출력

    # Iterate through dataloader
    for it, obj in enumerate(ld_or_itrt):
        data_time.update(time.time() - iter_end_t)

        if it < start_it:
            continue

        if is_first_ep and it == start_it:
            warnings.resetwarnings()

        # Compute global iteration for checkpoint saving
        g_it = ep * iters_train + it

        if use_cubemap_accum:
            # Cubemap batches have "images" [1,6,C,H,W] — skip the shared extraction.
            # The cubemap training branch (elif use_cubemap_accum) handles all of this.
            inp = None
            B = 1
            label = None
            prompt_embeds = None
        else:
            inp = obj["image"].to(args.device, non_blocking=True)
            obj["prompt_embeds"] = text_enc.extract_text_features(obj["prompt"])
            B = inp.shape[0]
            label = torch.tensor([args.default_label] * B).to(
                args.device, non_blocking=True
            )
            prompt_embeds = obj["prompt_embeds"]

        # Extract camera parameters if present (for ray adaptation training)
        cam_dir = None
        fov_deg = 120.0  # Default FOV
        if "cam_dir" in obj:
            # cam_dir from DataLoader default collate: list of 2 tensors [lons, lats]
            # where each tensor has batch_size elements
            from dataset.extract_tangents import lonlat_to_dir

            cam_dirs_list = []
            cam_dir_batch = obj["cam_dir"]

            # Handle PyTorch default collate output: list of [lon_tensor, lat_tensor]
            if isinstance(cam_dir_batch, (list, tuple)) and len(cam_dir_batch) == 2:
                lon_tensor, lat_tensor = cam_dir_batch
                if isinstance(lon_tensor, torch.Tensor) and isinstance(
                    lat_tensor, torch.Tensor
                ):
                    # Batch of (lon, lat) pairs
                    batch_size = lon_tensor.shape[0]
                    for i in range(batch_size):
                        lon = lon_tensor[i].item()
                        lat = lat_tensor[i].item()
                        dir_vec = lonlat_to_dir(lon, lat)
                        cam_dirs_list.append(dir_vec)
                else:
                    # Fallback: single (lon, lat) tuple
                    lon = (
                        lon_tensor
                        if isinstance(lon_tensor, float)
                        else lon_tensor.item()
                    )
                    lat = (
                        lat_tensor
                        if isinstance(lat_tensor, float)
                        else lat_tensor.item()
                    )
                    dir_vec = lonlat_to_dir(lon, lat)
                    cam_dirs_list.append(dir_vec)
            elif (
                isinstance(cam_dir_batch, torch.Tensor) and cam_dir_batch.shape[-1] == 2
            ):
                # Tensor of shape [batch_size, 2]
                for i in range(cam_dir_batch.shape[0]):
                    lon = cam_dir_batch[i, 0].item()
                    lat = cam_dir_batch[i, 1].item()
                    dir_vec = lonlat_to_dir(lon, lat)
                    cam_dirs_list.append(dir_vec)
            else:
                raise ValueError(
                    f"Unexpected cam_dir format. Type: {type(cam_dir_batch)}, "
                    f"Content: {cam_dir_batch}"
                )

            if len(cam_dirs_list) == 0:
                raise ValueError(
                    f"No valid camera directions found. cam_dir_batch: {cam_dir_batch}"
                )

            cam_dir = (
                torch.from_numpy(np.stack(cam_dirs_list, axis=0))
                .float()
                .to(args.device)
            )

        if "fov_deg" in obj:
            fov_deg = (
                obj["fov_deg"][0].item()
                if isinstance(obj["fov_deg"], torch.Tensor)
                else obj["fov_deg"]
            )

        args.cur_it = f"{update_step + 1}/{max_updates} (mb {it + 1}/{iters_train})"

        wp_it = args.wp * updates_per_epoch
        min_tlr, max_tlr, min_twd, max_twd = lr_wd_annealing(
            args.sche,
            trainer.var_opt.optimizer,
            args.tlr,
            args.twd,
            args.twde,
            update_step,
            wp_it,
            max_updates,
            wp0=args.wp0,
            wpe=args.wpe,
        )
        # warmup:迭代次数<wp_it时，学习率从wp0线性增加到1 (乘peak_lr=1e-5)
        args.cur_lr, args.cur_wd = max_tlr, max_twd

        if args.pg:  # default: 0.0, no progressive training, won't get into this
            if update_step <= wp_it:
                prog_si = args.pg0  # warmup阶段prog_si是默认值pg0=4
            elif update_step >= max_updates * args.pg:
                prog_si = (
                    len(args.patch_nums) - 1
                )  # iter大于args.pg指定的iter时，prog_si是scale个数
            else:
                delta = len(args.patch_nums) - 1 - args.pg0
                progress = min(
                    max((update_step - wp_it) / (max_updates * args.pg - wp_it), 0), 1
                )  # from 0 to 1
                prog_si = args.pg0 + round(
                    progress * delta
                )  # from args.pg0 to len(args.patch_nums)-1
        else:
            prog_si = -1  # prog_si似乎是指定不同训练stage focus在不同scale上

        # ERP same-scene gradient accumulation (split view groups)
        use_same_scene_accum = getattr(args, "dataset_type", "") == "erp" and getattr(
            args, "erp_same_scene_accum", False
        )

        if use_same_scene_accum:
            view_idx_full = obj.get("view_idx", None)
            if view_idx_full is None:
                raise ValueError("erp_same_scene_accum requires view_idx in the batch")

            if torch.is_tensor(view_idx_full):
                view_idx_tensor = view_idx_full.to(inp.device)
            else:
                view_idx_tensor = torch.tensor(view_idx_full, device=inp.device)
            cam_dir_raw = obj.get("cam_dir", None)

            # --- Dynamic grouping (replaces static erp_view_groups) ---
            group_size = getattr(args, "erp_group_size", 4)
            grouping_mode = getattr(args, "erp_grouping_mode", "spread")
            # Sampler guarantees canonical order: view_idx_tensor[i] == i
            canonical_view_ids = view_idx_tensor.tolist()
            total_views_for_scene = len(canonical_view_ids)

            # Assert same scene (all views come from one pano)
            file_keys = obj.get("file_key", None)
            if isinstance(file_keys, (list, tuple)) and len(file_keys) > 1:
                # Some datasets append view ids to file_key (e.g., "foo_000", "foo_001").
                # Normalize by stripping the trailing underscore segment and require all bases match.
                def _base_key(x: str):
                    return x.rsplit("_", 1)[0] if isinstance(x, str) else x

                base_keys = {_base_key(fk) for fk in file_keys}
                assert len(base_keys) == 1, (
                    f"[ERP accum] Mixed scenes in one batch: bases={base_keys} raw={set(file_keys)}"
                )

            groups = make_view_groups(
                canonical_view_ids, group_size=group_size, mode=grouping_mode
            )
            validate_groups(groups, canonical_view_ids, group_size)

            num_groups = len(groups)
            # Build list of (gi, positional_index_tensor, view_id_list) triples.
            # Positional index: since sampler provides canonical order, position i
            # in the batch has view_id == canonical_view_ids[i] == i.
            group_indices = []
            for gi, view_id_list in enumerate(groups):
                pos_idx = torch.tensor(
                    [canonical_view_ids.index(v) for v in view_id_list],
                    device=inp.device,
                )
                group_indices.append((gi, pos_idx, view_id_list))

            # Sample-aware loss weights: weight_i = group_size_i / total_views
            # Passed as loss_divisor so trainer applies: loss / loss_divisor
            # = loss * (group_size_i / total_views)  ← per-sample average weight
            # Summing across groups → full-scene sample average (not group average).
            group_loss_divisors = [
                total_views_for_scene / float(len(view_id_list))
                for _, _, view_id_list in group_indices
            ]
            # For ERP accum, every scene is one optimizer update (step on last group)
            base_stepping = True
            stepping = False
            grad_norm = scale_log2 = None

            # Per-update visualization: save every save_interval updates
            n_batch_saves = 5
            save_interval = max(1, updates_per_epoch // n_batch_saves)
            # Force-save on the very first update of this epoch
            is_first_update_this_ep = update_step == ep * updates_per_epoch
            do_save = (update_step % save_interval == 0) or is_first_update_this_ep

            # Build per-update directory (rank-0 only, but compute path on all ranks)
            update_dir = os.path.join(
                args.local_out_dir_path,
                "train_batch_vis",
                f"update_{update_step:07d}",
            )
            if do_save and dist.is_master():
                os.makedirs(update_dir, exist_ok=True)

            # Log grouping on first iteration for quick verification
            if dist.is_local_master() and it == start_it:
                group_sizes = {gi: len(vl) for gi, _, vl in group_indices}
                print(
                    f"[ERP accum] it={it} group_size={group_size} mode={grouping_mode} "
                    f"total_views={total_views_for_scene} view groups -> {group_sizes}",
                    flush=True,
                )

            group_records = []
            for idx_i, (gi, sel_idx, view_id_list) in enumerate(group_indices):
                sel_idx_cpu = sel_idx.detach().cpu()
                # Slice tensors/lists for this group
                inp_grp = inp[sel_idx]
                label_grp = label[sel_idx]
                prompt_embeds_grp = tuple(pe[sel_idx] for pe in prompt_embeds)
                cam_dir_grp = cam_dir[sel_idx] if cam_dir is not None else None
                view_idx_grp = view_idx_tensor[sel_idx]
                view_list = view_idx_grp.detach().cpu().tolist()

                if do_save:
                    cam_dir_grp_raw = None
                    if cam_dir_raw is not None:
                        if (
                            isinstance(cam_dir_raw, (list, tuple))
                            and len(cam_dir_raw) == 2
                        ):
                            cam_dir_grp_raw = [c[sel_idx_cpu] for c in cam_dir_raw]
                    obj_grp = {
                        "cam_dir": cam_dir_grp_raw,
                        "prompt": [obj["prompt"][i] for i in sel_idx_cpu.tolist()],
                        "fov_deg": obj.get("fov_deg", None),
                    }
                    fname = (
                        f"set{gi + 1}_views_{'_'.join(str(v) for v in view_list)}.png"
                    )
                    save_training_batch_images(
                        inp_grp,
                        obj_grp,
                        ep,
                        it,
                        update_step,
                        args.local_out_dir_path,
                        view_ids=view_list,
                        gather=False,
                        update_dir=update_dir,
                        filename=fname,
                    )

                    # Build per-group lon/lat for meta.json
                    lon_lat_for_group = []
                    if cam_dir_grp_raw is not None and len(cam_dir_grp_raw) == 2:
                        lon_t, lat_t = cam_dir_grp_raw
                        if isinstance(lon_t, torch.Tensor):
                            for k in range(lon_t.shape[0]):
                                lon_lat_for_group.append(
                                    {
                                        "lon_rad": lon_t[k].item(),
                                        "lat_rad": lat_t[k].item(),
                                    }
                                )
                    group_records.append(
                        {
                            "group": gi,
                            "view_ids": view_list,
                            "lon_lat": lon_lat_for_group,
                        }
                    )

                stepping = base_stepping and (idx_i == len(group_indices) - 1)
                # Sample-aware loss divisor: total_views / group_size_i
                # → loss_i / divisor_i = loss_i * (group_size_i / total_views)
                # Summed across groups gives full-scene sample average.
                loss_divisor = group_loss_divisors[idx_i]
                grad_norm, scale_log2 = trainer.train_step(
                    it=it,
                    g_it=update_step,
                    stepping=stepping,
                    metric_lg=me,
                    tb_lg=tb_lg,
                    inp_B3HW=inp_grp,
                    label_B=label_grp,
                    prompt_embeds=prompt_embeds_grp,
                    prog_si=prog_si,
                    prog_wp_it=args.pgwp * iters_train,
                    cam_dir=cam_dir_grp,
                    args=args,
                    fov_deg=fov_deg,
                    view_idx=view_idx_grp,
                    loss_divisor=loss_divisor,
                )

            # Write meta.json after all groups are processed
            if stepping and do_save and dist.is_master():
                file_key = obj.get("file_key", None)
                if isinstance(file_key, (list, tuple)):
                    pano_id = file_key[0] if len(file_key) > 0 else "?"
                elif file_key is not None:
                    pano_id = str(file_key)
                else:
                    pano_id = "?"
                prompt_val = obj.get("prompt", [""])
                if isinstance(prompt_val, (list, tuple)):
                    prompt_val = prompt_val[0] if len(prompt_val) > 0 else ""
                ws = dist.get_world_size() if dist.initialized() else 1
                meta = {
                    "optimizer_step": update_step,
                    "microbatch": int(it),
                    "pano_id": pano_id,
                    "prompt": prompt_val,
                    "view_groups": group_records,
                    "world_size": ws,
                    # Extended grouping metadata
                    "total_views_for_scene": total_views_for_scene,
                    "group_size": group_size,
                    "num_groups": num_groups,
                    "grouping_mode": grouping_mode,
                    "canonical_view_ids": canonical_view_ids,
                    "grouped_view_ids": [vl for _, _, vl in group_indices],
                }
                with open(os.path.join(update_dir, "meta.json"), "w") as f:
                    json.dump(meta, f, indent=2)

            if stepping:
                update_step += 1
            step_cnt += int(stepping)

        elif use_cubemap_accum:
            # ── Cubemap anchor-weighted 2-set accumulation ──────────────────────
            # Dataset yields: images[1,6,C,H,W], face_ids[1,6], prompt, scene_id
            # set1 = [front, back, left, right]; set2 = [front, back, up, down]
            # front + back appear in BOTH sets — INTENTIONAL anchor weighting, NOT a bug.
            # Each effective optimizer update = 2 microbatches (set1 then set2).
            from utils.cubemap_groups import (
                CANONICAL_FACES as _CANONICAL_FACES,
                CUBEMAP_SET1 as _CUBEMAP_SET1,
                CUBEMAP_SET2 as _CUBEMAP_SET2,
                FACE_TO_IDX as _FACE_TO_IDX,
                DUPLICATION_SUMMARY as _DUPLICATION_SUMMARY,
                make_cubemap_groups as _make_cubemap_groups,
                validate_cubemap_groups as _validate_cubemap_groups,
                cubemap_loss_divisors as _cubemap_loss_divisors,
            )
            from utils.face_id_embedding import FaceIdEmbedding as _FaceIdEmbedding

            # Squeeze batch-dim (DataLoader wraps single-item batch_sampler output)
            scene_images = obj["images"]  # [1, 6, C, H, W]
            if scene_images.dim() == 5:
                scene_images = scene_images.squeeze(0)  # [6, C, H, W]
            scene_images = scene_images.to(args.device, non_blocking=True)

            scene_face_ids = obj["face_ids"]  # [1, 6] or [6]
            if isinstance(scene_face_ids, torch.Tensor) and scene_face_ids.dim() == 2:
                scene_face_ids = scene_face_ids.squeeze(0)  # [6]

            scene_id = (
                obj["scene_id"][0]
                if isinstance(obj["scene_id"], (list, tuple))
                else obj["scene_id"]
            )
            scene_prompt = (
                obj["prompt"][0]
                if isinstance(obj["prompt"], (list, tuple))
                else obj["prompt"]
            )

            # ── Assertions ──────────────────────────────────────────────────────
            assert scene_images.shape[0] == 6, (
                f"[cubemap accum] Expected 6 faces, got {scene_images.shape[0]}"
            )
            assert len(scene_face_ids) == 6, (
                f"[cubemap accum] Expected 6 face_ids, got {len(scene_face_ids)}"
            )

            # Build groups and validate design invariants
            _groups = _make_cubemap_groups()  # [set1_names, set2_names]
            _validate_cubemap_groups(_groups)
            _loss_divisors = _cubemap_loss_divisors()  # [2.0, 2.0]

            # Verify front+back duplication (anchor weighting assertion)
            for _anchor in ("front", "back"):
                assert _anchor in _groups[0] and _anchor in _groups[1], (
                    f"[cubemap accum] Anchor '{_anchor}' missing from a group — "
                    "breaks intentional anchor weighting"
                )

            assert len(_groups) == 2, (
                f"[cubemap accum] Expected exactly 2 groups, got {len(_groups)}"
            )
            assert [sorted(g) for g in _groups] == [
                sorted(_CUBEMAP_SET1),
                sorted(_CUBEMAP_SET2),
            ], "[cubemap accum] Groups do not match CUBEMAP_SET1/SET2"

            # Map face name → position index in scene_images (canonical order)
            _face_name_to_pos = {name: i for i, name in enumerate(_CANONICAL_FACES)}

            # Lazy-init FaceIdEmbedding (persisted on args to avoid re-creating)
            if not hasattr(args, "_cubemap_face_id_embed"):
                _embed_dim = getattr(args, "cubemap_face_id_embed_dim", 3)
                args._cubemap_face_id_embed = _FaceIdEmbedding(embed_dim=_embed_dim).to(
                    args.device
                )

            # Per-update visualization bookkeeping
            n_batch_saves = 5
            save_interval = max(1, updates_per_epoch // n_batch_saves)
            is_first_update_this_ep = update_step == ep * updates_per_epoch
            do_save = (update_step % save_interval == 0) or is_first_update_this_ep

            update_dir = os.path.join(
                args.local_out_dir_path,
                "train_batch_vis",
                f"update_{update_step:07d}",
            )
            if do_save and dist.is_master():
                os.makedirs(update_dir, exist_ok=True)

            base_stepping = True
            stepping = False
            grad_norm = scale_log2 = None
            group_records = []

            for _idx_i, (_group_names, _loss_divisor) in enumerate(
                zip(_groups, _loss_divisors)
            ):
                _pos_indices = [_face_name_to_pos[n] for n in _group_names]
                _pos_tensor = torch.tensor(_pos_indices, device=args.device)

                inp_grp = scene_images[_pos_tensor]  # [4, C, H, W]
                B_grp = inp_grp.shape[0]
                label_grp = torch.tensor(
                    [args.default_label] * B_grp, device=args.device
                )

                # Text embeddings for group
                _group_prompts = [scene_prompt] * B_grp
                prompt_embeds_grp = text_enc.extract_text_features(_group_prompts)

                # face_id embeddings as conditioning signal (cam_dir slot)
                _face_ids_grp = torch.tensor(
                    [_FACE_TO_IDX[n] for n in _group_names],
                    device=args.device,
                    dtype=torch.long,
                )
                cam_dir_grp = args._cubemap_face_id_embed(
                    _face_ids_grp
                )  # [4, embed_dim]
                # Ensure unit-norm camera directions for camera_system assertions
                cam_dir_grp = F.normalize(cam_dir_grp, dim=-1, eps=1e-6)

                if do_save and dist.is_master():
                    _set_label = "set1" if _idx_i == 0 else "set2"
                    _fname = f"{_set_label}_faces_{'_'.join(_group_names)}.png"
                    save_training_batch_images(
                        inp_grp,
                        {
                            "prompt": _group_prompts,
                            "fov_deg": None,
                            "cam_dir": None,
                        },
                        ep,
                        it,
                        update_step,
                        args.local_out_dir_path,
                        view_ids=_pos_indices,
                        gather=False,
                        update_dir=update_dir,
                        filename=_fname,
                    )
                    group_records.append(
                        {
                            "set": _set_label,
                            "face_names": list(_group_names),
                            "face_positions": _pos_indices,
                        }
                    )

                stepping = base_stepping and (_idx_i == len(_groups) - 1)
                grad_norm, scale_log2 = trainer.train_step(
                    it=it,
                    g_it=update_step,
                    stepping=stepping,
                    metric_lg=me,
                    tb_lg=tb_lg,
                    inp_B3HW=inp_grp,
                    label_B=label_grp,
                    prompt_embeds=prompt_embeds_grp,
                    prog_si=prog_si,
                    prog_wp_it=args.pgwp * iters_train,
                    cam_dir=cam_dir_grp,
                    args=args,
                    fov_deg=getattr(args, "fov_deg", 90.0),
                    view_idx=_face_ids_grp,
                    loss_divisor=_loss_divisor,
                )

            # Write meta.json after both groups processed (optimizer step)
            if stepping and do_save and dist.is_master():
                _ws = dist.get_world_size() if dist.initialized() else 1
                _meta = {
                    "mode": "cubemap",
                    "optimizer_step": update_step,
                    "microbatch": int(it),
                    "scene_id": str(scene_id),
                    "prompt": str(scene_prompt),
                    "world_size": _ws,
                    "canonical_face_order": list(_CANONICAL_FACES),
                    "grouped_faces": {
                        "set1": list(_CUBEMAP_SET1),
                        "set2": list(_CUBEMAP_SET2),
                    },
                    "duplication_summary": _DUPLICATION_SUMMARY,
                    "group_records": group_records,
                    "note": (
                        "front and back appear in set1 AND set2. "
                        "This is intentional anchor weighting, NOT a bug."
                    ),
                }
                with open(os.path.join(update_dir, "meta.json"), "w") as f:
                    json.dump(_meta, f, indent=2)

            if stepping:
                update_step += 1
            step_cnt += int(stepping)

        else:
            # Save 3 training batch visualizations per epoch (evenly spaced)
            # All ranks must participate to avoid deadlock in collective operations
            n_batch_saves = 5
            batch_save_interval = max(1, iters_train // n_batch_saves)
            if (
                it % batch_save_interval == 0
                and it // batch_save_interval < n_batch_saves
            ):
                save_training_batch_images(
                    inp, obj, ep, it, g_it, args.local_out_dir_path
                )

            stepping = (g_it + 1) % args.ac == 0
            if stepping:
                update_step += 1
            step_cnt += int(stepping)

            grad_norm, scale_log2 = trainer.train_step(
                it=it,
                g_it=update_step,
                stepping=stepping,
                metric_lg=me,
                tb_lg=tb_lg,
                inp_B3HW=inp,
                label_B=label,
                prompt_embeds=prompt_embeds,
                prog_si=prog_si,
                prog_wp_it=args.pgwp * iters_train,
                cam_dir=cam_dir,
                args=args,
                fov_deg=fov_deg,
                view_idx=obj.get("view_idx", None),
            )

        me.update(tlr=max_tlr)
        tb_lg.set_step(step=update_step)
        tb_lg.update(head="AR_opt_lr/lr_min", sche_tlr=min_tlr)
        tb_lg.update(head="AR_opt_lr/lr_max", sche_tlr=max_tlr)
        tb_lg.update(head="AR_opt_wd/wd_max", sche_twd=max_twd)
        tb_lg.update(head="AR_opt_wd/wd_min", sche_twd=min_twd)
        tb_lg.update(head="AR_opt_grad/fp16", scale_log2=scale_log2)

        if args.tclip > 0:
            tb_lg.update(head="AR_opt_grad/grad", grad_norm=grad_norm)
            tb_lg.update(head="AR_opt_grad/grad", grad_clip=args.tclip)

        # Save mid-epoch checkpoint keyed to optimizer update_step
        # Force-save on first update; then every ckpt_save_it updates
        is_first_update = update_step == ep * updates_per_epoch + 1
        if (update_step > 0 and update_step % ckpt_save_it == 0) or is_first_update:
            if dist.is_master():
                local_out_ckpt = os.path.join(
                    args.local_out_dir_path, f"ar-ckpt-iter{update_step}.pth"
                )
                torch.save(
                    {
                        "epoch": ep,
                        "iter": it,
                        "global_iter": update_step,
                        "trainer": trainer.state_dict(),
                    },
                    local_out_ckpt,
                )

                # Keep only last 3 mid-epoch checkpoints (delete old ones)
                import glob

                ckpt_pattern = os.path.join(
                    args.local_out_dir_path, "ar-ckpt-iter*.pth"
                )
                all_ckpts = sorted(
                    glob.glob(ckpt_pattern),
                    key=lambda x: int(x.split("iter")[-1].split(".pth")[0]),
                )
                if len(all_ckpts) > 3:
                    for old_ckpt in all_ckpts[:-3]:
                        os.remove(old_ckpt)

                trainer.inference_pic(
                    args,
                    text_enc,
                    g_it=update_step,
                    top_k=600,
                    top_p=0.8,
                    w_mask=False,
                    tb_lg=tb_lg,
                )
                torch.cuda.empty_cache()

        # Update iteration timing
        iter_time.update(time.time() - iter_end_t)
        iter_end_t = time.time()

        # Print training progress every M iterations (rank 0 only)
        if dist.is_local_master() and (
            ((it + 1) % log_every_it == 0) or ((it + 1) == iters_train)
        ):
            avg_it_sec = iter_time.avg if iter_time.count > 0 else 0.0

            # remaining iterations
            remain_it_ep = iters_train - (it + 1)
            remain_it_total = (args.ep - ep - 1) * iters_train + remain_it_ep

            # ETA
            eta_ep_sec = remain_it_ep * avg_it_sec
            eta_total_sec = remain_it_total * avg_it_sec

            eta_ep_str = str(datetime.timedelta(seconds=int(eta_ep_sec)))
            eta_total_str = str(datetime.timedelta(seconds=int(eta_total_sec)))
            finish_time_str = time.strftime(
                "%Y-%m-%d %H:%M", time.localtime(time.time() + eta_total_sec)
            )

            print(
                f"{header} "
                f"it {it + 1}/{iters_train} | "
                f"Lm {me.meters['Lm'].median:.3f} | "
                f"Lt {me.meters['Lt'].median:.3f} | "
                f"Accm {me.meters['Accm'].median:.2f} | "
                f"Acct {me.meters['Acct'].median:.2f} | "
                f"lr {max_tlr:.2e} | "
                f"{iter_time.avg:.3f}s/it | "
                f"data {data_time.avg:.3f}s | "
                f"ETA(ep) {eta_ep_str} | "
                f"ETA(total) {eta_total_str} | "
                f"finish {finish_time_str}",
                flush=True,
            )

        # Break if we've completed all iterations
        if it + 1 >= iters_train:
            break

    # Print epoch summary (only on rank 0)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    me.synchronize_between_processes()

    if dist.is_local_master():
        print(
            f"{header} Completed - "
            f"Lm: {me.meters['Lm'].global_avg:.3f}, "
            f"Lt: {me.meters['Lt'].global_avg:.3f}, "
            f"Accm: {me.meters['Accm'].global_avg:.2f}, "
            f"Acct: {me.meters['Acct'].global_avg:.2f}, "
            f"Time: {total_time_str} ({total_time / iters_train:.3f} s/it)",
            flush=True,
        )

    return {
        k: meter.global_avg for k, meter in me.meters.items()
    }, iter_time.time_preds(
        max_updates - (update_step + 1) + (args.ep - ep) * 15
    )  # +15: other cost


class NullDDP(torch.nn.Module):
    def __init__(self, module, *args, **kwargs):
        super(NullDDP, self).__init__()
        self.module = module
        self.require_backward_grad_sync = False

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


if __name__ == "__main__":
    try:
        main_training()
    finally:
        dist.finalize()
        if isinstance(sys.stdout, misc.SyncPrint) and isinstance(
            sys.stderr, misc.SyncPrint
        ):
            sys.stdout.close(), sys.stderr.close()
