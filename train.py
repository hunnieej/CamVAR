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

# Fix CUDA multiprocessing issue - must be set before CUDA initialization
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass  # Already set

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# build models
from torch.nn.parallel import DistributedDataParallel as DDP
from models import (
    VAR,
    VQVAE,
    build_vae_var,
    build_var,
    build_vae_var_with_ray_adaptation,
)
from models.modified_var import ModifiedVAR
from trainer import VARTrainer
from utils.amp_sc import AmpOptimizer
from utils.lr_control import filter_params

import dist
from utils import arg_util, misc
from utils.wandb_logger import init_wandb_logger
from dataset.data import (
    build_dataset,
    build_dataset_webtar,
    build_erp_dataset,
    gather_file_keys,
)
from models.text_encoder import build_text
from utils.data_sampler import DistInfiniteBatchSampler, EvalDistributedSampler
from utils.misc import auto_resume
from utils.weights import zero_out_cross_attention_weights, apply_lvl_emb_and_pos_1LC
import torchvision


def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None  # Handle empty batch case
    return torch.utils.data.dataloader.default_collate(batch)


def save_training_batch_images(inp_B3HW, obj, ep, it, g_it, save_dir):
    """Save a batch of training input images with camera direction annotations.

    Args:
        inp_B3HW: Input image tensor (B, 3, H, W) in [-1, 1] range.
        obj: Data batch dict containing 'cam_dir', 'prompt', 'fov_deg', etc.
        ep: Current epoch.
        it: Current iteration within epoch.
        g_it: Global iteration.
        save_dir: Root output directory.
    """
    import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
    import PIL.ImageFont as PImageFont

    batch_dir = os.path.join(save_dir, "train_batch_vis")
    os.makedirs(batch_dir, exist_ok=True)

    # Denormalize from [-1, 1] to [0, 255]
    imgs = inp_B3HW.detach().cpu().float()
    imgs = (imgs + 1.0) * 0.5  # [-1,1] -> [0,1]
    imgs = imgs.clamp(0, 1)

    B = imgs.shape[0]
    H, W = imgs.shape[2], imgs.shape[3]

    # Extract camera info
    cam_dir_batch = obj.get("cam_dir", None)
    prompts = obj.get("prompt", [""] * B)
    fov_deg = obj.get("fov_deg", None)

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
        if fov_deg is not None:
            fov_val = (
                fov_deg[i].item() if isinstance(fov_deg, torch.Tensor) else fov_deg
            )
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

    # Make grid and save
    grid = torchvision.utils.make_grid(
        annotated_imgs, nrow=min(B, 4), padding=4, pad_value=1.0
    )
    grid_np = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    filename = f"ep{ep}_iter{it}_git{g_it}.png"
    PImage.fromarray(grid_np).save(os.path.join(batch_dir, filename))
    print(f"     [train_batch_vis] Saved {filename} ({B} images)")


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
    print(f"initial args:\n{str(args)}")

    # build data
    if not args.local_debug:
        print(f"[build PT data] ...\n")

        # Check if using ERP dataset
        dataset_type = getattr(args, "dataset_type", None)

        if dataset_type == "erp":
            print("[using ERP dataset for ray adaptation] ...\n")
            dataset_train, dataset_val = build_erp_dataset(args)
            types = str((type(dataset_train).__name__, type(dataset_val).__name__))

            # Use standard DataLoader for ERP dataset
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
            # for idx,item in enumerate(ld_train):
            #     print(f"Rank {dist.get_local_rank()}: Prompt={item['prompt']}, File Key={item['file_key']}, url={item['url']}")
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
                generator=args.get_different_generator_for_each_rank(),  # worker_init_fn=worker_init_fn,
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

        num_classes = 1000

        [print(line) for line in auto_resume_info]
        print(f"[dataloader multi processing] ...", end="", flush=True)
        try:
            stt = time.time()
            if dataset_type == "erp":
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
            print(
                f"[dataloader] gbs={args.glb_batch_size}, lbs={args.batch_size}, iters_train={iters_train}, types(tr, va)={types}"
            )
        except:
            print(sys.exc_info())

    else:
        num_classes = 1000
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

        print(
            f"[RAY ADAPTATION] patch_nums: {ray_patch_nums} (L={sum(pn**2 for pn in ray_patch_nums)})"
        )
        print(f"[RAY ADAPTATION] adapter_dim: {adapter_dim}")
        print(f"[RAY ADAPTATION] num_memory_tokens: {num_memory_tokens}")
        print(f"[RAY ADAPTATION] ray_adapter_num_heads: {ray_adapter_num_heads}\n")

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
    if trainer_state is not None and len(trainer_state):
        print("unsing strict=False in loading...")
        new_state_dict = apply_lvl_emb_and_pos_1LC(
            args, state_dict=trainer_state["var_wo_ddp"], patch_nums=args.patch_nums
        )
        missing, unexpected = var_wo_ddp.load_state_dict(new_state_dict, strict=False)
        print("checkpoints incompatible: ", missing, unexpected)

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

    print(f"[INIT] VAR model = {var_wo_ddp}\n\n")
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
    )
    # if trainer_state is not None and len(trainer_state):
    #     print('unsing strict=False in loading...')
    #     missing,unexpected=trainer.load_state_dict(trainer_state, strict=False, skip_vae=True) # don't load vae again
    #     print('checkpoints incompatible: ',missing,unexpected)

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

    for name, param in text_encoder.named_parameters():
        print(f"Parameter: {name}, Requires Grad: {param.requires_grad}")
    for name, param in trainer.vae_local.named_parameters():
        print(f"Parameter: {name}, Requires Grad: {param.requires_grad}")
    for name, param in trainer.var_wo_ddp.named_parameters():
        print(f"Parameter: {name}, Requires Grad: {param.requires_grad}")

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

    # Checkpoint saving setup
    ckpt_save_it = getattr(args, "ckpt_save_it", 150000)
    print(f"[CHECKPOINT SAVING] Saving checkpoints every {ckpt_save_it} iterations")

    print(f"===========> main training start")
    for ep in range(start_ep, args.ep):
        if hasattr(ld_train, "sampler") and hasattr(ld_train.sampler, "set_epoch"):
            print(f"has attr sampler")
            ld_train.sampler.set_epoch(ep)
            if ep < 3:
                print(f"ep < 3")
                # noinspection PyArgumentList
                print(
                    f"[{type(ld_train).__name__}] [ld_train.sampler.set_epoch({ep})]",
                    flush=True,
                    force=True,
                )
        print(f"===========> in main training")
        tb_lg.set_step(ep * iters_train)
        print(f"===========> epoch:{ep}, before eval")
        if ep > 0:
            val_loss_mean, val_loss_tail, val_acc_mean, val_acc_tail, tot, cost = (
                trainer.eval_ep(args, ld_val, text_encoder)
            )
        print(f"===========> epoch:{ep}, after eval")
        # def train_one_ep(ep: int, is_first_ep: bool, start_it: int, args: arg_util.Args, tb_lg: misc.TensorboardLogger, ld_or_itrt, iters_train: int, trainer):
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
        print(f"===========> epoch:{ep}, after eval")

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
                    cur_ep=ep,
                    cur_iter=-1,
                    top_k=600,
                    top_p=0.8,
                    w_mask=False,
                )
                torch.cuda.empty_cache()
                print(
                    f"     [saving ckpt](*) finished!  @ {local_out_ckpt}",
                    flush=True,
                    clean=True,
                )

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
                    f"     [✅ NEW BEST] Saved best checkpoint @ {local_out_ckpt_best} (val_loss_tail={val_loss_tail:.4f})"
                )

            # Early stopping check
            if val_loss_tail < best_val_loss_for_early_stop:
                best_val_loss_for_early_stop = val_loss_tail
                early_stop_counter = 0
                print(
                    f"     [✅ IMPROVEMENT] Validation loss improved: {val_loss_tail:.4f}"
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
        tb_lg.update(head="AR_ep_loss", step=ep + 1, **AR_ep_loss)
        tb_lg.update(
            head="AR_z_burnout", step=ep + 1, rest_hours=round(sec / 60 / 60, 2)
        )
        args.dump_log()
        tb_lg.flush()

    total_time = f"{(time.time() - start_time) / 60 / 60:.1f}h"
    print("\n\n")
    print(
        f"  [*] [PT finished]  Total cost: {total_time},   Lm: {best_L_mean:.3f} ({L_mean}),   Lt: {best_L_tail:.3f} ({L_tail})"
    )
    print("\n\n")

    # Clean up variables if they exist
    try:
        del stats
    except NameError:
        pass
    del iters_train, ld_train
    time.sleep(3), gc.collect(), torch.cuda.empty_cache(), time.sleep(3)

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
    print(f"===========> epoch:{ep}, train_one_ep")
    from trainer import VARTrainer
    from utils.lr_control import lr_wd_annealing

    trainer: VARTrainer

    step_cnt = 0
    me = misc.MetricLogger(delimiter="  ")
    me.add_meter("tlr", misc.SmoothedValue(window_size=1, fmt="{value:.2g}"))
    me.add_meter("tnm", misc.SmoothedValue(window_size=1, fmt="{value:.2f}"))
    # if args.using_webtar:
    #     me.add_meter('data_cnt', misc.SmoothedValue(window_size=1, fmt='{value:.2f}'))
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

    if args.using_webtar:
        all_file_keys = []
    for it, obj in me.log_every(
        start_it=start_it,
        max_iters=iters_train,
        itrt=ld_or_itrt,
        print_freq=math.floor(iters_train / args.print_every),
        header=header,
    ):
        # Compute global iteration for checkpoint saving
        g_it = ep * iters_train + it

        # Save checkpoint every ckpt_save_it global iterations
        if g_it > 0 and g_it % ckpt_save_it == 0:
            if dist.is_local_master():
                local_out_ckpt = os.path.join(
                    args.local_out_dir_path, f"ar-ckpt-iter{g_it}.pth"
                )
                print(f"[saving ckpt @ {g_it} iters] ...", end="", flush=True)
                torch.save(
                    {
                        "epoch": ep,
                        "iter": it,
                        "global_iter": g_it,
                        "trainer": trainer.state_dict(),
                    },
                    local_out_ckpt,
                )

                # Keep only last 3 checkpoints (delete old ones)
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
                        print(
                            f"     [cleanup] Removed old checkpoint: {os.path.basename(old_ckpt)}"
                        )

                trainer.inference_pic(
                    args,
                    text_enc,
                    cur_ep=ep,
                    cur_iter=it,
                    top_k=600,
                    top_p=0.8,
                    w_mask=False,
                )
                torch.cuda.empty_cache()
                print(
                    f"     [saving ckpt](*) finished!  @ {local_out_ckpt}",
                    flush=True,
                    clean=True,
                )

        if it < start_it:
            continue
        if is_first_ep and it == start_it:
            warnings.resetwarnings()

        # (inp, label)
        # if args.using_webtar:
        #     file_keys=obj['file_key']
        #     all_file_keys.extend(gather_file_keys(file_keys))
        #     # print('rank=',dist.get_rank(),file_keys)
        #     if dist.get_rank()==0:
        #         # print('total=', len(set(all_file_keys)),' file_keys')
        #         me.update(data_cnt=len(set(all_file_keys)))
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

        # Save 3 training batch visualizations per epoch (evenly spaced)
        n_batch_saves = 3
        batch_save_interval = max(1, iters_train // n_batch_saves)
        if (
            dist.is_local_master()
            and it % batch_save_interval == 0
            and it // batch_save_interval < n_batch_saves
        ):
            save_training_batch_images(inp, obj, ep, it, g_it, args.local_out_dir_path)

        args.cur_it = f"{it + 1}/{iters_train}"

        wp_it = args.wp * iters_train
        min_tlr, max_tlr, min_twd, max_twd = lr_wd_annealing(
            args.sche,
            trainer.var_opt.optimizer,
            args.tlr,
            args.twd,
            args.twde,
            g_it,
            wp_it,
            max_it,
            wp0=args.wp0,
            wpe=args.wpe,
        )
        # warmup:迭代次数<wp_it时，学习率从wp0线性增加到1 (乘peak_lr=1e-5)
        args.cur_lr, args.cur_wd = max_tlr, max_twd

        if args.pg:  # default: 0.0, no progressive training, won't get into this
            if g_it <= wp_it:
                prog_si = args.pg0  # warmup阶段prog_si是默认值pg0=4
            elif g_it >= max_it * args.pg:
                prog_si = (
                    len(args.patch_nums) - 1
                )  # iter大于args.pg指定的iter时，prog_si是scale个数
            else:
                delta = len(args.patch_nums) - 1 - args.pg0
                progress = min(
                    max((g_it - wp_it) / (max_it * args.pg - wp_it), 0), 1
                )  # from 0 to 1
                prog_si = args.pg0 + round(
                    progress * delta
                )  # from args.pg0 to len(args.patch_nums)-1
        else:
            prog_si = -1  # prog_si似乎是指定不同训练stage focus在不同scale上

        stepping = (g_it + 1) % args.ac == 0
        step_cnt += int(stepping)

        grad_norm, scale_log2 = trainer.train_step(
            it=it,
            g_it=g_it,
            stepping=stepping,
            metric_lg=me,
            tb_lg=tb_lg,
            inp_B3HW=inp,
            label_B=label,
            prompt_embeds=prompt_embeds,
            prog_si=prog_si,
            prog_wp_it=args.pgwp * iters_train,
            cam_dir=cam_dir,
            fov_deg=fov_deg,
        )

        me.update(tlr=max_tlr)
        tb_lg.set_step(step=g_it)
        tb_lg.update(head="AR_opt_lr/lr_min", sche_tlr=min_tlr)
        tb_lg.update(head="AR_opt_lr/lr_max", sche_tlr=max_tlr)
        tb_lg.update(head="AR_opt_wd/wd_max", sche_twd=max_twd)
        tb_lg.update(head="AR_opt_wd/wd_min", sche_twd=min_twd)
        tb_lg.update(head="AR_opt_grad/fp16", scale_log2=scale_log2)

        if args.tclip > 0:
            tb_lg.update(head="AR_opt_grad/grad", grad_norm=grad_norm)
            tb_lg.update(head="AR_opt_grad/grad", grad_clip=args.tclip)
            # t_ratio = 1 if grad_norm is None else min(1.0, args.tclip / (grad_norm + 1e-7))
            # tb_lg.update(head='AR_opt_lr/lr_max', actu_tlr=t_ratio*max_tlr)
            # tb_lg.update(head='AR_opt_lr/lr_min', actu_tlr=t_ratio*min_tlr)

    me.synchronize_between_processes()
    return {
        k: meter.global_avg for k, meter in me.meters.items()
    }, me.iter_time.time_preds(
        max_it - (g_it + 1) + (args.ep - ep) * 15
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
