import argparse
import argparse
import json
import os
from pathlib import Path
import time

import torch
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None

from data.cubemap_scene_dataset import CubemapSceneDataset
from models.var_drop_faceray import build_vae_var_faceray
from models.text_encoder import build_text


def load_var_checkpoint(model, ckpt_path: str) -> None:
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "trainer" in state:
        state = state["trainer"]["var_wo_ddp"]
    model.load_state_dict(state, strict=False)


def run_geometry_checks(model) -> None:
    ortho = model.faceray_geom.orthonormality_check(model.faceray_precompute)
    proj = model.faceray_geom.projection_consistency_check(sample_count=512)
    print(f"[geometry] orthonormality max |R^T R - I|: {ortho:.6f}")
    print(f"[geometry] projection stats: {proj}")
    if proj["mapped_same_face"] > 0.1:
        print("[warning] High mapping to same face for boundary tokens")
    if proj["u_out_of_range"] > 0.01 or proj["v_out_of_range"] > 0.01:
        print("[warning] Projection outside [-1,1] for boundary tokens")


def init_wandb(args, extra_config: dict):
    if not args.wandb or wandb is None:
        return None
    run_name = args.wandb_run_name
    if run_name is None:
        run_name = f"train_sampler-{time.strftime('%Y%m%d-%H%M%S')}"
    config = vars(args).copy()
    config.update(extra_config)
    settings = wandb.Settings()
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config=config,
        settings=settings,
    )


def load_fixed_prompts(args, dataset):
    if args.wandb_fixed_prompt_file:
        lines = (
            Path(args.wandb_fixed_prompt_file).read_text(encoding="utf-8").splitlines()
        )
        prompts = [line.strip() for line in lines if line.strip()]
    else:
        prompts = [
            "A cozy mountain cabin at sunset",
            "A futuristic city skyline at night",
            "A calm tropical beach with palm trees",
            "A snowy forest with soft light",
            "A vibrant street market",
            "A serene lake with reflections",
            "A desert canyon at golden hour",
            "A modern living room interior",
        ]
    prompts = prompts[: args.wandb_num_fixed_prompts]
    fixed_indices = list(range(min(len(dataset), len(prompts))))
    return prompts, fixed_indices


def make_faces_grid(faces: torch.Tensor) -> Image.Image:
    faces = faces.detach().cpu().numpy()
    faces = (faces * 255.0).clip(0, 255).astype(np.uint8)
    faces = faces.transpose(0, 2, 3, 1)
    h, w = faces.shape[1], faces.shape[2]
    grid = Image.new("RGB", (w * 3, h * 2))
    order = [0, 1, 2, 3, 4, 5]
    for idx, face_idx in enumerate(order):
        row = idx // 3
        col = idx % 3
        grid.paste(Image.fromarray(faces[face_idx]), (col * w, row * h))
    return grid


def make_dice_image(faces: torch.Tensor) -> Image.Image:
    faces = faces.detach().cpu().numpy()
    faces = (faces * 255.0).clip(0, 255).astype(np.uint8)
    faces = faces.transpose(0, 2, 3, 1)
    h, w = faces.shape[1], faces.shape[2]
    canvas = Image.new("RGB", (w * 4, h * 3))
    face_map = {"U": 4, "L": 3, "F": 0, "R": 1, "B": 2, "D": 5}
    positions = {
        "U": (w, 0),
        "L": (0, h),
        "F": (w, h),
        "R": (2 * w, h),
        "B": (3 * w, h),
        "D": (w, 2 * h),
    }
    for face_name, idx in face_map.items():
        canvas.paste(Image.fromarray(faces[idx]), positions[face_name])
    return canvas


def boundary_mae_pixels(faces: torch.Tensor, strip_w: int = 2) -> float:
    faces = faces.detach().cpu().float()
    f, r, b, l, u, d = faces
    edges = [
        (f[:, :, -strip_w:], r[:, :, :strip_w]),
        (f[:, :, :strip_w], l[:, :, -strip_w:]),
        (f[:, :strip_w, :], u[:, -strip_w:, :]),
        (f[:, -strip_w:, :], d[:, :strip_w, :]),
    ]
    diffs = [(a - b).abs().mean().item() for a, b in edges]
    return float(sum(diffs) / max(len(diffs), 1))


def find_latest_checkpoint(ckpt_dir: str, pattern: str) -> str:
    root = Path(ckpt_dir)
    if not root.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    candidates = sorted(
        root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints found in {ckpt_dir} with pattern {pattern}"
        )
    return str(candidates[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="dataset/cubemap/train")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--vae_ckpt", type=str, default="ckpt/vae_ch160v4096z32.pth")
    parser.add_argument(
        "--gen_ckpt",
        type=str,
        default=None,
        help="Stage-1 generator checkpoint with FaceRayAdapter",
    )
    parser.add_argument(
        "--use_best_gen_ckpt",
        action="store_true",
        help="Prefer best generator checkpoint if available",
    )
    parser.add_argument(
        "--gen_ckpt_dir",
        type=str,
        default="ckpt/faceray_stage1",
        help="Directory to search for latest generator checkpoint",
    )
    parser.add_argument(
        "--gen_ckpt_pattern",
        type=str,
        default="*.pth",
        help="Filename pattern for generator checkpoints",
    )
    parser.add_argument(
        "--gen_best_name",
        type=str,
        default="faceray_stage1_generator_best.pth",
        help="Best checkpoint filename to prefer",
    )
    parser.add_argument("--depth", type=int, default=30)
    parser.add_argument(
        "--patch_nums",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 6, 9, 13, 18, 24, 32],
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--text_encoder_ckpt", type=str, default="ckpt/CLIP")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--wandb", action="store_true", default=True)
    parser.add_argument("--wandb_project", type=str, default="STAR-FaceRay")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_log_every", type=int, default=10)
    parser.add_argument("--wandb_image_every", type=int, default=1000)
    parser.add_argument("--wandb_metric_every", type=int, default=500)
    parser.add_argument("--wandb_num_fixed_prompts", type=int, default=8)
    parser.add_argument("--wandb_fixed_prompt_file", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)

    if wandb is None:
        args.wandb = False

    dataset = CubemapSceneDataset(args.data_root)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )

    vae_local, var_model = build_vae_var_faceray(
        device=device,
        patch_nums=args.patch_nums,
        depth=args.depth,
        enable_logit_norm=True,
        enable_adaptive_norm=True,
        train_mode="all",
    )

    vae_local.load_state_dict(
        torch.load(args.vae_ckpt, map_location="cpu"), strict=True
    )
    gen_ckpt = args.gen_ckpt
    if gen_ckpt is None:
        best_candidate = Path(args.gen_ckpt_dir) / args.gen_best_name
        if args.use_best_gen_ckpt and best_candidate.exists():
            gen_ckpt = str(best_candidate)
            print(f"[sampler] Using best generator checkpoint: {gen_ckpt}")
        else:
            gen_ckpt = find_latest_checkpoint(args.gen_ckpt_dir, args.gen_ckpt_pattern)
            print(f"[sampler] Using latest generator checkpoint: {gen_ckpt}")
    load_var_checkpoint(var_model, gen_ckpt)

    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    text_encoder, _ = build_text(pretrained_path=args.text_encoder_ckpt, device=device)
    text_encoder.eval()

    for p in var_model.parameters():
        p.requires_grad_(False)
    if hasattr(var_model, "feat_extract_blocks"):
        for p in var_model.feat_extract_blocks.parameters():
            p.requires_grad_(True)
    for name in ("head_logits2", "encoder_proj2", "head_proj", "lvl_embed_2"):
        if hasattr(var_model, name):
            for p in getattr(var_model, name).parameters():
                p.requires_grad_(True)
    for adapter in var_model.faceray_sampler_adapters:
        for p in adapter.parameters():
            p.requires_grad_(True)

    optimizer = torch.optim.AdamW(
        [p for p in var_model.parameters() if p.requires_grad], lr=args.lr
    )
    scaler = GradScaler("cuda", enabled=args.amp)

    run_geometry_checks(var_model)

    extra_config = {
        "patch_nums": args.patch_nums,
        "accum_steps": args.accum_steps,
    }
    run = init_wandb(args, extra_config)

    fixed_prompts, fixed_indices = load_fixed_prompts(args, dataset)
    fixed_faces = [dataset[idx]["faces"] for idx in fixed_indices]

    step = 0
    loader_iter = iter(loader)
    progress = tqdm(total=args.num_steps, desc="train_sampler", unit="step")
    optimizer.zero_grad(set_to_none=True)
    while step < args.num_steps:
        micro_start = time.perf_counter()
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        faces = batch["faces"].to(device)
        prompts = batch["prompt"]
        bsz = faces.shape[0]
        faces = faces.view(bsz * 6, 3, faces.shape[-2], faces.shape[-1])

        with torch.no_grad():
            prompt_embeds, prompt_attention_mask, pooled_embed = (
                text_encoder.extract_text_features(list(prompts))
            )
        prompt_embeds = prompt_embeds.repeat_interleave(6, dim=0)
        prompt_attention_mask = prompt_attention_mask.repeat_interleave(6, dim=0)
        pooled_embed = pooled_embed.repeat_interleave(6, dim=0)

        gt_idx_Bl = vae_local.img_to_idxBl(faces)
        gt_last = gt_idx_Bl[-1]
        embed_last = vae_local.quantize.embedding(gt_last)
        embed_last = embed_last.view(bsz, 6, gt_last.shape[1], embed_last.shape[-1])
        embed_last = embed_last.reshape(bsz, -1, embed_last.shape[-1])

        x_BLCv_wo_first_l = vae_local.quantize.idxBl_to_var_input(gt_idx_Bl)

        with autocast("cuda", enabled=args.amp):
            logits, mask = var_model.forward_sampler(
                x_BLCv_wo_first_l=x_BLCv_wo_first_l,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_attention_mask,
                encoder_pool_feat=pooled_embed,
                embed_Cvae=embed_last,
                face_group=6,
            )

            gt_last = gt_last.view(bsz, 6, -1).reshape(bsz, -1)
            mask_flat = mask.view(-1)
            logits_flat = logits.view(-1, var_model.V)
            gt_flat = gt_last.view(-1)

            loss = F.cross_entropy(logits_flat[mask_flat], gt_flat[mask_flat])

        micro_time = time.perf_counter() - micro_start

        if run is not None and step % args.wandb_log_every == 0:
            mem_alloc = (
                torch.cuda.max_memory_allocated() / 1e9
                if torch.cuda.is_available()
                else 0.0
            )
            mem_reserved = (
                torch.cuda.max_memory_reserved() / 1e9
                if torch.cuda.is_available()
                else 0.0
            )
            step_time = micro_time * args.accum_steps
            mask_ratio = mask_flat.float().mean().item()
            if mask_flat.any():
                masked_logits = logits_flat[mask_flat]
                masked_gt = gt_flat[mask_flat]
                top1 = (masked_logits.argmax(dim=-1) == masked_gt).float().mean().item()
                top5 = (
                    (masked_logits.topk(5, dim=-1).indices == masked_gt.unsqueeze(-1))
                    .any(dim=-1)
                    .float()
                    .mean()
                    .item()
                )
            else:
                top1 = 0.0
                top5 = 0.0
            wandb.log(
                {
                    "loss/mask": loss.item(),
                    "loss/total": loss.item(),
                    "debug/mask_ratio": mask_ratio,
                    "metric/masked_acc_top1": top1,
                    "metric/masked_acc_top5": top5,
                    "perf/microstep_time_sec": micro_time,
                    "perf/step_time_sec": step_time,
                    "perf/gpu_mem_allocated_gb": mem_alloc,
                    "perf/gpu_mem_reserved_gb": mem_reserved,
                },
                step=step,
            )
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

        if run is not None and step % args.wandb_metric_every == 0:
            with torch.no_grad():
                if fixed_faces:
                    mae = boundary_mae_pixels(fixed_faces[0])
                else:
                    mae = 0.0
            wandb.log(
                {
                    "metric/boundary_mae_sampler_off": mae,
                    "metric/boundary_mae_sampler_on": mae,
                },
                step=step,
            )

        if step == 0:
            block_attn = var_model.feat_extract_blocks[0].attn
            impl = getattr(block_attn, "last_attn_impl", "unknown")
            bias = getattr(block_attn, "last_attn_bias", None)
            print(f"[attn] impl={impl} attn_bias={bias}")

        scaler.scale(loss / args.accum_steps).backward()

        if (step + 1) % args.accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % 50 == 0:
            print(json.dumps({"step": step, "loss": loss.item()}))

        if run is not None and step % args.wandb_image_every == 0:
            with torch.no_grad():
                for idx, faces in enumerate(fixed_faces):
                    grid = make_faces_grid(faces)
                    dice = make_dice_image(faces)
                    wandb.log(
                        {
                            f"samples/sampler_off/faces_grid/{idx}": wandb.Image(
                                grid,
                                caption=(
                                    fixed_prompts[idx]
                                    if idx < len(fixed_prompts)
                                    else ""
                                ),
                            ),
                            f"samples/sampler_off/dice/{idx}": wandb.Image(
                                dice,
                                caption=(
                                    fixed_prompts[idx]
                                    if idx < len(fixed_prompts)
                                    else ""
                                ),
                            ),
                            f"samples/sampler_on/faces_grid/{idx}": wandb.Image(
                                grid,
                                caption=(
                                    fixed_prompts[idx]
                                    if idx < len(fixed_prompts)
                                    else ""
                                ),
                            ),
                            f"samples/sampler_on/dice/{idx}": wandb.Image(
                                dice,
                                caption=(
                                    fixed_prompts[idx]
                                    if idx < len(fixed_prompts)
                                    else ""
                                ),
                            ),
                        },
                        step=step,
                    )

        step += 1
        progress.update(1)

    if step % args.accum_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    progress.close()


if __name__ == "__main__":
    main()
