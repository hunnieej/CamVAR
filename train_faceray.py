import argparse
import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

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
except Exception:  # pragma: no cover - optional dependency
    wandb = None

from data.cubemap_scene_dataset import CubemapSceneDataset
from models.var_drop_faceray import build_vae_var_faceray
from models.text_encoder import build_text


def load_var_checkpoint(model, ckpt_path: str) -> None:
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "trainer" in state:
        state = state["trainer"]["var_wo_ddp"]
    model.load_state_dict(state, strict=False)


def save_var_checkpoint(
    model, ckpt_path: str, step: int, loss: Optional[float] = None
) -> None:
    trainer_state = {
        "var_wo_ddp": model.state_dict(),
        "step": step,
    }
    if loss is not None:
        trainer_state["loss"] = float(loss)
    ckpt = {"trainer": trainer_state}
    torch.save(ckpt, ckpt_path)


def compute_seam_loss(
    feat_last, precompute, tau: float, face_group: int = 6
) -> torch.Tensor:
    b6, n, c = feat_last.shape
    b = b6 // face_group
    feat_last = feat_last.view(b, face_group, n, c)

    device = feat_last.device
    q_faces = precompute.boundary_face_ids_last.to(device)
    q_idx = precompute.boundary_local_indices_last.to(device)
    k_faces = precompute.neighbor_face_ids_last.to(device)
    k_idx = precompute.neighbor_local_indices_last.to(device)
    mask = precompute.neighbor_mask_last.to(device)  # [Q,K], bool

    safe_k_faces = k_faces.clamp(min=0)
    safe_k_idx = k_idx.clamp(min=0)

    q_feat = feat_last[:, q_faces, q_idx, :]  # [B,Q,C]
    k_feat = feat_last[:, safe_k_faces, safe_k_idx, :]  # [B,Q,K,C]

    has_neighbor = mask.any(dim=1)  # [Q]
    mask_b = mask.unsqueeze(0)  # [1,Q,K] broadcast

    q_exp = q_feat.unsqueeze(2).expand_as(k_feat)  # [B,Q,K,C] (no warning)
    diff = F.smooth_l1_loss(q_exp, k_feat, reduction="none").mean(dim=-1)  # [B,Q,K]

    large = torch.tensor(1e6, device=diff.device, dtype=diff.dtype)
    diff = diff + (~mask_b).to(diff.dtype) * large

    weights = torch.softmax(-diff / max(tau, 1e-6), dim=2)  # [B,Q,K]
    loss_t = (weights * diff).sum(dim=2)  # [B,Q]

    valid = has_neighbor.unsqueeze(0).expand(b, -1)  # [B,Q]
    return loss_t[valid].mean()


def run_geometry_checks(model) -> None:
    ortho = model.faceray_geom.orthonormality_check(model.faceray_precompute)
    proj = model.faceray_geom.projection_consistency_check(sample_count=512)
    print(f"[geometry] orthonormality max |R^T R - I|: {ortho:.6f}")
    print(f"[geometry] projection stats: {proj}")
    counts = model.faceray_precompute.boundary_count_per_stage
    print(f"[geometry] boundary counts per stage: {counts}")
    if proj["mapped_same_face"] > 0.1:
        print("[warning] High mapping to same face for boundary tokens")
    if proj["u_out_of_range"] > 0.01 or proj["v_out_of_range"] > 0.01:
        print("[warning] Projection outside [-1,1] for boundary tokens")


def check_adapter_zero_init(model) -> None:
    adapter = model.faceray_adapters[0]
    x = torch.randn(2, 6, model.L, model.C, device=next(model.parameters()).device)
    delta = adapter(x, face_group=6).abs().mean().item()
    print(f"[adapter] zero-init delta mean abs: {delta:.8f}")


def init_wandb(args, extra_config: dict):
    if not args.wandb or wandb is None:
        return None
    run_name = args.wandb_run_name
    if run_name is None:
        run_name = f"train_faceray-{time.strftime('%Y%m%d-%H%M%S')}"
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
        prompts = prompts[: args.wandb_num_fixed_prompts]
        fixed_indices = list(range(min(len(dataset), len(prompts))))
        return prompts, fixed_indices

    fixed_indices = list(range(min(len(dataset), args.wandb_num_fixed_prompts)))
    prompts = []
    for idx in fixed_indices:
        item = dataset[idx]
        prompts.append(item.get("prompt", ""))
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
    face_map = {
        "U": 4,
        "L": 3,
        "F": 0,
        "R": 1,
        "B": 2,
        "D": 5,
    }
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


def adapter_stats(adapter: torch.nn.Module) -> dict:
    params = [p for p in adapter.parameters() if p.requires_grad]
    param_norm = (
        torch.sqrt(sum((p.detach() ** 2).sum() for p in params)).item()
        if params
        else 0.0
    )
    grad_norm = 0.0
    for p in params:
        if p.grad is not None:
            grad_norm += (p.grad.detach() ** 2).sum().item()
    grad_norm = grad_norm**0.5
    out_proj_norm = 0.0
    if hasattr(adapter, "out_proj"):
        out_proj_norm = adapter.out_proj.weight.detach().norm().item()
    return {
        "param_norm": param_norm,
        "grad_norm": grad_norm,
        "out_proj_norm": out_proj_norm,
    }


def boundary_mae_by_edge(feat_last: torch.Tensor, precompute) -> dict:
    b6, n, c = feat_last.shape
    b = b6 // 6
    feat_last = feat_last.view(b, 6, n, c)

    q_faces = precompute.boundary_face_ids_last.to(feat_last.device)
    q_idx = precompute.boundary_local_indices_last.to(feat_last.device)
    neighbor_faces = precompute.neighbor_face_ids_last.to(feat_last.device)
    neighbor_idx = precompute.neighbor_local_indices_last.to(feat_last.device)
    mask = precompute.neighbor_mask_last.to(feat_last.device)

    q_feat = feat_last[:, q_faces, q_idx, :]
    k_feat = feat_last[:, neighbor_faces.clamp(min=0), neighbor_idx.clamp(min=0), :]

    diff = (q_feat.unsqueeze(2) - k_feat).abs().mean(dim=-1)
    diff = diff.masked_fill(~mask.unsqueeze(0), float("inf"))

    edges = [
        ("F-R", 0, 1),
        ("F-L", 0, 3),
        ("F-U", 0, 4),
        ("F-D", 0, 5),
        ("R-B", 1, 2),
        ("L-B", 3, 2),
        ("U-R", 4, 1),
        ("U-L", 4, 3),
        ("D-R", 5, 1),
        ("D-L", 5, 3),
        ("U-B", 4, 2),
        ("D-B", 5, 2),
    ]
    out = {}
    for name, src, tgt in edges:
        q_mask = q_faces.unsqueeze(1) == src
        edge_mask = q_mask & (neighbor_faces == tgt)
        if not edge_mask.any():
            continue
        edge_diff = diff[:, edge_mask, :]
        edge_diff = edge_diff.min(dim=2).values
        out[f"metric/boundary_mae/{name}"] = edge_diff.mean().item()
    return out


def main() -> None:
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="dataset/cubemap/train")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lambda_seam", type=float, default=0.1)
    parser.add_argument("--seam_warmup_ratio", type=float, default=0.10)
    parser.add_argument("--seam_tau", type=float, default=0.2)
    parser.add_argument("--demo_last_scale_only", action="store_true", default=False)
    parser.add_argument("--demo_boundary_only", action="store_true", default=False)
    parser.add_argument("--demo_subset_list", type=str, default="")
    parser.add_argument("--demo_max_steps", type=int, default=5000)
    parser.add_argument("--demo_seam_warmup_ratio", type=float, default=0.05)
    parser.add_argument("--demo_face_loss_weight", type=float, default=1.0)
    parser.add_argument("--demo_seam_loss_weight", type=float, default=0.1)
    parser.add_argument("--vae_ckpt", type=str, default="ckpt/vae_ch160v4096z32.pth")
    parser.add_argument(
        "--var_ckpt",
        type=str,
        default="ckpt/star_rope_d30_512-ar-ckpt-ep1-iter30000.pth",
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
    parser.add_argument("--accum_steps", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="ckpt/faceray_stage1")
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument(
        "--save_name",
        type=str,
        default="faceray_stage1_generator.pth",
    )
    parser.add_argument(
        "--save_best",
        action="store_true",
        help="Save best checkpoint based on training loss",
    )
    parser.add_argument(
        "--save_best_name",
        type=str,
        default="faceray_stage1_generator_best.pth",
    )
    parser.add_argument("--wandb", action="store_true", default=True)
    parser.add_argument("--wandb_project", type=str, default="STAR-FaceRay")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_log_every", type=int, default=10)
    parser.add_argument("--wandb_image_every", type=int, default=100)
    parser.add_argument("--wandb_metric_every", type=int, default=100)
    parser.add_argument("--wandb_num_fixed_prompts", type=int, default=8)
    parser.add_argument("--wandb_fixed_prompt_file", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)

    if wandb is None:
        args.wandb = False

    dataset = CubemapSceneDataset(args.data_root)
    if args.demo_subset_list:
        subset_ids = [
            line.strip()
            for line in Path(args.demo_subset_list)
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        subset_set = set(subset_ids)
        dataset.scene_dirs = [p for p in dataset.scene_dirs if p.name in subset_set]
        if not dataset.scene_dirs:
            raise ValueError("demo_subset_list yielded empty dataset")
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
    load_var_checkpoint(var_model, args.var_ckpt)

    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    text_encoder, _ = build_text(pretrained_path=args.text_encoder_ckpt, device=device)
    text_encoder.eval()

    for p in var_model.parameters():
        p.requires_grad_(False)
    for adapter in var_model.faceray_adapters:
        for p in adapter.parameters():
            p.requires_grad_(True)

    optimizer = torch.optim.AdamW(
        [p for p in var_model.parameters() if p.requires_grad], lr=args.lr
    )
    scaler = GradScaler("cuda", enabled=args.amp)

    run_geometry_checks(var_model)
    neighbor_mask_last = var_model.faceray_precompute.neighbor_mask_last
    avg_neighbors = (
        neighbor_mask_last.float().sum(dim=1).mean().item()
        if neighbor_mask_last.numel()
        else 0.0
    )
    print(f"[seam] avg_neighbors={avg_neighbors:.2f} tau={args.seam_tau}")
    check_adapter_zero_init(var_model)

    extra_config = {
        "lambda_seam": args.lambda_seam,
        "seam_warmup_ratio": args.seam_warmup_ratio,
        "seam_tau": args.seam_tau,
        "strip_w": 2,
        "k": 2,
        "patch_nums": args.patch_nums,
        "accum_steps": args.accum_steps,
    }
    run = init_wandb(args, extra_config)

    fixed_prompts, fixed_indices = load_fixed_prompts(args, dataset)
    fixed_faces = [dataset[idx]["faces"] for idx in fixed_indices]

    is_demo = args.demo_last_scale_only or args.demo_boundary_only
    num_steps = args.num_steps
    seam_warmup_ratio = args.seam_warmup_ratio
    if is_demo:
        num_steps = min(num_steps, args.demo_max_steps)
        seam_warmup_ratio = args.demo_seam_warmup_ratio
        args.wandb_metric_every = max(args.wandb_metric_every, 2000)
        args.wandb_image_every = max(args.wandb_image_every, 2000)

    warmup_steps = max(1, int(num_steps * seam_warmup_ratio))
    step = 0
    best_loss = float("inf")
    loss_value = None
    loader_iter = iter(loader)
    progress = tqdm(total=num_steps, desc="train_faceray", unit="step")
    optimizer.zero_grad(set_to_none=True)
    while step < num_steps:
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
        prompt_list = list(prompts)

        with torch.no_grad():
            prompt_embeds, prompt_attention_mask, pooled_embed = (
                text_encoder.extract_text_features(prompt_list)
            )
        prompt_embeds = prompt_embeds.repeat_interleave(6, dim=0)
        prompt_attention_mask = prompt_attention_mask.repeat_interleave(6, dim=0)
        pooled_embed = pooled_embed.repeat_interleave(6, dim=0)

        gt_idx_Bl = vae_local.img_to_idxBl(faces)
        gt_BL = torch.cat(gt_idx_Bl, dim=1)
        x_BLCv_wo_first_l = vae_local.quantize.idxBl_to_var_input(gt_idx_Bl)

        with autocast("cuda", enabled=args.amp):
            logits_BLV, feat_BlC, _ = var_model(
                x_BLCv_wo_first_l=x_BLCv_wo_first_l,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_attention_mask,
                encoder_pool_feat=pooled_embed,
                face_group=6,
            )

            bg_last, ed_last = var_model.begin_ends[-1]
            logits_last = logits_BLV[:, bg_last:ed_last, :]
            target_last = gt_BL[:, bg_last:ed_last]

            if args.demo_last_scale_only:
                if args.demo_boundary_only:
                    boundary_local = (
                        var_model.faceray_precompute.boundary_local_indices_last.to(
                            logits_last.device
                        )
                    )
                    logits_use = logits_last[:, boundary_local, :]
                    target_use = target_last[:, boundary_local]
                else:
                    logits_use = logits_last
                    target_use = target_last
            else:
                logits_use = logits_BLV
                target_use = gt_BL

            loss_face = F.cross_entropy(
                logits_use.reshape(-1, var_model.V), target_use.reshape(-1)
            )

            bg, ed = var_model.begin_ends[-1]
            feat_last = feat_BlC[:, bg:ed]
            seam_loss = compute_seam_loss(
                feat_last, var_model.faceray_precompute, tau=args.seam_tau
            )

            seam_weight = args.lambda_seam * min(1.0, step / warmup_steps)
            if args.demo_last_scale_only:
                loss = (
                    args.demo_face_loss_weight * loss_face
                    + seam_weight * args.demo_seam_loss_weight * seam_loss
                )
            else:
                loss = loss_face + seam_weight * seam_loss

        if step == 0:
            block_attn = var_model.blocks[0].attn
            impl = getattr(block_attn, "last_attn_impl", "unknown")
            bias = getattr(block_attn, "last_attn_bias", None)
            print(f"[attn] impl={impl} attn_bias={bias}")

        micro_time = time.perf_counter() - micro_start

        if run is not None and step % args.wandb_log_every == 0:
            stats = adapter_stats(var_model.faceray_adapters[0])
            delta = var_model.faceray_adapters[0](feat_BlC.detach(), face_group=6).abs()
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
            faces_per_sec = (6 * args.batch_size * args.accum_steps) / max(
                step_time, 1e-6
            )
            scenes_per_sec = (args.batch_size * args.accum_steps) / max(step_time, 1e-6)
            wandb.log(
                {
                    "loss/total": loss.item(),
                    "loss/face": loss_face.item(),
                    "loss/seam_raw": seam_loss.item(),
                    "loss/seam_weight_base": args.lambda_seam,
                    "loss/seam_weight_eff": seam_weight,
                    "debug/neighbor_same_face_ratio": float(
                        (
                            var_model.faceray_precompute.neighbor_face_ids_last
                            == var_model.faceray_precompute.boundary_face_ids_last.unsqueeze(
                                1
                            )
                        )
                        .float()
                        .mean()
                        .item()
                    ),
                    "debug/neighbor_invalid_ratio": float(
                        1.0
                        - var_model.faceray_precompute.neighbor_mask_last.float()
                        .mean()
                        .item()
                    ),
                    "debug/d_ray_orthonormal_maxerr": float(
                        var_model.faceray_geom.orthonormality_check(
                            var_model.faceray_precompute
                        )
                    ),
                    "adapter/delta_mean_abs": float(delta.mean().item()),
                    "adapter/delta_max_abs": float(delta.max().item()),
                    "adapter/grad_norm": stats["grad_norm"],
                    "adapter/param_norm": stats["param_norm"],
                    "adapter/out_proj_norm": stats["out_proj_norm"],
                    "perf/microstep_time_sec": micro_time,
                    "perf/step_time_sec": step_time,
                    "perf/gpu_mem_allocated_gb": mem_alloc,
                    "perf/gpu_mem_reserved_gb": mem_reserved,
                    "perf/throughput_faces_per_sec": faces_per_sec,
                    "perf/throughput_scenes_per_sec": scenes_per_sec,
                    "metric/boundary_mae": seam_loss.item(),
                },
                step=step,
            )
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            loss_value = loss.item()

        if run is not None and step % args.wandb_metric_every == 0:
            with torch.no_grad():
                edge_metrics = boundary_mae_by_edge(
                    feat_last, var_model.faceray_precompute
                )
            wandb.log(edge_metrics, step=step)

        scaler.scale(loss / args.accum_steps).backward()

        if (step + 1) % args.accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % 50 == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": loss.item(),
                        "loss_face": loss_face.item(),
                        "loss_seam": seam_loss.item(),
                        "seam_weight": seam_weight,
                    }
                )
            )

        if run is not None and step % args.wandb_image_every == 0:
            with torch.no_grad():
                for idx, faces in enumerate(fixed_faces):
                    grid = make_faces_grid(faces)
                    dice = make_dice_image(faces)
                    wandb.log(
                        {
                            f"samples/faces_grid/{idx}": wandb.Image(
                                grid,
                                caption=(
                                    fixed_prompts[idx]
                                    if idx < len(fixed_prompts)
                                    else ""
                                ),
                            ),
                            f"samples/dice/{idx}": wandb.Image(
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

        if args.save_every > 0 and step > 0 and step % args.save_every == 0:
            save_path = Path(args.save_dir) / args.save_name
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_var_checkpoint(var_model, str(save_path), step, loss=loss_value)

        if args.save_best and loss_value is not None and loss_value < best_loss:
            best_loss = loss_value
            best_path = Path(args.save_dir) / args.save_best_name
            best_path.parent.mkdir(parents=True, exist_ok=True)
            save_var_checkpoint(var_model, str(best_path), step, loss=best_loss)

        step += 1
        progress.update(1)

    if step % args.accum_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    progress.close()
    save_path = Path(args.save_dir) / args.save_name
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_var_checkpoint(var_model, str(save_path), step, loss=loss_value)


if __name__ == "__main__":
    main()
