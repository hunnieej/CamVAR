import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from models.var_drop_faceray import build_vae_var_faceray
from models.text_encoder import build_text


FACE_ORDER = ["F", "R", "B", "L", "U", "D"]


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


def load_var_checkpoint(model, ckpt_path: str) -> None:
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "trainer" in state:
        state = state["trainer"]["var_wo_ddp"]
    model.load_state_dict(state, strict=False)


def make_faces_grid(faces: list[Image.Image]) -> Image.Image:
    w, h = faces[0].size
    grid = Image.new("RGB", (w * 3, h * 2))
    for idx, img in enumerate(faces):
        row = idx // 3
        col = idx % 3
        grid.paste(img, (col * w, row * h))
    return grid


def make_dice_image(faces: list[Image.Image]) -> Image.Image:
    w, h = faces[0].size
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
        canvas.paste(faces[idx], positions[face_name])
    return canvas

def load_var_checkpoint(model, ckpt_path: str) -> None:
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "trainer" in state:
        state = state["trainer"]["var_wo_ddp"]
    ret = model.load_state_dict(state, strict=False)
    print("[CKPT] missing:", len(ret.missing_keys), "unexpected:", len(ret.unexpected_keys))
    print("[CKPT] missing sample:", ret.missing_keys[:20])
    print("[CKPT] unexpected sample:", ret.unexpected_keys[:20])

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="A cozy mountain cabin at sunset")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--cfg", type=float, default=4.0)
    parser.add_argument("--top_k", type=int, default=600)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--more_smooth", action="store_true")
    parser.add_argument("--vae_ckpt", type=str, default="ckpt/vae_ch160v4096z32.pth")
    parser.add_argument("--gen_ckpt", type=str, default=None)
    parser.add_argument("--gen_ckpt_dir", type=str, default="ckpt/faceray_stage1")
    parser.add_argument("--gen_ckpt_pattern", type=str, default="*.pth")
    parser.add_argument("--text_encoder_ckpt", type=str, default="ckpt/CLIP")
    parser.add_argument(
        "--patch_nums",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 6, 9, 13, 18, 24, 32],
    )
    parser.add_argument("--depth", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.batch_size != 6:
        raise ValueError("batch_size must be 6 for cubemap faces")

    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    gen_ckpt = args.gen_ckpt
    if gen_ckpt is None:
        gen_ckpt = find_latest_checkpoint(args.gen_ckpt_dir, args.gen_ckpt_pattern)

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
    load_var_checkpoint(var_model, gen_ckpt)

    text_encoder, _ = build_text(pretrained_path=args.text_encoder_ckpt, device=device)
    text_encoder.eval()

    prompts = [args.prompt] * args.batch_size
    with torch.no_grad():
        prompt_embeds, prompt_attention_mask, pooled_embed = (
            text_encoder.extract_text_features(prompts + [""] * args.batch_size)
        )

    with torch.no_grad():
        with torch.autocast("cuda", enabled=device.type == "cuda", dtype=torch.float16):
            recon = var_model.autoregressive_infer_cfg(
                B=args.batch_size,
                label_B=None,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_attention_mask,
                encoder_pool_feat=pooled_embed,
                g_seed=args.seed,
                cfg=args.cfg,
                top_k=args.top_k,
                top_p=args.top_p,
                more_smooth=args.more_smooth,
                w_mask=False,
                sample_version="1024",
            )
    
    images = []
    for i in range(args.batch_size):
        x = recon[i].float()
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
        x = x.clamp(0.0, 1.0)
        img = (x.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
        images.append(Image.fromarray(img))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    faces_dir = output_dir / "faces"
    faces_dir.mkdir(parents=True, exist_ok=True)

    for name, img in zip(FACE_ORDER, images):
        img.save(faces_dir / f"{name}.png")

    grid = make_faces_grid(images)
    dice = make_dice_image(images)
    grid.save(output_dir / "faces_grid.png")
    dice.save(output_dir / "dice.png")

    print("recon stats:", recon.min().item(), recon.max().item(), torch.isfinite(recon).all().item())
    print(f"Saved faces to {faces_dir}")
    print(f"Saved grid to {output_dir / 'faces_grid.png'}")
    print(f"Saved dice to {output_dir / 'dice.png'}")
    print(f"Generator checkpoint: {gen_ckpt}")


if __name__ == "__main__":
    main()
