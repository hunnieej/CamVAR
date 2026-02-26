import argparse
import argparse
import math
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import torch
from PIL import Image

from models.vqvae import VQVAE


def load_image(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC in [0,1]
    return torch.from_numpy(arr).permute(2, 0, 1)  # C,H,W


def to_range(x: torch.Tensor, mode: Literal["0_1", "neg1_1"]) -> torch.Tensor:
    if mode == "0_1":
        return x
    return x * 2.0 - 1.0


def from_neg1_1(x: torch.Tensor) -> torch.Tensor:
    return (x + 1.0) * 0.5


@torch.no_grad()
def roundtrip(
    vae: VQVAE,
    img_chw: torch.Tensor,
    range_mode: Literal["0_1", "neg1_1"],
) -> Tuple[torch.Tensor, float]:
    device = next(vae.parameters()).device
    x = to_range(img_chw, range_mode).unsqueeze(0).to(device)
    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        idx_bl = vae.img_to_idxBl(x)
        recon = vae.idxBl_to_img(idx_bl, same_shape=True, last_one=True)
    recon = recon.squeeze(0)
    if range_mode == "neg1_1":
        recon = from_neg1_1(recon)
    recon = recon.clamp(0.0, 1.0)
    # stats
    mse = torch.mean((recon - img_chw.to(device)) ** 2).item()
    psnr = float("inf") if mse == 0 else 10 * math.log10(1.0 / mse)
    return recon.cpu(), psnr


def save_image(t: torch.Tensor, path: Path) -> None:
    arr = (t.clamp(0, 1) * 255.0).round().byte().permute(1, 2, 0).numpy()
    Image.fromarray(arr).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_ckpt", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--scene_id", type=str, required=True)
    parser.add_argument("--face_name", type=str, default="F")
    parser.add_argument(
        "--vae_input_range", type=str, choices=["auto", "0_1", "neg1_1"], default="auto"
    )
    parser.add_argument("--output_dir", type=str, default="outputs/vae_verify")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--patch_nums", type=int, nargs="+", default=[1, 2, 3, 4, 6, 9, 13, 18, 24, 32]
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    vae = VQVAE(
        vocab_size=4096,
        z_channels=32,
        ch=160,
        test_mode=True,
        share_quant_resi=4,
        v_patch_nums=tuple(args.patch_nums),
    ).to(device)
    state = torch.load(args.vae_ckpt, map_location="cpu")
    vae.load_state_dict(state, strict=True)
    vae.eval()

    img_path = Path(args.data_root) / args.scene_id / "faces" / f"{args.face_name}.png"
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    img_chw = load_image(img_path)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_image(img_chw, out_dir / "original.png")

    modes = (
        ["0_1", "neg1_1"] if args.vae_input_range == "auto" else [args.vae_input_range]
    )
    best = None
    for mode in modes:
        recon, psnr = roundtrip(vae, img_chw, mode)
        mse = torch.mean((recon - img_chw) ** 2).item()
        recon_min = recon.min().item()
        recon_max = recon.max().item()
        recon_mean = recon.mean().item()
        idx_bl = vae.img_to_idxBl(to_range(img_chw, mode).unsqueeze(0).to(device))
        idx_cat = torch.cat(idx_bl, dim=1).cpu()
        idx_min = idx_cat.min().item()
        idx_max = idx_cat.max().item()
        V = vae.vocab_size
        out_of_range = ((idx_cat < 0) | (idx_cat >= V)).float().mean().item()
        emb_shape = tuple(vae.quantize.embedding.weight.shape)

        tag = mode.replace("_", "")
        save_image(recon, out_dir / f"recon_{tag}.png")
        print(f"=== Range {mode} ===")
        print(
            f"recon stats: min={recon_min:.4f} max={recon_max:.4f} mean={recon_mean:.4f}"
        )
        print(f"MSE={mse:.6f} PSNR={psnr:.2f} dB")
        print(f"codebook: V={V} C_vae={vae.Cvae} emb_shape={emb_shape}")
        print(
            f"idx range: min={idx_min} max={idx_max} out_of_range_frac={out_of_range:.6f}"
        )

        if best is None or psnr > best[0]:
            best = (psnr, mode, out_of_range)

    best_psnr, best_mode, best_oor = best
    verdict = "PASS" if (best_psnr > 20.0 and best_oor == 0.0) else "FAIL"
    print(
        f"VAE_VERIFY: {verdict} (range={best_mode}, psnr={best_psnr:.2f}dB, oor={best_oor:.6f})"
    )


if __name__ == "__main__":
    main()
