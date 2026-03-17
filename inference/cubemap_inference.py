"""
Cubemap inference: generate all 6 canonical faces for one or more scenes.

For each scene (identified by a prompt):
  1. Generate 6 face images using face_id embeddings as conditioning.
  2. Save individual face PNGs: front.png, right.png, back.png, left.png, up.png, down.png
  3. Save a 2×3 grid preview: cubemap_grid.png
  4. Optionally stitch faces into an ERP panorama: erp_preview.png  (evaluation only)
  5. Write meta.json with canonical face order, grouped faces, duplication summary.

Face canonical order: front, right, back, left, up, down (indices 0-5)
  set1 = [front, back, left, right]  (horizontal belt)
  set2 = [front, back, up, down]     (vertical cross)
  front + back appear in both sets — intentional anchor weighting, NOT a bug.

Usage:
    python inference/cubemap_inference.py \\
        --model_path ckpt/cubemap_checkpoint.pth \\
        --vae_path ckpt/vae_ch160v4096z32.pth \\
        --text_model_path ckpt/CLIP \\
        --prompt "A cozy mountain cabin interior with wooden walls and a fireplace" \\
        --output_dir outputs/cubemap_scene_001 \\
        --erp_reconstruction
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torchvision

# Allow imports from the repo root regardless of invocation directory
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PIL import Image

from models import build_vae_var
from models.text_encoder import build_text
from utils.cubemap_groups import (
    CANONICAL_FACES,
    CUBEMAP_SET1,
    CUBEMAP_SET2,
    DUPLICATION_SUMMARY,
    FACE_TO_IDX,
)
from utils.face_id_embedding import FaceIdEmbedding

PATCH_PRESETS = {
    "512": [1, 2, 3, 4, 6, 9, 13, 18, 24, 32],
    "1024": [1, 2, 3, 4, 5, 7, 9, 12, 16, 21, 27, 36, 48, 64],
}


# ── ERP reconstruction ────────────────────────────────────────────────────────


def stitch_cubemap_to_erp(
    face_images: dict,
    erp_width: int = 2048,
    erp_height: int = 1024,
) -> Image.Image:
    """
    Reconstruct an ERP panorama from the 6 cubemap faces.

    This is a simple equirectangular projection reconstruction for
    **evaluation/visualization only** — it is not used during training.

    Args:
        face_images: dict mapping face name → PIL.Image (RGB, square).
        erp_width:  Width of the output ERP image.
        erp_height: Height of the output ERP image (= erp_width / 2).

    Returns:
        PIL.Image of shape (erp_height, erp_width, 3).
    """
    # Convert faces to numpy float32 [0,1]
    face_np: dict = {}
    face_size = None
    for name in CANONICAL_FACES:
        img = face_images[name].convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        face_np[name] = arr
        if face_size is None:
            face_size = arr.shape[0]  # square

    H, W = erp_height, erp_width
    output = np.zeros((H, W, 3), dtype=np.float32)

    # ERP → unit direction vector
    lon = (np.arange(W) / W) * 2.0 * math.pi - math.pi  # [-π, π]
    lat = (np.arange(H) / H) * math.pi - math.pi / 2.0  # [-π/2, π/2]
    lon_grid, lat_grid = np.meshgrid(lon, lat)  # [H, W]

    x = np.cos(lat_grid) * np.cos(lon_grid)
    y = np.cos(lat_grid) * np.sin(lon_grid)
    z = np.sin(lat_grid)

    abs_x, abs_y, abs_z = np.abs(x), np.abs(y), np.abs(z)

    def _sample(face_arr: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Bilinear sample from face_arr using UV coords in [-1,1]."""
        N = face_arr.shape[0]
        px = (u * 0.5 + 0.5) * (N - 1)
        py = (v * 0.5 + 0.5) * (N - 1)
        px = np.clip(px, 0, N - 1)
        py = np.clip(py, 0, N - 1)
        x0 = np.floor(px).astype(int)
        y0 = np.floor(py).astype(int)
        x1 = np.minimum(x0 + 1, N - 1)
        y1 = np.minimum(y0 + 1, N - 1)
        dx = (px - x0)[..., np.newaxis]
        dy = (py - y0)[..., np.newaxis]
        return (
            face_arr[y0, x0] * (1 - dy) * (1 - dx)
            + face_arr[y0, x1] * (1 - dy) * dx
            + face_arr[y1, x0] * dy * (1 - dx)
            + face_arr[y1, x1] * dy * dx
        )

    # +X face = right (lon ≈ 0, dominant x positive)
    mask_right = (abs_x >= abs_y) & (abs_x >= abs_z) & (x > 0)
    # -X face = left
    mask_left = (abs_x >= abs_y) & (abs_x >= abs_z) & (x < 0)
    # +Y face = back (lon ≈ π/2, dominant y positive in our convention: +Y=east/front)
    # Convention: front = -Y direction (camera looking south), back = +Y
    mask_back = (abs_y >= abs_x) & (abs_y >= abs_z) & (y > 0)
    mask_front = (abs_y >= abs_x) & (abs_y >= abs_z) & (y < 0)
    # +Z face = up
    mask_up = (abs_z >= abs_x) & (abs_z >= abs_y) & (z > 0)
    mask_down = (abs_z >= abs_x) & (abs_z >= abs_y) & (z < 0)

    def _fill(mask, face_name, u_arr, v_arr):
        if mask.any():
            output[mask] = _sample(face_np[face_name], u_arr[mask], v_arr[mask])

    # right (+X): u = -z/x, v = y/x
    scale_r = np.where(abs_x > 0, x, 1)
    _fill(
        mask_right,
        "right",
        -z / np.where(mask_right, scale_r, 1),
        y / np.where(mask_right, scale_r, 1),
    )

    # left (-X): u = z/|x|, v = y/|x|
    scale_l = np.where(abs_x > 0, -x, 1)
    _fill(
        mask_left,
        "left",
        z / np.where(mask_left, scale_l, 1),
        y / np.where(mask_left, scale_l, 1),
    )

    # back (+Y): u = x/y, v = z/y
    scale_b = np.where(abs_y > 0, y, 1)
    _fill(
        mask_back,
        "back",
        x / np.where(mask_back, scale_b, 1),
        z / np.where(mask_back, scale_b, 1),
    )

    # front (-Y): u = -x/|y|, v = z/|y|
    scale_f = np.where(abs_y > 0, -y, 1)
    _fill(
        mask_front,
        "front",
        -x / np.where(mask_front, scale_f, 1),
        z / np.where(mask_front, scale_f, 1),
    )

    # up (+Z): u = y/z, v = -x/z
    scale_u = np.where(abs_z > 0, z, 1)
    _fill(
        mask_up,
        "up",
        y / np.where(mask_up, scale_u, 1),
        -x / np.where(mask_up, scale_u, 1),
    )

    # down (-Z): u = -y/|z|, v = -x/|z|
    scale_d = np.where(abs_z > 0, -z, 1)
    _fill(
        mask_down,
        "down",
        -y / np.where(mask_down, scale_d, 1),
        -x / np.where(mask_down, scale_d, 1),
    )

    output = np.clip(output * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(output)


# ── Grid preview ──────────────────────────────────────────────────────────────


def make_cubemap_grid(face_images: dict, cell_size: int = 256) -> Image.Image:
    """
    Arrange 6 faces in a 2×3 grid for quick visual inspection.

    Layout (row × col):
        row 0: front, right, back
        row 1: left,  up,   down

    Args:
        face_images: dict mapping face name → PIL.Image (RGB).
        cell_size: Size to resize each face to.

    Returns:
        PIL.Image of shape (2*cell_size, 3*cell_size).
    """
    grid_layout = [
        ["front", "right", "back"],
        ["left", "up", "down"],
    ]
    rows, cols = 2, 3
    grid = Image.new("RGB", (cols * cell_size, rows * cell_size))
    for r, row in enumerate(grid_layout):
        for c, face_name in enumerate(row):
            img = (
                face_images[face_name]
                .convert("RGB")
                .resize((cell_size, cell_size), Image.LANCZOS)
            )
            grid.paste(img, (c * cell_size, r * cell_size))
    return grid


# ── Core inference ────────────────────────────────────────────────────────────


def generate_cubemap_scene(
    var_model,
    vae_model,
    text_encoder,
    face_id_embed: FaceIdEmbedding,
    prompt: str,
    device: torch.device,
    cfg: float = 4.5,
    top_k: int = 600,
    top_p: float = 0.8,
    seed: int = 1,
    sample_version: str = "512",
) -> dict:
    """
    Generate 6 cubemap face images for a single scene.

    Each face is generated independently using its discrete face_id embedding
    as the conditioning signal (passed through the cam_dir slot).

    Args:
        var_model: The VAR model (eval mode).
        vae_model: The VAE model (eval mode).
        text_encoder: Text encoder (eval mode).
        face_id_embed: FaceIdEmbedding module for discrete face conditioning.
        prompt: Text description of the scene.
        device: Torch device.
        cfg: Classifier-free guidance scale.
        top_k: Top-k sampling.
        top_p: Top-p sampling.
        seed: Random seed for reproducibility.
        sample_version: "512" or "1024".

    Returns:
        dict mapping face_name → PIL.Image (RGB, square).
    """
    torch.manual_seed(seed)

    # Encode text prompt (with CFG null condition)
    with torch.no_grad():
        prompt_embeds, prompt_attention_mask, pooled_embed = (
            text_encoder.extract_text_features([prompt, ""])
        )
        # Take only the conditional embedding (index 0)
        cond_embeds = prompt_embeds[:1]
        cond_mask = prompt_attention_mask[:1]
        cond_pool = pooled_embed[:1]

    face_pil_images = {}

    with torch.no_grad():
        with torch.inference_mode():
            for face_name in CANONICAL_FACES:
                face_idx = FACE_TO_IDX[face_name]
                face_id_tensor = torch.tensor(
                    [face_idx], device=device, dtype=torch.long
                )

                # Get face_id embedding as cam_dir conditioning
                cam_dir_cond = face_id_embed(face_id_tensor)  # [1, embed_dim]

                with torch.autocast(
                    "cuda", enabled=True, dtype=torch.float16, cache_enabled=True
                ):
                    # Check if model supports cam_dir parameter
                    import inspect

                    infer_sig = inspect.signature(var_model.autoregressive_infer_cfg)
                    infer_kwargs = dict(
                        B=1,
                        label_B=None,
                        encoder_hidden_states=cond_embeds,
                        encoder_attention_mask=cond_mask,
                        encoder_pool_feat=cond_pool,
                        cfg=cfg,
                        top_k=top_k,
                        top_p=top_p,
                        g_seed=seed + face_idx,  # deterministic per-face seed offset
                        more_smooth=False,
                        w_mask=True,
                        sample_version=sample_version,
                    )
                    if "cam_dir" in infer_sig.parameters:
                        infer_kwargs["cam_dir"] = cam_dir_cond

                    recon_B3HW = var_model.autoregressive_infer_cfg(**infer_kwargs)

                # Decode to PIL
                img_tensor = recon_B3HW[0].float().clamp(0.0, 1.0)  # [3, H, W]
                img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 255).astype(np.uint8)
                face_pil_images[face_name] = Image.fromarray(img_np)

    return face_pil_images


# ── Save outputs ──────────────────────────────────────────────────────────────


def save_cubemap_outputs(
    face_images: dict,
    output_dir: str,
    prompt: str,
    scene_id: str = "scene_000",
    erp_reconstruction: bool = True,
    erp_width: int = 2048,
    grid_cell_size: int = 256,
) -> dict:
    """
    Save all cubemap inference outputs for a single scene.

    Outputs:
        {output_dir}/
            front.png, right.png, back.png, left.png, up.png, down.png
            set1_faces_front_back_left_right.png   (4-face group preview)
            set2_faces_front_back_up_down.png      (4-face group preview)
            cubemap_grid.png                       (2×3 face grid)
            erp_preview.png                        (ERP stitch, if enabled)
            meta.json                              (full metadata)

    Returns:
        dict with all saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved = {}

    # 1. Individual face PNGs (canonical order preserved)
    for face_name in CANONICAL_FACES:
        img = face_images[face_name]
        path = os.path.join(output_dir, f"{face_name}.png")
        img.save(path)
        saved[face_name] = path

    # 2. Set group previews (face-order stable, matches training grouping)
    def _save_group_grid(group_names, label):
        grid_imgs = [face_images[n].convert("RGB") for n in group_names]
        W = grid_imgs[0].width
        H = grid_imgs[0].height
        strip = Image.new("RGB", (len(group_names) * W, H))
        for i, img in enumerate(grid_imgs):
            strip.paste(img, (i * W, 0))
        fname = f"{label}_faces_{'_'.join(group_names)}.png"
        path = os.path.join(output_dir, fname)
        strip.save(path)
        saved[label] = path
        return path

    _save_group_grid(CUBEMAP_SET1, "set1")
    _save_group_grid(CUBEMAP_SET2, "set2")

    # 3. 2×3 grid preview
    grid = make_cubemap_grid(face_images, cell_size=grid_cell_size)
    grid_path = os.path.join(output_dir, "cubemap_grid.png")
    grid.save(grid_path)
    saved["cubemap_grid"] = grid_path

    # 4. ERP reconstruction (evaluation only)
    if erp_reconstruction:
        erp_img = stitch_cubemap_to_erp(face_images, erp_width=erp_width)
        erp_path = os.path.join(output_dir, "erp_preview.png")
        erp_img.save(erp_path)
        saved["erp_preview"] = erp_path

    # 5. Metadata JSON
    meta = {
        "mode": "cubemap",
        "scene_id": scene_id,
        "prompt": prompt,
        "canonical_face_order": list(CANONICAL_FACES),
        "grouped_faces": {
            "set1": list(CUBEMAP_SET1),
            "set2": list(CUBEMAP_SET2),
        },
        "duplication_summary": DUPLICATION_SUMMARY,
        "saved_files": {k: os.path.basename(v) for k, v in saved.items()},
        "erp_reconstruction_enabled": erp_reconstruction,
        "note": (
            "front and back appear in set1 AND set2. "
            "This is intentional anchor weighting, NOT a bug."
        ),
    }
    meta_path = os.path.join(output_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    saved["meta"] = meta_path

    return saved


# ── Main ──────────────────────────────────────────────────────────────────────


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load checkpoint
    print(f"[cubemap_inference] Loading model from {args.model_path}")
    model_ckpt = torch.load(args.model_path, map_location="cpu")
    var_state = model_ckpt["trainer"]["var_wo_ddp"]

    # Auto-detect patch_nums from checkpoint
    if args.patch_nums is not None and len(args.patch_nums) > 0:
        patch_nums = args.patch_nums
    else:
        num_levels = var_state["lvl_embed.weight"].shape[0]
        if num_levels == 10:
            patch_nums = PATCH_PRESETS["512"]
            sample_version = "512"
        elif num_levels == 14:
            patch_nums = PATCH_PRESETS["1024"]
            sample_version = "1024"
        else:
            raise ValueError(
                f"Unsupported number of levels in checkpoint: {num_levels}. "
                "Please pass --patch_nums explicitly."
            )
        print(f"[cubemap_inference] Auto-detected patch_nums: {patch_nums}")

    # Build VAE + VAR
    vae_model, var_model = build_vae_var(
        V=4096,
        Cvae=32,
        ch=160,
        share_quant_resi=4,
        device=device,
        patch_nums=patch_nums,
        depth=args.depth,
        shared_aln=False,
        attn_l2_norm=True,
        enable_cross=True,
        in_dim_cross=1024,
        flash_if_available=False,
        fused_if_available=True,
        init_adaln=0.5,
        init_adaln_gamma=5e-5,
        init_head=0.02,
        init_std=-1,
        rope_emb=True,
        lvl_emb=True,
        enable_logit_norm=True,
        enable_adaptive_norm=False,
        train_mode="none",
        rope_theta=10000,
        rope_norm=64,
        sample_from_idx=9,
    )

    vae_model.load_state_dict(
        torch.load(args.vae_path, map_location="cpu"), strict=True
    )
    var_model.load_state_dict(var_state, strict=True)
    vae_model.eval()
    var_model.eval()

    # Text encoder
    text_encoder, _ = build_text(pretrained_path=args.text_model_path, device=device)
    text_encoder.eval()

    # Face-ID embedding (loaded from checkpoint if present, else fresh)
    embed_dim = getattr(args, "face_id_embed_dim", 3)
    face_id_embed = FaceIdEmbedding(embed_dim=embed_dim).to(device)
    if "face_id_embed" in model_ckpt.get("trainer", {}):
        face_id_embed.load_state_dict(model_ckpt["trainer"]["face_id_embed"])
        print("[cubemap_inference] Loaded face_id_embed from checkpoint.")
    else:
        print(
            "[cubemap_inference] face_id_embed not in checkpoint — using random init."
        )
    face_id_embed.eval()

    # Collect prompts
    prompts = []
    if args.prompt:
        prompts = [args.prompt]
    elif args.prompt_file:
        with open(args.prompt_file, "r") as f:
            prompts = [ln.strip() for ln in f if ln.strip()]
    else:
        prompts = [
            "A cozy mountain cabin interior with wooden walls, a stone fireplace, and soft warm lighting."
        ]

    # Generate scenes
    for scene_idx, prompt in enumerate(prompts):
        scene_id = f"scene_{scene_idx:04d}"
        scene_out_dir = (
            os.path.join(args.output_dir, scene_id)
            if len(prompts) > 1
            else args.output_dir
        )
        print(
            f"\n[cubemap_inference] Scene {scene_idx + 1}/{len(prompts)}: {prompt[:80]}"
        )

        face_images = generate_cubemap_scene(
            var_model=var_model,
            vae_model=vae_model,
            text_encoder=text_encoder,
            face_id_embed=face_id_embed,
            prompt=prompt,
            device=device,
            cfg=args.cfg,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed + scene_idx,
            sample_version=sample_version,
        )

        saved = save_cubemap_outputs(
            face_images=face_images,
            output_dir=scene_out_dir,
            prompt=prompt,
            scene_id=scene_id,
            erp_reconstruction=args.erp_reconstruction,
            erp_width=args.erp_width,
            grid_cell_size=args.grid_cell_size,
        )

        print(f"[cubemap_inference] Saved {len(saved)} files to {scene_out_dir}")
        for key, path in saved.items():
            print(f"  {key}: {os.path.basename(path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate 6-face cubemap scenes using STAR-T2I with anchor-weighted cubemap conditioning."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to cubemap-trained STAR checkpoint (.pth).",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default="ckpt/vae_ch160v4096z32.pth",
        help="Path to VAE checkpoint.",
    )
    parser.add_argument(
        "--text_model_path",
        type=str,
        default="ckpt/CLIP",
        help="Path to text encoder (CLIP) directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/cubemap_inference",
        help="Directory to write output images and metadata.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Single text prompt for the scene.",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="",
        help="Path to a text file with one prompt per line.",
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=4.5,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=600,
        help="Top-k sampling parameter.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.8,
        help="Top-p sampling parameter.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=30,
        help="Depth of the VAR model.",
    )
    parser.add_argument(
        "--patch_nums",
        type=int,
        nargs="+",
        default=None,
        help="Patch numbers (auto-detected from checkpoint if omitted).",
    )
    parser.add_argument(
        "--face_id_embed_dim",
        type=int,
        default=3,
        help="Embedding dimension for face_id conditioning.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed.",
    )
    parser.add_argument(
        "--erp_reconstruction",
        action="store_true",
        default=False,
        help="Stitch generated faces into an ERP panorama (evaluation only).",
    )
    parser.add_argument(
        "--erp_width",
        type=int,
        default=2048,
        help="Width of ERP reconstruction output.",
    )
    parser.add_argument(
        "--grid_cell_size",
        type=int,
        default=256,
        help="Cell size (px) for the 2×3 face grid preview.",
    )

    args = parser.parse_args()
    main(args)
