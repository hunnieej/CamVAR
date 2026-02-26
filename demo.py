import argparse
import os
import random
import time

import numpy as np
import torch
from PIL import Image

from models import build_vae_var
from models.text_encoder import build_text


torch.set_grad_enabled(False)


def build_models(args, device):
    var_state = torch.load(args.var_ckpt, map_location="cpu")
    var_weights = var_state["trainer"]["var_wo_ddp"]
    trainer_config = var_state["trainer"].get("config", {})

    if args.patch_nums is None:
        args.patch_nums = list(trainer_config.get("patch_nums", []))
        if not args.patch_nums:
            raise ValueError(
                "patch_nums not provided and not found in checkpoint config"
            )

    adaptive_norm = args.adaptive_norm
    if adaptive_norm == "auto":
        adaptive_norm = any(
            key.startswith(
                (
                    "word_embed_head.",
                    "feat_extract_blocks.",
                    "head_logits2.",
                    "encoder_proj2.",
                    "head_proj.",
                    "lvl_embed_2.",
                    "pos_start_last",
                )
            )
            for key in var_weights
        )
    else:
        adaptive_norm = adaptive_norm == "true"

    vae_model, var_model = build_vae_var(
        V=4096,
        Cvae=32,
        ch=160,
        share_quant_resi=4,
        device=device,
        patch_nums=args.patch_nums,
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
        enable_logit_norm=args.enable_logit_norm,
        enable_adaptive_norm=adaptive_norm,
        train_mode="none",
        rope_theta=10000,
        rope_norm=64.0,
        sample_from_idx=9,
    )

    var_model.load_state_dict(var_weights, strict=True)
    vae_model.load_state_dict(
        torch.load(args.vae_ckpt, map_location="cpu"), strict=True
    )

    vae_model.eval()
    var_model.eval()

    text_encoder, _ = build_text(pretrained_path=args.text_encoder_ckpt, device=device)
    text_encoder.eval()

    return vae_model, var_model, text_encoder


def run_model(
    prompt, more_smooth, cfg, top_k, top_p, seed, batch_size, var_model, text_encoder
):
    if seed < 0:
        raise ValueError("Seed must be greater than or equal to 0")

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    start_time = time.time()

    with torch.no_grad():
        with torch.inference_mode():
            prompt_list = [prompt] * batch_size
            prompt_embeds, prompt_attention_mask, pooled_embed = (
                text_encoder.extract_text_features(prompt_list + [""] * batch_size)
            )
            with torch.autocast(
                "cuda", enabled=True, dtype=torch.float16, cache_enabled=True
            ):
                recon_B3HW = var_model.autoregressive_infer_cfg(
                    B=batch_size,
                    label_B=None,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    encoder_pool_feat=pooled_embed,
                    cfg=cfg,
                    top_k=top_k,
                    top_p=top_p,
                    g_seed=seed,
                    more_smooth=more_smooth,
                    w_mask=True,
                    sample_version="1024",
                )

    images = []
    for i in range(batch_size):
        img_pred = (recon_B3HW[i].permute(1, 2, 0).cpu().numpy() * 255.0).astype(
            np.uint8
        )
        images.append(Image.fromarray(img_pred))

    inference_time = time.time() - start_time
    return images, inference_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="A red car.")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--cfg", type=float, default=4.0)
    parser.add_argument("--top_k", type=int, default=600)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--more_smooth", action="store_true")
    parser.add_argument(
        "--var_ckpt",
        type=str,
        default="ckpt/star_rope_d30_512-ar-ckpt-ep1-iter30000.pth",
    )
    parser.add_argument(
        "--vae_ckpt",
        type=str,
        default="ckpt/vae_ch160v4096z32.pth",
    )
    parser.add_argument(
        "--text_encoder_ckpt",
        type=str,
        default="ckpt/CLIP",
    )
    parser.add_argument("--patch_nums", type=int, nargs="+", default=None)
    parser.add_argument("--depth", type=int, default=30)
    parser.add_argument("--enable_logit_norm", type=bool, default=True)
    parser.add_argument(
        "--adaptive_norm",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
    )

    args = parser.parse_args()

    tf32 = True
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.set_float32_matmul_precision("high" if tf32 else "highest")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(args.text_encoder_ckpt):
        raise FileNotFoundError(
            f"Text encoder checkpoint not found: {args.text_encoder_ckpt}"
        )

    _, var_model, text_encoder = build_models(args, device)

    images, inference_time = run_model(
        args.prompt,
        args.more_smooth,
        args.cfg,
        args.top_k,
        args.top_p,
        args.seed,
        args.batch_size,
        var_model,
        text_encoder,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    for idx, image in enumerate(images):
        output_path = os.path.join(args.output_dir, f"sample_{args.seed}_{idx}.png")
        image.save(output_path)
        print(f"Saved: {output_path}")

    print(f"Inference Time: {inference_time:.2f} seconds")


if __name__ == "__main__":
    main()
