import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(args):
    print("[two-stage]", " ".join(args))
    subprocess.run(args, check=True)


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
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--gen_steps", type=int, default=20000)
    parser.add_argument("--sampler_steps", type=int, default=20000)
    parser.add_argument("--gen_save_dir", type=str, default="ckpt/faceray_stage1")
    parser.add_argument(
        "--gen_save_name", type=str, default="faceray_stage1_generator.pth"
    )
    parser.add_argument(
        "--gen_save_best_name",
        type=str,
        default="faceray_stage1_generator_best.pth",
    )
    parser.add_argument("--gen_save_every", type=int, default=500)
    parser.add_argument("--gen_ckpt", type=str, default=None)
    parser.add_argument("--use_best_gen_ckpt", action="store_true")
    parser.add_argument("--gen_ckpt_pattern", type=str, default="*.pth")
    parser.add_argument("--text_encoder_ckpt", type=str, default="ckpt/CLIP")
    parser.add_argument("--vae_ckpt", type=str, default="ckpt/vae_ch160v4096z32.pth")
    parser.add_argument(
        "--var_ckpt",
        type=str,
        default="ckpt/star_rope_d30_512-ar-ckpt-ep1-iter30000.pth",
    )
    parser.add_argument("--patch_nums", type=int, nargs="+", default=None)
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    gen_cmd = [
        sys.executable,
        "train_faceray.py",
        "--data_root",
        args.data_root,
        "--num_steps",
        str(args.gen_steps),
        "--device",
        args.device,
        "--text_encoder_ckpt",
        args.text_encoder_ckpt,
        "--vae_ckpt",
        args.vae_ckpt,
        "--var_ckpt",
        args.var_ckpt,
        "--save_dir",
        args.gen_save_dir,
        "--save_name",
        args.gen_save_name,
        "--save_every",
        str(args.gen_save_every),
    ]
    if args.amp:
        gen_cmd.append("--amp")
    if args.use_best_gen_ckpt:
        gen_cmd.append("--save_best")
        gen_cmd.extend(["--save_best_name", args.gen_save_best_name])
    if args.patch_nums is not None:
        gen_cmd.append("--patch_nums")
        gen_cmd.extend([str(p) for p in args.patch_nums])

    run_command(gen_cmd)

    gen_ckpt = args.gen_ckpt
    if gen_ckpt is None:
        best_candidate = Path(args.gen_save_dir) / args.gen_save_best_name
        if args.use_best_gen_ckpt and best_candidate.exists():
            gen_ckpt = str(best_candidate)
        else:
            gen_ckpt = find_latest_checkpoint(args.gen_save_dir, args.gen_ckpt_pattern)

    sampler_cmd = [
        sys.executable,
        "train_sampler_faceray.py",
        "--data_root",
        args.data_root,
        "--num_steps",
        str(args.sampler_steps),
        "--device",
        args.device,
        "--text_encoder_ckpt",
        args.text_encoder_ckpt,
        "--vae_ckpt",
        args.vae_ckpt,
        "--gen_ckpt",
        gen_ckpt,
    ]
    if args.use_best_gen_ckpt:
        sampler_cmd.append("--use_best_gen_ckpt")
    if args.amp:
        sampler_cmd.append("--amp")
    if args.patch_nums is not None:
        sampler_cmd.append("--patch_nums")
        sampler_cmd.extend([str(p) for p in args.patch_nums])

    run_command(sampler_cmd)


if __name__ == "__main__":
    main()
