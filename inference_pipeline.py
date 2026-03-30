"""Run SRDiff baseline and SIG sampler evaluations."""

import argparse
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

from dataset_setup import ensure_benchmark_datasets
from evaluator import SRDiffEvaluator


warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "diffsr_df2k4x.yaml"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SRDiff samplers with spectral-aligned inference guidance.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the SRDiff checkpoint."
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        required=True,
        help="Root directory containing test datasets.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Path to the SRDiff config.",
    )
    parser.add_argument(
        "--rrdb_path",
        type=str,
        default=None,
        help="Optional RRDB checkpoint path. The evaluator normally loads RRDB from the main checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory used for metrics and images.",
    )
    parser.add_argument(
        "--samplers",
        nargs="*",
        default=None,
        help="Subset of sampler names to run. Leave empty to evaluate all registered samplers.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["Set5", "Set14", "BSD100", "Urban100"],
        help="Datasets to evaluate from the test root.",
    )
    parser.add_argument(
        "--scale_factor", type=int, default=4, help="Super-resolution scale factor."
    )
    parser.add_argument(
        "--max_images", type=int, default=None, help="Optional cap per dataset."
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Computation device."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--no_save_images", action="store_true", help="Skip writing SR images to disk."
    )
    parser.add_argument("--skip_fid", action="store_true", help="Skip FID computation.")
    parser.add_argument(
        "--use_patches", action="store_true", help="Force patch-based processing."
    )
    parser.add_argument(
        "--patch_size", type=int, default=160, help="Patch size for tiled evaluation."
    )
    parser.add_argument(
        "--patch_overlap", type=int, default=40, help="Patch overlap in pixels."
    )
    parser.add_argument(
        "--patch_batch_size",
        type=int,
        default=4,
        help="Number of patches processed together.",
    )
    parser.add_argument(
        "--auto_patch_threshold",
        type=int,
        default=2048,
        help="Automatically enable patches for images larger than this threshold.",
    )
    parser.add_argument(
        "--no_auto_download_datasets",
        action="store_true",
        help="Disable automatic benchmark dataset download and preparation.",
    )
    return parser.parse_args()


def validate_paths(args: argparse.Namespace) -> None:
    missing = []
    for label, path in (
        ("model checkpoint", args.model_path),
        ("config file", args.config_path),
    ):
        if not Path(path).exists():
            missing.append(f"{label}: {path}")

    if args.rrdb_path and not Path(args.rrdb_path).exists():
        missing.append(f"RRDB checkpoint: {args.rrdb_path}")

    if missing:
        raise FileNotFoundError("Missing required paths:\n- " + "\n- ".join(missing))


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_config_summary(
    args: argparse.Namespace, evaluator: SRDiffEvaluator, sampler_names
) -> None:
    print("SRDiff + SIG evaluation")
    print(f"- model_path: {args.model_path}")
    print(f"- config_path: {args.config_path}")
    print(f"- test_dir: {args.test_dir}")
    print(f"- output_dir: {args.output_dir}")
    print(f"- device: {args.device}")
    print(f"- datasets: {', '.join(args.datasets)}")
    print(f"- samplers: {', '.join(sampler_names)}")
    print(f"- save_images: {not args.no_save_images}")
    print(f"- calculate_fid: {not args.skip_fid}")
    print(f"- use_patches: {args.use_patches}")
    print(f"- available_samplers: {len(evaluator.samplers)}")


def main() -> int:
    args = parse_arguments()

    try:
        validate_paths(args)
    except FileNotFoundError as exc:
        print(exc)
        return 1

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        args.device = "cpu"

    set_random_seed(args.seed)

    try:
        ensure_benchmark_datasets(
            dataset_root=args.test_dir,
            datasets=args.datasets,
            scale_factor=args.scale_factor,
            auto_download=not args.no_auto_download_datasets,
        )
    except Exception as exc:
        print(f"Failed to prepare benchmark datasets: {exc}")
        return 1

    try:
        evaluator = SRDiffEvaluator(
            config_path=args.config_path,
            model_ckpt_path=args.model_path,
            rrdb_ckpt_path=args.rrdb_path,
            device=args.device,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"Failed to initialize evaluator: {exc}")
        return 1

    sampler_names = args.samplers or list(evaluator.samplers.keys())
    unknown_samplers = [
        name for name in sampler_names if name not in evaluator.samplers
    ]
    if unknown_samplers:
        print("Unknown samplers:")
        for name in unknown_samplers:
            print(f"- {name}")
        print("Available samplers:")
        for name in evaluator.samplers:
            print(f"- {name}")
        return 1

    print_config_summary(args, evaluator, sampler_names)

    try:
        results = evaluator.evaluate_all_datasets(
            dataset_root=args.test_dir,
            datasets=args.datasets,
            scale_factor=args.scale_factor,
            sampler_names=sampler_names,
            max_images_per_dataset=args.max_images,
            save_images=not args.no_save_images,
            calculate_fid=not args.skip_fid,
            use_patches=args.use_patches,
            patch_size=args.patch_size,
            patch_overlap=args.patch_overlap,
            patch_batch_size=args.patch_batch_size,
            auto_patch_threshold=args.auto_patch_threshold,
        )
    except KeyboardInterrupt:
        print("Evaluation interrupted by user.")
        return 130
    except Exception as exc:
        print(f"Evaluation failed: {exc}")
        return 1

    print("Evaluation completed.")
    print(f"- csv_path: {results['csv_path']}")
    print(f"- info_path: {results['info_path']}")
    print(f"- total_results: {len(results['results'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
