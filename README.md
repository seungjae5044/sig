# SIG : Spectral-Aligned Inference Guidance for Diffusion-Based Image Super-Resolution

Official implementation of `Spectral-Aligned Inference Guidance for Diffusion-Based Image Super-Resolution`, accepted to ICASSP 2026.

This `sig` folder keeps only the SRDiff-based inference and evaluation path used for the paper:

- `inference_pipeline.py`: main evaluation entry point
- `evaluator.py`: dataset loop, sampler registry, metric export
- `samplers/`: DDPM, DDIM, DPM-Solver++, and SIG guidance
- `models/`, `utils/`, `configs/`: SRDiff backbone pieces required for inference

The previous SinSR-based workspace was archived to `../discard/sig_sinsr_archive`.

## Setup

```bash
uv sync
```

This creates `.venv/` from `pyproject.toml` and `uv.lock`.

## Models And Datasets

- Model checkpoints are not downloaded automatically. Place the SRDiff checkpoint somewhere local and pass it with `--model_path`.
- Benchmark datasets are downloaded automatically on first use when `--test_dir` does not already contain `Set5`, `Set14`, `BSD100`, and `Urban100` in the expected layout.
- Only the datasets requested in `--datasets` are downloaded.
- The downloader uses the public SwinIR benchmark archives and converts them into this repo's canonical layout: `DATASET/image_SRF_<scale>/{HR,LR}`.
- HR images are normalized with `modcrop`, and LR inputs are generated with bicubic downsampling so no manual preprocessing is required.
- To disable this behavior, pass `--no_auto_download_datasets`.

## Core SIG Files

- `samplers/unified_frequency_guidance.py`: spectral-aligned guidance core and sampler wrappers
- `evaluator.py`: registers `freq_guided_ddpm`, `freq_guided_ddim_*`, `freq_guided_dpmpp_*`
- `models/diffusion.py`: SRDiff diffusion backbone used by the samplers

## Run

```bash
uv run python inference_pipeline.py \
  --model_path /path/to/model_ckpt_steps_400000.ckpt \
  --test_dir ./datasets/test_set \
  --config_path configs/diffsr_df2k4x.yaml \
  --datasets Set5 Set14 BSD100 Urban100 \
  --samplers ddpm_100 freq_guided_ddpm ddim_25 freq_guided_ddim_25 dpmpp_10 freq_guided_dpmpp_10
```

Patch-based evaluation for large images:

```bash
uv run python inference_pipeline.py \
  --model_path /path/to/model_ckpt_steps_400000.ckpt \
  --test_dir ./datasets/test_set \
  --use_patches \
  --patch_size 160 \
  --patch_overlap 40
```

Quick smoke test with automatic `Set5` download:

```bash
uv run python inference_pipeline.py \
  --model_path /path/to/model_ckpt_steps_400000.ckpt \
  --test_dir ./datasets/test_set \
  --datasets Set5 \
  --samplers ddpm_100 freq_guided_ddpm \
  --max_images 1 \
  --skip_fid
```

Results are written under `results/` as:

- `results/metrics/*.csv`
- `results/metrics/*.json`
- `results/images/<dataset>/<sampler>/*.png`
