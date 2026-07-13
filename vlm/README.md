# InfoFlow KV: VLM Pipeline

This directory contains the InfoFlow KV vision-language pipeline for Qwen3-VL-8B-Instruct. It implements selective KV cache recomputation for mixed image-and-text inputs, including image chunking and multimodal benchmark evaluation.

## Environment

Install the dependencies listed in [`requirements.txt`](requirements.txt) in a dedicated environment with a compatible CUDA-enabled PyTorch build.

Model weights and benchmark datasets are not included. Each configuration under `configs/` contains placeholder paths that must be updated for the local environment.

## Quick start

Edit `configs/blink_counting.yaml` and set:

- `model`: local path or Hugging Face identifier for Qwen3-VL-8B-Instruct.
- `cache_dir`: model cache directory.
- `dataset_dir`: benchmark dataset directory.
- `output_dir`: directory for predictions and metrics.
- `num_samples`: optional sample limit for a smoke test.

Then run:

```bash
python scripts/evaluate.py --config configs/blink_counting.yaml
```

The evaluator expands lists of datasets, recomputation ratios, and image chunk counts into individual runs. Use `run_baseline` and `run_recompute` in the YAML file to select the desired paths.

## Additional benchmarks

Configurations are provided for BLINK, ChartQA, DocVQA, MathVista, MMBench, OCRBench, and RealWorldQA. Run any configuration with the same entry point:

```bash
python scripts/evaluate.py --config configs/docvqa.yaml
```

For VLMEvalKit integration, install VLMEvalKit separately, set `VLMEVALKIT_DIR`, and use `scripts/eval_vlmeval.py`.

## Entry points

| Script | Purpose |
|---|---|
| `scripts/evaluate.py` | Expand a YAML configuration into baseline and recomputation runs |
| `scripts/inference_with_recompute_kv.py` | Run the selective VLM KV-recomputation pipeline |
| `scripts/run_blink.py` | Run the full-prefill BLINK baseline |
| `scripts/eval_vlmeval.py` | Evaluate supported datasets through VLMEvalKit |

## Layout

- `benchmarks/`: dataset adapters and evaluation helpers.
- `configs/`: per-benchmark YAML configurations.
- `inference/`: reusable inference runner.
- `models/qwen/kv_cache/`: extraction, scoring, chunking, and recomputation.
- `models/qwen/patches/`: Qwen3-VL attention, text, and visual patches.
- `scripts/`: primary evaluation and inference entry points.
