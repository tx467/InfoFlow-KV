# InfoFlow KV: LLM Pipeline

This directory contains the text-only InfoFlow KV implementation for Qwen3, Llama, and ChatGLM. It supports selective KV cache recomputation, single-GPU evaluation, and Ring-Attention sequence parallelism.

## Environment

The paper experiments used Python 3.10 and CUDA 12.1. Core package versions are recorded in [`requirements.txt`](requirements.txt). Install CUDA-dependent packages for the CUDA and PyTorch versions available on your system.

Model weights are not included. Replace the model placeholders in `configs/*.yaml` or pass a model path to scripts that expose `--model`.

## Single-GPU inference

Set the `models` list in a configuration file, then run:

```bash
python scripts/inference_with_recompute_kv.py configs/2wikimqa_eval.yaml
```

Configurations are also provided for HotpotQA and MuSiQue. Each configuration controls the dataset, recomputation ratio, chunking, generation length, sample limit, and strategies to evaluate.

## Distributed LongBench evaluation

The distributed methods require NCCL and multiple CUDA GPUs. For a four-GPU smoke test:

```bash
torchrun --nproc_per_node=4 scripts/eval_longbench.py \
  --model /path/to/Qwen3-14B \
  --tasks hotpotqa \
  --methods sp_guided_recompute sp_cacheblend sp_lego ring_attention \
  --max_samples 5
```

Supported tasks include `hotpotqa`, `2wikimqa`, `musique`, `narrativeqa`, `qasper`, `multifieldqa_en`, and `longbenchv2`.

## Paper experiment drivers

Set `MODEL_PATH` before running the shell drivers:

```bash
export MODEL_PATH=/path/to/Qwen3-14B

bash scripts/run_eval_stride1.sh
bash scripts/run_eval_stride8.sh
bash scripts/run_eval_all_methods.sh
bash scripts/run_eval_longbenchv2.sh
```

Review each script's task list, GPU count, sample count, and output directory before starting a full experiment.

## Layout

- `benchmarks/`: LongBench v1/v2 and needle-in-a-haystack dataset adapters.
- `configs/`: YAML experiment configurations.
- `inputs/`: bundled LongBench samples used by the provided configurations.
- `models/`: model-specific KV cache implementations and distributed parallel components.
- `scripts/`: inference, evaluation, benchmark, and cluster entry points.
- `tests/parallel/`: distributed recomputation tests.
