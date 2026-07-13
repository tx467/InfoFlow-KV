# LLM YAML Configuration Reference

The YAML files in this directory configure `scripts/inference_with_recompute_kv.py`. Run the commands below from the `llm/` directory.

## Quick start

Edit the model path in a configuration file, then run:

```bash
python scripts/inference_with_recompute_kv.py configs/2wikimqa_eval.yaml
```

Equivalent evaluation configurations are available for HotpotQA and MuSiQue.

## Configuration structure

```yaml
# Models to evaluate
models:
  - /path/to/Qwen3-14B

# Dataset and device
dataset: 2wikimqa
device: "cuda:0"

# Recomputation settings
top_p: 0.15
lego_k: 4
batch_size: [1, 4, 8]
default_split: true
chunk_size: 1024
layer_indices: null

# Generation and evaluation
max_new_tokens: 32
num_samples: 200

# Strategies
strategies:
  - name: baseline
  - name: no_recompute
  - name: guided_recompute
    method: norm
  - name: double_guided
    method: entropy
  - name: cacheblend
  - name: lego
  - name: lego2
```

## Parameters

| Parameter | Description |
|---|---|
| `models` | Model paths or Hugging Face model identifiers to evaluate |
| `dataset` | Benchmark name, such as `2wikimqa`, `hotpotqa`, or `musique` |
| `device` | Execution device, for example `cuda:0`, `cpu`, or `auto` |
| `top_p` | Fraction of positions selected by guided-recomputation strategies |
| `lego_k` | Number of positions used by the LEGO strategy when applicable |
| `batch_size` | One batch size or a list of batch sizes to compare |
| `default_split` | Use passage boundaries when true; otherwise use fixed-size chunks |
| `chunk_size` | Fixed chunk length used when `default_split` is false |
| `layer_indices` | Layers used for importance scoring; `null` selects the implementation default |
| `max_new_tokens` | Maximum number of generated tokens |
| `num_samples` | Optional evaluation-sample limit |
| `strategies` | Strategies and scoring methods evaluated in the run |

## Strategies

- `baseline`: standard full-context inference.
- `no_recompute`: reuse the extracted cache without selective recomputation.
- `guided_recompute`: select positions using an importance score such as `norm`, `vatp`, or `entropy`.
- `double_guided`: apply the two-stage guided-recomputation variant.
- `cacheblend`: run the CacheBlend comparison.
- `lego` and `lego2`: run the LEGO comparison variants.

Model paths and device identifiers in the supplied configurations are environment-specific and should be updated before running an evaluation.
