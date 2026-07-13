# YAML Configuration Reference

> [!NOTE]
> These YAML files belong to the standard, non-sequence-parallel recomputation runner on the [`main` branch](https://github.com/tx467/InfoFlow-KV/tree/main/llm). On this branch, SP/Ring experiments use `scripts/eval_longbench.py` and command-line arguments.

## Configuration structure

A typical evaluation configuration looks like this:

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

For the maintained YAML-driven runner, use the [main-branch LLM implementation](https://github.com/tx467/InfoFlow-KV/tree/main/llm).
