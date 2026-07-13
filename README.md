# [ICML 2026] InfoFlow KV: Information-Flow-Aware KV Recomputation for Long Context

[Paper](https://arxiv.org/abs/2603.05353) · [Project page](https://infoflow-kv.github.io)

> [!IMPORTANT]
> The sequence-parallel (SP) and Ring-Attention experiments are available on the [`ring` branch](https://github.com/tx467/InfoFlow-KV/tree/ring). Use that branch for `sp_guided_recompute`, `sp_cacheblend`, `sp_lego`, and `ring_attention`.

Official implementation of **InfoFlow KV**, an information-flow-aware approach to selective key-value (KV) cache recomputation for long-context inference.

Reusing document-level KV caches avoids repeated prefilling, but independently cached chunks lose global causal dependencies. InfoFlow KV identifies tokens that can effectively carry query-relevant information, restores an inference-consistent RoPE geometry, and reorders chunks to improve information propagation. This repository contains the language-model and vision-language-model experiments from the paper.

## Branches

| Branch | Purpose |
| --- | --- |
| [`ring`](https://github.com/tx467/InfoFlow-KV/tree/ring) | Ring-Attention-style sequence-parallel implementation |
| [`main`](https://github.com/tx467/InfoFlow-KV/tree/main) | LLM and VLM InfoFlow KV single-GPU implementation |

## Repository contents on `main`

| Directory | Scope | Models and benchmarks |
| --- | --- | --- |
| [`llm/`](llm/) | Text-only LLM implementation; SP/Ring experiments are on the `ring` branch | Qwen3, Llama, ChatGLM; LongBench v1 and v2 |
| [`vlm/`](vlm/) | Selective KV recomputation with image chunking | Qwen3-VL-8B-Instruct; BLINK and VLMEvalKit |

The `ring` branch includes the `sp_guided_recompute`, `sp_cacheblend`, `sp_lego`, and `ring_attention` evaluation methods. The VLM implementation on `main` provides the corresponding recomputation pipeline for multimodal inputs.

## Setup

### SP and Ring-Attention experiments

Clone the `ring` branch:

```bash
git clone --branch ring --recurse-submodules https://github.com/tx467/InfoFlow-KV.git
cd InfoFlow-KV
```

### LLM and VLM experiments

```bash
git clone --recurse-submodules https://github.com/tx467/InfoFlow-KV.git
cd InfoFlow-KV
```

If the repository was cloned without submodules, initialize them with:

```bash
git submodule update --init --recursive
```

### Prepare an environment

The LLM experiments were run with Python 3.10, CUDA 12.1, PyTorch 2.4.0, Transformers 4.53.0, FlashAttention 2.6.3, FlashInfer 0.2.0, and Triton 3.0.0. Core LLM versions are recorded in [`llm/requirements.txt`](llm/requirements.txt). VLM dependencies are listed in [`vlm/requirements.txt`](vlm/requirements.txt).

The two pipelines have different dependency sets. Use separate environments and install CUDA-specific packages that match the CUDA and PyTorch versions on your system.

### Provide models and datasets

Model weights and full benchmark datasets are not bundled. Replace the `/path/to/...` values in the selected YAML configuration before running an experiment. Cluster launchers additionally use environment variables such as `MODEL_PATH`, `DATASET_DIR`, `OUTPUT_DIR`, and `VLMEVALKIT_DIR`.

## Quick start

### Ring-Attention sequence-parallel inference

The sequence-parallel inference implementation is available on the [`ring` branch](https://github.com/tx467/InfoFlow-KV/tree/ring).

### LLM single-GPU inference

Edit the `models` entry in [`llm/configs/2wikimqa_eval.yaml`](llm/configs/2wikimqa_eval.yaml), then run:

```bash
cd llm
python scripts/inference_with_recompute_kv.py configs/2wikimqa_eval.yaml
```

Equivalent configurations are available for HotpotQA and MuSiQue.

### VLM inference

Set `model`, `cache_dir`, `dataset_dir`, and `output_dir` in [`vlm/configs/blink_counting.yaml`](vlm/configs/blink_counting.yaml), then run:

```bash
cd vlm
python scripts/evaluate.py --config configs/blink_counting.yaml
```

See [`llm/README.md`](llm/README.md) and [`vlm/README.md`](vlm/README.md) for pipeline-specific notes.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{teng2026infoflowkv,
  title   = {InfoFlow KV: Information-Flow-Aware KV Recomputation for Long Context},
  author  = {Teng, Xin and Zhang, Canyu and Zheng, Shaoyi and Zhuo, Danyang and Zhou, Tianyi and Wan, Shenji},
  booktitle = {Forty-third International Conference on Machine Learning},
  year    = {2026},
  url     = {https://openreview.net/forum?id=o8y6CoJsWA}
}
```

## License

This project is released under the [MIT License](LICENSE). Third-party components retain their original licenses; see [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md) for details.
