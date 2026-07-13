#!/usr/bin/env python
"""
TTFT vs Sequence Length benchmark.

Measures Time-To-First-Token scaling behavior across sequence lengths using
synthetic data. Compares single-GPU prefill, ring attention (library), and
sp_guided_recompute.

Methods are benchmarked in interleaved fashion (ring → sp_guided per iteration)
to match eval_longbench.py's per-sample execution pattern, ensuring consistent
GPU/NCCL state between methods.

Usage:
    # Multi-GPU (primary use case):
    torchrun --nproc_per_node=4 scripts/benchmark_ttft_scaling.py \
        --model /scratch/xt2251/models/Qwen3-14B \
        --seq_lengths 2048 4096 8192 16384 32768

    # Single-GPU (only runs single_gpu_prefill):
    python scripts/benchmark_ttft_scaling.py \
        --model /scratch/xt2251/models/Qwen3-14B
"""

import gc
import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.distributed as dist

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    parser = argparse.ArgumentParser(description="TTFT scaling benchmark")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--seq_lengths",
        type=int,
        nargs="+",
        default=[2048, 4096, 8192, 16384, 32768],
        help="Sequence lengths to benchmark",
    )
    parser.add_argument("--recompute_ratio", type=float, default=0.15)
    parser.add_argument("--num_warmup", type=int, default=1)
    parser.add_argument("--num_iterations", type=int, default=3)
    parser.add_argument("--output", type=str, default="results/ttft_scaling.json")
    return parser.parse_args()


def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return dist.get_rank(), dist.get_world_size(), local_rank
    return 0, 1, 0


def generate_context(tokenizer, target_length):
    """Generate deterministic synthetic context (identical on all GPUs)."""
    passage = """
    This is a passage about artificial intelligence and machine learning.
    Deep learning models have revolutionized many fields including natural
    language processing, computer vision, and speech recognition. Transformer
    architectures have become the foundation of modern large language models.
    """
    passage_tokens = len(tokenizer.encode(passage, add_special_tokens=False))
    num_passages = (target_length // passage_tokens) + 1
    passages = [f"\nPassage {i+1}: {passage.strip()}" for i in range(num_passages)]
    context = "".join(passages)
    tokens = tokenizer.encode(context, add_special_tokens=False)[:target_length]
    return tokenizer.decode(tokens)


def benchmark_single_gpu_prefill(model, tokenizer, seq_len, device, num_warmup, num_iterations):
    """Single GPU full prefill baseline."""
    query_ids = tokenizer("What is the main topic discussed in the passages above?", return_tensors="pt").input_ids.to(device)
    context_text = generate_context(tokenizer, seq_len)
    context_ids = tokenizer(context_text, return_tensors="pt").input_ids.to(device)
    full_ids = torch.cat([context_ids, query_ids], dim=1)

    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            outputs = model(full_ids, use_cache=True)
        del outputs
        torch.cuda.empty_cache()

    # Timed iterations
    timings = []
    for _ in range(num_iterations):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        with torch.no_grad():
            outputs = model(full_ids, use_cache=True)

        torch.cuda.synchronize(device)
        timings.append((time.perf_counter() - t0) * 1000)
        del outputs
        torch.cuda.empty_cache()

    del full_ids, context_ids, query_ids
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "mean_ms": sum(timings) / len(timings),
        "std_ms": torch.tensor(timings).std().item() if len(timings) > 1 else 0.0,
    }


def benchmark_interleaved(model, tokenizer, seq_len, device, config, group,
                          recompute_ratio, num_warmup, num_iterations):
    """Interleaved ring attention + sp_guided_recompute benchmark.

    Matches eval_longbench.py execution pattern: for each iteration, run
    ring_attention first, then sp_guided_recompute immediately after. This
    ensures sp_guided runs with the same GPU/NCCL state as in actual evaluation.
    """
    from ring_flash_attn import update_ring_flash_attn_params
    from ring_flash_attn.adapters.hf_adapter import use_ring_attn
    from transformers.cache_utils import DynamicCache
    from models.qwen.kv_cache import KVCacheExtractor
    from models.parallel import (
        DistributedExtractor,
        DistributedScorer,
        RingAttentionRecomputer,
    )
    from models.parallel.recomputer import all_gather_kv

    world_size = config.world_size
    rank = config.rank

    query_text = "What is the main topic discussed in the passages above?"
    query_ids = tokenizer(query_text, return_tensors="pt").input_ids.to(device)
    context_text = generate_context(tokenizer, seq_len)
    context_ids = tokenizer(context_text, return_tensors="pt").input_ids.to(device)

    # ---- Ring attention setup ----
    ring_input_ids = torch.cat([context_ids, query_ids], dim=1)
    actual_seq_len = ring_input_ids.shape[1]

    # Pad to world_size-divisible
    if actual_seq_len % world_size != 0:
        pad_len = world_size - (actual_seq_len % world_size)
        ring_input_ids = torch.cat([
            ring_input_ids,
            torch.full((1, pad_len), tokenizer.pad_token_id, device=device, dtype=ring_input_ids.dtype),
        ], dim=1)
        padded_seq_len = actual_seq_len + pad_len
    else:
        padded_seq_len = actual_seq_len

    position_ids = torch.arange(padded_seq_len, device=device).unsqueeze(0)
    cu_seqlens = torch.tensor([0, padded_seq_len], device=device, dtype=torch.int32)
    update_ring_flash_attn_params(cu_seqlens, group)

    chunk_size = padded_seq_len // world_size
    start_idx = rank * chunk_size
    end_idx = start_idx + chunk_size
    input_ids_chunk = ring_input_ids[:, start_idx:end_idx]
    position_ids_chunk = position_ids[:, start_idx:end_idx]

    last_rank_start = (world_size - 1) * chunk_size
    last_real_local_idx = actual_seq_len - 1 - last_rank_start

    def all_gather_kv_cache(local_kv):
        """All-gather KV cache from all GPUs to reconstruct full cache."""
        if isinstance(local_kv, DynamicCache):
            key_cache = local_kv.key_cache
            value_cache = local_kv.value_cache
        else:
            key_cache = [kv[0] for kv in local_kv]
            value_cache = [kv[1] for kv in local_kv]

        num_layers = len(key_cache)
        local_len = key_cache[0].shape[2]
        local_lens_tensor = torch.tensor([local_len], device=device, dtype=torch.long)
        all_lens = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
        dist.all_gather(all_lens, local_lens_tensor, group=group)
        all_lens = [int(l.item()) for l in all_lens]
        max_len = max(all_lens)

        full_cache = DynamicCache()
        full_cache.key_cache = []
        full_cache.value_cache = []

        for layer_idx in range(num_layers):
            k = key_cache[layer_idx]
            v = value_cache[layer_idx]
            B, H, local_T, D = k.shape

            k_padded = torch.zeros(B, H, max_len, D, device=device, dtype=k.dtype)
            v_padded = torch.zeros(B, H, max_len, D, device=device, dtype=v.dtype)
            k_padded[:, :, :local_T, :] = k
            v_padded[:, :, :local_T, :] = v

            k_gathered = [torch.zeros_like(k_padded) for _ in range(world_size)]
            v_gathered = [torch.zeros_like(v_padded) for _ in range(world_size)]

            dist.all_gather(k_gathered, k_padded.contiguous(), group=group)
            dist.all_gather(v_gathered, v_padded.contiguous(), group=group)

            k_full = torch.cat([k_gathered[r][:, :, :all_lens[r], :] for r in range(world_size)], dim=2)
            v_full = torch.cat([v_gathered[r][:, :, :all_lens[r], :] for r in range(world_size)], dim=2)

            full_cache.key_cache.append(k_full)
            full_cache.value_cache.append(v_full)

        return full_cache

    # ---- SP guided recompute setup ----
    base_extractor = KVCacheExtractor(model, tokenizer, model_type="qwen")
    extractor = DistributedExtractor(base_extractor, config)
    scorer = DistributedScorer(model, config, method="norm", layer_indices=[22, 23, 24, 25])
    recomputer = RingAttentionRecomputer(model, config, use_ring_attention=True)

    # ---- Define run_once functions ----
    def ring_run_once():
        """One iteration of ring attention: forward → first-token → all-gather KV."""
        use_ring_attn(True)

        torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids_chunk,
                position_ids=position_ids_chunk,
                use_cache=True,
            )

        # First-token extraction + broadcast (matches eval_longbench.py:335-341)
        if rank == world_size - 1:
            next_token = outputs.logits[:, last_real_local_idx, :].argmax(dim=-1, keepdim=True)
        else:
            next_token = torch.zeros((1, 1), dtype=torch.long, device=device)
        dist.broadcast(next_token, src=world_size - 1, group=group)

        full_kv = all_gather_kv_cache(outputs.past_key_values)

        torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        use_ring_attn(False)

        del outputs, full_kv, next_token
        torch.cuda.empty_cache()

        return elapsed_ms

    def sp_guided_run_once():
        """One iteration of sp_guided_recompute: extract → score → recompute → gather."""
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        # Step 1: Independent chunk extraction
        local_kv = extractor.extract_distributed(context_ids)

        # Step 2: Distributed scoring
        exclude_first = local_kv.local_seq_len if rank == 0 else 0
        exclude_tensor = torch.tensor([exclude_first], device=device, dtype=torch.long)
        dist.broadcast(exclude_tensor, src=0, group=config.process_group)
        exclude_first = int(exclude_tensor.item())

        local_important, global_important = scorer.score_distributed(
            local_kv, query_ids, top_ratio=recompute_ratio,
            exclude_first_tokens=exclude_first,
        )

        # Step 3: Distributed recompute
        updated_kv = recomputer.recompute_distributed(
            local_kv, local_important, global_important
        )
        del local_kv

        # Step 4: All-gather final KV
        full_kv = all_gather_kv(updated_kv, config)

        torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        del updated_kv, full_kv
        torch.cuda.empty_cache()

        return elapsed_ms

    # ---- Warmup (both methods) ----
    for _ in range(num_warmup):
        ring_run_once()
        sp_guided_run_once()
        gc.collect()
        torch.cuda.empty_cache()

    # ---- Timed iterations (interleaved: ring → sp_guided, matching eval pattern) ----
    ring_timings = []
    sp_guided_timings = []
    for _ in range(num_iterations):
        if dist.is_initialized():
            dist.barrier()
        ring_timings.append(ring_run_once())
        sp_guided_timings.append(sp_guided_run_once())
        gc.collect()
        torch.cuda.empty_cache()

    # Cleanup
    del ring_input_ids, input_ids_chunk, position_ids, position_ids_chunk
    del context_ids, query_ids
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "ring_attention": {
            "mean_ms": sum(ring_timings) / len(ring_timings),
            "std_ms": torch.tensor(ring_timings).std().item() if len(ring_timings) > 1 else 0.0,
        },
        "sp_guided_recompute": {
            "mean_ms": sum(sp_guided_timings) / len(sp_guided_timings),
            "std_ms": torch.tensor(sp_guided_timings).std().item() if len(sp_guided_timings) > 1 else 0.0,
        },
    }


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    device = f"cuda:{local_rank}"
    is_main = rank == 0

    if is_main:
        print("=" * 70)
        print("TTFT Scaling Benchmark (interleaved)")
        print("=" * 70)
        print(f"Model: {args.model}")
        print(f"World size: {world_size}")
        print(f"Sequence lengths: {args.seq_lengths}")
        print(f"Recompute ratio: {args.recompute_ratio}")
        print(f"Warmup: {args.num_warmup}, Iterations: {args.num_iterations}")
        print("=" * 70)

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if is_main:
        print("\nLoading model...")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": local_rank},
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Setup distributed config and ring attention
    config = None
    ring_group = None
    if world_size > 1:
        from models.parallel.config import DistributedConfig

        config = DistributedConfig.from_env(recompute_ratio=args.recompute_ratio)
        ring_group = dist.new_group(ranks=list(range(world_size)), backend="nccl")
        config.process_group = ring_group

        # Setup ring attention library (substitute once)
        from ring_flash_attn import substitute_hf_flash_attn
        from ring_flash_attn.adapters.hf_adapter import use_ring_attn

        substitute_hf_flash_attn(ring_group, heads_k_stride=1)
        use_ring_attn(False)  # Default OFF

    if is_main:
        print("Model loaded.\n")

    all_results = {
        "metadata": {
            "model": args.model,
            "world_size": world_size,
            "recompute_ratio": args.recompute_ratio,
            "num_iterations": args.num_iterations,
            "timestamp": datetime.now().isoformat(),
        },
        "results": [],
    }

    for seq_len in args.seq_lengths:
        if is_main:
            print(f"\n{'='*70}")
            print(f"Sequence length: {seq_len}")
            print(f"{'='*70}")

        # --- Single GPU prefill (rank 0 only, others wait) ---
        if is_main:
            print(f"\n  Single GPU Prefill:")

        if rank == 0:
            try:
                result = benchmark_single_gpu_prefill(
                    model, tokenizer, seq_len, device, args.num_warmup, args.num_iterations
                )
                if is_main:
                    print(f"    TTFT: {result['mean_ms']:.1f}ms (+/- {result['std_ms']:.1f}ms)")
                all_results["results"].append({
                    "seq_len": seq_len,
                    "method": "single_gpu_prefill",
                    "ttft_ms": result["mean_ms"],
                    "std_ms": result["std_ms"],
                })
            except Exception as e:
                if is_main:
                    print(f"    Failed: {e}")

        if dist.is_initialized():
            dist.barrier()

        # --- Interleaved ring attention + sp_guided_recompute (all ranks) ---
        if world_size > 1:
            try:
                results = benchmark_interleaved(
                    model, tokenizer, seq_len, device, config, ring_group,
                    args.recompute_ratio, args.num_warmup, args.num_iterations,
                )

                for method_name in ["ring_attention", "sp_guided_recompute"]:
                    r = results[method_name]
                    if is_main:
                        label = "Ring Attention" if method_name == "ring_attention" else f"SP Guided Recompute (ratio={args.recompute_ratio})"
                        print(f"\n  {label}:")
                        print(f"    TTFT: {r['mean_ms']:.1f}ms (+/- {r['std_ms']:.1f}ms)")
                        all_results["results"].append({
                            "seq_len": seq_len,
                            "method": method_name,
                            "ttft_ms": r["mean_ms"],
                            "std_ms": r["std_ms"],
                        })
            except Exception as e:
                if is_main:
                    print(f"    Interleaved benchmark failed: {e}")
                    import traceback
                    traceback.print_exc()

            if dist.is_initialized():
                dist.barrier()

    # Save results and print summary
    if is_main:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

        # Print summary table
        print(f"\n{'='*70}")
        print("Summary")
        print(f"{'='*70}")
        print(f"{'Seq Len':<10} {'Method':<25} {'TTFT (ms)':<15} {'vs 1-GPU':<10}")
        print("-" * 62)

        for seq_len in args.seq_lengths:
            seq_results = [r for r in all_results["results"] if r["seq_len"] == seq_len]
            single_gpu = next((r for r in seq_results if r["method"] == "single_gpu_prefill"), None)
            single_gpu_ttft = single_gpu["ttft_ms"] if single_gpu else None

            for r in seq_results:
                if r["method"] == "single_gpu_prefill":
                    speedup = "1.00x"
                elif single_gpu_ttft:
                    speedup = f"{single_gpu_ttft / r['ttft_ms']:.2f}x"
                else:
                    speedup = "N/A"
                print(f"{r['seq_len']:<10} {r['method']:<25} {r['ttft_ms']:<15.1f} {speedup:<10}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
