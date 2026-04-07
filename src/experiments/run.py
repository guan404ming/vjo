"""
VJO Experiment Runner on Modal

Target model and judge model run in parallel on separate GPUs:
  - Target (1x A100): VJO search + BOA sampling
  - Judge (2x A100): Llama-3.1-70B nuanced evaluator (matching BOA)

The search calls the judge in real-time via Modal remote calls.

Usage:
  uv run modal run src/experiments/run.py
  uv run modal run src/experiments/run.py --model llama3 --n-prompts 10
  uv run modal run src/experiments/run.py --model qwen3 --time-budget 300
"""

from __future__ import annotations

import json

import modal

app = modal.App("vjo")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm>=0.6.0", "torch>=2.4.0", "transformers>=4.44.0", "numpy")
    .add_local_python_source("vjo")
)

model_cache = modal.Volume.from_name("vjo-model-cache", create_if_missing=True)

from vjo.config import load_all_prompts

PROMPTS = load_all_prompts()


# ---------------------------------------------------------------------------
# Judge service: Llama-3.1-70B on 2x A100, stays loaded between calls
# ---------------------------------------------------------------------------

@app.cls(image=image, gpu="A100-80GB:2", volumes={"/models": model_cache})
class JudgeService:
    @modal.enter()
    def load(self):
        from vjo.backend import NuancedJudge
        self.judge = NuancedJudge(download_dir="/models", tensor_parallel_size=2)

    @modal.method()
    def is_jailbreak(self, prompt: str, response: str) -> bool:
        return self.judge.is_jailbreak(prompt, response)


# ---------------------------------------------------------------------------
# Search: target model on 1x A100, calls judge service during search
# ---------------------------------------------------------------------------

@app.function(image=image, gpu="A100-80GB", timeout=3600, volumes={"/models": model_cache})
def run_search(
    model_key: str = "vicuna",
    prompts: list[str] | None = None,
    time_budget: float = 300.0,
    epsilon: float = 1e-4,
    n_max: int = 64,
    top_k: int = 50,
    n_boa_samples: int = 10,
) -> dict:
    """Run VJO search + BOA baseline, judging in real-time via remote judge."""
    import time
    from vllm import LLM

    from vjo.config import MODELS
    from vjo.backend import VLLMOracle
    from vjo.search import search

    cfg = MODELS[model_key]
    prompts = prompts or PROMPTS

    # Load target model
    print(f"Loading target: {cfg.name} ({cfg.hf_id})...")
    llm = LLM(
        model=cfg.hf_id,
        download_dir="/models",
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        enable_prefix_caching=True,
    )
    oracle = VLLMOracle(llm, cfg)

    # Connect to judge service (already loaded on separate GPUs)
    judge_service = JudgeService()

    class RemoteJudge:
        """Wraps the Modal judge service as a local JudgeOracle."""
        def is_jailbreak(self, prompt: str, response: str) -> bool:
            return judge_service.is_jailbreak.remote(prompt, response)

    judge = RemoteJudge()

    vjo_results = []
    boa_results = []

    for i, prompt in enumerate(prompts):
        print(f"\n{'='*50}")
        print(f"[{i+1}/{len(prompts)}] {prompt[:60]}...")

        # VJO: probability-ordered complete search
        stats = search(
            oracle, judge, prompt,
            epsilon=epsilon, n_max=n_max, top_k=top_k,
            time_budget=time_budget, block_list=oracle.block_list,
        )

        vjo_r = {
            "prompt_idx": i,
            "result": stats.result.value,
            "nodes_expanded": stats.nodes_expanded,
            "nodes_pruned": stats.nodes_pruned,
            "time": round(stats.time_elapsed, 2),
            "max_depth": stats.max_depth_reached,
        }
        if stats.witness:
            vjo_r["witness_length"] = stats.witness.length
            vjo_r["witness_log_prob"] = round(stats.witness.response_log_prob, 4)
            vjo_r["witness_text"] = stats.witness.response_text[:150]

        print(f"  VJO: {stats.result.value:16s} | "
              f"{stats.nodes_expanded} nodes | {stats.time_elapsed:.1f}s")

        # BOA baseline: random sampling + judge
        t0 = time.time()
        samples = oracle.sample_responses(prompt, n=n_boa_samples)
        boa_sat = False
        boa_witness = None
        for s in samples:
            if any(tid in oracle.block_list for tid in s["token_ids"]):
                continue
            if judge.is_jailbreak(prompt, s["text"]):
                boa_sat = True
                boa_witness = s
                break
        boa_time = time.time() - t0

        boa_r = {
            "prompt_idx": i,
            "result": "sat" if boa_sat else "unsat",
            "nodes_expanded": len(samples),
            "time": round(boa_time, 2),
        }
        if boa_witness:
            boa_r["witness_log_prob"] = round(boa_witness["total_log_prob"], 4)
            boa_r["witness_text"] = boa_witness["text"][:150]

        print(f"  BOA: {'sat' if boa_sat else 'unsat':16s} | "
              f"{len(samples)} samples | {boa_time:.1f}s")

        vjo_results.append(vjo_r)
        boa_results.append(boa_r)

    vjo_sat = sum(1 for r in vjo_results if r["result"] == "sat")
    vjo_unsat = sum(1 for r in vjo_results if r["result"] == "unsat")
    boa_sat_n = sum(1 for r in boa_results if r["result"] == "sat")

    return {
        "model": cfg.name,
        "model_key": model_key,
        "n_prompts": len(prompts),
        "config": {
            "epsilon": epsilon, "n_max": n_max, "top_k": top_k,
            "time_budget": time_budget,
        },
        "vjo": vjo_results,
        "boa": boa_results,
        "summary": {
            "vjo_sat": vjo_sat,
            "vjo_certified_unsat": vjo_unsat,
            "boa_sat": boa_sat_n,
            "vjo_total_time": round(sum(r["time"] for r in vjo_results), 1),
            "boa_total_time": round(sum(r["time"] for r in boa_results), 1),
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    model: str = "vicuna",
    n_prompts: int = 5,
    time_budget: float = 60.0,
    epsilon: float = 1e-4,
):
    from vjo.config import MODELS

    print(f"VJO | {MODELS[model].name} | {n_prompts} prompts | {time_budget}s")
    print(f"Judge: Llama-3.1-70B-Instruct (Zhu et al.)\n")

    result = run_search.remote(
        model_key=model,
        prompts=PROMPTS[:n_prompts],
        time_budget=time_budget,
        epsilon=epsilon,
    )

    s = result["summary"]
    print(f"\n{'='*60}")
    print(f"RESULTS ({result['n_prompts']} prompts)")
    print(f"{'='*60}")
    print(f"  {'':20s} {'VJO':>10s} {'BOA':>10s}")
    print(f"  {'SAT':20s} {s['vjo_sat']:>10d} {s['boa_sat']:>10d}")
    print(f"  {'Certified UNSAT':20s} {s['vjo_certified_unsat']:>10d} {'N/A':>10s}")
    print(f"  {'Time (s)':20s} {s['vjo_total_time']:>10.1f} {s['boa_total_time']:>10.1f}")

    import os
    os.makedirs("src/experiments/results", exist_ok=True)
    path = f"src/experiments/results/{model}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved to {path}")
