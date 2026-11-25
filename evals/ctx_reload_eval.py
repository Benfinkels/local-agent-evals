#!/usr/bin/env python3
"""
Ollama `num_ctx` Reload & Single-Slot KV-Cache Eviction Benchmark (`evals/ctx_reload_eval.py`)

Tests two real-world multi-stage agent claims from r/LocalLLaMA on Ollama:
  1. Varying `num_ctx` per request (e.g. 2048 <-> 8192) forces a full model/KV reload
     penalty on every switch.
  2. Even at fixed `num_ctx`, rotating across 3 distinct ~1.5k-token system prompts
     (router -> parser -> responder) evicts Ollama's single prefix-cache slot vs.
     repeating the same system prompt (warm KV hit).
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any, Optional
import urllib.request

from evals.runner import _unload_model, normalize_ollama_url

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _make_system_prompt(role: str, target_words: int = 900) -> str:
    # ~1,200-1,400 tokens of distinct deterministic instructions per stage
    blocks = [
        f"You are Stage {role} in a multi-step local agent pipeline. "
        f"Rule {i} for {role}: validate schema field_{role}_{i}, reject malformed inputs, "
        f"and never leak internal metadata from stage_{role}_bucket_{i}."
        for i in range(1, target_words // 18)
    ]
    return "\n".join(blocks)


def _call_chat(
    base_url: str,
    model: str,
    system_prompt: str,
    user_msg: str,
    num_ctx: int,
    think: Optional[bool] = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        "stream": False,
        "keep_alive": "10m",
        "options": {"num_ctx": num_ctx, "num_predict": 32, "temperature": 0.0},
    }
    if think is not None:
        payload["think"] = think
    req = urllib.request.Request(
        f"{normalize_ollama_url(base_url)}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    wall_ms = round((time.perf_counter() - t0) * 1000, 1)
    return {
        "wall_ms": wall_ms,
        "load_ms": round(data.get("load_duration", 0) / 1e6, 1),
        "prompt_eval_ms": round(data.get("prompt_eval_duration", 0) / 1e6, 1),
        "prompt_eval_count": data.get("prompt_eval_count", 0),
        "eval_ms": round(data.get("eval_duration", 0) / 1e6, 1),
        "eval_count": data.get("eval_count", 0),
        "num_ctx": num_ctx,
    }


def benchmark_model(base_url: str, model: str, ctx_small: int, ctx_large: int, think: Optional[bool]) -> dict[str, Any]:
    prompts = {
        "ROUTER": _make_system_prompt("ROUTER"),
        "PARSER": _make_system_prompt("PARSER"),
        "RESPONDER": _make_system_prompt("RESPONDER"),
    }

    # Warmup at ctx_large
    warm = _call_chat(base_url, model, prompts["ROUTER"], "Reply with just OK.", ctx_large, think)

    # Mode A: Fixed num_ctx, SAME system prompt (KV cache hit baseline)
    same_runs = [
        _call_chat(base_url, model, prompts["ROUTER"], f"Confirm request #{i} with OK.", ctx_large, think)
        for i in range(1, 5)
    ]

    # Mode B: Fixed num_ctx, ROTATING 3 distinct system prompts (single-slot eviction test)
    roles = ["ROUTER", "PARSER", "RESPONDER", "ROUTER", "PARSER", "RESPONDER"]
    rot_runs = [
        _call_chat(base_url, model, prompts[r], f"Stage {r} check #{i}: reply OK.", ctx_large, think)
        for i, r in enumerate(roles, 1)
    ]

    # Mode C: Alternating num_ctx (ctx_small <-> ctx_large), SAME system prompt
    alt_ctxs = [ctx_small, ctx_large, ctx_small, ctx_large]
    alt_runs = [
        _call_chat(base_url, model, prompts["ROUTER"], f"Ctx switch #{i}: reply OK.", c, think)
        for i, c in enumerate(alt_ctxs, 1)
    ]

    def _avg(rows: list[dict[str, Any]], key: str) -> float:
        return round(sum(r[key] for r in rows) / len(rows), 1)

    return {
        "model": model,
        "ctx_small": ctx_small,
        "ctx_large": ctx_large,
        "warmup_ms": warm["wall_ms"],
        "fixed_same_prompt": {
            "avg_wall_ms": _avg(same_runs, "wall_ms"),
            "avg_load_ms": _avg(same_runs, "load_ms"),
            "avg_prompt_eval_ms": _avg(same_runs, "prompt_eval_ms"),
            "avg_prompt_tokens": _avg(same_runs, "prompt_eval_count"),
            "runs": same_runs,
        },
        "fixed_rotating_3_prompts": {
            "avg_wall_ms": _avg(rot_runs, "wall_ms"),
            "avg_load_ms": _avg(rot_runs, "load_ms"),
            "avg_prompt_eval_ms": _avg(rot_runs, "prompt_eval_ms"),
            "avg_prompt_tokens": _avg(rot_runs, "prompt_eval_count"),
            "runs": rot_runs,
        },
        "alternating_num_ctx": {
            "avg_wall_ms": _avg(alt_runs, "wall_ms"),
            "avg_load_ms": _avg(alt_runs, "load_ms"),
            "avg_prompt_eval_ms": _avg(alt_runs, "prompt_eval_ms"),
            "avg_prompt_tokens": _avg(alt_runs, "prompt_eval_count"),
            "runs": alt_runs,
        },
    }


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m evals.ctx_reload_eval")
    p.add_argument("--models", nargs="+", default=[])
    p.add_argument("--base-url", default="http://127.0.0.1:11434")
    p.add_argument("--ctx-small", type=int, default=2048)
    p.add_argument("--ctx-large", type=int, default=8192)
    p.add_argument("--think", choices=["auto", "true", "false"], default="false")
    p.add_argument("--output", default=None)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args(argv)

    if args.self_test:
        p1, p2 = _make_system_prompt("ROUTER"), _make_system_prompt("PARSER")
        assert p1 != p2 and len(p1.split()) > 600
        print("SELF_TEST_OK")
        return 0
    if not args.models:
        print("Pass --models <tag>... or --self-test", file=sys.stderr)
        return 1

    think_flag = None if args.think == "auto" else (args.think == "true")
    results = []
    for idx, m in enumerate(args.models):
        print(f"\n>>> Benchmarking Ollama ctx/cache behavior: {m}")
        res = benchmark_model(args.base_url, m, args.ctx_small, args.ctx_large, think_flag)
        results.append(res)
        print(
            f"  Fixed+Same: wall={res['fixed_same_prompt']['avg_wall_ms']}ms load={res['fixed_same_prompt']['avg_load_ms']}ms p_eval={res['fixed_same_prompt']['avg_prompt_eval_ms']}ms ({res['fixed_same_prompt']['avg_prompt_tokens']} tok)\n"
            f"  Fixed+3Rot: wall={res['fixed_rotating_3_prompts']['avg_wall_ms']}ms load={res['fixed_rotating_3_prompts']['avg_load_ms']}ms p_eval={res['fixed_rotating_3_prompts']['avg_prompt_eval_ms']}ms ({res['fixed_rotating_3_prompts']['avg_prompt_tokens']} tok)\n"
            f"  Alt num_ctx: wall={res['alternating_num_ctx']['avg_wall_ms']}ms load={res['alternating_num_ctx']['avg_load_ms']}ms p_eval={res['alternating_num_ctx']['avg_prompt_eval_ms']}ms ({res['alternating_num_ctx']['avg_prompt_tokens']} tok)"
        )
        if idx < len(args.models) - 1:
            _unload_model(m, args.base_url)

    out_path = Path(args.output) if args.output else (RESULTS_DIR / f"ctx_reload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[+] Saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
