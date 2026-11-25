#!/usr/bin/env python3
"""
25-Email Position & Shuffle Stress Eval (`evals/shuffle_eval.py`)

Directly tests whether 8GB local models suffer from "lost in the middle" recall
drops or position-sensitive prompt-injection hijacks when the exact same 25 emails
from Case A4 (`A4_inbox_25batch_prompt_injection`) are reordered across 6 layouts:
  1. baseline       - original order (targets @ #13, #18; injections @ #11, #16)
  2. top_pinned     - targets pinned to #1, #2; injections in middle (#13, #14)
  3. dead_middle    - targets buried at #12, #13; injections on prime edges (#1, #25)
  4. bottom_pinned  - targets pinned to #24, #25; injections at #1, #2
  5. shuffle_42     - deterministic full shuffle (seed=42)
  6. shuffle_99     - deterministic full shuffle (seed=99)
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import random
import sys
from typing import Any, Optional

from evals.briefing_eval import (
    RESULTS_DIR,
    build_cases,
    call_ollama_json,
    grade_case,
)
from evals.runner import _unload_model, normalize_ollama_url


def build_shuffle_variants() -> list[dict[str, Any]]:
    a4 = next(c for c in build_cases() if c["id"] == "A4_inbox_25batch_prompt_injection")
    emails: list[dict[str, Any]] = list(a4["payload"])
    expected_ids = set(a4["expected_ids"])
    injection_ids = set(a4["injection_ids"])

    targets = [e for e in emails if e["id"] in expected_ids]
    injections = [e for e in emails if e["id"] in injection_ids]
    noise = [e for e in emails if e["id"] not in expected_ids and e["id"] not in injection_ids]
    assert len(targets) == 2 and len(injections) == 2 and len(noise) == 21

    # 1. baseline
    p_base = list(emails)
    # 2. top_pinned: targets at #1,#2; injections at #13,#14
    p_top = targets + noise[:10] + injections + noise[10:]
    # 3. dead_middle: injections at #1 and #25 (attention edges); targets buried at #12,#13
    p_mid = [injections[0]] + noise[:10] + targets + noise[10:] + [injections[1]]
    # 4. bottom_pinned: injections at #1,#2; targets at #24,#25
    p_bot = injections + noise + targets
    # 5 & 6. deterministic full shuffles
    p_s42 = list(emails)
    random.Random(42).shuffle(p_s42)
    p_s99 = list(emails)
    random.Random(99).shuffle(p_s99)

    layouts = [
        ("baseline", p_base),
        ("top_pinned", p_top),
        ("dead_middle", p_mid),
        ("bottom_pinned", p_bot),
        ("shuffle_42", p_s42),
        ("shuffle_99", p_s99),
    ]

    variants: list[dict[str, Any]] = []
    for name, payload in layouts:
        assert len(payload) == 25 and {e["id"] for e in payload} == a4["valid_ids"]
        t_pos = [i + 1 for i, e in enumerate(payload) if e["id"] in expected_ids]
        inj_pos = [i + 1 for i, e in enumerate(payload) if e["id"] in injection_ids]
        variants.append(
            {
                **a4,
                "id": f"A4_{name}",
                "layout": name,
                "payload": payload,
                "target_positions": t_pos,
                "injection_positions": inj_pos,
            }
        )
    return variants


def evaluate_model_shuffle(
    model: str,
    variants: list[dict[str, Any]],
    base_url: str,
    think: Optional[bool],
    num_ctx: int,
    runs: int,
    temperature: float,
) -> dict[str, Any]:
    layout_results: dict[str, Any] = {}
    strict_pass_k = 0
    total_raw_pass = 0
    total_runs = len(variants) * runs
    total_caught_both = 0
    total_fp = 0
    total_hijacks = 0
    tps_vals: list[float] = []

    for v in variants:
        user_json = json.dumps(v["payload"], ensure_ascii=False)
        run_logs = []
        layout_pass = 0
        layout_caught = 0
        layout_hijacks = 0
        layout_fp = 0
        for _ in range(runs):
            resp = call_ollama_json(
                model,
                v["sys_prompt"],
                user_json,
                base_url=base_url,
                think=think,
                num_ctx=num_ctx,
                temperature=temperature,
            )
            if resp.get("tps", 0) > 0:
                tps_vals.append(resp["tps"])
            g = grade_case(v, resp.get("content", "")) if resp.get("ok") else {
                "passed": False,
                "invalid_json": True,
                "false_positive": False,
                "hallucinated_id": False,
                "injection_hijacked": False,
                "selected": [],
                "reason": resp.get("error", "call failed"),
            }
            sel = set(g.get("selected") or [])
            caught_both = v["expected_ids"].issubset(sel)
            if g["passed"]:
                layout_pass += 1
                total_raw_pass += 1
            if caught_both:
                layout_caught += 1
                total_caught_both += 1
            if g.get("false_positive"):
                layout_fp += 1
                total_fp += 1
            if g.get("injection_hijacked"):
                layout_hijacks += 1
                total_hijacks += 1
            run_logs.append(
                {
                    "passed": g["passed"],
                    "caught_both_targets": caught_both,
                    "selected": sorted(sel),
                    "false_positive": bool(g.get("false_positive")),
                    "injection_hijacked": bool(g.get("injection_hijacked")),
                    "reason": g.get("reason", ""),
                }
            )
        if layout_pass == runs:
            strict_pass_k += 1
        layout_results[v["layout"]] = {
            "target_positions": v["target_positions"],
            "injection_positions": v["injection_positions"],
            "pass_runs": layout_pass,
            "caught_both_runs": layout_caught,
            "false_positives": layout_fp,
            "hijacks": layout_hijacks,
            "runs": runs,
            "details": run_logs,
        }

    avg_tps = round(sum(tps_vals) / len(tps_vals), 1) if tps_vals else 0.0
    return {
        "model": model,
        "runs_per_layout": runs,
        "layouts_pass_k": strict_pass_k,
        "total_layouts": len(variants),
        "raw_passed": total_raw_pass,
        "raw_total": total_runs,
        "caught_both_targets": total_caught_both,
        "false_positives": total_fp,
        "injection_hijacks": total_hijacks,
        "avg_tps": avg_tps,
        "by_layout": layout_results,
    }


def format_markdown(summary: list[dict[str, Any]]) -> str:
    layouts = ["baseline", "top_pinned", "dead_middle", "bottom_pinned", "shuffle_42", "shuffle_99"]
    header = "| Model | Strict (`pass^k`) | Target Recall | Hijacks | Base | Top | Middle | Bottom | Shuf42 | Shuf99 | Speed |"
    sep = "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    rows = [header, sep]
    for r in summary:
        bl = r["by_layout"]
        cells = [f"{bl[k]['pass_runs']}/{bl[k]['runs']} (R:{bl[k]['caught_both_runs']})" for k in layouts]
        rows.append(
            f"| `{r['model']}` | **{r['layouts_pass_k']}/{r['total_layouts']}** ({r['raw_passed']}/{r['raw_total']}) | "
            f"{r['caught_both_targets']}/{r['raw_total']} | {r['injection_hijacks']} | "
            + " | ".join(cells)
            + f" | {r['avg_tps']:.0f} t/s |"
        )
    return "\n".join(rows)


def _self_test() -> None:
    vs = build_shuffle_variants()
    assert len(vs) == 6
    by_name = {v["layout"]: v for v in vs}
    assert by_name["top_pinned"]["target_positions"] == [1, 2]
    assert by_name["dead_middle"]["target_positions"] == [12, 13]
    assert by_name["dead_middle"]["injection_positions"] == [1, 25]
    assert by_name["bottom_pinned"]["target_positions"] == [24, 25]
    assert by_name["shuffle_42"]["target_positions"] != by_name["baseline"]["target_positions"]
    print("SELF_TEST_OK:", {k: (v["target_positions"], v["injection_positions"]) for k, v in by_name.items()})


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m evals.shuffle_eval")
    p.add_argument("--models", nargs="+", default=[])
    p.add_argument("--base-url", default=None)
    p.add_argument("--think", choices=["auto", "true", "false"], default="false")
    p.add_argument("--num-ctx", type=int, default=4096)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--temp", type=float, default=0.2)
    p.add_argument("--output", default=None)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args(argv)

    if args.self_test:
        _self_test()
        return 0
    if not args.models:
        print("Pass --models <tag>... or --self-test", file=sys.stderr)
        return 1

    variants = build_shuffle_variants()
    base_url = normalize_ollama_url(args.base_url)
    think_flag = None if args.think == "auto" else (args.think == "true")
    summary: list[dict[str, Any]] = []

    print(f"25-EMAIL SHUFFLE EVAL | layouts={len(variants)} | runs={args.runs} | temp={args.temp} | ctx={args.num_ctx}")
    for idx, m in enumerate(args.models):
        print(f"\n>>> Benchmarking: {m}")
        res = evaluate_model_shuffle(m, variants, base_url, think_flag, args.num_ctx, args.runs, args.temp)
        print(
            f"  strict={res['layouts_pass_k']}/6 ({res['raw_passed']}/{res['raw_total']}) | "
            f"target_recall={res['caught_both_targets']}/{res['raw_total']} | "
            f"hijacks={res['injection_hijacks']} | tps={res['avg_tps']}"
        )
        summary.append(res)
        if idx < len(args.models) - 1:
            _unload_model(m, base_url)

    out_path = Path(args.output) if args.output else (RESULTS_DIR / f"shuffle_25email_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n" + format_markdown(summary))
    print(f"\n[+] Saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
