"""Generate publication-ready PNG charts from verified benchmark JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT_DIR = ROOT / "docs" / "charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#ffffff",
        "axes.edgecolor": "#d1d5db",
        "axes.labelcolor": "#111827",
        "xtick.color": "#374151",
        "ytick.color": "#374151",
        "text.color": "#111827",
        "font.sans-serif": ["DejaVu Sans", "Arial"],
        "axes.grid": True,
        "grid.color": "#e5e7eb",
        "grid.linestyle": "--",
        "grid.linewidth": 0.7,
    }
)


def make_model_scorecard_chart() -> None:
    """Plot Composite Agent Score vs. Speed & VRAM footprint."""
    models = [
        ("nemotron-3.5-bolt\n(20.5 GB, 4090)", 100.0, 87.5, 20.5, "#6366f1"),
        ("gemma4:e2b-it-q4_K_M\n(7.2 GB, 8GB GPU)", 96.7, 68.5, 7.2, "#10b981"),
        ("qwen3.5:9b\n(6.6 GB, 8GB GPU)", 60.0, 29.0, 6.6, "#f59e0b"),
        ("phi4-mini:latest\n(2.5 GB, 8GB GPU)", 17.8, 127.5, 2.5, "#64748b"),
        ("nemotron-mini:4b\n(2.7 GB, 8GB GPU)", 11.1, 126.5, 2.7, "#94a3b8"),
        ("deepseek-r1:8b\n(5.2 GB, Starved)", 0.0, 66.5, 5.2, "#ef4444"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1.35, 1]})
    fig.suptitle(
        "Local Agentic Benchmark Scorecard (Tool Calling + 2-Turn Self-Healing + Bayesian Math)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    labels = [m[0] for m in models][::-1]
    scores = [m[1] for m in models][::-1]
    colors = [m[4] for m in models][::-1]
    y = np.arange(len(labels))

    bars = ax1.barh(y, scores, color=colors, height=0.62, edgecolor="none")
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.set_xlim(0, 114)
    ax1.set_xlabel("Composite Agent Score (%)", fontsize=10, fontweight="bold")
    ax1.set_title("Deterministic Pass Rate", fontsize=11, pad=8)
    ax1.grid(axis="y", visible=False)
    for bar, s in zip(bars, scores):
        ax1.text(s + 1.8, bar.get_y() + bar.get_height() / 2, f"{s:.1f}%", va="center", fontsize=9, fontweight="bold")

    # Right panel: Pareto frontier (VRAM vs Score, bubble size = speed)
    offsets = {
        "nemotron-3.5-bolt": (0.2, -17, "right"),
        "gemma4:e2b-it-q4_K_M": (1.1, -3, "left"),
        "qwen3.5:9b": (1.0, 2, "left"),
        "phi4-mini:latest": (1.1, 8, "left"),
        "nemotron-mini:4b": (1.1, -2, "left"),
        "deepseek-r1:8b": (1.1, -4, "left"),
    }
    for name, score, tps, vram, col in models:
        short = name.split("\n")[0]
        ax2.scatter(vram, score, s=tps * 4.5, color=col, alpha=0.88, edgecolors="#111827", linewidth=0.8, zorder=3)
        dx, dy, ha = offsets[short]
        ax2.annotate(
            f"{short}\n({tps:.0f} t/s)",
            (vram, score),
            xytext=(vram + dx, score + dy),
            ha=ha,
            fontsize=8,
            fontweight="bold" if "gemma4" in short else "normal",
        )

    ax2.axvline(8.0, color="#10b981", linestyle=":", linewidth=1.3, label="8 GB GPU VRAM Ceiling")
    ax2.set_xlim(0, 24.5)
    ax2.set_ylim(-14, 114)
    ax2.set_xlabel("Model Footprint (GB VRAM)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Composite Agent Score (%)", fontsize=10, fontweight="bold")
    ax2.set_title("Pareto Frontier (Size = tok/s)", fontsize=11, pad=8)
    ax2.legend(loc="lower right", fontsize=8.5)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_DIR / "model_scorecard.png", dpi=160)
    plt.close(fig)


def make_thinking_starvation_chart() -> None:
    """Plot Thinking-Budget Starvation from thinking_budget_eval.json."""
    rows = json.loads((RESULTS / "thinking_budget_eval.json").read_text(encoding="utf-8"))
    tasks = [
        ("task1_concurrency", "Task 1:\nConcurrency Audit"),
        ("task2_indexer", "Task 2:\nIndex Refactor"),
        ("task3_parser", "Task 3:\nHardware Parser"),
    ]
    regimes = [
        ("regime_zero", "Zero Thinking\n(think=False)", "#10b981"),
        ("regime_budgeted", "Prompt-Budgeted\n(~150w, 3k cap)", "#f59e0b"),
        ("regime_deep", "Deep Unconstrained\n(think=True, 4k cap)", "#ef4444"),
    ]

    lookup = {(r["task_id"], r["regime_id"]): r for r in rows}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    fig.suptitle(
        "Empirical CoT Token Starvation: Prompt Thinking Budgets Still Emit 0 Output Chars (6/6 Runs)",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )

    x = np.arange(len(tasks))
    w = 0.25

    for idx, (rid, rlabel, col) in enumerate(regimes):
        resp_k = [lookup[(tid, rid)]["response_chars"] / 1000.0 for tid, _ in tasks]
        think_k = [lookup[(tid, rid)]["thinking_chars"] / 1000.0 for tid, _ in tasks]
        offset = (idx - 1) * w

        b1 = ax1.bar(x + offset, resp_k, width=w, color=col, label=rlabel)
        for bar, val in zip(b1, resp_k):
            lbl = "0!" if val == 0 else f"{val:.1f}k"
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.3,
                lbl,
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
                color="#b91c1c" if val == 0 else "#111827",
            )

        b2 = ax2.bar(x + offset, think_k, width=w, color=col, label=rlabel)
        for bar, val in zip(b2, think_k):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.4,
                f"{val:.1f}k" if val > 0 else "0",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

    ax1.set_xticks(x)
    ax1.set_xticklabels([t[1] for t in tasks], fontsize=9)
    ax1.set_ylabel("Final Usable Response (k chars)", fontsize=10, fontweight="bold")
    ax1.set_ylim(0, 14)
    ax1.set_title("Usable Final Output Emitted", fontsize=11, pad=8)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(axis="x", visible=False)

    ax2.set_xticks(x)
    ax2.set_xticklabels([t[1] for t in tasks], fontsize=9)
    ax2.set_ylabel("Internal <think> Tokens Consumed (k chars)", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 21.5)
    ax2.set_title("Hidden Reasoning Overhead (Exhausted Token Cap)", fontsize=11, pad=8)
    ax2.grid(axis="x", visible=False)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT_DIR / "thinking_starvation.png", dpi=160)
    plt.close(fig)


def make_speculative_4090_chart() -> None:
    """Plot RTX 4090 Speculative Decoding & Task Throughput from rtx4090_speculative_benchmark.json."""
    d = json.loads((RESULTS / "rtx4090_speculative_benchmark.json").read_text(encoding="utf-8"))
    spec = d["speculative_benchmark"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    fig.suptitle(
        "RTX 4090 (24 GB) Local Inference Benchmarks: Speculative Decoding (+37.5%) & Task Throughput",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )

    labels = ["Baseline\n(draft=0)", "Speculative\n(draft=7)"]
    tps_vals = [spec["draft_0_tps"], spec["draft_7_tps"]]
    wall_vals = [spec["draft_0_wall_sec"], spec["draft_7_wall_sec"]]

    bars = ax1.bar(labels, tps_vals, color=["#64748b", "#2563eb"], width=0.5)
    ax1.set_ylim(0, 130)
    ax1.set_ylabel("Generation Speed (tok/s)", fontsize=10, fontweight="bold")
    ax1.set_title(f"Speculative Decoding Speedup (+{spec['speedup_pct']:.1f}%)", fontsize=11, pad=8)
    ax1.grid(axis="x", visible=False)
    for bar, t, wall in zip(bars, tps_vals, wall_vals):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            t + 2.5,
            f"{t:.1f} tok/s\n({wall:.1f}s wall)",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    tasks = ["JSON Extraction\n(100% valid, 0.8s)", "Code Synthesis\n(9/10 tests)", "Deep Reasoning\n(0 output chars!)"]
    t_vals = [
        d["json_extraction_eval"]["tps"],
        d["code_synthesis_eval"]["tps"],
        d["deep_reasoning_eval"]["tps"],
    ]
    t_cols = ["#10b981", "#2563eb", "#ef4444"]
    bars2 = ax2.bar(tasks, t_vals, color=t_cols, width=0.52)
    ax2.set_ylim(0, 168)
    ax2.set_ylabel("Throughput (tok/s)", fontsize=10, fontweight="bold")
    ax2.set_title("Throughput by Task Class (23.5 GB VRAM, 45W Peak)", fontsize=11, pad=8)
    ax2.grid(axis="x", visible=False)
    for bar, v in zip(bars2, t_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 3, f"{v:.1f} tok/s", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT_DIR / "speculative_4090.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    make_model_scorecard_chart()
    make_thinking_starvation_chart()
    make_speculative_4090_chart()
    print("Generated charts in", OUT_DIR)
