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


def make_showdown_8gb_charts() -> None:
    """Plot the 8GB RTX 2080 10-case autonomous tool showdown & speed-vs-score chart."""
    v2_path = RESULTS / "showdown_8gb_v2.json"
    qwen_on_path = RESULTS / "showdown_qwen35_think_on.json"
    if not (v2_path.exists() and qwen_on_path.exists()):
        return

    rows = json.loads(v2_path.read_text(encoding="utf-8"))
    qwen_on = json.loads(qwen_on_path.read_text(encoding="utf-8"))
    all_rows = rows + qwen_on

    size_gb = {
        "gemma4:e2b-it-q4_K_M": 7.2,
        "ministral-3:8b": 6.0,
        "ministral-3:3b": 3.0,
        "granite4:micro": 2.1,
        "qwen3.5:9b": 6.6,
        "phi4-mini:latest": 2.5,
    }
    verdict_colors = {
        "Unattended Ready": "#10b981",
        "Validator + Retry": "#f59e0b",
        "Net Frontier Tax": "#ef4444",
    }

    def fmt_label(r: dict) -> str:
        m = r["model"]
        gb = size_gb.get(m, 4.0)
        short = m.replace("-it-q4_K_M", "").replace(":latest", "")
        if m == "qwen3.5:9b":
            tag = "think=on" if r.get("think") == "true" else "think=off"
            return f"{short} ({tag}, {gb:.1f} GB)"
        return f"{short} ({gb:.1f} GB)"

    # Sort descending by composite score, then speed
    all_rows.sort(
        key=lambda r: (r["composite_agent_score"], r["tool_calling"]["avg_tps"]),
        reverse=True,
    )

    # Chart 1: Horizontal 2-panel Scorecard (10 Hard Tool Tests + Composite Score)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    fig.suptitle(
        "8GB GPU Autonomous Agent Showdown (RTX 2080, ctx=4096, Zero Frontier Help)",
        fontsize=13,
        fontweight="bold",
        y=0.97,
    )

    labels = [fmt_label(r) for r in all_rows][::-1]
    tool_pcts = [r["tool_calling"]["score_pct"] for r in all_rows][::-1]
    tool_raw = [f"{r['tool_calling']['passed']}/{r['tool_calling']['total']}" for r in all_rows][::-1]
    comp_pcts = [r["composite_agent_score"] for r in all_rows][::-1]
    verdicts = [r["offload_verdict"] for r in all_rows][::-1]
    cols = [verdict_colors.get(v, "#64748b") for v in verdicts]
    y = np.arange(len(labels))

    b1 = ax1.barh(y, tool_pcts, color=cols, height=0.6)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=9.5)
    ax1.set_xlim(0, 122)
    ax1.set_xlabel("Pass Rate (%)", fontsize=10, fontweight="bold")
    ax1.set_title("Track 1: 10 Hard Tool-Calling Traps", fontsize=11, pad=8)
    ax1.grid(axis="y", visible=False)
    for bar, pct, raw in zip(b1, tool_pcts, tool_raw):
        ax1.text(
            pct + 2.0,
            bar.get_y() + bar.get_height() / 2,
            f"{raw} ({pct:.0f}%)",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    b2 = ax2.barh(y, comp_pcts, color=cols, height=0.6)
    ax2.set_xlim(0, 125)
    ax2.set_xlabel("Composite Score (%)", fontsize=10, fontweight="bold")
    ax2.set_title("Overall Composite (Tools + Self-Heal + Math)", fontsize=11, pad=8)
    ax2.grid(axis="y", visible=False)
    for bar, pct, v in zip(b2, comp_pcts, verdicts):
        ax2.text(
            pct + 2.0,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor="#10b981", label="Unattended Ready"),
        Patch(facecolor="#f59e0b", label="Validator + Retry"),
        Patch(facecolor="#ef4444", label="Net Frontier Tax"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=9.5,
        bbox_to_anchor=(0.5, 0.01),
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.92])
    fig.savefig(OUT_DIR / "showdown_8gb_scorecard.png", dpi=170)
    plt.close(fig)

    # Chart 2: Speed vs Composite Score (bubble size = model GB) + Qwen think arrow
    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    ax.axhspan(85, 105, xmin=0.0, xmax=1.0, color="#10b981", alpha=0.07, zorder=0)
    ax.axhline(85, color="#10b981", linestyle=":", linewidth=1.2, zorder=1)
    ax.text(8, 102.2, "Unattended Ready Zone (85%+)", ha="left", fontsize=8.8, color="#047857", fontweight="bold")

    offsets = {
        ("gemma4:e2b-it-q4_K_M", "false"): (-14, 22, "right"),
        ("ministral-3:8b", "false"): (16, -24, "left"),
        ("ministral-3:3b", "false"): (0, 24, "center"),
        ("granite4:micro", "false"): (-14, -24, "right"),
        ("qwen3.5:9b", "false"): (16, 4, "left"),
        ("qwen3.5:9b", "true"): (16, -4, "left"),
        ("phi4-mini:latest", "false"): (-14, 14, "right"),
    }

    q_on_pt = None
    q_off_pt = None
    for r in all_rows:
        m = r["model"]
        th = r.get("think", "false")
        tps = r["tool_calling"]["avg_tps"]
        sc = r["composite_agent_score"]
        gb = size_gb.get(m, 4.0)
        col = verdict_colors.get(r["offload_verdict"], "#64748b")
        ax.scatter(
            tps,
            sc,
            s=gb * 75,
            color=col,
            alpha=0.88,
            edgecolors="#111827",
            linewidth=1.0,
            zorder=4,
        )
        if m == "qwen3.5:9b" and th == "true":
            q_on_pt = (tps, sc)
        elif m == "qwen3.5:9b" and th == "false":
            q_off_pt = (tps, sc)

        dx, dy, ha = offsets.get((m, th), (8, 6, "left"))
        short = m.replace("-it-q4_K_M", "").replace(":latest", "")
        if m == "qwen3.5:9b":
            short += " (think=on)" if th == "true" else " (think=off)"
        ax.annotate(
            f"{short}\n{sc:.1f}% | {tps:.0f} tok/s | {gb:.1f} GB",
            (tps, sc),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=8.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#e5e7eb", alpha=0.88),
            zorder=5,
        )

    if q_on_pt and q_off_pt:
        ax.annotate(
            "",
            xy=(q_off_pt[0], q_off_pt[1] - 2.2),
            xytext=(q_on_pt[0], q_on_pt[1] + 2.2),
            arrowprops=dict(arrowstyle="->", color="#2563eb", lw=2.0, ls="--"),
            zorder=3,
        )
        ax.text(
            q_on_pt[0] + 3.0,
            (q_on_pt[1] + q_off_pt[1]) / 2,
            "+26.7 pts\n(think=False)",
            fontsize=8.5,
            color="#1d4ed8",
            fontweight="bold",
            va="center",
        )

    ax.set_xlim(5, 155)
    ax.set_ylim(32, 105)
    ax.set_xlabel("Generation Speed on RTX 2080 8GB (tokens/sec)", fontsize=10.5, fontweight="bold")
    ax.set_ylabel("Composite Agent Score (%)", fontsize=10.5, fontweight="bold")
    ax.set_title(
        "Speed vs. Reliability on an 8GB GPU (Bubble Size = Model Size in GB)",
        fontsize=12.5,
        fontweight="bold",
        pad=12,
    )
    ax.legend(handles=legend_handles, loc="lower left", frameon=True, facecolor="white", edgecolor="#e5e7eb", fontsize=9)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "showdown_8gb_speed_vs_score.png", dpi=170)
    plt.close(fig)


def make_ep1_9model_charts() -> None:
    """Plot Episode 1 (9 models, 3 runs @ temp=0.2): Action vs Restraint + Speed vs Composite."""
    ep1_path = RESULTS / "showdown_8gb_ep1_3runs.json"
    if not ep1_path.exists():
        return

    rows = json.loads(ep1_path.read_text(encoding="utf-8"))
    size_gb = {
        "gemma4:e2b-it-q4_K_M": 7.2,
        "ministral-3:8b": 6.0,
        "ministral-3:3b": 3.0,
        "qwen3.5:9b": 6.6,
        "qwen3:8b": 5.2,
        "llama3.1:8b": 4.9,
        "nemotron-3-nano:4b": 2.8,
        "granite4:3b-h": 1.9,
        "granite4:micro": 2.1,
    }
    verdict_colors = {
        "Unattended Ready": "#10b981",
        "Validator + Retry": "#f59e0b",
        "Net Frontier Tax": "#ef4444",
    }

    def short_name(m: str) -> str:
        return m.replace("-it-q4_K_M", "").replace(":latest", "")

    # Sort descending by composite, then tool score, then speed
    rows.sort(
        key=lambda r: (
            r["composite_agent_score"],
            r["tool_calling"]["score_pct"],
            r["tool_calling"]["avg_tps"],
        ),
        reverse=True,
    )

    from matplotlib.patches import Patch

    # Chart 1: Action Tool Accuracy vs No-Call Restraint (3 runs @ temp=0.2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.8), sharey=True)
    fig.suptitle(
        "8GB GPU Tool Calling vs. No-Call Restraint (9 Models, 3 Runs @ temp=0.2, Exact Call Count)",
        fontsize=12.5,
        fontweight="bold",
        y=0.97,
    )

    labels = [f"{short_name(r['model'])} ({size_gb.get(r['model'], 4.0):.1f} GB)" for r in rows][::-1]
    act_pcts = [r["tool_calling"]["action_score_pct"] for r in rows][::-1]
    act_raw = [f"{r['tool_calling']['action_passed']}/{r['tool_calling']['action_total']}" for r in rows][::-1]
    extras = [r["tool_calling"]["extra_calls_rejected"] for r in rows][::-1]
    abs_pcts = [r["tool_calling"]["abstention_score_pct"] for r in rows][::-1]
    abs_raw = [f"{r['tool_calling']['abstention_passed']}/{r['tool_calling']['abstention_total']}" for r in rows][::-1]
    y = np.arange(len(labels))

    act_cols = ["#10b981" if p >= 95 else ("#f59e0b" if p >= 80 else "#ef4444") for p in act_pcts]
    abs_cols = ["#10b981" if p == 100 else ("#f59e0b" if p >= 75 else "#ef4444") for p in abs_pcts]

    b1 = ax1.barh(y, act_pcts, color=act_cols, height=0.62)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=9.5)
    ax1.set_xlim(0, 132)
    ax1.set_xlabel("Pass Rate (%)", fontsize=10, fontweight="bold")
    ax1.set_title("Track 1A: Action Tool Calls (21 checks / model)", fontsize=10.8, pad=8)
    ax1.grid(axis="y", visible=False)
    for bar, pct, raw, ex in zip(b1, act_pcts, act_raw, extras):
        note = f"  ({ex} extra calls!)" if ex > 0 else ""
        ax1.text(
            pct + 1.8,
            bar.get_y() + bar.get_height() / 2,
            f"{raw} ({pct:.0f}%){note}",
            va="center",
            fontsize=8.6,
            fontweight="bold",
            color="#b91c1c" if ex > 0 else "#111827",
        )

    b2 = ax2.barh(y, abs_pcts, color=abs_cols, height=0.62)
    ax2.set_xlim(0, 128)
    ax2.set_xlabel("Restraint Rate (%)", fontsize=10, fontweight="bold")
    ax2.set_title("Track 1B: No-Call Restraint (15 traps / model)", fontsize=10.8, pad=8)
    ax2.grid(axis="y", visible=False)
    for bar, pct, raw in zip(b2, abs_pcts, abs_raw):
        ax2.text(
            pct + 1.8,
            bar.get_y() + bar.get_height() / 2,
            f"{raw} ({pct:.0f}%)",
            va="center",
            fontsize=8.8,
            fontweight="bold",
            color="#b91c1c" if pct < 75 else "#111827",
        )

    legend_handles = [
        Patch(facecolor="#10b981", label="Rock Solid (95-100%)"),
        Patch(facecolor="#f59e0b", label="Borderline (75-94%)"),
        Patch(facecolor="#ef4444", label="Trigger-Happy / Flaky (<75%)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False, fontsize=9.2, bbox_to_anchor=(0.5, 0.01))
    plt.tight_layout(rect=[0, 0.06, 1, 0.92])
    fig.savefig(OUT_DIR / "ep1_action_vs_restraint.png", dpi=170)
    plt.close(fig)

    # Chart 2: Speed vs Composite Score across all 9 models (bubble size = model GB)
    fig, ax = plt.subplots(figsize=(10.4, 6.2))
    ax.axhspan(85, 103, xmin=0.0, xmax=1.0, color="#10b981", alpha=0.07, zorder=0)
    ax.axhline(85, color="#10b981", linestyle=":", linewidth=1.2, zorder=1)
    ax.text(8, 100.5, "Unattended Ready Zone (85%+ Composite, 0 Starvation)", ha="left", fontsize=8.8, color="#047857", fontweight="bold")

    offsets = {
        "ministral-3:8b": (-14, 18, "right"),
        "nemotron-3-nano:4b": (0, 22, "center"),
        "qwen3:8b": (16, 4, "left"),
        "qwen3.5:9b": (16, 0, "left"),
        "gemma4:e2b-it-q4_K_M": (-16, -6, "right"),
        "ministral-3:3b": (-8, 22, "right"),
        "llama3.1:8b": (-16, -24, "right"),
        "granite4:micro": (0, -26, "center"),
        "granite4:3b-h": (-14, 12, "right"),
    }

    for r in rows:
        m = r["model"]
        tps = r["tool_calling"]["avg_tps"]
        sc = r["composite_agent_score"]
        gb = size_gb.get(m, 4.0)
        col = verdict_colors.get(r["offload_verdict"], "#64748b")
        ax.scatter(
            tps,
            sc,
            s=gb * 78,
            color=col,
            alpha=0.88,
            edgecolors="#111827",
            linewidth=1.0,
            zorder=4,
        )
        dx, dy, ha = offsets.get(m, (10, 8, "left"))
        label_extra = " | 20% restraint!" if m == "llama3.1:8b" else ""
        ax.annotate(
            f"{short_name(m)}\n{sc:.1f}% | {tps:.0f} t/s | {gb:.1f} GB{label_extra}",
            (tps, sc),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=8.2,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#e5e7eb", alpha=0.90),
            zorder=5,
        )

    v_handles = [
        Patch(facecolor="#10b981", label="Unattended Ready"),
        Patch(facecolor="#f59e0b", label="Validator + Retry"),
        Patch(facecolor="#ef4444", label="Net Frontier Tax"),
    ]
    ax.set_xlim(5, 154)
    ax.set_ylim(42, 103)
    ax.set_xlabel("Generation Speed on RTX 2080 8GB (tokens/sec)", fontsize=10.5, fontweight="bold")
    ax.set_ylabel("Composite Agent Score (%)", fontsize=10.5, fontweight="bold")
    ax.set_title(
        "9-Model 8GB GPU Leaderboard: Speed vs. Agent Reliability (3 Runs @ temp=0.2)",
        fontsize=12.2,
        fontweight="bold",
        pad=12,
    )
    ax.legend(handles=v_handles, loc="lower left", frameon=True, facecolor="white", edgecolor="#e5e7eb", fontsize=9)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "ep1_9model_speed_vs_score.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    make_model_scorecard_chart()
    make_thinking_starvation_chart()
    make_speculative_4090_chart()
    make_showdown_8gb_charts()
    make_ep1_9model_charts()
    print("Generated charts in", OUT_DIR)
