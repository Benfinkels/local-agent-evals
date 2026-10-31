# Local Agentic LLM Evaluation Harness (`local-agent-evals`)

[![CI](https://github.com/Benfinkels/local-agent-evals/actions/workflows/ci.yml/badge.svg)](https://github.com/Benfinkels/local-agent-evals/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A deterministic, unit-test-driven evaluation framework designed to benchmark **small local language models (SLMs, 2.5GB–9B class)** against **workstation MoE baselines (74B-A9B)** on **real agentic execution loops** rather than subjective "LLM-as-a-Judge" prose grading.

---

## Why This Benchmark Exists

Most LLM benchmarks measure single-turn chat fluency or static multiple-choice recall. When deploying local models inside autonomous agent workflows, models fail in three specific operational ways that standard benchmarks miss:
1. **Runaway CoT Token Starvation:** Reasoning models burning 100% of their `num_predict` token budget inside `<think>...</think>` blocks and emitting zero executable code or tool payloads.
2. **Tool Schema Wrapper Hallucinations:** Models supporting `/api/chat` `tools` in theory, but wrapping arguments inside hallucinated nested keys (e.g., `{"parameters": {...}, "type": "..."}`) or failing to abstain when no tool is needed.
3. **Zero Self-Healing Capability:** Inability to read a real `unittest` / `pytest` stack trace on Turn 2 and patch a failing edge case.

---

## Benchmark Architecture (3 Deterministic Tracks)

| Track | Capability Tested | Verification Method |
| :--- | :--- | :--- |
| **Track 1: Native Tool Calling & Abstention** | Structured function dispatch (`run_pytest`, `query_device_telemetry`) + negative abstention control via Ollama `/api/chat` `tools` API | Strict AST/JSON parameter type verification + false-positive tool call check (`3/3` cases) |
| **Track 2: 2-Turn Self-Healing Code Loop** | Multi-constraint regex & boundary parser (`parse_hardware_identifiers`). If Turn 1 scores `<10/10`, the raw `unittest` failure traceback is fed back on Turn 2 | `py_compile` + 10-case `unittest` battery (`Turn 1` vs. `Turn 2` self-healing delta) |
| **Track 3: Sequential Bayesian & Z-Score Math** | Population Z-score outlier filtering ($|x - \mu| > z\sigma$) coupled with $k$-step sequential Bayesian posterior fault updating | 5 exact floating-point `assertAlmostEqual` unit tests (`TestSensorBayes`) |

---

## Empirical Scorecard (October 2026 Hardware Run)

Evaluated across two local hardware tiers:
* **Edge / Server Node (`Sheldon`):** Dual 8GB GPUs (NVIDIA RTX 2080 8GB + GTX 1080 8GB)
* **Workstation Baseline (`Hyperion`):** NVIDIA RTX 4090 (24GB VRAM)

| Model | Node / Footprint | Speed (tok/s) | Track 1: Tool Calling | Track 2: Self-Healing (`Turn 1` $\rightarrow$ `Turn 2`) | Track 3: Bayesian Math | CoT Token Starvation? | **Composite Agent Score** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nemotron-3.5-bolt`** *(74B-A9B)* | Hyperion (20.5 GB) | 84–91 | **3/3 (100%)** | `8/10` $\rightarrow$ **`10/10`** *(+2)* | **5/5 (100%)** | Yes *(if `think=True` $\le$ 4k)* | **100.0%** |
| **`gemma4:e2b-it-q4_K_M`** | **Sheldon (7.2 GB)** | **60–77** | **3/3 (100%)** | `8/10` $\rightarrow$ **`9/10`** *(+1)* | **5/5 (100%)** | **No (0%)** | **96.7%** |
| **`qwen3.5:9b`** | **Sheldon (6.6 GB)** | 22–36 | **3/3 (100%)** | `0/10` $\rightarrow$ **`8/10`** *(+8)* | 0/5 (0%) | **Yes** *(Starved Track 3)* | **60.0%** |
| **`phi4-mini:latest`** | Sheldon (2.5 GB) | 125–130 | 1/3 (33.3%) | `0/10` $\rightarrow$ `0/10` | 1/5 (20%) | No | **17.8%** |
| **`nemotron-mini:4b`** | Sheldon (2.7 GB) | 126–127 | 1/3 (33.3%) | `0/10` $\rightarrow$ `0/10` | 0/5 (0%) | No | **11.1%** |
| **`deepseek-r1:8b`** | Sheldon (5.2 GB) | 66–67 | 0/3 *(HTTP 400)* | `0/10` $\rightarrow$ `0/10` | 0/5 (0%) | **Yes (100% Starved)** | **0.0%** |
| **`nemotron-3-nano`** *(30B-A3B)* | Sheldon (24.0 GB) | — | *OOM (HTTP 500)* | *OOM (HTTP 500)* | *OOM (HTTP 500)* | — | *VRAM OOM* |

---

## Key Engineering Takeaways

1. **`gemma4:e2b-it-q4_K_M` (7.2 GB) is the Pareto-Optimal 8GB Agent Model:**
   * Achieved a **96.7% Composite Agent Score** at **60–77 tok/s** inside a single 8GB GPU—matching 97% of the capability of a 24GB 74B-A9B workstation model (`nemotron-3.5-bolt`) at one-third the VRAM cost.
   * Zero CoT token starvation, 100% tool-calling precision, 100% Bayesian math accuracy, and demonstrated autonomous Turn-2 self-healing from `unittest` tracebacks.
2. **The "Runaway CoT Starvation" Trap in Reasoning Models (`deepseek-r1:8b`, `qwen3.5:9b`, `nemotron-3.5-bolt`):**
   * Unconstrained Chain-of-Thought models frequently exhaust standard 2,048–4,096 output token caps inside `<think>` blocks before emitting the final code block or JSON payload.
   * **Mitigation:** Production agent loops must either disable thinking (`think=False`) for deterministic extraction/coding tasks or enforce a split token budget for reasoning vs. final synthesis.
3. **Why Ultra-Small 4B Models (`phi4-mini`, `nemotron-mini:4b`) Fail Strict Tool Routers:**
   * While delivering **125+ tok/s**, 4B models frequently hallucinate extra JSON nesting (`{"parameters": {"metric": "vram_free_mb"}, "type": "..."}`) that breaks strict Pydantic/JSON-Schema tool validators.

---

## Quick Start

### 1. Run Offline Harness Verification (No GPU Required)
Verify the deterministic grading engine, regex test suites, and Bayesian math verifiers:
```bash
pip install -r requirements.txt
PYTHONPATH=. pytest tests/ -v
```

### 2. Run Live Ollama Benchmark Suite
```bash
python3 evals/runner.py
```
Full telemetry is written to [`results/benchmark_summary.json`](results/benchmark_summary.json).

---

## 4. Phase 2 Expansion: Multimodal Vision-to-Tool & Verify-Then-Escalate Cascade

### Track 4: Edge Multimodal Vision $\rightarrow$ Structured Tool Dispatch (`gemma4:e2b-it-q4_K_M` on 8GB RTX 2080)
Using a deterministic, zero-dependency PNG OSD synthesizer (`evals/cascade.py`), we evaluated whether an 8GB edge model (`gemma4:e2b-it-q4_K_M`, 5.1B active, `think: false`) can inspect raw image frames and directly emit structured `/api/chat` tool calls in a single pass without waking a 24GB workstation GPU:

| Test Case | Synthetic OSD Frame | Expected Behavior | `gemma4:e2b` Result | Latency (RTX 2080) |
| :--- | :--- | :--- | :--- | :--- |
| **`thermal_critical`** | `NODE: GPU-4090 / TEMP: 98C / STATE: CRIT` | Call `trigger_thermal_shutdown(node="GPU-4090", temperature_c=98)` | **PASS (Exact Match)** | `7.17s` |
| **`driveway_sentry_alert`** | `CAM: GATE-01 / EVENT: VEHICLE / TIME: 23:45` | Call `dispatch_sentry_alert(camera_id="GATE-01", event_type="VEHICLE")` | **PASS (Exact Match)** | `12.42s` |
| **`normal_telemetry_abstain`** | `NODE: SHELDON-2080 / TEMP: 42C NORMAL` | **Abstain** from tool call & reply in plain text | **PASS (0 False Positives)** | `11.96s` |

* **Track 4 Score:** **3 / 3 (`100.0%`)** on 8GB VRAM (`$0.00` API cost).

### Track 5: `Gemma 4` ($0 Local) $\rightarrow$ `MiniMax-M2.7` Verify-Then-Escalate Cascade (`evals/cascade.py`)
Instead of routing every coding/agent prompt to a cloud model, `run_verify_then_escalate_cascade()` executes Tier 0 (`gemma4:e2b-it-q4_K_M` on local GPU) first, grades the output against deterministic unit tests (`run_code_tests`), and **only escalates failed assertions to Tier 1 (`MiniMax-M2.7`)**:

| Pipeline Mode | Tool Calling (Track 1) | Self-Healing Parser (Track 2) | Bayesian Math (Track 3) | Multimodal Vision (Track 4) | Composite Accuracy | Cloud Token Reduction |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 0 Standalone (`Gemma 4 e2B` Local)** | 100.0% (3/3) | 90.0% (9/10) | 100.0% (4/4) | 100.0% (3/3) | **97.5%** | **100.0% ($0)** |
| **Tier 0 $\rightarrow$ Tier 1 Cascade (`Gemma 4` + `MiniMax-M2.7`)** | **100.0% (3/3)** | **100.0% (10/10)** | **100.0% (4/4)** | **100.0% (3/3)** | **100.0%** | **95.0% (19/20 local)** |
