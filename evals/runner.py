#!/usr/bin/env python3
"""
Local Agentic LLM Benchmark Suite
Evaluates local Ollama models on:
1. Native Tool-Calling Precision & Abstention (/api/chat tools API)
2. Multi-Turn Self-Healing Code Generation (Turn 1 -> unittest traceback -> Turn 2 Patch)
3. Deterministic Applied Math & Bayesian Verification (unit-test graded)
4. Thinking Token Efficiency & Runaway CoT Starvation Rate
"""

import argparse
from datetime import datetime
import json
import math
import os
import py_compile
import re
import subprocess
import sys
import tempfile
import time
from typing import Optional
import urllib.error
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"


def normalize_ollama_url(url: Optional[str] = None) -> str:
    raw = (url or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").strip()
    if not raw.startswith(("http://", "https://")):
        raw = f"http://{raw}"
    return raw.rstrip("/")


OLLAMA_BASE = normalize_ollama_url()


def list_local_ollama_models(base_url: Optional[str] = None) -> list[str]:
    """Return locally installed Ollama model names without pulling anything."""
    url = f"{normalize_ollama_url(base_url or OLLAMA_BASE)}/api/tags"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return [m["name"] for m in data.get("models", []) if m.get("name")]


def ollama_chat(
    model: str,
    messages: list,
    tools: list = None,
    options: dict = None,
    timeout: int = 240,
    think: Optional[bool] = None,
    base_url: Optional[str] = None,
) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options or {"temperature": 0.2, "num_predict": 2048, "num_ctx": 8192},
    }
    if tools:
        payload["tools"] = tools
    if think is not None:
        payload["think"] = think

    endpoint = f"{normalize_ollama_url(base_url or OLLAMA_BASE)}/api/chat"
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        elapsed = time.perf_counter() - t0
        msg = data.get("message", {})
        content = msg.get("content", "") or ""
        thinking = msg.get("thinking", "") or ""
        # Also detect inline <think>...</think> blocks
        think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        if think_match and not thinking:
            thinking = think_match.group(1)
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        eval_count = data.get("eval_count", 0)
        eval_dur = data.get("eval_duration", 0)
        tps = (eval_count / (eval_dur / 1e9)) if eval_dur > 0 else (eval_count / max(elapsed, 0.01))

        return {
            "ok": True,
            "elapsed_sec": round(elapsed, 2),
            "eval_tokens": eval_count,
            "tps": round(tps, 2),
            "content": content,
            "thinking": thinking,
            "thinking_chars": len(thinking),
            "content_chars": len(content),
            "tool_calls": msg.get("tool_calls", []),
            "starved": (len(thinking) > 500 and len(content.strip()) == 0 and not msg.get("tool_calls")),
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "elapsed_sec": round(time.perf_counter() - t0, 2),
            "eval_tokens": 0,
            "tps": 0.0,
            "content": "",
            "thinking": "",
            "thinking_chars": 0,
            "content_chars": 0,
            "tool_calls": [],
            "starved": False,
        }


# ============================================================================
# TRACK 1: NATIVE TOOL CALLING & ABSTENTION PRECISION
# ============================================================================
AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_pytest",
            "description": "Execute pytest on a specific file or directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_path": {"type": "string", "description": "File or folder path to test"},
                    "maxfail": {"type": "integer", "description": "Stop after N failures"},
                    "verbose": {"type": "boolean", "description": "Enable verbose output"},
                },
                "required": ["target_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_device_telemetry",
            "description": "Query hardware sensor telemetry for a specific node.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_name": {"type": "string", "enum": ["sheldon", "hyperion", "sentry"]},
                    "metric": {"type": "string", "enum": ["gpu_temp", "vram_free_mb", "power_watts"]},
                },
                "required": ["node_name", "metric"],
            },
        },
    },
]


def eval_tool_calling(
    model: str,
    think: Optional[bool] = None,
    base_url: Optional[str] = None,
) -> dict:
    cases = [
        {
            "id": "pytest_dispatch",
            "prompt": "Run pytest on 'tests/test_parser.py' with verbose enabled and stop after 1 failure.",
            "expect_tool": "run_pytest",
            "check_args": lambda a: a.get("target_path") == "tests/test_parser.py" and int(a.get("maxfail", 0)) == 1,
        },
        {
            "id": "telemetry_lookup",
            "prompt": "Check how much free VRAM in MB is currently available on the 'sheldon' node.",
            "expect_tool": "query_device_telemetry",
            "check_args": lambda a: str(a.get("node_name", "")).lower() == "sheldon" and a.get("metric") == "vram_free_mb",
        },
        {
            "id": "abstention_check",
            "prompt": "Do NOT call any tools. In one short sentence, what does HTTP status code 429 mean?",
            "expect_tool": None,
            "check_args": lambda a: True,
        },
    ]

    passed = 0
    details = []
    total_tps = []

    for c in cases:
        res = ollama_chat(
            model,
            messages=[{"role": "user", "content": c["prompt"]}],
            tools=AGENT_TOOLS,
            options={"temperature": 0.0, "num_predict": 512, "num_ctx": 4096},
            think=think,
            base_url=base_url,
        )
        if res["tps"] > 0:
            total_tps.append(res["tps"])

        tcalls = res.get("tool_calls") or []
        success = False
        if c["expect_tool"] is None:
            # Should abstain from calling tools and answer in text
            success = (len(tcalls) == 0 and len(res["content"].strip()) > 5)
        else:
            if len(tcalls) >= 1:
                fn = tcalls[0].get("function", {})
                fname = fn.get("name")
                fargs = fn.get("arguments", {})
                if isinstance(fargs, str):
                    try:
                        fargs = json.loads(fargs)
                    except Exception:
                        fargs = {}
                try:
                    if isinstance(fargs, dict) and fname == c["expect_tool"] and c["check_args"](fargs):
                        success = True
                except (TypeError, ValueError):
                    success = False

        if success:
            passed += 1
        details.append({
            "case": c["id"],
            "passed": success,
            "tool_calls": tcalls,
            "elapsed_sec": res["elapsed_sec"],
            "error": res.get("error"),
        })

    return {
        "supported": any(len(d["tool_calls"]) > 0 for d in details[:2]),
        "passed": passed,
        "total": len(cases),
        "score_pct": round((passed / len(cases)) * 100, 1),
        "avg_tps": round(sum(total_tps) / len(total_tps), 2) if total_tps else 0.0,
        "details": details,
    }


# ============================================================================
# TRACK 2: 2-TURN SELF-HEALING CODE VERIFICATION LOOP
# ============================================================================
PARSER_PROMPT = """Implement a Python function `parse_hardware_identifiers(text: str) -> dict`.

Requirements:
Return a dictionary with strictly these 3 keys:
{
    'mac_address': str or None,   # Normalized lowercase colon-separated 12-hex-digit MAC e.g. '08:bf:b8:04:f4:8c', or None
    'serial_number': str or None, # Alphanumeric serial (>=6 chars) preceded by 'SN:', 'S/N:', 'Serial:', or 'SN ', stripped of trailing punctuation (!.,), or None
    'model_code': str or None     # Match known model code from ['AW3423DW', 'RTX 4090', 'RTX 4080', 'WYZECP1', 'HERO12'] or None
}

Rules:
- Accept MAC addresses separated by ':' or '-' (6 pairs of valid hex digits 0-9a-f). Reject invalid hex like 'GG' or short lengths.
- Do NOT extract a serial number from inside a hyphenated SKU like 'GTX1660-SN12345' unless 'SN:' / 'S/N:' / 'Serial:' / 'SN ' is preceded by start-of-string or whitespace.
- Output ONLY valid Python code inside a single ```python ... ``` block.
"""

PARSER_UNITTESTS = """
import unittest

class TestHardwareParser(unittest.TestCase):
    def test_01_mac_colons(self):
        self.assertEqual(parse_hardware_identifiers("MAC: 08:BF:B8:04:F4:8C")['mac_address'], "08:bf:b8:04:f4:8c")

    def test_02_mac_hyphens(self):
        self.assertEqual(parse_hardware_identifiers("Addr 08-bf-b8-04-f4-8c")['mac_address'], "08:bf:b8:04:f4:8c")

    def test_03_mac_invalid_hex(self):
        self.assertIsNone(parse_hardware_identifiers("MAC: 08:GG:B8:04:F4:8C")['mac_address'])

    def test_04_mac_short(self):
        self.assertIsNone(parse_hardware_identifiers("MAC: 08:BF:B8:04:F4")['mac_address'])

    def test_05_serial_sn(self):
        self.assertEqual(parse_hardware_identifiers("Unit SN: ABC123456")['serial_number'], "ABC123456")

    def test_06_serial_slash(self):
        self.assertEqual(parse_hardware_identifiers("Item S/N: XYZ987654!")['serial_number'], "XYZ987654")

    def test_07_serial_sku_rejection(self):
        self.assertIsNone(parse_hardware_identifiers("Part: GTX1660-SN12345")['serial_number'])

    def test_08_model_monitor(self):
        self.assertEqual(parse_hardware_identifiers("Monitor AW3423DW OLED")['model_code'], "AW3423DW")

    def test_09_model_gpu(self):
        self.assertEqual(parse_hardware_identifiers("Card: NVIDIA RTX 4090")['model_code'], "RTX 4090")

    def test_10_empty_noise(self):
        res = parse_hardware_identifiers("Random noise @#$")
        self.assertEqual(res, {'mac_address': None, 'serial_number': None, 'model_code': None})

if __name__ == '__main__':
    unittest.main()
"""


def extract_code(text: str) -> str:
    m = re.findall(r"```python\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        return m[-1].strip()
    m = re.findall(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        return m[-1].strip()
    return text.strip()


def run_code_tests(code_str: str, test_suite: str, total_tests: int, tag: str) -> dict:
    with tempfile.TemporaryDirectory(prefix="local_agent_evals_") as tmp_dir:
        test_file = Path(tmp_dir) / f"tmp_test_{tag}.py"
        test_file.write_text(f"{code_str}\n\n{test_suite}", encoding="utf-8")
        try:
            py_compile.compile(str(test_file), doraise=True)
        except Exception as e:
            return {"compiled": False, "passed": 0, "total": total_tests, "score_pct": 0.0, "trace": str(e)}

        try:
            proc = subprocess.run(
                [sys.executable, str(test_file)],
                capture_output=True,
                text=True,
                timeout=15,
            )
        except subprocess.TimeoutExpired:
            return {
                "compiled": True,
                "passed": 0,
                "total": total_tests,
                "score_pct": 0.0,
                "trace": "Execution timed out after 15s",
            }

        out = proc.stderr
        if proc.returncode == 0:
            return {"compiled": True, "passed": total_tests, "total": total_tests, "score_pct": 100.0, "trace": "OK"}

        fail_m = re.findall(r"FAILED \((?:failures=(\d+))?(?:, )?(?:errors=(\d+))?\)", out)
        failures = int(fail_m[0][0]) if (fail_m and fail_m[0][0]) else 0
        errors = int(fail_m[0][1]) if (fail_m and fail_m[0][1]) else 0
        if not fail_m:
            passed = 0
        else:
            passed = max(0, total_tests - failures - errors)
        return {
            "compiled": True,
            "passed": passed,
            "total": total_tests,
            "score_pct": round((passed / total_tests) * 100, 1),
            "trace": out[-1500:],
        }


def eval_self_healing_code(
    model: str,
    think: Optional[bool] = None,
    base_url: Optional[str] = None,
) -> dict:
    safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", model)
    messages = [{"role": "user", "content": PARSER_PROMPT}]
    t1 = ollama_chat(
        model,
        messages,
        options={"temperature": 0.2, "num_predict": 2048, "num_ctx": 8192},
        think=think,
        base_url=base_url,
    )
    c1 = extract_code(t1["content"])
    r1 = run_code_tests(c1, PARSER_UNITTESTS, 10, f"{safe_name}_t1")

    # Turn 2: Self-Healing if Turn 1 < 100%
    if r1["passed"] < 10:
        feedback = (
            f"Your Turn 1 implementation scored {r1['passed']}/10 unit tests.\n"
            f"Here is the exact test failure output:\n```\n{r1['trace']}\n```\n"
            "Fix the bugs and return the complete corrected `parse_hardware_identifiers` function in a single ```python block."
        )
        messages.append({"role": "assistant", "content": t1["content"] or "```python\n# empty\n```"})
        messages.append({"role": "user", "content": feedback})
        t2 = ollama_chat(
            model,
            messages,
            options={"temperature": 0.2, "num_predict": 2048, "num_ctx": 8192},
            think=think,
            base_url=base_url,
        )
        c2 = extract_code(t2["content"])
        r2 = run_code_tests(c2, PARSER_UNITTESTS, 10, f"{safe_name}_t2")
        t2_elapsed = t2["elapsed_sec"]
        t2_starved = t2["starved"]
    else:
        r2 = r1
        t2_elapsed = 0.0
        t2_starved = False

    return {
        "turn1_passed": r1["passed"],
        "turn1_score_pct": r1["score_pct"],
        "turn1_compiled": r1["compiled"],
        "turn1_elapsed_sec": t1["elapsed_sec"],
        "turn1_tps": t1["tps"],
        "turn1_starved": t1["starved"],
        "turn2_passed": r2["passed"],
        "turn2_score_pct": r2["score_pct"],
        "turn2_elapsed_sec": t2_elapsed,
        "turn2_starved": t2_starved,
        "healed_delta": r2["passed"] - r1["passed"],
    }


# ============================================================================
# TRACK 3: DETERMINISTIC APPLIED MATH & BAYESIAN SENSOR FILTERING
# ============================================================================
MATH_PROMPT = """Write a Python function `analyze_sensor_bayes(readings: list[float], z_threshold: float, prior_fault: float, sensitivity: float, false_pos_rate: float) -> dict`.

Mathematical Specification:
1. Outlier Filtering (Z-Score):
   - Compute population mean `mu = sum(readings) / len(readings)` and population standard deviation `sigma = sqrt(sum((x - mu)**2 for x in readings) / len(readings))`.
   - Identify outliers where `abs(x - mu) > z_threshold * sigma` (if sigma == 0, no outliers).
   - Let `k` be the integer count of outlier readings.
   - Compute `clean_mean` as the mean of the non-outlier readings (rounded to 4 decimal places).
2. Sequential Bayesian Updating:
   - Starting from `p = prior_fault`, perform `k` sequential independent Bayesian updates for each outlier observed:
     `p = (p * sensitivity) / (p * sensitivity + (1 - p) * false_pos_rate)`
   - Round final `posterior_fault_prob` to 6 decimal places.
3. Return dictionary:
   `{'outlier_count': int, 'clean_mean': float, 'posterior_fault_prob': float}`

Output ONLY valid Python code inside a single ```python ... ``` block.
"""

MATH_UNITTESTS = """
import unittest

class TestSensorBayes(unittest.TestCase):
    def test_01_single_outlier_bayes(self):
        # readings: 9 values of 10.0 and 1 value of 50.0 -> mu=14.0, sigma=12.0
        # |50-14| = 36 > 2.0*12 -> k=1 outlier (50.0), clean_mean=10.0
        # Bayes update: (0.05 * 0.90) / (0.05*0.90 + 0.95*0.10) = 0.045 / 0.14 = 0.321429
        res = analyze_sensor_bayes([10.0]*9 + [50.0], z_threshold=2.0, prior_fault=0.05, sensitivity=0.90, false_pos_rate=0.10)
        self.assertEqual(res['outlier_count'], 1)
        self.assertAlmostEqual(res['clean_mean'], 10.0, places=4)
        self.assertAlmostEqual(res['posterior_fault_prob'], 0.321429, places=5)

    def test_02_zero_outliers(self):
        res = analyze_sensor_bayes([20.0, 20.0, 20.0, 20.0], z_threshold=2.0, prior_fault=0.12, sensitivity=0.8, false_pos_rate=0.05)
        self.assertEqual(res['outlier_count'], 0)
        self.assertAlmostEqual(res['clean_mean'], 20.0, places=4)
        self.assertAlmostEqual(res['posterior_fault_prob'], 0.12, places=6)

    def test_03_two_outliers_sequential_bayes(self):
        # 8 values of 0.0, 2 values of 100.0 -> mu=20.0, sigma=40.0
        # z_threshold=1.5 -> 1.5*40 = 60.0 -> |100-20|=80 > 60 -> k=2 outliers
        # Update 1 from p=0.10, sens=0.9, fpr=0.1: p1 = 0.09 / (0.09 + 0.09) = 0.5
        # Update 2 from p1=0.5: p2 = 0.45 / (0.45 + 0.05) = 0.9
        res = analyze_sensor_bayes([0.0]*8 + [100.0, 100.0], z_threshold=1.5, prior_fault=0.10, sensitivity=0.90, false_pos_rate=0.10)
        self.assertEqual(res['outlier_count'], 2)
        self.assertAlmostEqual(res['clean_mean'], 0.0, places=4)
        self.assertAlmostEqual(res['posterior_fault_prob'], 0.90, places=5)

    def test_04_negative_outlier(self):
        res = analyze_sensor_bayes([5.0]*9 + [-45.0], z_threshold=2.0, prior_fault=0.20, sensitivity=0.80, false_pos_rate=0.20)
        # k=1: p = (0.2*0.8) / (0.16 + 0.8*0.2) = 0.5
        self.assertEqual(res['outlier_count'], 1)
        self.assertAlmostEqual(res['clean_mean'], 5.0, places=4)
        self.assertAlmostEqual(res['posterior_fault_prob'], 0.50, places=5)

    def test_05_high_threshold_no_triggers(self):
        res = analyze_sensor_bayes([1.0, 2.0, 3.0, 4.0, 5.0], z_threshold=3.0, prior_fault=0.01, sensitivity=0.99, false_pos_rate=0.01)
        self.assertEqual(res['outlier_count'], 0)
        self.assertAlmostEqual(res['clean_mean'], 3.0, places=4)
        self.assertAlmostEqual(res['posterior_fault_prob'], 0.01, places=6)

if __name__ == '__main__':
    unittest.main()
"""


def eval_math_verification(
    model: str,
    think: Optional[bool] = None,
    base_url: Optional[str] = None,
) -> dict:
    safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", model)
    messages = [{"role": "user", "content": MATH_PROMPT}]
    t1 = ollama_chat(
        model,
        messages,
        options={"temperature": 0.1, "num_predict": 2048, "num_ctx": 8192},
        think=think,
        base_url=base_url,
    )
    code = extract_code(t1["content"])
    r1 = run_code_tests(code, MATH_UNITTESTS, 5, f"{safe_name}_math")
    return {
        "passed": r1["passed"],
        "total": 5,
        "score_pct": r1["score_pct"],
        "compiled": r1["compiled"],
        "elapsed_sec": t1["elapsed_sec"],
        "tps": t1["tps"],
        "starved": t1["starved"],
    }


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m evals.runner",
        description="Run deterministic agentic benchmarks against any local Ollama model.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="One or more Ollama model tags to evaluate (e.g. --models gemma4:e2b qwen3.5:9b). "
        "If omitted, evaluates all currently installed local Ollama models (never auto-pulls).",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Ollama base URL (default: $OLLAMA_HOST or http://localhost:11434).",
    )
    parser.add_argument(
        "--think",
        choices=["auto", "true", "false"],
        default="auto",
        help="Control Ollama's `think` parameter (default: auto = omit parameter).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write JSON results (default: results/run_<timestamp>.json).",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List locally installed Ollama models and exit.",
    )
    return parser.parse_args(argv)


def format_markdown_table(summary: list[dict]) -> str:
    lines = [
        "| Model | Track 1: Tool Calling | Track 2: Self-Healing (T1 -> T2) | Track 3: Bayesian Math | CoT Starved? | Composite |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for row in summary:
        tc = row["tool_calling"]
        sh = row["self_healing"]
        mv = row["bayesian_math"]
        starved = "Yes" if (sh["turn1_starved"] or sh["turn2_starved"] or mv["starved"]) else "No"
        lines.append(
            f"| `{row['model']}` | {tc['passed']}/{tc['total']} ({tc['score_pct']}%) | "
            f"{sh['turn1_passed']}/10 -> {sh['turn2_passed']}/10 (+{sh['healed_delta']}) | "
            f"{mv['passed']}/{mv['total']} ({mv['score_pct']}%) | {starved} | **{row['composite_agent_score']}%** |"
        )
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    base_url = normalize_ollama_url(args.base_url)
    think_flag = None if args.think == "auto" else (args.think == "true")

    if args.list_models:
        models = list_local_ollama_models(base_url)
        print(f"Installed Ollama models at {base_url}:")
        for m in models:
            print(f"  - {m}")
        return 0

    models = args.models
    if not models:
        try:
            models = list_local_ollama_models(base_url)
        except Exception as e:
            print(f"[!] Could not reach Ollama at {base_url}: {e}", file=sys.stderr)
            print("    Start Ollama or pass --base-url http://<host>:11434 --models <tag>", file=sys.stderr)
            return 1
        if not models:
            print(f"[!] No models found at {base_url}. Pull one first (e.g. `ollama pull gemma4:e2b`).", file=sys.stderr)
            return 1

    print("=" * 72)
    print(f"LOCAL AGENTIC BENCHMARK SUITE | endpoint={base_url} | think={args.think}")
    print("=" * 72)

    summary = []
    for m in models:
        print(f"\n>>> Benchmarking Model: {m}")
        tc = eval_tool_calling(m, think=think_flag, base_url=base_url)
        print(f"  [Track 1 - Tool Calling]  Score: {tc['passed']}/{tc['total']} ({tc['score_pct']}%) | Supported: {tc['supported']}")

        sh = eval_self_healing_code(m, think=think_flag, base_url=base_url)
        print(f"  [Track 2 - Self-Healing]  Turn 1: {sh['turn1_passed']}/10 -> Turn 2: {sh['turn2_passed']}/10 (Delta: +{sh['healed_delta']}) | TPS: {sh['turn1_tps']}")

        mv = eval_math_verification(m, think=think_flag, base_url=base_url)
        print(f"  [Track 3 - Bayesian Math] Score: {mv['passed']}/5 ({mv['score_pct']}%) | TPS: {mv['tps']}")

        composite = round((tc["score_pct"] + sh["turn2_score_pct"] + mv["score_pct"]) / 3.0, 1)
        summary.append({
            "model": m,
            "endpoint": base_url,
            "think": args.think,
            "tool_calling": tc,
            "self_healing": sh,
            "bayesian_math": mv,
            "composite_agent_score": composite,
        })

    out_path = Path(args.output) if args.output else (RESULTS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + format_markdown_table(summary))
    print(f"\n[+] Saved benchmark results to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
