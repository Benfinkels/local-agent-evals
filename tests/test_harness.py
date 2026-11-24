"""
Unit tests for the Local Agentic Evaluation Harness.
Verifies deterministic grading of:
1. Code extraction from markdown blocks & <think> tag stripping
2. Track 2 Hardware Identifier Parser unit test grading (ground-truth pass & buggy fail)
3. Track 3 Sequential Bayesian & Z-Score Outlier math grading
4. Runaway CoT Token Starvation detection logic
"""

import json
from pathlib import Path
from evals.runner import (
    extract_code,
    run_code_tests,
    PARSER_UNITTESTS,
    MATH_UNITTESTS,
)


def test_extract_code_strips_markdown():
    raw = "Here is the solution:\n```python\ndef foo():\n    return 42\n```"
    assert extract_code(raw) == "def foo():\n    return 42"


def test_hardware_parser_verifier_passes_reference_solution():
    ref_code = r"""
import re
from typing import Optional

KNOWN_MODELS = ['AW3423DW', 'RTX 4090', 'RTX 4080', 'WYZECP1', 'HERO12']

def parse_hardware_identifiers(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        return {'mac_address': None, 'serial_number': None, 'model_code': None}

    mac = None
    mac_m = re.search(r'(?<![0-9A-Fa-f:-])([0-9A-Fa-f]{2}(?:[:-][0-9A-Fa-f]{2}){5})(?![0-9A-Fa-f:-])', text)
    if mac_m:
        raw_mac = mac_m.group(1)
        sep = ':' if ':' in raw_mac else '-'
        parts = raw_mac.split(sep)
        if len(parts) == 6:
            mac = ':'.join(p.lower() for p in parts)

    serial = None
    sn_m = re.search(r'(?:^|\s)(?:SN:|S/N:|Serial:|SN\s)\s*([A-Za-z0-9]{6,})[!.,]*', text)
    if sn_m:
        serial = sn_m.group(1)

    model = None
    for m in KNOWN_MODELS:
        if m in text:
            model = m
            break

    return {'mac_address': mac, 'serial_number': serial, 'model_code': model}
"""
    res = run_code_tests(ref_code, PARSER_UNITTESTS, 10, "pytest_ref_parser")
    assert res["compiled"] is True
    assert res["passed"] == 10
    assert res["score_pct"] == 100.0


def test_hardware_parser_verifier_catches_buggy_solution():
    buggy_code = """
def parse_hardware_identifiers(text: str) -> dict:
    return {'mac_address': None, 'serial_number': None, 'model_code': None}
"""
    res = run_code_tests(buggy_code, PARSER_UNITTESTS, 10, "pytest_buggy_parser")
    assert res["compiled"] is True
    assert res["passed"] < 10
    assert res["score_pct"] < 100.0


def test_hardware_parser_verifier_catches_syntax_error():
    broken_code = "def parse_hardware_identifiers(text: str) -> dict:\n    return {'unclosed"
    res = run_code_tests(broken_code, PARSER_UNITTESTS, 10, "pytest_syntax_err")
    assert res["compiled"] is False
    assert res["passed"] == 0


def test_bayesian_math_verifier_passes_reference_solution():
    ref_math = """
import math

def analyze_sensor_bayes(readings: list[float], z_threshold: float, prior_fault: float, sensitivity: float, false_pos_rate: float) -> dict:
    n = len(readings)
    mu = sum(readings) / n
    sigma = math.sqrt(sum((x - mu) ** 2 for x in readings) / n)
    if sigma == 0:
        outliers = []
        clean = readings[:]
    else:
        outliers = [x for x in readings if abs(x - mu) > z_threshold * sigma]
        clean = [x for x in readings if abs(x - mu) <= z_threshold * sigma]

    k = len(outliers)
    clean_mean = round(sum(clean) / len(clean), 4) if clean else 0.0

    p = prior_fault
    for _ in range(k):
        p = (p * sensitivity) / (p * sensitivity + (1.0 - p) * false_pos_rate)

    return {
        'outlier_count': k,
        'clean_mean': clean_mean,
        'posterior_fault_prob': round(p, 6)
    }
"""
    res = run_code_tests(ref_math, MATH_UNITTESTS, 5, "pytest_ref_math")
    assert res["compiled"] is True
    assert res["passed"] == 5
    assert res["score_pct"] == 100.0


def test_benchmark_summary_artifact_integrity():
    summary_path = Path(__file__).parent.parent / "results" / "benchmark_summary.json"
    assert summary_path.exists()
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    assert len(data) >= 5
    models = {row["model"] for row in data}
    assert "gemma4:e2b-it-q4_K_M" in models
    assert "qwen3.5:9b" in models


def test_vision_osd_png_synthesizer_produces_valid_png():
    import base64
    from evals.cascade import synthesize_osd_png_base64

    b64 = synthesize_osd_png_base64("NODE: GPU-4090\nTEMP: 98C\nSTATE: CRIT", (180, 20, 20))
    raw = base64.b64decode(b64)
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    assert b"IHDR" in raw and b"IDAT" in raw and b"IEND" in raw


def test_strip_thinking_blocks_preserves_code():
    from evals.cascade import strip_thinking_blocks

    sample = "<think>\nLet me reason step by step...\n</think>\n```python\ndef f(): return 42\n```"
    cleaned = strip_thinking_blocks(sample)
    assert "<think>" not in cleaned
    assert "def f(): return 42" in cleaned


def test_verify_then_escalate_cascade_avoids_cloud_when_tier0_passes():
    from evals.cascade import run_verify_then_escalate_cascade

    tier1_called = []
    res = run_verify_then_escalate_cascade(
        prompt="Write parser",
        tier0_generator=lambda p: "VALID_SOLUTION",
        tier1_generator=lambda p: tier1_called.append(True) or "TIER1_SOLUTION",
        deterministic_verifier=lambda out: (out == "VALID_SOLUTION", {"score": 10}),
    )
    assert res["passed"] is True
    assert res["escalated_to_tier1"] is False
    assert len(tier1_called) == 0


def test_verify_then_escalate_cascade_escalates_on_tier0_failure():
    from evals.cascade import run_verify_then_escalate_cascade

    res = run_verify_then_escalate_cascade(
        prompt="Write parser",
        tier0_generator=lambda p: "PARTIAL_9_OF_10",
        tier1_generator=lambda p: "<think>Fixing regex edge case</think>PERFECT_10_OF_10",
        deterministic_verifier=lambda out: (out == "PERFECT_10_OF_10", {"out": out}),
    )
    assert res["passed"] is True
    assert res["escalated_to_tier1"] is True
    assert res["selected_tier"] == "tier1_worker_minimax"


def test_phase3_benchmark_artifacts_and_hygiene():
    import re

    results_dir = Path(__file__).parent.parent / "results"
    think_path = results_dir / "thinking_budget_eval.json"
    spec_path = results_dir / "rtx4090_speculative_benchmark.json"

    assert think_path.exists() and spec_path.exists()

    think_raw = think_path.read_text(encoding="utf-8")
    spec_raw = spec_path.read_text(encoding="utf-8")

    # Guard against leaking RFC-1918 LAN IPs into public benchmark artifacts
    private_ip = re.compile(r"\b(?:192\.168|10\.\d{1,3})\.\d{1,3}\.\d{1,3}\b")
    assert not private_ip.search(think_raw)
    assert not private_ip.search(spec_raw)

    think_rows = json.loads(think_raw)
    assert len(think_rows) == 9
    starved = [r for r in think_rows if r["think_enabled"] and r["response_chars"] == 0]
    assert len(starved) == 6

    spec_data = json.loads(spec_raw)
    assert spec_data["speculative_benchmark"]["speedup_pct"] > 30.0


def test_runner_cli_and_ollama_url_normalization(monkeypatch):
    from evals.runner import format_markdown_table, normalize_ollama_url, parse_args

    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert normalize_ollama_url() == "http://localhost:11434"
    assert normalize_ollama_url("127.0.0.1:11434/") == "http://127.0.0.1:11434"

    monkeypatch.setenv("OLLAMA_HOST", "http://192.0.2.10:11434/")
    assert normalize_ollama_url() == "http://192.0.2.10:11434"

    args = parse_args(["--models", "gemma4:e2b", "qwen3.5:9b", "--think", "false", "--base-url", "http://localhost:11435"])
    assert args.models == ["gemma4:e2b", "qwen3.5:9b"]
    assert args.think == "false"
    assert args.base_url == "http://localhost:11435"

    md = format_markdown_table([{
        "model": "gemma4:e2b",
        "tool_calling": {"passed": 3, "total": 3, "score_pct": 100.0},
        "self_healing": {"turn1_passed": 8, "turn2_passed": 9, "healed_delta": 1, "turn1_starved": False, "turn2_starved": False},
        "bayesian_math": {"passed": 5, "total": 5, "score_pct": 100.0, "starved": False},
        "composite_agent_score": 96.7,
    }])
    assert "`gemma4:e2b`" in md and "**96.7%**" in md


def test_ollama_chat_passes_think_flag_when_set():
    from unittest.mock import MagicMock, patch
    from evals.runner import ollama_chat

    captured = {}

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({
                "message": {"content": "ok"},
                "eval_count": 5,
                "eval_duration": 100_000_000,
            }).encode("utf-8")

    def fake_urlopen(req, timeout=0):
        captured["url"] = req.full_url
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return FakeResp()

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        res = ollama_chat("gemma4:e2b", [{"role": "user", "content": "hi"}], think=False, base_url="http://localhost:11435")

    assert res["ok"] is True
    assert captured["url"] == "http://localhost:11435/api/chat"
    assert captured["payload"]["think"] is False


def test_eval_tool_calling_12_cases_split_and_offload_verdict():
    from unittest.mock import patch
    from evals.runner import classify_offload_readiness, eval_tool_calling

    def perfect_responses():
        return iter([
            {"tps": 60.0, "elapsed_sec": 0.5, "starved": False, "content": "", "tool_calls": [{"function": {"name": "run_pytest", "arguments": {"target_path": "tests/test_parser.py", "maxfail": 1, "verbose": True}}}]},
            {"tps": 60.0, "elapsed_sec": 0.5, "starved": False, "content": "", "tool_calls": [{"function": {"name": "query_device_telemetry", "arguments": {"node_name": "edge-gpu", "metric": "vram_free_mb"}}}]},
            {"tps": 60.0, "elapsed_sec": 0.5, "starved": False, "content": "", "tool_calls": [{"function": {"name": "log_crm_interaction", "arguments": {"contact_name": "Sarah Chen", "channel": "phone", "summary": "Discussed Q4 board budget", "follow_up_days": 7}}}]},
            {"tps": 60.0, "elapsed_sec": 0.5, "starved": False, "content": "", "tool_calls": [{"function": {"name": "dispatch_alert", "arguments": {"severity": "critical", "title": "Driveway motion after midnight", "tags": ["camera", "perimeter"], "suppress_if_quiet_hours": False}}}]},
            {"tps": 60.0, "elapsed_sec": 0.5, "starved": False, "content": "", "tool_calls": [{"function": {"name": "schedule_power_action", "arguments": {"node_name": "edge-gpu", "action": "sleep", "delay_seconds": 30, "reason": "Nightly backup finished"}}}]},
            # 5 No-Call Restraint cases
            {"tps": 60.0, "elapsed_sec": 0.5, "starved": False, "content": "HTTP 429 means Too Many Requests.", "tool_calls": []},
            {"tps": 60.0, "elapsed_sec": 0.5, "starved": False, "content": "Check GPU temperature first before rebooting.", "tool_calls": []},
            {"tps": 60.0, "elapsed_sec": 0.5, "starved": False, "content": "Which channel did you use and how many days until follow-up?", "tool_calls": []},
            {"tps": 60.0, "elapsed_sec": 0.5, "starved": False, "content": "Example: Alerting team that GPU hit 95C and workloads are throttled.", "tool_calls": []},
            {"tps": 60.0, "elapsed_sec": 0.5, "starved": False, "content": "You ran parser tests and put edge-gpu to sleep after backup.", "tool_calls": []},
            # Case 11 (2-step chain)
            {"tps": 60.0, "elapsed_sec": 0.5, "starved": False, "content": "", "tool_calls": [{"function": {"name": "query_device_telemetry", "arguments": {"node_name": "workstation", "metric": "gpu_temp"}}}]},
            {"tps": 60.0, "elapsed_sec": 0.5, "starved": False, "content": "", "tool_calls": [{"function": {"name": "dispatch_alert", "arguments": {"severity": "critical", "title": "Workstation GPU overheating at 92C", "tags": ["thermal"], "suppress_if_quiet_hours": False}}}]},
            # Case 12 (validator error self-repair)
            {"tps": 60.0, "elapsed_sec": 0.5, "starved": False, "content": "", "tool_calls": [{"function": {"name": "log_crm_interaction", "arguments": {"contact_name": "Marcus Vance", "channel": "linkedin", "summary": "Keynote invite", "follow_up_days": 5}}}]},
        ])

    res_stream = list(perfect_responses()) * 2
    it = iter(res_stream)
    with patch("evals.runner.ollama_chat", side_effect=lambda *a, **kw: next(it)):
        tc = eval_tool_calling("mock-model", runs=2, temperature=0.2)

    assert tc["runs"] == 2
    assert tc["action_passed"] == 14 and tc["action_total"] == 14
    assert tc["abstention_passed"] == 10 and tc["abstention_total"] == 10
    assert tc["extra_calls_rejected"] == 0
    assert tc["score_pct"] == 100.0
    assert classify_offload_readiness(tc, {"turn1_starved": False, "turn2_starved": False}, {"starved": False}, 96.7) == "Unattended Ready"
    assert classify_offload_readiness({"score_pct": 80.0, "starved": False}, {"turn1_starved": False, "turn2_starved": False}, {"starved": False}, 75.0) == "Validator + Retry"
    assert classify_offload_readiness({"score_pct": 60.0, "starved": False}, {"turn1_starved": False, "turn2_starved": False}, {"starved": False}, 70.0) == "Net Frontier Tax"


def test_extra_unwanted_tool_call_is_rejected():
    """Negative fixture (per u/MongooseWonderful198): valid run_pytest + unwanted schedule_power_action must FAIL."""
    from unittest.mock import patch
    from evals.runner import eval_tool_calling_once

    bad_pair = [
        {"function": {"name": "run_pytest", "arguments": {"target_path": "tests/test_parser.py", "maxfail": 1, "verbose": True}}},
        {"function": {"name": "schedule_power_action", "arguments": {"node_name": "edge-gpu", "action": "poweroff", "delay_seconds": 0, "reason": "unwanted"}}},
    ]
    responses = iter([
        {"tps": 60.0, "elapsed_sec": 0.5, "starved": False, "content": "", "tool_calls": bad_pair},
        {"tps": 60.0, "elapsed_sec": 0.5, "starved": False, "content": "", "tool_calls": [{"function": {"name": "query_device_telemetry", "arguments": {"node_name": "edge-gpu", "metric": "vram_free_mb"}}}]},
        {"tps": 60.0, "elapsed_sec": 0.5, "starved": False, "content": "", "tool_calls": [{"function": {"name": "log_crm_interaction", "arguments": {"contact_name": "Sarah Chen", "channel": "phone", "summary": "Discussed Q4 board budget", "follow_up_days": 7}}}]},
        {"tps": 60.0, "elapsed_sec": 0.5, "starved": False, "content": "", "tool_calls": [{"function": {"name": "dispatch_alert", "arguments": {"severity": "critical", "title": "Driveway motion", "tags": ["camera", "perimeter"], "suppress_if_quiet_hours": False}}}]},
        {"tps": 60.0, "elapsed_sec": 0.5, "starved": False, "content": "", "tool_calls": [{"function": {"name": "schedule_power_action", "arguments": {"node_name": "edge-gpu", "action": "sleep", "delay_seconds": 30, "reason": "Backup done"}}}]},
        {"tps": 60.0, "elapsed_sec": 0.5, "starved": False, "content": "HTTP 429 means rate limited.", "tool_calls": []},
        {"tps": 60.0, "elapsed_sec": 0.5, "starved": False, "content": "Check GPU temp first.", "tool_calls": []},
        {"tps": 60.0, "elapsed_sec": 0.5, "starved": False, "content": "Which channel and how many days?", "tool_calls": []},
        {"tps": 60.0, "elapsed_sec": 0.5, "starved": False, "content": "Sample thermal alert documentation text.", "tool_calls": []},
        {"tps": 60.0, "elapsed_sec": 0.5, "starved": False, "content": "You ran tests and slept edge-gpu.", "tool_calls": []},
        # Chain step 1 with extra unwanted call -> must fail chain!
        {"tps": 60.0, "elapsed_sec": 0.5, "starved": False, "content": "", "tool_calls": [
            {"function": {"name": "query_device_telemetry", "arguments": {"node_name": "workstation", "metric": "gpu_temp"}}},
            {"function": {"name": "schedule_power_action", "arguments": {"node_name": "workstation", "action": "reboot", "delay_seconds": 0, "reason": "extra"}}},
        ]},
        # Repair step with extra unwanted call -> must fail repair!
        {"tps": 60.0, "elapsed_sec": 0.5, "starved": False, "content": "", "tool_calls": [
            {"function": {"name": "log_crm_interaction", "arguments": {"contact_name": "Marcus Vance", "channel": "linkedin", "summary": "Keynote invite", "follow_up_days": 5}}},
            {"function": {"name": "dispatch_alert", "arguments": {"severity": "info", "title": "extra", "tags": ["spam"], "suppress_if_quiet_hours": True}}},
        ]},
    ])

    with patch("evals.runner.ollama_chat", side_effect=lambda *a, **kw: next(responses)):
        tc = eval_tool_calling_once("mock-model")

    assert tc["extra_calls_rejected"] == 3
    assert tc["action_passed"] == 4 and tc["action_total"] == 7
    assert tc["abstention_passed"] == 5 and tc["abstention_total"] == 5
    case_map = {d["case"]: d for d in tc["details"]}
    assert case_map["01_pytest_strict_types"]["passed"] is False
    assert case_map["01_pytest_strict_types"]["extra_calls_rejected"] is True
    assert case_map["11_two_step_autonomous_chain"]["passed"] is False
    assert case_map["12_validator_error_self_repair"]["passed"] is False


