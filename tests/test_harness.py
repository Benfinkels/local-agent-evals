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
    ref_code = """
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
