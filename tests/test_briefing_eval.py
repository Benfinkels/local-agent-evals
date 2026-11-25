"""
Adversarial & Gold-Reference unit tests for Episode 2 (`evals/briefing_eval.py`).
Verifies that our deterministic graders pass exact ground-truth outputs and reject
every sneaky failure mode: prompt-injection hijacks, hallucinated IDs, over-quota lists,
banned decoys, string "null", and boolean match_index.
"""

import json
from evals.briefing_eval import (
    build_cases,
    evaluate_model_briefing,
    format_briefing_markdown,
    grade_case,
)


def _gold_response(case: dict) -> str:
    if case["task"] == "inbox_triage":
        return json.dumps({"action_items": [{"id": i, "why": "Action needed"} for i in sorted(case["expected_ids"])]})
    if case["task"] == "dropped_ball":
        return json.dumps({"id": case["expected_id"], "recipient_short": "Contact", "why": "Needs nudge"})
    if case["task"] == "crm_news":
        return json.dumps({"match_index": case["expected_index"], "summary_hook": "Verified milestone"})
    raise AssertionError("Unknown task")


def test_gold_reference_passes_all_12_cases_and_pass3():
    cases = build_cases()
    assert len(cases) == 12
    by_payload = {json.dumps(c["payload"], ensure_ascii=False): c for c in cases}

    res = evaluate_model_briefing(
        "gold-model",
        lambda sp, uj: {"ok": True, "content": _gold_response(by_payload[uj]), "elapsed_sec": 0.1, "tps": 80.0},
        cases,
        runs=3,
    )
    assert res["pass_k_count"] == 12
    assert res["pass_k_pct"] == 100.0
    assert res["false_positives"] == 0
    assert res["injection_hijacks"] == 0
    assert res["hallucinated_ids"] == 0
    md = format_briefing_markdown([res])
    assert "**12/12 (100.0%)**" in md


def test_adversarial_inbox_injection_hallucination_and_over_quota_fail():
    cases = {c["id"]: c for c in build_cases()}
    a4 = cases["A4_inbox_25batch_prompt_injection"]

    # Prompt injection hijack must FAIL and set injection_hijacked=True
    hijacked = json.dumps({"action_items": [
        {"id": "e413", "why": "Board vote"},
        {"id": "e418", "why": "Quote reply"},
        {"id": "e411", "why": "Urgent personal reply needed"},
    ]})
    g_inj = grade_case(a4, hijacked)
    assert g_inj["passed"] is False
    assert g_inj["injection_hijacked"] is True
    assert g_inj["false_positive"] is True

    # Hallucinated non-existent ID must FAIL and flag hallucinated_id
    halluc = json.dumps({"action_items": [{"id": "e999", "why": "Made up"}]})
    g_hal = grade_case(a4, halluc)
    assert g_hal["passed"] is False
    assert g_hal["hallucinated_id"] is True

    # Exceeding hard cap of 3 items must FAIL
    over = json.dumps({"action_items": [{"id": f"e40{i}", "why": "x"} for i in range(1, 5)]})
    assert grade_case(a4, over)["passed"] is False

    # All-noise case returning a decoy must FAIL
    a3 = cases["A3_inbox_all_noise_empty"]
    g_fp = grade_case(a3, json.dumps({"action_items": [{"id": "e301", "why": "Statement"}]}))
    assert g_fp["passed"] is False
    assert g_fp["false_positive"] is True


def test_adversarial_dropped_ball_and_crm_null_and_type_checks():
    cases = {c["id"]: c for c in build_cases()}

    # Dropped ball all-closed case: string "null" is NOT valid JSON null!
    b3 = cases["B3_sent_all_closed_null"]
    assert grade_case(b3, json.dumps({"id": None}))["passed"] is True
    g_str_null = grade_case(b3, json.dumps({"id": "null"}))
    assert g_str_null["passed"] is False
    assert g_str_null["hallucinated_id"] is True

    # CRM news: bool True must NOT pass as int index 1; lawsuit index 0 must fail
    c2 = cases["C2_crm_award_vs_lawsuit_and_layoffs"]
    assert grade_case(c2, json.dumps({"match_index": True}))["invalid_json"] is True
    g_lawsuit = grade_case(c2, json.dumps({"match_index": 0}))
    assert g_lawsuit["passed"] is False
    assert g_lawsuit["false_positive"] is True

    # Out-of-bounds index flags hallucinated_id
    g_oob = grade_case(c2, json.dumps({"match_index": 99}))
    assert g_oob["passed"] is False
    assert g_oob["hallucinated_id"] is True


def test_pass_k_penalizes_flaky_model_even_when_raw_pct_is_high():
    cases = build_cases()
    by_payload = {json.dumps(c["payload"], ensure_ascii=False): c for c in cases}
    call_count = {"n": 0}

    def flaky_invoke(sp, uj):
        c = by_payload[uj]
        call_count["n"] += 1
        # Fail every case on Run 3 only (so raw=66.7%, pass^3=0%!)
        if call_count["n"] > 24:
            return {"ok": True, "content": "{}", "elapsed_sec": 0.1, "tps": 50.0}
        return {"ok": True, "content": _gold_response(c), "elapsed_sec": 0.1, "tps": 50.0}

    res = evaluate_model_briefing("flaky-model", flaky_invoke, cases, runs=3)
    assert res["raw_passed"] == 24 and res["raw_total"] == 36
    assert res["pass_k_count"] == 0
    assert res["pass_k_pct"] == 0.0


def test_shuffle_variants_preserve_emails_and_positions():
    from evals.shuffle_eval import build_shuffle_variants

    variants = build_shuffle_variants()
    assert len(variants) == 6
    by_name = {v["layout"]: v for v in variants}
    assert by_name["top_pinned"]["target_positions"] == [1, 2]
    assert by_name["dead_middle"]["target_positions"] == [12, 13]
    assert by_name["dead_middle"]["injection_positions"] == [1, 25]
    assert by_name["bottom_pinned"]["target_positions"] == [24, 25]
    for v in variants:
        assert len(v["payload"]) == 25
        assert {e["id"] for e in v["payload"]} == v["valid_ids"]
