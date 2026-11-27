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


def _openai_v1_base(base_url: str) -> str:
    norm = normalize_ollama_url(base_url)
    return norm if norm.endswith("/v1") else f"{norm}/v1"


def list_local_ollama_models(base_url: Optional[str] = None, api: str = "ollama") -> list[str]:
    """Return locally installed model names from Ollama (/api/tags) or OpenAI-compatible (/v1/models)."""
    base = normalize_ollama_url(base_url or OLLAMA_BASE)
    if api == "openai":
        url = f"{_openai_v1_base(base)}/models"
        headers = {"Accept": "application/json"}
        key = os.environ.get("OPENAI_API_KEY")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [m["id"] for m in data.get("data", []) if m.get("id")]
    url = f"{base}/api/tags"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return [m["name"] for m in data.get("models", []) if m.get("name")]


def _normalize_openai_messages(messages: list) -> list:
    """Convert internal/Ollama tool-call history into strict OpenAI /v1/chat/completions format."""
    out = []
    last_call_id = "call_0"
    for m in messages:
        msg = dict(m)
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            norm_calls = []
            for idx, tc in enumerate(msg["tool_calls"]):
                fn = dict(tc.get("function") or {})
                args = fn.get("arguments", {})
                if not isinstance(args, str):
                    args = json.dumps(args, ensure_ascii=False)
                cid = tc.get("id") or f"call_{idx}"
                last_call_id = cid
                norm_calls.append({"id": cid, "type": "function", "function": {"name": fn.get("name", ""), "arguments": args}})
            msg["tool_calls"] = norm_calls
        elif msg.get("role") == "tool" and not msg.get("tool_call_id"):
            msg["tool_call_id"] = last_call_id
        out.append(msg)
    return out


def _openai_chat(
    model: str,
    messages: list,
    tools: list = None,
    options: dict = None,
    timeout: int = 240,
    think: Optional[bool] = None,
    base_url: Optional[str] = None,
) -> dict:
    opts = options or {"temperature": 0.2, "num_predict": 2048}
    payload: dict = {
        "model": model,
        "messages": _normalize_openai_messages(messages),
        "stream": False,
        "temperature": opts.get("temperature", 0.2),
        "max_tokens": opts.get("num_predict", 2048),
    }
    if tools:
        payload["tools"] = tools
    if think is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": bool(think)}

    endpoint = f"{_openai_v1_base(base_url or OLLAMA_BASE)}/chat/completions"
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"

    t0 = time.perf_counter()
    try:
        try:
            req = urllib.request.Request(endpoint, data=json.dumps(payload).encode("utf-8"), headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as he:
            err_body = he.read().decode("utf-8", errors="replace")
            if he.code == 400 and "chat_template_kwargs" in payload:
                payload.pop("chat_template_kwargs", None)
                req = urllib.request.Request(endpoint, data=json.dumps(payload).encode("utf-8"), headers=headers)
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
            else:
                raise RuntimeError(f"HTTP {he.code}: {err_body[:200]}") from he

        elapsed = time.perf_counter() - t0
        msg = (data.get("choices") or [{}])[0].get("message") or {}
        content = msg.get("content", "") or ""
        thinking = msg.get("reasoning_content") or msg.get("reasoning") or ""
        think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        if think_match and not thinking:
            thinking = think_match.group(1)
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        raw_calls = msg.get("tool_calls") or []
        norm_calls = []
        for tc in raw_calls:
            fn = tc.get("function") or {}
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    pass
            norm_calls.append({"id": tc.get("id"), "function": {"name": fn.get("name"), "arguments": args}})

        usage = data.get("usage") or {}
        eval_count = usage.get("completion_tokens", 0) or 0
        timings = data.get("timings") or {}
        tps = timings.get("predicted_per_second") or (eval_count / max(elapsed, 0.01))
        return {
            "ok": True,
            "elapsed_sec": round(elapsed, 2),
            "eval_tokens": eval_count,
            "tps": round(float(tps), 2),
            "content": content,
            "thinking": thinking,
            "thinking_chars": len(thinking),
            "content_chars": len(content),
            "tool_calls": norm_calls,
            "starved": (len(thinking) > 500 and len(content.strip()) == 0 and not norm_calls),
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


def ollama_chat(
    model: str,
    messages: list,
    tools: list = None,
    options: dict = None,
    timeout: int = 240,
    think: Optional[bool] = None,
    base_url: Optional[str] = None,
    keep_alive: str = "5m",
    api: str = "ollama",
) -> dict:
    if api == "openai":
        return _openai_chat(model, messages, tools=tools, options=options, timeout=timeout, think=think, base_url=base_url)
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "keep_alive": keep_alive,
        "options": options or {"temperature": 0.2, "num_predict": 2048, "num_ctx": 4096},
    }
    if tools:
        payload["tools"] = tools
    if think is not None:
        payload["think"] = think

    endpoint = f"{normalize_ollama_url(base_url or OLLAMA_BASE)}/api/chat"
    t0 = time.perf_counter()
    try:
        try:
            req = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as he:
            err_body = he.read().decode("utf-8", errors="replace")
            if he.code == 400 and "think" in payload and "think" in err_body.lower():
                payload.pop("think", None)
                req = urllib.request.Request(
                    endpoint,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
            else:
                raise RuntimeError(f"HTTP {he.code}: {err_body[:200]}") from he
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
# TRACK 1: AUTONOMOUS TOOL CALLING, STRICT SCHEMAS & ZERO-FRONTIER CHAINS (10)
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
                    "node_name": {"type": "string", "enum": ["edge-gpu", "workstation", "sentry"]},
                    "metric": {"type": "string", "enum": ["gpu_temp", "vram_free_mb", "power_watts"]},
                },
                "required": ["node_name", "metric"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "log_crm_interaction",
            "description": "Record a structured contact interaction and follow-up interval in the CRM.",
            "parameters": {
                "type": "object",
                "properties": {
                    "contact_name": {"type": "string", "description": "Full name of the contact"},
                    "channel": {"type": "string", "enum": ["email", "linkedin", "phone", "in_person"]},
                    "summary": {"type": "string", "description": "Short summary of the interaction"},
                    "follow_up_days": {"type": "integer", "description": "Days until next follow-up"},
                },
                "required": ["contact_name", "channel", "summary", "follow_up_days"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dispatch_alert",
            "description": "Send a structured alert notification with tags and quiet-hours handling.",
            "parameters": {
                "type": "object",
                "properties": {
                    "severity": {"type": "string", "enum": ["info", "warning", "critical"]},
                    "title": {"type": "string", "description": "Concise alert title"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of lowercase category tags",
                    },
                    "suppress_if_quiet_hours": {
                        "type": "boolean",
                        "description": "Whether to suppress notification during quiet hours",
                    },
                },
                "required": ["severity", "title", "tags", "suppress_if_quiet_hours"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_power_action",
            "description": "Schedule a sleep, reboot, or poweroff action on a compute node.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_name": {"type": "string", "enum": ["edge-gpu", "workstation"]},
                    "action": {"type": "string", "enum": ["sleep", "reboot", "poweroff"]},
                    "delay_seconds": {"type": "integer", "description": "Delay in seconds before action"},
                    "reason": {"type": "string", "description": "Short reason for the power action"},
                },
                "required": ["node_name", "action", "delay_seconds", "reason"],
            },
        },
    },
]


def _extract_exact_single_tool_call(tcalls: list) -> tuple[Optional[str], Optional[dict], bool]:
    """
    Validate that the model emitted EXACTLY one tool call.
    Returns (function_name, arguments_dict, had_extra_calls).
    Rejects responses that bundle an unwanted extra tool call alongside a valid one.
    """
    if not tcalls:
        return None, None, False
    if len(tcalls) != 1:
        return None, None, True
    fn = tcalls[0].get("function", {})
    fname = fn.get("name")
    fargs = fn.get("arguments", {})
    if isinstance(fargs, str):
        try:
            fargs = json.loads(fargs)
        except Exception:
            return fname, None, False
    return fname, (fargs if isinstance(fargs, dict) else None), False


def _is_strict_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def eval_tool_calling_once(
    model: str,
    think: Optional[bool] = None,
    base_url: Optional[str] = None,
    num_ctx: int = 4096,
    temperature: float = 0.0,
    api: str = "ollama",
) -> dict:
    """
    Run 12 deterministic Track 1 cases split cleanly into:
      - Action Accuracy (7 cases): 5 single-turn tool calls, 1 two-step autonomous chain,
        and 1 mechanical validator-error self-repair. Every positive step requires EXACTLY
        one valid tool call (extra unwanted calls fail).
      - No-Call Restraint / Abstention (5 cases): prompts where the model MUST emit 0 tool calls.
    """
    opts = {"temperature": temperature, "num_predict": 512, "num_ctx": num_ctx}
    action_cases = [
        {
            "id": "01_pytest_strict_types",
            "prompt": "Run pytest on 'tests/test_parser.py' with verbose enabled and stop after 1 failure.",
            "expect_tool": "run_pytest",
            "check_args": lambda a: (
                a.get("target_path") == "tests/test_parser.py"
                and _is_strict_int(a.get("maxfail")) and a.get("maxfail") == 1
                and a.get("verbose") is True
            ),
        },
        {
            "id": "02_telemetry_enum_exact",
            "prompt": "Check how much free VRAM in MB is currently available on the 'edge-gpu' node.",
            "expect_tool": "query_device_telemetry",
            "check_args": lambda a: (
                a.get("node_name") == "edge-gpu" and a.get("metric") == "vram_free_mb"
            ),
        },
        {
            "id": "03_crm_prose_extraction",
            "prompt": (
                "Log that I had a 20-minute phone call with Sarah Chen about the Q4 board budget, "
                "and remind me to follow up in 7 days."
            ),
            "expect_tool": "log_crm_interaction",
            "check_args": lambda a: (
                "sarah chen" in str(a.get("contact_name", "")).lower()
                and a.get("channel") == "phone"
                and len(str(a.get("summary", "")).strip()) >= 8
                and _is_strict_int(a.get("follow_up_days")) and a.get("follow_up_days") == 7
            ),
        },
        {
            "id": "04_alert_array_and_bool",
            "prompt": (
                "Send a critical alert titled 'Driveway motion after midnight' tagged with both "
                "'camera' and 'perimeter', and make sure it is NOT suppressed during quiet hours."
            ),
            "expect_tool": "dispatch_alert",
            "check_args": lambda a: (
                a.get("severity") == "critical"
                and "driveway" in str(a.get("title", "")).lower()
                and isinstance(a.get("tags"), list)
                and {"camera", "perimeter"}.issubset({str(x).lower() for x in a.get("tags", [])})
                and a.get("suppress_if_quiet_hours") is False
            ),
        },
        {
            "id": "05_disambiguate_power_tool",
            "prompt": "Put the edge-gpu node to sleep in 30 seconds because the nightly backup finished.",
            "expect_tool": "schedule_power_action",
            "check_args": lambda a: (
                a.get("node_name") == "edge-gpu"
                and a.get("action") == "sleep"
                and _is_strict_int(a.get("delay_seconds")) and a.get("delay_seconds") == 30
                and len(str(a.get("reason", "")).strip()) >= 4
            ),
        },
    ]

    abstention_cases = [
        {
            "id": "06_abstain_general_qa",
            "prompt": "Do NOT call any tools. In one short sentence, what does HTTP status code 429 mean?",
        },
        {
            "id": "07_abstain_advisory_no_action",
            "prompt": (
                "The workstation feels slow. Should I reboot it or check its GPU temperature first? "
                "Do not run any tools or take any action yet — just advise me in two short sentences."
            ),
        },
        {
            "id": "08_abstain_missing_required_fields",
            "prompt": (
                "I want to log a CRM note that I spoke with Alex Rivera about the partnership, "
                "but I haven't told you the channel or follow-up days yet. "
                "Ask me for those missing fields instead of guessing or calling any tool."
            ),
        },
        {
            "id": "09_abstain_hypothetical_doc_example",
            "prompt": (
                "Write a two-sentence documentation example showing how an on-call engineer would "
                "describe a critical thermal alert to a teammate. This is only sample doc text — "
                "do NOT dispatch any real alert or call any tool."
            ),
        },
        {
            "id": "10_abstain_past_tense_summary",
            "prompt": (
                "Earlier today I already ran pytest on tests/test_parser.py and put edge-gpu to sleep "
                "after the backup finished. Just summarize what I did in one sentence without calling any tools."
            ),
        },
    ]

    action_passed = 0
    abstention_passed = 0
    extra_calls_rejected = 0
    details = []
    total_tps = []
    starved_any = False

    def _record_tps(res: dict):
        nonlocal starved_any
        if res["tps"] > 0:
            total_tps.append(res["tps"])
        if res.get("starved"):
            starved_any = True

    for c in action_cases:
        res = ollama_chat(
            model,
            messages=[{"role": "user", "content": c["prompt"]}],
            tools=AGENT_TOOLS,
            options=opts,
            think=think,
            base_url=base_url,
            api=api,
        )
        _record_tps(res)
        tcalls = res.get("tool_calls") or []
        fname, fargs, extra = _extract_exact_single_tool_call(tcalls)
        if extra:
            extra_calls_rejected += 1
        success = False
        try:
            if not extra and fargs is not None and fname == c["expect_tool"] and c["check_args"](fargs):
                success = True
        except (TypeError, ValueError):
            success = False

        if success:
            action_passed += 1
        details.append({
            "case": c["id"],
            "category": "action",
            "passed": success,
            "extra_calls_rejected": extra,
            "tool_calls": tcalls,
            "elapsed_sec": res["elapsed_sec"],
            "starved": res.get("starved", False),
            "error": res.get("error"),
        })

    for c in abstention_cases:
        res = ollama_chat(
            model,
            messages=[{"role": "user", "content": c["prompt"]}],
            tools=AGENT_TOOLS,
            options=opts,
            think=think,
            base_url=base_url,
            api=api,
        )
        _record_tps(res)
        tcalls = res.get("tool_calls") or []
        success = len(tcalls) == 0 and len(res["content"].strip()) > 5
        if success:
            abstention_passed += 1
        details.append({
            "case": c["id"],
            "category": "abstention",
            "passed": success,
            "over_triggered": len(tcalls) > 0,
            "tool_calls": tcalls,
            "elapsed_sec": res["elapsed_sec"],
            "starved": res.get("starved", False),
            "error": res.get("error"),
        })

    # Case 11: 2-Step Autonomous Tool Chain (exact single call per step, zero frontier steering)
    chain_msgs = [{
        "role": "user",
        "content": (
            "First check the gpu_temp on 'workstation'. "
            "After you receive the temperature, if it is above 85C, immediately dispatch a "
            "critical alert with tag 'thermal' and suppress_if_quiet_hours set to false."
        ),
    }]
    r_step1 = ollama_chat(model, chain_msgs, tools=AGENT_TOOLS, options=opts, think=think, base_url=base_url, api=api)
    _record_tps(r_step1)
    fn1, args1, extra1 = _extract_exact_single_tool_call(r_step1.get("tool_calls") or [])
    if extra1:
        extra_calls_rejected += 1
    step1_ok = (not extra1) and fn1 == "query_device_telemetry" and args1 is not None and (
        args1.get("node_name") == "workstation" and args1.get("metric") == "gpu_temp"
    )
    step2_ok = False
    extra2 = False
    r_step2 = {"tool_calls": [], "elapsed_sec": 0.0, "starved": False, "error": None}
    if step1_ok:
        chain_msgs.append({"role": "assistant", "content": r_step1["content"], "tool_calls": r_step1["tool_calls"]})
        chain_msgs.append({
            "role": "tool",
            "content": json.dumps({"node_name": "workstation", "metric": "gpu_temp", "value": 92, "unit": "C"}),
        })
        r_step2 = ollama_chat(model, chain_msgs, tools=AGENT_TOOLS, options=opts, think=think, base_url=base_url, api=api)
        _record_tps(r_step2)
        fn2, args2, extra2 = _extract_exact_single_tool_call(r_step2.get("tool_calls") or [])
        if extra2:
            extra_calls_rejected += 1
        step2_ok = (not extra2) and fn2 == "dispatch_alert" and args2 is not None and (
            args2.get("severity") == "critical"
            and isinstance(args2.get("tags"), list)
            and "thermal" in {str(x).lower() for x in args2.get("tags", [])}
            and args2.get("suppress_if_quiet_hours") is False
        )
    chain_ok = step1_ok and step2_ok
    if chain_ok:
        action_passed += 1
    details.append({
        "case": "11_two_step_autonomous_chain",
        "category": "action",
        "passed": chain_ok,
        "step1_ok": step1_ok,
        "step2_ok": step2_ok,
        "extra_calls_rejected": extra1 or extra2,
        "tool_calls": (r_step1.get("tool_calls") or []) + (r_step2.get("tool_calls") or []),
        "elapsed_sec": round(r_step1["elapsed_sec"] + r_step2["elapsed_sec"], 2),
        "starved": r_step1.get("starved", False) or r_step2.get("starved", False),
        "error": r_step1.get("error") or r_step2.get("error"),
    })

    # Case 12: Mechanical Validator-Error Self-Correction (exact single repaired call)
    repair_msgs = [
        {
            "role": "user",
            "content": "Log my LinkedIn DM with Marcus Vance about the keynote invite, follow up in 5 days.",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "function": {
                    "name": "log_crm_interaction",
                    "arguments": {
                        "contact_name": "Marcus Vance",
                        "channel": "dm",
                        "summary": "Keynote invite",
                        "follow_up_days": "5",
                    },
                }
            }],
        },
        {
            "role": "tool",
            "content": (
                "ValidationError: 'channel' must be one of ['email', 'linkedin', 'phone', 'in_person'] "
                "(got 'dm'); 'follow_up_days' must be an integer (got string '5'). Retry with valid arguments."
            ),
        },
    ]
    r_fix = ollama_chat(model, repair_msgs, tools=AGENT_TOOLS, options=opts, think=think, base_url=base_url, api=api)
    _record_tps(r_fix)
    fn_fix, args_fix, extra_fix = _extract_exact_single_tool_call(r_fix.get("tool_calls") or [])
    if extra_fix:
        extra_calls_rejected += 1
    fix_ok = (not extra_fix) and fn_fix == "log_crm_interaction" and args_fix is not None and (
        "marcus vance" in str(args_fix.get("contact_name", "")).lower()
        and args_fix.get("channel") == "linkedin"
        and len(str(args_fix.get("summary", "")).strip()) >= 4
        and _is_strict_int(args_fix.get("follow_up_days")) and args_fix.get("follow_up_days") == 5
    )
    if fix_ok:
        action_passed += 1
    details.append({
        "case": "12_validator_error_self_repair",
        "category": "action",
        "passed": fix_ok,
        "extra_calls_rejected": extra_fix,
        "tool_calls": r_fix.get("tool_calls") or [],
        "elapsed_sec": r_fix["elapsed_sec"],
        "starved": r_fix.get("starved", False),
        "error": r_fix.get("error"),
    })

    action_total = len(action_cases) + 2
    abstention_total = len(abstention_cases)
    total_cases = action_total + abstention_total
    total_passed = action_passed + abstention_passed
    return {
        "supported": any(len(d["tool_calls"]) > 0 for d in details),
        "passed": total_passed,
        "total": total_cases,
        "score_pct": round((total_passed / total_cases) * 100, 1),
        "action_passed": action_passed,
        "action_total": action_total,
        "action_score_pct": round((action_passed / action_total) * 100, 1),
        "abstention_passed": abstention_passed,
        "abstention_total": abstention_total,
        "abstention_score_pct": round((abstention_passed / abstention_total) * 100, 1),
        "extra_calls_rejected": extra_calls_rejected,
        "avg_tps": round(sum(total_tps) / len(total_tps), 2) if total_tps else 0.0,
        "starved": starved_any,
        "details": details,
    }


def eval_tool_calling(
    model: str,
    think: Optional[bool] = None,
    base_url: Optional[str] = None,
    num_ctx: int = 4096,
    runs: int = 1,
    temperature: float = 0.0,
    api: str = "ollama",
) -> dict:
    """Run Track 1 once or N times and aggregate action accuracy, no-call restraint, and consistency."""
    runs = max(1, int(runs))
    if runs == 1:
        res = eval_tool_calling_once(model, think=think, base_url=base_url, num_ctx=num_ctx, temperature=temperature, api=api)
        res["runs"] = 1
        res["tool_temp"] = temperature
        return res

    per_run = [
        eval_tool_calling_once(model, think=think, base_url=base_url, num_ctx=num_ctx, temperature=temperature, api=api)
        for _ in range(runs)
    ]
    action_passed = sum(r["action_passed"] for r in per_run)
    action_total = sum(r["action_total"] for r in per_run)
    abstention_passed = sum(r["abstention_passed"] for r in per_run)
    abstention_total = sum(r["abstention_total"] for r in per_run)
    total_passed = action_passed + abstention_passed
    total_cases = action_total + abstention_total
    tps_vals = [r["avg_tps"] for r in per_run if r["avg_tps"] > 0]
    return {
        "runs": runs,
        "tool_temp": temperature,
        "supported": any(r["supported"] for r in per_run),
        "passed": total_passed,
        "total": total_cases,
        "score_pct": round((total_passed / total_cases) * 100, 1),
        "action_passed": action_passed,
        "action_total": action_total,
        "action_score_pct": round((action_passed / action_total) * 100, 1),
        "abstention_passed": abstention_passed,
        "abstention_total": abstention_total,
        "abstention_score_pct": round((abstention_passed / abstention_total) * 100, 1),
        "extra_calls_rejected": sum(r["extra_calls_rejected"] for r in per_run),
        "avg_tps": round(sum(tps_vals) / len(tps_vals), 2) if tps_vals else 0.0,
        "starved": any(r["starved"] for r in per_run),
        "per_run_scores": [
            {
                "run": i + 1,
                "action": f"{r['action_passed']}/{r['action_total']}",
                "abstention": f"{r['abstention_passed']}/{r['abstention_total']}",
                "extra_calls_rejected": r["extra_calls_rejected"],
            }
            for i, r in enumerate(per_run)
        ],
        "details": per_run[-1]["details"],
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
    num_ctx: int = 4096,
    api: str = "ollama",
) -> dict:
    safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", model)
    messages = [{"role": "user", "content": PARSER_PROMPT}]
    t1 = ollama_chat(
        model,
        messages,
        options={"temperature": 0.2, "num_predict": 2048, "num_ctx": num_ctx},
        think=think,
        base_url=base_url,
        api=api,
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
            options={"temperature": 0.2, "num_predict": 2048, "num_ctx": num_ctx},
            think=think,
            base_url=base_url,
            api=api,
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
    num_ctx: int = 4096,
    api: str = "ollama",
) -> dict:
    safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", model)
    messages = [{"role": "user", "content": MATH_PROMPT}]
    t1 = ollama_chat(
        model,
        messages,
        options={"temperature": 0.1, "num_predict": 2048, "num_ctx": num_ctx},
        think=think,
        base_url=base_url,
        api=api,
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


# ============================================================================
# TRACK 4: STRICT JSON REPAIR AND STRUCTURED EXTRACTION
# ============================================================================

JSON_REPAIR_PROMPT = """Repair the malformed JSON record below and return the
corrected data as one valid JSON object.

Malformed record:
{
    'customer_name': 'Ava Chen',
    'order_id': 'ORD-1042',
    'item': 'USB-C hub',
    'quantity': '2',
    'delivery_date': '2026-11-30',
    'tracking_number':
}

Required schema (exactly these keys):
{
    "customer_name": string,
    "order_id": string,
    "item": string,
    "quantity": integer,
    "delivery_date": string or null,
    "tracking_number": string or null
}

Rules:
- Convert quantity from a numeric string to an integer.
- The tracking number is missing. Return JSON null for it.
- Do not invent or infer missing information.
- Do not add or remove keys.
- Return ONLY valid JSON. No Markdown, code fences, or explanations.
"""

JSON_REPAIR_EXPECTED = {
    "customer_name": "Ava Chen",
    "order_id": "ORD-1042",
    "item": "USB-C hub",
    "quantity": 2,
    "delivery_date": "2026-11-30",
    "tracking_number": None,
}


def grade_json_repair(output: str) -> dict:
    """Grade JSON syntax, exact schema, types, and reference values."""
    try:
        actual = json.loads(output)
    except (json.JSONDecodeError, TypeError):
        return {
            "passed": 0,
            "total": 1,
            "score_pct": 0.0,
            "valid_json": False,
            "schema_valid": False,
            "matches_reference": False,
        }

    expected = JSON_REPAIR_EXPECTED
    exact_keys = isinstance(actual, dict) and set(actual) == set(expected)

    exact_types = exact_keys and all(
        type(actual[key]) is type(expected[key])
        for key in expected
    )

    matches_reference = (
        exact_types
        and all(actual[key] == expected[key] for key in expected)
    )

    passed = int(matches_reference)

    return {
        "passed": passed,
        "total": 1,
        "score_pct": 100.0 if passed else 0.0,
        "valid_json": True,
        "schema_valid": bool(exact_types),
        "matches_reference": bool(matches_reference),
    }


def eval_json_repair(
    model: str,
    think: Optional[bool] = None,
    base_url: Optional[str] = None,
    num_ctx: int = 4096,
    api: str = "ollama",
) -> dict:
    """Run the deterministic Track 4 JSON repair evaluation."""
    response = ollama_chat(
        model,
        [{"role": "user", "content": JSON_REPAIR_PROMPT}],
        options={"temperature": 0.0, "num_predict": 512, "num_ctx": num_ctx},
        think=think,
        base_url=base_url,
        api=api,
    )

    if not response["ok"]:
        result = grade_json_repair("")
        result["error"] = response["error"]
    else:
        result = grade_json_repair(response["content"])

    result["elapsed_sec"] = response["elapsed_sec"]
    result["tps"] = response["tps"]
    result["starved"] = response["starved"]
    return result


def classify_offload_readiness(tc: dict, sh: dict, mv: dict, composite: float) -> str:
    """Classify whether a local model can run unattended without paying a frontier babysitter tax."""
    starved = bool(tc.get("starved") or sh.get("turn1_starved") or sh.get("turn2_starved") or mv.get("starved"))
    if tc["score_pct"] >= 90.0 and composite >= 85.0 and not starved:
        return "Unattended Ready"
    if tc["score_pct"] >= 70.0 and composite >= 60.0 and not starved:
        return "Validator + Retry"
    return "Net Frontier Tax"


def _unload_model(model: str, base_url: str) -> None:
    """Ask Ollama to evict weights before switching to the next model on constrained GPUs."""
    try:
        req = urllib.request.Request(
            f"{normalize_ollama_url(base_url)}/api/chat",
            data=json.dumps({"model": model, "messages": [], "keep_alive": 0}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=15):
            pass
    except Exception:
        pass


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
        help="Backend base URL (default: $OLLAMA_HOST or http://localhost:11434).",
    )
    parser.add_argument(
        "--api",
        choices=["ollama", "openai"],
        default="ollama",
        help="Backend protocol: ollama (/api/chat) or openai (/v1/chat/completions for llama-server/vLLM/LM Studio).",
    )
    parser.add_argument(
        "--think",
        choices=["auto", "true", "false"],
        default="auto",
        help="Control Ollama's `think` parameter (default: auto = omit parameter).",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=4096,
        help="Context window passed to Ollama (default: 4096 to stay 100%% GPU on 8GB cards).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of repeated Track 1 tool-calling runs per model to measure consistency (default: 1).",
    )
    parser.add_argument(
        "--tool-temp",
        type=float,
        default=0.0,
        help="Sampling temperature for Track 1 tool-calling runs (default: 0.0; try 0.2 with --runs 3).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write JSON results (default: results/run_<timestamp>.json).",
    )

    parser.add_argument(
        "--track4",
        action="store_true",
        help="Run Track 4: strict JSON repair and structured extraction.",
    )

    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List locally installed Ollama models and exit.",
    )
    return parser.parse_args(argv)


def format_markdown_table(summary: list[dict]) -> str:
    has_track4 = any(row.get("json_repair") is not None for row in summary)
    if has_track4:
        lines = [
            "| Model | Action Tools | No-Call Restraint | Extra Calls | Code (T1->T2) | Math | JSON Repair | Starved? | Composite | Verdict |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
        ]
    else:
        lines = [
            "| Model | Action Tools | No-Call Restraint | Extra Calls | Code (T1->T2) | Math | Starved? | Composite | Verdict |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
        ]
    for row in summary:
        tc = row["tool_calling"]
        sh = row["self_healing"]
        mv = row["bayesian_math"]
        j4 = row.get("json_repair")

        starved = "Yes" if (
            tc.get("starved")
            or sh["turn1_starved"]
            or sh["turn2_starved"]
            or mv["starved"]
            or (j4 and j4.get("starved"))
        ) else "No"

        verdict = row.get("offload_verdict") or classify_offload_readiness(
            tc, sh, mv, row["composite_agent_score"]
        )

        act_str = (
            f"{tc['action_passed']}/{tc['action_total']} ({tc['action_score_pct']}%)"
            if "action_passed" in tc else f"{tc['passed']}/{tc['total']}"
        )
        abs_str = (
            f"{tc['abstention_passed']}/{tc['abstention_total']} ({tc['abstention_score_pct']}%)"
            if "abstention_passed" in tc else "n/a"
        )
        extra_str = str(tc.get("extra_calls_rejected", 0))

        line = (
            f"| `{row['model']}` | {act_str} | {abs_str} | {extra_str} | "
            f"{sh['turn1_passed']}/10 -> {sh['turn2_passed']}/10 (+{sh['healed_delta']}) | "
            f"{mv['passed']}/{mv['total']} ({mv['score_pct']}%) |"
        )

        if has_track4:
            json_str = (
                f"{j4['passed']}/{j4['total']} ({j4['score_pct']}%)"
                if j4 else "n/a"
            )
            line += f" {json_str} |"

        line += f" {starved} | **{row['composite_agent_score']}%** | {verdict} |"
        lines.append(line)

    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    base_url = normalize_ollama_url(args.base_url)
    think_flag = None if args.think == "auto" else (args.think == "true")

    if args.list_models:
        models = list_local_ollama_models(base_url, api=args.api)
        print(f"Installed models at {base_url} ({args.api}):")
        for m in models:
            print(f"  - {m}")
        return 0

    models = args.models
    try:
        installed = list_local_ollama_models(base_url, api=args.api)
    except Exception as e:
        print(f"[!] Could not reach {args.api} backend at {base_url}: {e}", file=sys.stderr)
        print("    Start your server or pass --base-url http://<host>:<port>", file=sys.stderr)
        return 1

    if not models:
        models = installed
        if not models:
            print(f"[!] No models found at {base_url}. Pull or load one first.", file=sys.stderr)
            return 1

    print("=" * 72)
    print(f"LOCAL AGENTIC BENCHMARK SUITE | api={args.api} | endpoint={base_url} | think={args.think} | ctx={args.num_ctx} | runs={args.runs} | tool_temp={args.tool_temp}")
    print("=" * 72)

    summary = []
    for idx, m in enumerate(models):
        print(f"\n>>> Benchmarking Model: {m}")
        tc = eval_tool_calling(
            m,
            think=think_flag,
            base_url=base_url,
            num_ctx=args.num_ctx,
            runs=args.runs,
            temperature=args.tool_temp,
            api=args.api,
        )
        print(
            f"  [Track 1A - Action Tools]    {tc['action_passed']}/{tc['action_total']} ({tc['action_score_pct']}%) | "
            f"Extra Calls Rejected: {tc['extra_calls_rejected']} | TPS: {tc['avg_tps']}"
        )
        print(
            f"  [Track 1B - No-Call Restr.]  {tc['abstention_passed']}/{tc['abstention_total']} ({tc['abstention_score_pct']}%)"
        )

        sh = eval_self_healing_code(m, think=think_flag, base_url=base_url, num_ctx=args.num_ctx, api=args.api)
        print(f"  [Track 2 - Self-Healing]     Turn 1: {sh['turn1_passed']}/10 -> Turn 2: {sh['turn2_passed']}/10 (Delta: +{sh['healed_delta']}) | TPS: {sh['turn1_tps']}")

        mv = eval_math_verification(m, think=think_flag, base_url=base_url, num_ctx=args.num_ctx, api=args.api)
        print(f"  [Track 3 - Bayesian Math]    Score: {mv['passed']}/5 ({mv['score_pct']}%) | TPS: {mv['tps']}")

        j4 = None
        if args.track4:
            j4 = eval_json_repair(
                m,
                think=think_flag,
                base_url=base_url,
                num_ctx=args.num_ctx,
                api=args.api,
            )
            print(
                f"  [Track 4 - JSON Repair]      "
                f"Score: {j4['passed']}/{j4['total']} "
                f"({j4['score_pct']}%)"
            )


        composite = round((tc["score_pct"] + sh["turn2_score_pct"] + mv["score_pct"]) / 3.0, 1)
        verdict = classify_offload_readiness(tc, sh, mv, composite)
        print(f"  [Composite / Verdict]        {composite}% -> {verdict}")
        summary.append({
            "model": m,
            "api": args.api,
            "endpoint": base_url,
            "think": args.think,
            "num_ctx": args.num_ctx,
            "runs": args.runs,
            "tool_temp": args.tool_temp,
            "tool_calling": tc,
            "self_healing": sh,
            "bayesian_math": mv,
            "json_repair": j4,
            "composite_agent_score": composite,
            "offload_verdict": verdict,
        })
        if args.api == "ollama" and idx < len(models) - 1:
            _unload_model(m, base_url)

    out_path = Path(args.output) if args.output else (RESULTS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + format_markdown_table(summary))
    print(f"\n[+] Saved benchmark results to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
