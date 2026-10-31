"""
Phase 2 Expansion for `local-agent-evals`:
- Track 4: Multimodal Edge Vision -> Structured Tool Dispatch (Zero-Dependency PNG Synthesizer)
- Track 5: Tier 0 (`Gemma 4` Local $0) -> Tier 1 (`MiniMax-M2.7`) Verify-Then-Escalate Cascade
  with CoT `<think>` Anti-Starvation Extraction.
"""

from __future__ import annotations

import base64
import re
import struct
import zlib
from typing import Any, Callable

FONT_5X7 = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "D": ["11100", "10010", "10001", "10001", "10001", "10010", "11100"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01110"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "J": ["00111", "00010", "00010", "00010", "10010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "11011", "10001"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    ":": ["00000", "00100", "00100", "00000", "00100", "00100", "00000"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
}


def synthesize_osd_png_base64(label_text: str, bg_rgb: tuple[int, int, int] = (20, 40, 140)) -> str:
    """
    Generate a deterministic, zero-dependency PNG image with a 5x7 bitmap font scaled 4x
    for evaluating Multimodal Edge Vision -> Tool Call models without external image assets.
    """
    w, h = 160, 60
    scale = 4
    grid = [[bg_rgb for _ in range(w)] for _ in range(h)]
    for x in range(2, w - 2):
        grid[2][x] = (255, 255, 255)
        grid[h - 3][x] = (255, 255, 255)
    for y in range(2, h - 2):
        grid[y][2] = (255, 255, 255)
        grid[y][w - 3] = (255, 255, 255)

    for line_idx, line in enumerate(label_text.upper().split("\n")):
        y0 = 8 + line_idx * 15
        x0 = 8
        for ch in line:
            glyph = FONT_5X7.get(ch, FONT_5X7[" "])
            for gy in range(7):
                for gx in range(5):
                    if glyph[gy][gx] == "1" and 0 <= y0 + gy < h and 0 <= x0 + gx < w:
                        grid[y0 + gy][x0 + gx] = (255, 255, 255)
            x0 += 6

    W, H = w * scale, h * scale
    raw_rows = bytearray()
    for y in range(H):
        raw_rows.append(0)
        sy = y // scale
        for x in range(W):
            raw_rows.extend(grid[sy][x // scale])

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack("!I", len(data)) + tag + data + struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack("!IIBBBBB", W, H, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(raw_rows), 9))
        + chunk(b"IEND", b"")
    )
    return base64.b64encode(png).decode("ascii")


def strip_thinking_blocks(raw_output: str) -> str:
    """
    Strip inline `<think>...</think>` reasoning traces emitted by reasoning/MoE models
    (such as `MiniMax-M2.7`, `Qwen 3.5`, and `DeepSeek-R1`) before extracting code or JSON.
    """
    return re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()


def run_verify_then_escalate_cascade(
    prompt: str,
    tier0_generator: Callable[[str], str],
    tier1_generator: Callable[[str], str],
    deterministic_verifier: Callable[[str], tuple[bool, dict[str, Any]]],
) -> dict[str, Any]:
    """
    Execute Tier 0 (`Gemma 4` local $0 GPU worker) first and run `deterministic_verifier`.
    Only escalate to Tier 1 (`MiniMax-M2.7` cloud/LiteLLM worker) if Tier 0 fails verification.
    """
    tier0_raw = tier0_generator(prompt)
    tier0_clean = strip_thinking_blocks(tier0_raw)
    t0_passed, t0_meta = deterministic_verifier(tier0_clean)

    if t0_passed:
        return {
            "passed": True,
            "escalated_to_tier1": False,
            "selected_tier": "tier0_local_gemma4",
            "cloud_tokens_saved": True,
            "verification": t0_meta,
        }

    tier1_raw = tier1_generator(prompt)
    tier1_clean = strip_thinking_blocks(tier1_raw)
    t1_passed, t1_meta = deterministic_verifier(tier1_clean)
    return {
        "passed": t1_passed,
        "escalated_to_tier1": True,
        "selected_tier": "tier1_worker_minimax",
        "cloud_tokens_saved": False,
        "tier0_failure": t0_meta,
        "verification": t1_meta,
    }
