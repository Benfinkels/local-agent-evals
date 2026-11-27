import json
from pathlib import Path

from scripts.generate_leaderboard import (
    build_row,
    generate_markdown,
    main,
)


def test_build_row_from_runner_record():
    record = {
        "model": "test-model",
        "tool_calling": {"score_pct": 90.0},
        "self_healing": {
            "turn2_score_pct": 80.0,
            "turn2_elapsed_sec": 10.0,
        },
        "bayesian_math": {
            "score_pct": 100.0,
            "elapsed_sec": 5.0,
        },
        "composite_agent_score": 90.0,
    }

    row = build_row(record)

    assert row["model"] == "test-model"
    assert row["tool"] == 90.0
    assert row["healing"] == 80.0
    assert row["math"] == 100.0
    assert row["overall"] == 90.0
    assert row["pass_rate"] is None


def test_build_row_from_briefing_record():
    record = {
        "model": "briefing-model",
        "pass_k_pct": 85.0,
        "avg_latency_sec": 2.5,
    }

    row = build_row(record)

    assert row["model"] == "briefing-model"
    assert row["overall"] == 85.0
    assert row["pass_rate"] == 85.0
    assert row["latency"] == 2.5


def test_build_row_without_model():
    assert build_row({"tool_calling": {}}) is None


def test_generate_markdown_sorts_by_overall_score():
    records = [
        {"model": "model-low", "pass_k_pct": 60.0},
        {"model": "model-high", "pass_k_pct": 95.0},
    ]

    markdown = generate_markdown(records)

    assert markdown.index("model-high") < markdown.index("model-low")


def test_generate_markdown_shows_missing_values():
    markdown = generate_markdown([{"model": "test-model"}])

    assert "test-model" in markdown
    assert "-" in markdown


def test_load_records_and_cli_print(tmp_path, capsys):
    input_file = tmp_path / "results.json"
    input_file.write_text(
        json.dumps([{"model": "cli-model", "pass_k_pct": 75.0}]),
        encoding="utf-8",
    )

    result = main(["--input", str(input_file)])

    captured = capsys.readouterr()

    assert result == 0
    assert "cli-model" in captured.out
    assert "75.0%" in captured.out


def test_cli_writes_markdown_output(tmp_path):
    input_file = tmp_path / "results.json"
    output_file = tmp_path / "leaderboard.md"

    input_file.write_text(
        json.dumps([{"model": "output-model", "pass_k_pct": 80.0}]),
        encoding="utf-8",
    )

    result = main(
        ["--input", str(input_file), "--output", str(output_file)]
    )

    assert result == 0
    assert "output-model" in output_file.read_text(encoding="utf-8")


def test_existing_results_smoke():
    results_file = Path("results/showdown_8gb_v2.json")

    assert results_file.is_file()

    records = json.loads(results_file.read_text(encoding="utf-8"))
    markdown = generate_markdown(records)

    assert "| Model |" in markdown
    assert "gemma4:e2b-it-q4_K_M" in markdown