import argparse
import json
from pathlib import Path


def load_records(path):
    """Load supported model records from a JSON file."""
    with path.open(encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict) and "model" in data:
        return [data]
    return []


def average(values):
    """Return the average of available numeric values."""
    numbers = [float(value) for value in values if isinstance(value, (int, float))]
    return sum(numbers) / len(numbers) if numbers else None


def build_row(record):
    """Normalize runner and briefing result records."""
    model = record.get("model")
    if not model:
        return None

    tool = record.get("tool_calling", {})
    healing = record.get("self_healing", {})
    math = record.get("bayesian_math", {})

    tool_score = tool.get("score_pct", tool.get("action_score_pct"))
    healing_score = healing.get("turn2_score_pct")
    math_score = math.get("score_pct")

    pass_rate = record.get("pass_k_pct")

    latency = record.get("avg_latency_sec")
    if latency is None:
        tool_details = tool.get("details", [])
        tool_latency = average(
            detail.get("elapsed_sec")
            for detail in tool_details
            if isinstance(detail, dict)
        )
        track_latencies = [
            tool_latency,
            healing.get("turn2_elapsed_sec"),
            math.get("elapsed_sec"),
        ]
        latency = average(track_latencies)

    overall = record.get("composite_agent_score")
    if overall is None:
        overall = record.get("pass_k_pct", pass_rate)

    return {
        "model": str(model),
        "tool": tool_score,
        "healing": healing_score,
        "math": math_score,
        "overall": overall,
        "latency": latency,
        "pass_rate": pass_rate,
    }


def display(value, suffix="%"):
    """Format a numeric value or show a marker for missing data."""
    if value is None:
        return "—"
    return f"{value:.1f}{suffix}"


def generate_markdown(records):
    """Build a leaderboard sorted by overall score, highest first."""
    rows = [build_row(record) for record in records]
    rows = [row for row in rows if row is not None]
    rows.sort(
        key=lambda row: (
            row["overall"] is not None,
            row["overall"] if row["overall"] is not None else 0,
        ),
        reverse=True,
    )

    lines = [
        "| Model | Tool Calling | Self Healing | Bayesian Math | Overall | Latency (s) | Pass Rate |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in rows:
        lines.append(
            f"| {row['model']} "
            f"| {display(row['tool'])} "
            f"| {display(row['healing'])} "
            f"| {display(row['math'])} "
            f"| **{display(row['overall'])}** "
            f"| {display(row['latency'], ' s')} "
            f"| {display(row['pass_rate'])} |"
        )

    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate a Markdown leaderboard from saved evaluation JSON files."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        type=Path,
        help="One or more JSON result files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional Markdown output path. Prints to the terminal if omitted.",
    )
    args = parser.parse_args(argv)

    records = []
    for path in args.input:
        if not path.is_file():
            parser.error(f"Input file does not exist: {path}")
        try:
            records.extend(load_records(path))
        except (OSError, json.JSONDecodeError) as error:
            parser.error(f"Could not read {path}: {error}")

    markdown = generate_markdown(records)

    if args.output:
        args.output.write_text(markdown + "\n", encoding="utf-8")
    else:
        print(markdown)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())