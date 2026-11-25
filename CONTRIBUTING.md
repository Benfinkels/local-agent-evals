# Contributing to local-agent-evals

Contributions are welcome, including new deterministic eval tasks, reporting improvements, and agent-assisted PRs. Start with open issues tagged `good first issue` or `help wanted`, or open a quick issue before adding a new eval track.

## Offline test verification

You do not need a GPU, Ollama, or paid API keys to run the verification suite.

```bash
pip install -r requirements.txt
PYTHONPATH=. pytest tests/ -v
```

## What makes a PR easy to merge

- Scope each PR to a single issue and avoid unrelated edits across existing benchmark results in `results/`.
- Every new eval task or parser helper must be deterministically graded (unit tests or exact assertions, not LLM-as-a-judge) and covered by offline tests in `tests/`.
- Keep the harness dependency-light (prefer Python standard library inside `evals/`).
- Do not commit personal hostnames, private LAN IPs, Tailscale addresses, or API keys.
- In your PR description, state what changed and how you verified it.

## Using AI coding agents

AI-assisted PRs are welcome here. Stick to the files named in the issue, keep diffs minimal, and verify `PYTHONPATH=. pytest tests/ -v` passes before opening the PR.
