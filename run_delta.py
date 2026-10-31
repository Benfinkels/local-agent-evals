import json
import sys
sys.path.insert(0, "/home/alfred/projects/local-agent-evals")
import run_sheldon_agent_evals as b

res_file = b.RESULTS_DIR / "benchmark_summary.json"
summary = json.loads(res_file.read_text())

m = "nemotron-mini:4b"
print(f">>> Benchmarking {m}...")
tc = b.eval_tool_calling(m)
sh = b.eval_self_healing_code(m)
mv = b.eval_math_verification(m)
comp = round((tc["score_pct"] + sh["turn2_score_pct"] + mv["score_pct"]) / 3.0, 1)
print(f"  [Track 1 - Tool Calling]  Score: {tc['passed']}/{tc['total']} ({tc['score_pct']}%) | Supported: {tc['supported']}")
print(f"  [Track 2 - Self-Healing]  Turn 1: {sh['turn1_passed']}/10 -> Turn 2: {sh['turn2_passed']}/10 | TPS: {sh['turn1_tps']}")
print(f"  [Track 3 - Bayesian Math] Score: {mv['passed']}/5 ({mv['score_pct']}%) | TPS: {mv['tps']}")

for i, item in enumerate(summary):
    if item["model"] == "nemotron-3-nano":
        summary[i] = {
            "model": m,
            "node": "Sheldon (RTX 2080 + GTX 1080)",
            "tool_calling": tc,
            "self_healing": sh,
            "bayesian_math": mv,
            "composite_agent_score": comp,
            "note": "Replaced 24GB nemotron-3-nano (HTTP 500 OOM on 8GB GPU) with 2.7GB nemotron-mini:4b"
        }

res_file.write_text(json.dumps(summary, indent=2))
print("[+] Updated benchmark_summary.json")
