import json
from pathlib import Path

rep = json.loads(Path("eval/results/remap_gold_audit.json").read_text(encoding="utf-8"))
for r in rep:
    bad = [j for j in r["expected_judgments"] if j["exists"] and not j["ok"]]
    good = [j for j in r["expected_judgments"] if j["exists"] and j["ok"]]
    miss = sum(1 for j in r["expected_judgments"] if not j["exists"])
    print(
        f"{r['id']:10} verdict={r['verdict']:28} "
        f"good={len(good)} bad={len(bad)} missing={miss} "
        f"good_sec={[j['section'] for j in good]} "
        f"bad_sec={[j['section'] for j in bad]}"
    )
