# -*- coding: utf-8 -*-
"""Measure how much of a rule bank's supporting evidence was produced by the constraint itself.

In the rule-constrained arm the rule bank is re-induced every 50 episodes from the *constrained*
trajectory, and that trajectory contains only the action the mask permitted at covered symbols. The
observations a rule rests on are therefore partly its own output. This script quantifies that share
directly from the logged trace, without any retraining or counterfactual.

For every condition in the final bank we count (a) all training observations of that condition and
(b) those that carried an applied constraint rule. The ratio is the fraction of the rule's evidential
base that the mask generated. It is a direct measurement, not a controlled experiment: it shows that
the observer's input is self-generated, not that removing the loop would change the bank.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = ROOT / "runs" / "shared_multienv_gpi" / "CartPole-v1" / "grammar_constrained"
OUTPUT = ROOT / "results" / "constraint_observer_self_support.json"
SEEDS = [11, 29, 43, 71, 101, 149, 211, 307]


def final_bank_conditions(run_dir: Path) -> set[tuple[int, ...]]:
    grammar = json.loads((run_dir / "induced_grammar.json").read_text(encoding="utf-8"))
    conditions = set()
    for production in grammar.get("productions", []):
        state_symbol = production["rhs"][0]
        conditions.add(tuple(int(part) for part in state_symbol.removeprefix("STATE_").split("_")))
    return conditions


def main() -> int:
    rows = []
    for seed in SEEDS:
        matches = sorted(TASK_ROOT.glob(f"seed-{seed}_*"))
        completed = [path for path in matches if (path / "trajectory.jsonl").exists()]
        if not completed:
            raise FileNotFoundError(f"no completed constrained run for seed {seed}")
        run_dir = completed[-1]

        conditions = final_bank_conditions(run_dir)
        observed: Counter = Counter()
        forced: Counter = Counter()
        total_steps = 0
        with (run_dir / "trajectory.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                total_steps += 1
                condition = tuple(record["state_symbol"])
                observed[condition] += 1
                if record.get("constraint_rule"):
                    forced[condition] += 1

        covered_observations = sum(observed[c] for c in conditions)
        forced_observations = sum(forced[c] for c in conditions)
        rows.append({
            "seed": seed,
            "run_dir": str(run_dir),
            "rules_in_final_bank": len(conditions),
            "training_steps": total_steps,
            "covered_condition_observations": covered_observations,
            "mask_generated_observations": forced_observations,
            "self_generated_support_fraction": forced_observations / max(covered_observations, 1),
            "masked_step_fraction_all_states": forced_observations / max(total_steps, 1),
        })

    fractions = [row["self_generated_support_fraction"] for row in rows]
    summary = {
        "purpose": (
            "Quantify the share of each final rule's evidential base that the action mask itself "
            "produced, in the rule-constrained arm where the bank is re-induced from the constrained "
            "trajectory every 50 episodes."
        ),
        "environment": "CartPole-v1",
        "reinduction_interval_episodes": 50,
        "interpretation_limit": (
            "Direct measurement, not a controlled experiment. It shows the observer's input is "
            "self-generated; it does not show that removing the loop would change the bank."
        ),
        "seeds": SEEDS,
        "per_seed": rows,
        "aggregate": {
            "mean_self_generated_support_fraction": sum(fractions) / len(fractions),
            "min_self_generated_support_fraction": min(fractions),
            "max_self_generated_support_fraction": max(fractions),
        },
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("%-5s %7s %13s %13s %12s" % ("seed", "rules", "covered_obs", "mask_gen", "self_gen"))
    for row in rows:
        print("%-5d %7d %13d %13d %11.1f%%" % (
            row["seed"], row["rules_in_final_bank"], row["covered_condition_observations"],
            row["mask_generated_observations"], 100.0 * row["self_generated_support_fraction"],
        ))
    print("\nmean self-generated support fraction: %.3f (range %.3f-%.3f)" % (
        summary["aggregate"]["mean_self_generated_support_fraction"],
        summary["aggregate"]["min_self_generated_support_fraction"],
        summary["aggregate"]["max_self_generated_support_fraction"],
    ))
    print("WROTE", OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
