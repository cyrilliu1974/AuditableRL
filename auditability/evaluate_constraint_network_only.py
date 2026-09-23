# -*- coding: utf-8 -*-
"""B1: decompose the reported rule-constrained return into network vs substituted-rule parts.

Motivation. ``evaluate_policy`` replaces the learned action with the rule's action whenever a rule
covers the state symbol, and in the constrained arm that happens on 61-95% of held-out steps. The
reported constrained return is therefore a hybrid of the learned policy and the frozen rule bank, not
the network's own return. This script re-evaluates the same saved checkpoints on the same held-out
reset seeds with the substitution disabled, so the two contributions can be separated.

No training is performed: every number is read from artifacts already on disk.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auditability.grammar_audit_experiment import (  # noqa: E402
    Actor,
    StateSymbolizer,
    evaluate_policy,
)
from auditability.run_shared_multienv_gpi import load_actor, load_rules  # noqa: E402

ENVIRONMENT = "CartPole-v1"
SEEDS = [11, 29, 43, 71, 101, 149, 211, 307]
EPISODES = 20
TASK_ROOT = ROOT / "runs" / "shared_multienv_gpi" / ENVIRONMENT
OUTPUT = ROOT / "results" / "constraint_network_only_eval.json"


def symbolizer_from_metadata(metadata: dict) -> StateSymbolizer:
    """Rebuild the frozen shared symbolizer recorded in a run's metadata."""
    bins = metadata["config"]["state_bins"]
    symbolizer = StateSymbolizer(n_bins=int(bins["n_bins"]))
    symbolizer.edges = torch.tensor(bins["edges"], dtype=torch.float64)
    return symbolizer


def latest_run(arm: str, seed: int) -> Path:
    matches = sorted((TASK_ROOT / arm).glob(f"seed-{seed}_*"))
    completed = [path for path in matches if (path / "metadata.json").exists()]
    if not completed:
        raise FileNotFoundError(f"no completed {arm} run for seed {seed}")
    return completed[-1]


def main() -> int:
    rows = []
    for seed in SEEDS:
        constrained_dir = latest_run("grammar_constrained", seed)
        audit_dir = latest_run("grammar_audit_only", seed)

        metadata = json.loads((constrained_dir / "metadata.json").read_text(encoding="utf-8"))
        state_dim = int(metadata["config"]["state_dim"])
        action_dim = int(metadata["config"]["action_dim"])
        symbolizer = symbolizer_from_metadata(metadata)
        rules = load_rules(constrained_dir)

        constrained_actor: Actor = load_actor(constrained_dir, state_dim, action_dim)
        audit_actor: Actor = load_actor(audit_dir, state_dim, action_dim)

        # Reference: reproduce the recorded constrained number (substitution ON) as a harness check.
        hybrid = evaluate_policy(
            constrained_actor, symbolizer, rules, "grammar_constrained", seed,
            n_episodes=EPISODES, environment_name=ENVIRONMENT,
        )
        # B1: same checkpoint, same reset seeds, substitution OFF.
        network_only = evaluate_policy(
            constrained_actor, symbolizer, rules, "baseline", seed,
            n_episodes=EPISODES, environment_name=ENVIRONMENT,
        )
        # Control: the unconstrained actor, which needs no substitution.
        audit_reference = evaluate_policy(
            audit_actor, symbolizer, rules, "baseline", seed,
            n_episodes=EPISODES, environment_name=ENVIRONMENT,
        )

        recorded = float(metadata["evaluation"]["mean_return"])
        row = {
            "seed": seed,
            "run_dir": str(constrained_dir),
            "recorded_constrained_return": recorded,
            "reproduced_hybrid_return": hybrid["mean_return"],
            "harness_reproduces_recorded": abs(hybrid["mean_return"] - recorded) < 1e-9,
            "constrained_network_only_return": network_only["mean_return"],
            "unconstrained_actor_return": audit_reference["mean_return"],
            "grammar_state_coverage": hybrid["grammar_state_coverage"],
            "grammar_action_agreement_when_covered": hybrid["grammar_action_agreement_when_covered"],
            "covered_steps": hybrid["covered_steps"],
            "total_steps": hybrid["total_steps"],
            "substitution_gain": hybrid["mean_return"] - network_only["mean_return"],
            "network_deficit_vs_unconstrained": (
                audit_reference["mean_return"] - network_only["mean_return"]
            ),
            "rule_count": len(rules),
        }
        rows.append(row)

    summary = {
        "purpose": (
            "Separate the rule-substitution contribution from the learned network's own held-out "
            "return in the rule-constrained arm. Substitution is disabled by evaluating with "
            "arm='baseline', which skips the rule override in evaluate_policy."
        ),
        "environment": ENVIRONMENT,
        "episodes_per_policy": EPISODES,
        "reset_seed_formula": "seed * 10000 + 900000 + episode_index",
        "substitution_rule": (
            "evaluate_policy substitutes the rule action only when arm == 'grammar_constrained'"
        ),
        "seeds": SEEDS,
        "per_seed": rows,
    }

    all_reproduced = all(row["harness_reproduces_recorded"] for row in rows)
    summary["harness_reproduces_all_recorded_values"] = all_reproduced
    if not all_reproduced:
        print("WARNING: harness did not reproduce every recorded constrained return", file=sys.stderr)

    mean = lambda key: sum(row[key] for row in rows) / len(rows)  # noqa: E731
    summary["aggregate"] = {
        "mean_recorded_constrained_return": mean("recorded_constrained_return"),
        "mean_hybrid_return": mean("reproduced_hybrid_return"),
        "mean_network_only_return": mean("constrained_network_only_return"),
        "mean_unconstrained_actor_return": mean("unconstrained_actor_return"),
        "mean_substitution_gain": mean("substitution_gain"),
        "mean_network_deficit_vs_unconstrained": mean("network_deficit_vs_unconstrained"),
        "mean_coverage": mean("grammar_state_coverage"),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    header = "%-5s %9s %9s %11s %11s %9s %7s" % (
        "seed", "hybrid", "network", "unconstr.", "subst.gain", "coverage", "reprod",
    )
    print(header)
    for row in rows:
        print("%-5d %9.1f %9.1f %11.1f %11.1f %9.4f %7s" % (
            row["seed"],
            row["reproduced_hybrid_return"],
            row["constrained_network_only_return"],
            row["unconstrained_actor_return"],
            row["substitution_gain"],
            row["grammar_state_coverage"],
            "yes" if row["harness_reproduces_recorded"] else "NO",
        ))
    print()
    print("means: hybrid %.2f | network-only %.2f | unconstrained %.2f | substitution gain %.2f"
          % (
              summary["aggregate"]["mean_hybrid_return"],
              summary["aggregate"]["mean_network_only_return"],
              summary["aggregate"]["mean_unconstrained_actor_return"],
              summary["aggregate"]["mean_substitution_gain"],
          ))
    print("WROTE", OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
