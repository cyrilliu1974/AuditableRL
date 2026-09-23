# -*- coding: utf-8 -*-
"""Test whether the rule-constrained arm's bank is monotone.

The observer accumulates evidence in a table that is never reset
(``counts[condition][action] += 1``), and ``_eligible_rules`` derives every rule from that whole
cumulative table. In the constrained arm the hard mask forces the rule's action on any covered
condition, so once a condition is covered the observer can only ever record that one action for it.
Two consequences follow, and both are falsifiable from the saved traces:

1. **No retraction.** A rule, once admitted, can never lose eligibility, because its confidence
   cannot fall: every further visit to its condition re-records the same action. So if a condition
   was covered in generation ``g`` and is visited again in a later generation, it must still be
   covered there.

2. **No action flip.** The rule's action is the argmax over the cumulative counts for its condition.
   With only one action ever recorded after coverage begins, the argmax cannot change. So the action
   the mask forces on a condition must be identical in every generation in which it is covered.

Either violation falsifies the fixed-point account of the self-confirming loop. Neither requires
training: the traces on disk contain ``constraint_rule`` per step.

Generation index. The trace's ``grammar_generation`` field is written but never updated: it is ``0``
for every one of the 74320 steps of seed 11, so it cannot be used to segment the run. The generation
is therefore derived from the recorded episode index using the configured re-induction interval
(``grammar_update_interval = 50``), i.e. ``generation = episode // 50``. That interval is read from
the run's own metadata rather than assumed.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT = "CartPole-v1"
SEEDS = [11, 29, 43, 71, 101, 149, 211, 307]
TASK_ROOT = ROOT / "runs" / "shared_multienv_gpi" / ENVIRONMENT
OUTPUT = ROOT / "results" / "constraint_rule_retraction.json"


def latest_run(arm: str, seed: int) -> Path:
    matches = sorted((TASK_ROOT / arm).glob(f"seed-{seed}_*"))
    completed = [p for p in matches if (p / "metadata.json").exists()]
    if not completed:
        raise FileNotFoundError(f"no completed {arm} run for seed {seed}")
    return completed[-1]


def analyse(run_dir: Path) -> dict:
    """Stream one constrained run and look for retractions and action flips."""
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    interval = int(metadata["config"]["grammar_update_interval"])

    # condition -> generation -> set of actions executed while a rule was applied
    covered: dict[tuple, dict[int, set]] = defaultdict(lambda: defaultdict(set))
    # condition -> generations in which the condition was visited at all
    visited: dict[tuple, set] = defaultdict(set)
    generations = set()
    steps = 0

    with (run_dir / "trajectory.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            generation = int(record["episode"]) // interval
            condition = tuple(record["state_symbol"])
            generations.add(generation)
            visited[condition].add(generation)
            if record["constraint_rule"] is not None:
                covered[condition][generation].add(int(record["action"]))
            steps += 1

    retractions = []
    flips = []
    for condition, by_generation in covered.items():
        covered_gens = sorted(by_generation)
        first_covered = covered_gens[0]
        # (1) a condition covered here, visited later, must still be covered later
        for generation in sorted(visited[condition]):
            if generation > first_covered and generation not in by_generation:
                retractions.append({
                    "condition": list(condition),
                    "first_covered_generation": first_covered,
                    "visited_uncovered_generation": generation,
                })
        # (2) the forced action must be identical in every covered generation
        actions = set()
        for generation in covered_gens:
            actions |= by_generation[generation]
        if len(actions) > 1:
            flips.append({
                "condition": list(condition),
                "actions": sorted(actions),
                "by_generation": {str(g): sorted(by_generation[g]) for g in covered_gens},
            })

    return {
        "run_dir": str(run_dir),
        "reinduction_interval_episodes": interval,
        "training_steps": steps,
        "generations_observed": sorted(generations),
        "conditions_ever_covered": len(covered),
        "conditions_ever_visited": len(visited),
        "retraction_events": len(retractions),
        "action_flip_events": len(flips),
        "retraction_examples": retractions[:5],
        "action_flip_examples": flips[:5],
    }


def main() -> int:
    rows = []
    print("%-6s %-9s %-9s %-14s %-13s %-11s" % (
        "seed", "gens", "steps", "cond.covered", "retractions", "action flips"))
    for seed in SEEDS:
        row = {"seed": seed, **analyse(latest_run("grammar_constrained", seed))}
        rows.append(row)
        print("%-6d %-9d %-9d %-14d %-13d %-11d" % (
            seed,
            len(row["generations_observed"]),
            row["training_steps"],
            row["conditions_ever_covered"],
            row["retraction_events"],
            row["action_flip_events"],
        ))

    total_retractions = sum(r["retraction_events"] for r in rows)
    total_flips = sum(r["action_flip_events"] for r in rows)
    print()
    print("total retraction events: %d" % total_retractions)
    print("total action flips:      %d" % total_flips)
    verdict = (
        "MONOTONE: no rule was ever retracted and no forced action ever changed, in any seed"
        if total_retractions == 0 and total_flips == 0
        else "NON-MONOTONE: the fixed-point account is falsified"
    )
    print("verdict:", verdict)

    summary = {
        "purpose": (
            "Test the fixed-point prediction of the self-confirming loop: in the rule-constrained "
            "arm, an admitted rule can neither be retracted nor change its action, because the mask "
            "prevents any contradicting observation from being recorded."
        ),
        "environment": ENVIRONMENT,
        "seeds": SEEDS,
        "per_seed": rows,
        "total_retraction_events": total_retractions,
        "total_action_flip_events": total_flips,
        "bank_is_monotone": total_retractions == 0 and total_flips == 0,
        "verdict": verdict,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("WROTE", OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
