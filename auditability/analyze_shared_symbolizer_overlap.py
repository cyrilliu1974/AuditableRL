"""Measure cross-seed rule overlap in a single frozen-symbolizer experiment."""

from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
from pathlib import Path

from auditability.grammar_audit_experiment import canonical_json


def load_rules(run_dir: Path) -> tuple[dict, dict[tuple[tuple[int, ...], int], dict]]:
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    grammar = json.loads((run_dir / "induced_grammar.json").read_text(encoding="utf-8"))
    rules = {}
    for production in grammar.get("productions", []):
        state, action = production["rhs"]
        condition = tuple(int(part) for part in state.removeprefix("STATE_").split("_"))
        action_id = int(action.removeprefix("ACTION_"))
        rules[(condition, action_id)] = production
    return metadata, rules


def find_run(task_root: Path, seed: int, arm: str = "grammar_audit_only") -> Path:
    candidates = [path for path in (task_root / arm).glob(f"seed-{seed}_*") if (path / "metadata.json").is_file()]
    if len(candidates) != 1:
        raise ValueError(f"expected one complete {arm} run for seed {seed}; found {len(candidates)}")
    return candidates[0]


def pair_overlap(left: dict, right: dict) -> dict:
    left_keys, right_keys = set(left), set(right)
    left_conditions = {condition for condition, _ in left_keys}
    right_conditions = {condition for condition, _ in right_keys}
    shared_conditions = left_conditions & right_conditions
    conflicts = [condition for condition in shared_conditions if
                 {action for cond, action in left_keys if cond == condition} !=
                 {action for cond, action in right_keys if cond == condition}]
    intersection, union = left_keys & right_keys, left_keys | right_keys
    condition_union = left_conditions | right_conditions
    return {
        "left_rule_count": len(left_keys),
        "right_rule_count": len(right_keys),
        "exact_state_action_intersection": len(intersection),
        "exact_state_action_union": len(union),
        "state_action_jaccard": len(intersection) / len(union) if union else 1.0,
        "condition_intersection": len(shared_conditions),
        "condition_union": len(condition_union),
        "condition_jaccard": len(shared_conditions) / len(condition_union) if condition_union else 1.0,
        "shared_condition_action_conflict_count": len(conflicts),
        "shared_conditions_same_action_count": len(shared_conditions) - len(conflicts),
        "shared_exact_rules": [
            {"condition": list(condition), "action": action}
            for condition, action in sorted(intersection)
        ],
        "conflicting_conditions": [list(condition) for condition in sorted(conflicts)],
    }


def cross_apply_training_states(left: dict, right: dict, left_trace: Path, right_trace: Path) -> dict:
    left_by_condition = {condition: action for condition, action in left}
    right_by_condition = {condition: action for condition, action in right}

    def score(trace_path: Path) -> dict:
        counts = collections.Counter()
        with trace_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                condition = tuple(int(x) for x in row["state_symbol"])
                left_action = left_by_condition.get(condition)
                right_action = right_by_condition.get(condition)
                counts["states"] += 1
                counts["left_rule_covered"] += left_action is not None
                counts["right_rule_covered"] += right_action is not None
                if left_action is not None and right_action is not None:
                    counts["both_rules_covered"] += 1
                    counts["same_rule_action"] += left_action == right_action
                    counts["different_rule_action"] += left_action != right_action
        joint = counts["both_rules_covered"]
        total = counts["states"]
        return {
            **dict(counts),
            "left_rule_coverage": counts["left_rule_covered"] / max(1, total),
            "right_rule_coverage": counts["right_rule_covered"] / max(1, total),
            "joint_rule_coverage": joint / max(1, total),
            "action_agreement_when_both_rules_cover": counts["same_rule_action"] / max(1, joint),
            "state_source_role": "training trajectory state occupancy; not an independent held-out policy agreement test",
        }

    return {
        "seed_11_training_states": score(left_trace),
        "seed_29_training_states": score(right_trace),
    }


def analyze(task_root: Path, seeds: list[int]) -> dict:
    runs = {seed: find_run(task_root, seed) for seed in seeds}
    metadata = {}
    rules = {}
    for seed, run_dir in runs.items():
        metadata[seed], rules[seed] = load_rules(run_dir)
    quantizers = {
        hashlib.sha256(canonical_json(meta["config"]["state_bins"]).encode("utf-8")).hexdigest()
        for meta in metadata.values()
    }
    if len(quantizers) != 1:
        raise ValueError("seed grammars do not share exactly the same frozen symbolizer")
    all_pairs = []
    for left_seed, right_seed in itertools.combinations(seeds, 2):
        overlap = pair_overlap(rules[left_seed], rules[right_seed])
        all_pairs.append({"left_seed": left_seed, "right_seed": right_seed, **{
            key: overlap[key] for key in (
                "exact_state_action_intersection", "exact_state_action_union", "state_action_jaccard",
                "condition_intersection", "condition_union", "condition_jaccard",
                "shared_condition_action_conflict_count", "shared_conditions_same_action_count",
            )
        }})
    key_pair = (11, 29)
    if not set(key_pair).issubset(rules):
        raise ValueError("the requested 11/29 pair is missing from the seed set")
    pair = pair_overlap(rules[11], rules[29])
    left_trace = runs[11] / "trajectory.jsonl"
    right_trace = runs[29] / "trajectory.jsonl"
    return {
        "environment": json.loads((task_root / "shared_calibration.json").read_text(encoding="utf-8"))["environment"],
        "confidence_threshold": metadata[11]["config"]["grammar_min_confidence"],
        "support_threshold": metadata[11]["config"]["grammar_min_support"],
        "shared_symbolizer_sha256": next(iter(quantizers)),
        "shared_symbolizer_verified_across_seeds": True,
        "seed_11_vs_29": {
            "seed_11_run": str(runs[11]),
            "seed_29_run": str(runs[29]),
            **pair,
            "cross_application_on_training_state_occupancy": cross_apply_training_states(
                rules[11], rules[29], left_trace, right_trace
            ),
        },
        "all_pairwise_seed_overlaps": all_pairs,
        "all_pairwise_jaccard_mean": sum(row["state_action_jaccard"] for row in all_pairs) / max(1, len(all_pairs)),
        "all_pairwise_jaccard_median": sorted(row["state_action_jaccard"] for row in all_pairs)[len(all_pairs) // 2] if all_pairs else None,
        "claim_boundary": "Shared bins remove seed-specific quantizer-edge mismatch. Rule-set overlap and rule predictions on logged training-state occupancy do not establish causal explanations or independent held-out actor action agreement.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.task_root, args.seeds)
    output = args.output or args.task_root / "shared_symbolizer_rule_overlap.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "environment": result["environment"],
        "seed_11_vs_29": {
            key: result["seed_11_vs_29"][key]
            for key in ("exact_state_action_intersection", "exact_state_action_union", "state_action_jaccard", "condition_intersection", "shared_condition_action_conflict_count")
        },
        "pairwise_mean_jaccard": result["all_pairwise_jaccard_mean"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
