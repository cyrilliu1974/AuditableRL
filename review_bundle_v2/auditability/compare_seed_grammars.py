"""Compare two induced state-action grammars and quantify their shared rule core."""

from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path


def load_run(run_dir: Path) -> tuple[dict, dict[str, dict]]:
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    grammar = json.loads((run_dir / "induced_grammar.json").read_text(encoding="utf-8"))
    rules = {}
    for production in grammar.get("productions", []):
        state_symbol, action_symbol = production["rhs"]
        rules[(state_symbol, action_symbol)] = production
    return metadata, rules


def compare_runs(left_dir: Path, right_dir: Path) -> dict:
    left_meta, left_rules = load_run(left_dir)
    right_meta, right_rules = load_run(right_dir)
    left_set, right_set = set(left_rules), set(right_rules)
    shared = left_set & right_set
    union = left_set | right_set
    left_conditions = {state for state, _ in left_set}
    right_conditions = {state for state, _ in right_set}
    shared_conditions = left_conditions & right_conditions
    conflicts = [
        {
            "state_symbol": state,
            "left_actions": sorted(action for condition, action in left_set if condition == state),
            "right_actions": sorted(action for condition, action in right_set if condition == state),
        }
        for state in sorted(shared_conditions)
        if {action for condition, action in left_set if condition == state}
        != {action for condition, action in right_set if condition == state}
    ]

    left_edges = left_meta["config"]["state_bins"]["edges"]
    right_edges = right_meta["config"]["state_bins"]["edges"]
    edge_distances = [
        {"dimension": dimension, "absolute_edge_differences": [abs(a - b) for a, b in zip(left, right)]}
        for dimension, (left, right) in enumerate(zip(left_edges, right_edges))
    ]
    all_diffs = [difference for row in edge_distances for difference in row["absolute_edge_differences"]]
    shared_rows = []
    for state, action in sorted(shared):
        left, right = left_rules[(state, action)], right_rules[(state, action)]
        shared_rows.append({
            "state_symbol": state,
            "action_symbol": action,
            "left_support": left["support"],
            "right_support": right["support"],
            "left_confidence": left["confidence"],
            "right_confidence": right["confidence"],
            "left_source_step_count": len(left.get("source_step_ids", [])),
            "right_source_step_count": len(right.get("source_step_ids", [])),
        })

    return {
        "comparison_scope": "exact symbol/action production identity; equal state-symbol indices are not guaranteed to denote identical raw-state intervals",
        "left": {
            "run_dir": str(left_dir),
            "seed": left_meta["config"]["seed"],
            "arm": left_meta["config"]["arm"],
            "state_bins": left_meta["config"]["state_bins"]["n_bins"],
            "min_support": left_meta["config"]["grammar_min_support"],
            "min_confidence": left_meta["config"]["grammar_min_confidence"],
            "rule_count": len(left_set),
        },
        "right": {
            "run_dir": str(right_dir),
            "seed": right_meta["config"]["seed"],
            "arm": right_meta["config"]["arm"],
            "state_bins": right_meta["config"]["state_bins"]["n_bins"],
            "min_support": right_meta["config"]["grammar_min_support"],
            "min_confidence": right_meta["config"]["grammar_min_confidence"],
            "rule_count": len(right_set),
        },
        "rule_overlap": {
            "intersection_count": len(shared),
            "union_count": len(union),
            "jaccard": len(shared) / len(union) if union else 1.0,
            "left_contained_in_right": len(left_set - right_set) == 0,
            "right_contained_in_left": len(right_set - left_set) == 0,
            "shared_rules": shared_rows,
        },
        "condition_overlap": {
            "left_condition_count": len(left_conditions),
            "right_condition_count": len(right_conditions),
            "shared_condition_count": len(shared_conditions),
            "same_action_for_shared_condition_count": len(shared_conditions) - len(conflicts),
            "shared_condition_action_conflicts": conflicts,
        },
        "calibration_edge_alignment": {
            "exactly_equal": left_edges == right_edges,
            "per_dimension_absolute_differences": edge_distances,
            "maximum_absolute_edge_difference": max(all_diffs, default=0.0),
            "mean_absolute_edge_difference": sum(all_diffs) / len(all_diffs) if all_diffs else 0.0,
        },
        "interpretation_limit": "Raw set overlap is a fast syntactic estimate of a shared core. Because each seed currently fits its own quantile boundaries, semantic overlap requires a follow-up with a shared frozen symbolizer or raw-state remapping.",
    }


def compare_on_raw_state_traces(left_dir: Path, right_dir: Path, left_trace: Path, right_trace: Path) -> dict:
    """Compare rule actions on identical raw states after each seed's own bin mapping."""
    left_meta, left_rules = load_run(left_dir)
    right_meta, right_rules = load_run(right_dir)
    left_edges = left_meta["config"]["state_bins"]["edges"]
    right_edges = right_meta["config"]["state_bins"]["edges"]

    def action_for(state: list[float], edges: list[list[float]], rules: dict[str, int]) -> int | None:
        bins = [bisect.bisect_right(edge, value) for edge, value in zip(edges, state)]
        return rules.get("STATE_" + "_".join(map(str, bins)))

    left_actions = {state: int(action.removeprefix("ACTION_")) for state, action in left_rules}
    right_actions = {state: int(action.removeprefix("ACTION_")) for state, action in right_rules}

    def score(trace: Path) -> dict:
        counts = {"total_steps": 0, "left_covered": 0, "right_covered": 0, "both_covered": 0, "same_action": 0}
        with trace.open("r", encoding="utf-8") as stream:
            for line in stream:
                event = json.loads(line)
                state = event["state"]
                left_action = action_for(state, left_edges, left_actions)
                right_action = action_for(state, right_edges, right_actions)
                counts["total_steps"] += 1
                counts["left_covered"] += left_action is not None
                counts["right_covered"] += right_action is not None
                if left_action is not None and right_action is not None:
                    counts["both_covered"] += 1
                    counts["same_action"] += left_action == right_action
        return {
            **counts,
            "left_coverage": counts["left_covered"] / max(1, counts["total_steps"]),
            "right_coverage": counts["right_covered"] / max(1, counts["total_steps"]),
            "joint_coverage": counts["both_covered"] / max(1, counts["total_steps"]),
            "agreement_when_both_covered": counts["same_action"] / max(1, counts["both_covered"]),
        }

    return {
        "comparison_scope": "both induced grammars are applied to the exact raw states in each recorded training trace, each with its own fitted quantile edges",
        "both_grammars_on_left_seed_trace": score(left_trace),
        "both_grammars_on_right_seed_trace": score(right_trace),
        "interpretation_limit": "This is a state-occupancy-weighted cross-seed rule-agreement estimate on recorded training states, not an independent held-out causal test.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-run", type=Path, required=True)
    parser.add_argument("--right-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--left-trace", type=Path)
    parser.add_argument("--right-trace", type=Path)
    args = parser.parse_args()
    result = compare_runs(args.left_run, args.right_run)
    if args.left_trace and args.right_trace:
        result["raw_state_trace_overlap"] = compare_on_raw_state_traces(
            args.left_run, args.right_run, args.left_trace, args.right_trace
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
