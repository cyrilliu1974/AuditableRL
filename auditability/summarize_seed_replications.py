"""Collect per-seed controlled-game evidence into a compact JSON artifact."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from validate_symbol_trace import validate_trace


def read_run(run_dir: Path) -> dict:
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    training = json.loads((run_dir / "training_summary.json").read_text(encoding="utf-8"))
    evaluation = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
    matched = json.loads((run_dir / "matched_prefix_effect.json").read_text(encoding="utf-8"))
    trace_audit = validate_trace(run_dir / "symbol_training_trace.jsonl", metadata["config"]["K"])
    causal = evaluation["causal_intervention"]
    active = matched["active_position_matched_support_causal_contrast"]
    return {
        "seed": metadata["seed"],
        "run_dir": str(run_dir),
        "episodes": metadata["config"]["episodes"],
        "optimizer_updates": metadata["config"]["optimizer_updates"],
        "training_sampled_accuracy": training["sampled_accuracy_over_training_episodes"],
        "sampled_sender_eval_accuracy": evaluation["sampled_sender_expected_receiver_accuracy"],
        "shuffled_message_accuracy": evaluation["shuffled_message_expected_receiver_accuracy"],
        "no_message_accuracy": evaluation["no_message_receiver_accuracy"],
        "whole_message_delta_P_B_C": causal["delta_P_C_even_minus_odd"],
        "whole_message_bootstrap_95pct_ci": causal["delta_bootstrap_95pct_ci"],
        "active_token_position": matched[
            "active_position_selected_by_highest_observed_mutual_information"
        ],
        "active_token_MI_bits": matched["per_position_parity_mutual_information_bits"][
            str(matched["active_position_selected_by_highest_observed_mutual_information"])
        ]["observed_mutual_information_bits"],
        "matched_other_token_overlap_mass": active["shared_other_token_probability_mass"],
        "matched_token_delta_P_B_C": active["delta_P_B_C_even_minus_odd"],
        "matched_even_P_B_C": active["P_B_C_under_even_token_given_matched_other"],
        "matched_odd_P_B_C": active["P_B_C_under_odd_token_given_matched_other"],
        "symbol_trace_audit": trace_audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = sorted((read_run(path) for path in args.run_dirs), key=lambda item: item["seed"])

    metrics = (
        "sampled_sender_eval_accuracy",
        "shuffled_message_accuracy",
        "no_message_accuracy",
        "whole_message_delta_P_B_C",
        "matched_other_token_overlap_mass",
        "matched_token_delta_P_B_C",
    )
    aggregates = {}
    for metric in metrics:
        values = [float(run[metric]) for run in runs]
        aggregates[metric] = {
            "mean_across_seeds": statistics.mean(values),
            "minimum": min(values),
            "maximum": max(values),
        }

    result = {
        "claim_scope": "controlled private-parity AIM communication game only",
        "seed_count": len(runs),
        "runs": runs,
        "aggregate_metrics": aggregates,
        "all_seeds_pass_descriptive_gate": all(
            run["sampled_sender_eval_accuracy"] >= 0.90
            and abs(run["shuffled_message_accuracy"] - 0.5) <= 0.05
            and run["matched_other_token_overlap_mass"] >= 0.80
            and run["matched_token_delta_P_B_C"] >= 0.90
            for run in runs
        ),
        "interpretation_limit": (
            "The symbol trace reconstructs recorded decisions and rewards; it does not reconstruct weights, "
            "gradients, optimizer updates, or the full training computation. Causal symbol evidence is for "
            "this designed task and does not establish unrestricted RL auditability or human-readable semantics."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
