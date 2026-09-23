"""Cold-process same-seed reproducibility check for complete RL training runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from auditability.grammar_audit_experiment import (
    calibration_symbolizer,
    replay_environment_trace,
    train_one,
    verify_trace,
    verify_update_trace,
)


def run_check(output_root: Path, seed: int, episodes: int, calibration_episodes: int) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    symbolizer = calibration_symbolizer(seed, episodes=calibration_episodes)
    first = train_one(seed, "baseline", episodes, output_root / "repeat_a", symbolizer)
    second = train_one(seed, "baseline", episodes, output_root / "repeat_b", symbolizer)

    first_meta = json.loads((first / "metadata.json").read_text(encoding="utf-8"))
    second_meta = json.loads((second / "metadata.json").read_text(encoding="utf-8"))
    fields = (
        "actor_sha256",
        "trace_sha256",
        "training_updates_sha256",
        "training_checkpoint_sha256",
    )
    comparisons = {field: first_meta[field] == second_meta[field] for field in fields}
    audits = []
    for run_dir in (first, second):
        trace = verify_trace(run_dir / "trajectory.jsonl")
        updates = verify_update_trace(run_dir / "training_updates.jsonl")
        environment = replay_environment_trace(run_dir / "trajectory.jsonl")
        audits.append({"trace": trace, "updates": updates, "environment_replay": environment})

    result = {
        "claim_scope": "same-host, same-software CPU rerun of CartPole-v1 REINFORCE; not cross-hardware equivalence",
        "seed": seed,
        "episodes": episodes,
        "run_a": str(first),
        "run_b": str(second),
        "identical_hashes": comparisons,
        "all_hashes_identical": all(comparisons.values()),
        "both_runs_passed_trace_update_environment_audits": all(
            item["trace"]["valid"]
            and item["updates"]["valid"]
            and item["environment_replay"]["valid"]
            for item in audits
        ),
        "run_audits": audits,
    }
    report_path = output_root / "repeatability_report.json"
    report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 29])
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--calibration-episodes", type=int, default=24)
    args = parser.parse_args()
    results = [run_check(args.output_root / f"seed-{seed}", seed, args.episodes, args.calibration_episodes)
               for seed in args.seeds]
    print(json.dumps({"runs": results, "all_seeds_repeat_exactly": all(
        result["all_hashes_identical"] and result["both_runs_passed_trace_update_environment_audits"]
        for result in results
    )}, indent=2))


if __name__ == "__main__":
    main()
