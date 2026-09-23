"""Verify hash chains, update equivalence, replay, and compressed rule provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from auditability.grammar_audit_experiment import (
    replay_environment_trace,
    verify_trace,
    verify_update_trace,
)
from auditability.run_shared_multienv_gpi import ARMS, verify_constraint_snapshots


def find_complete(task_root: Path, seed: int, arm: str) -> Path:
    candidates = [
        path for path in (task_root / arm).glob(f"seed-{seed}_*")
        if (path / "metadata.json").is_file()
    ]
    if len(candidates) != 1:
        raise ValueError(f"expected one completed run for seed={seed} arm={arm}; found {len(candidates)}")
    return candidates[0]


def audit_task(task_root: Path, replay_constraint_runs: bool = True) -> dict:
    summary_path = task_root / "shared_multiseed_gpi_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    seeds = summary["experiment_design"]["seeds"]
    environment = summary["environment"]
    rows = []
    total_trace_rows = total_update_rows = total_replay_rows = total_rule_refs = 0
    for seed in seeds:
        run_paths = {arm: find_complete(task_root, seed, arm) for arm in ARMS}
        run_audits = {}
        metadata = {}
        for arm, run_dir in run_paths.items():
            metadata[arm] = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
            trace = verify_trace(run_dir / "trajectory.jsonl")
            updates = verify_update_trace(run_dir / "training_updates.jsonl")
            if trace["final_hash"] != metadata[arm]["trace_audit"]["final_hash"]:
                raise ValueError(f"trace final hash mismatch: {run_dir}")
            if updates["final_hash"] != metadata[arm]["training_update_audit"]["final_hash"]:
                raise ValueError(f"update final hash mismatch: {run_dir}")
            total_trace_rows += trace["records"]
            total_update_rows += updates["records"]
            run_audits[arm] = {"trace_chain": trace, "update_chain": updates}
        equivalence = all(
            metadata["baseline"][key] == metadata["grammar_audit_only"][key]
            for key in ("actor_sha256", "trace_sha256", "training_update_core_sha256")
        )
        if not equivalence:
            raise ValueError(f"baseline/audit-only mismatch for {environment} seed {seed}")
        baseline_replay = replay_environment_trace(run_paths["baseline"] / "trajectory.jsonl", environment)
        total_replay_rows += baseline_replay["transition_count"]
        run_audits["baseline"]["environment_replay"] = baseline_replay
        if replay_constraint_runs:
            constrained_replay = replay_environment_trace(run_paths["grammar_constrained"] / "trajectory.jsonl", environment)
            total_replay_rows += constrained_replay["transition_count"]
            run_audits["grammar_constrained"]["environment_replay"] = constrained_replay
        provenance = verify_constraint_snapshots(run_paths["grammar_constrained"])
        total_rule_refs += provenance["decision_references"]
        run_audits["grammar_constrained"]["rule_snapshot_audit"] = provenance
        rows.append({
            "seed": seed,
            "baseline_audit_only_exact_equivalence": equivalence,
            "runs": run_audits,
        })
    audit = {
        "environment": environment,
        "seeds": seeds,
        "all_hash_chains_valid": True,
        "all_update_ledgers_valid": True,
        "all_baseline_audit_only_pairs_exactly_equivalent": True,
        "all_baseline_traces_environment_replayed": True,
        "all_constraint_traces_environment_replayed": replay_constraint_runs,
        "all_constraint_rule_references_resolve_and_verify": True,
        "trace_records_across_all_arms": total_trace_rows,
        "optimizer_update_records_across_all_arms": total_update_rows,
        "environment_replay_records": total_replay_rows,
        "constraint_rule_decision_references": total_rule_refs,
        "per_seed": rows,
    }
    (task_root / "artifact_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    summary["posthoc_artifact_audit"] = {
        key: value for key, value in audit.items() if key != "per_seed"
    }
    summary["posthoc_artifact_audit_file"] = "artifact_audit.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_roots", type=Path, nargs="+")
    parser.add_argument("--skip-constraint-replay", action="store_true")
    args = parser.parse_args()
    results = [audit_task(root, replay_constraint_runs=not args.skip_constraint_replay) for root in args.task_roots]
    print(json.dumps([{
        key: result[key] for key in (
            "environment", "all_hash_chains_valid", "all_baseline_audit_only_pairs_exactly_equivalent",
            "all_baseline_traces_environment_replayed", "all_constraint_traces_environment_replayed",
            "all_constraint_rule_references_resolve_and_verify", "trace_records_across_all_arms",
            "environment_replay_records", "constraint_rule_decision_references",
        )
    } for result in results], indent=2))


if __name__ == "__main__":
    main()
