"""Replace repeated inline rule provenance with hash-addressed snapshots, preserving replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from auditability.grammar_audit_experiment import (
    ZERO_HASH,
    canonical_json,
    chain_record,
    sha256_file,
    verify_trace,
)


def compact_run(run_dir: Path) -> dict:
    run_dir = run_dir.resolve()
    trace_path = run_dir / "trajectory.jsonl"
    metadata_path = run_dir / "metadata.json"
    snapshot_path = run_dir / "constraint_rule_snapshots.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    before_bytes = trace_path.stat().st_size
    existing_snapshots = {}
    if snapshot_path.exists():
        existing_snapshots = json.loads(snapshot_path.read_text(encoding="utf-8"))
    temp_path = trace_path.with_suffix(".compact.tmp")
    previous = ZERO_HASH
    records = 0
    references = 0
    with trace_path.open("r", encoding="utf-8") as source, temp_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as target:
        for line_no, line in enumerate(source, 1):
            row = json.loads(line)
            row.pop("record_hash", None)
            rule = row.get("constraint_rule")
            if isinstance(rule, dict) and "source_step_ids" in rule:
                source_steps = rule.pop("source_step_ids")
                snapshot = {
                    "state_symbol": "STATE_" + "_".join(map(str, row["state_symbol"])),
                    "action": int(rule["action"]),
                    "support": int(rule["support"]),
                    "condition_count": int(rule["condition_count"]),
                    "confidence": float(rule["confidence"]),
                    "mean_reward": float(rule["mean_reward"]),
                    "source_step_ids": source_steps,
                }
                snapshot_id = hashlib.sha256(canonical_json(snapshot).encode("utf-8")).hexdigest()
                existing_snapshots[snapshot_id] = snapshot
                rule["source_step_count"] = len(source_steps)
                rule["source_step_ids_sha256"] = hashlib.sha256(
                    canonical_json(source_steps).encode("utf-8")
                ).hexdigest()
                rule["snapshot_id"] = snapshot_id
                references += 1
            elif isinstance(rule, dict) and "snapshot_id" in rule:
                references += 1
            chained = chain_record(row, previous)
            previous = chained["record_hash"]
            target.write(canonical_json(chained) + "\n")
            records += 1

    snapshot_path.write_text(
        json.dumps(existing_snapshots, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    backup_path = trace_path.with_suffix(".precompact.jsonl")
    if backup_path.exists():
        raise FileExistsError(f"refusing to overwrite prior trace backup: {backup_path}")
    trace_path.replace(backup_path)
    temp_path.replace(trace_path)
    try:
        trace_audit = verify_trace(trace_path)
        for snapshot_id, snapshot in existing_snapshots.items():
            actual_id = hashlib.sha256(canonical_json(snapshot).encode("utf-8")).hexdigest()
            if actual_id != snapshot_id:
                raise ValueError(f"snapshot hash mismatch: {snapshot_id}")
        metadata["trace_sha256"] = sha256_file(trace_path)
        metadata["trace_audit"] = trace_audit
        metadata["constraint_rule_snapshots_file"] = snapshot_path.name
        metadata["constraint_rule_snapshots_sha256"] = sha256_file(snapshot_path)
        metadata["trace_provenance_compaction"] = {
            "method": "replace repeated full source-step lists in decisions with SHA-256 references to a deduplicated snapshot registry",
            "rule_snapshot_count": len(existing_snapshots),
            "rule_references": references,
            "hash_chain_recomputed_and_verified": True,
            "trace_bytes_before": before_bytes,
            "trace_bytes_after": trace_path.stat().st_size,
            "trace_byte_reduction_ratio": 1 - trace_path.stat().st_size / max(1, before_bytes),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    except Exception:
        trace_path.unlink(missing_ok=True)
        backup_path.replace(trace_path)
        raise
    backup_path.unlink()
    return {
        "run_dir": str(run_dir),
        "trace_records": records,
        "rule_snapshot_count": len(existing_snapshots),
        "rule_references": references,
        "trace_bytes_before": before_bytes,
        "trace_bytes_after": trace_path.stat().st_size,
        "trace_reduction_ratio": 1 - trace_path.stat().st_size / max(1, before_bytes),
        "trace_sha256": metadata["trace_sha256"],
        "snapshot_registry_sha256": metadata["constraint_rule_snapshots_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps([compact_run(path) for path in args.run_dirs], indent=2))


if __name__ == "__main__":
    main()
