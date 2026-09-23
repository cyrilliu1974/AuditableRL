"""Measure a two-part MDL code length for every 50-episode trace prefix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from auditability.grammar_audit_experiment import (
    PairGrammarCodec,
    measure_pair_grammar_mdl,
    tokenize_losslessly,
)


def episode_prefix_offsets(trace_path: Path, interval: int) -> list[tuple[int, int, int]]:
    """Return (completed episodes, transition records, byte offset) snapshots."""
    offsets = []
    last_completed = 0
    records = 0
    byte_offset = 0
    with trace_path.open("rb") as stream:
        for raw_line in stream:
            row = json.loads(raw_line)
            records += 1
            byte_offset += len(raw_line)
            if row["terminated"] or row["truncated"]:
                completed = int(row["episode"]) + 1
                if completed % interval == 0:
                    offsets.append((completed, records, byte_offset))
                    last_completed = completed
    if not offsets or offsets[-1][0] != last_completed:
        raise ValueError("no completed episode boundary found for an induction snapshot")
    return offsets


def analyze_run(run_dir: Path, output_path: Path, max_rules: int = 16) -> dict:
    trace_path = run_dir / "trajectory.jsonl"
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    interval = int(metadata["config"]["grammar_update_interval"])
    snapshots = episode_prefix_offsets(trace_path, interval)
    rows = []
    with trace_path.open("rb") as stream:
        for episodes, records, byte_offset in snapshots:
            stream.seek(0)
            prefix = stream.read(byte_offset).decode("utf-8")
            tokens = tokenize_losslessly(prefix)
            encoded = PairGrammarCodec(max_rules=max_rules).compress(tokens)
            mdl = measure_pair_grammar_mdl(encoded)
            rows.append({
                "grammar_round": episodes // interval,
                "completed_episodes": episodes,
                "transition_records": records,
                "source_prefix_bytes": byte_offset,
                "token_count": encoded["token_count"],
                "production_count": encoded["production_count"],
                "compressed_symbol_count": encoded["compressed_symbol_count"],
                "token_symbol_ratio": encoded["token_symbol_ratio"],
                "roundtrip_exact": encoded["roundtrip_exact"],
                **mdl,
            })
    result = {
        "claim_scope": "descriptive two-part code lengths for exact raw JSONL trace prefixes; compression grammar is Re-Pair-like pair substitution, not canonical SEQUITUR",
        "run_dir": str(run_dir),
        "seed": metadata["config"]["seed"],
        "arm": metadata["config"]["arm"],
        "grammar_update_interval_episodes": interval,
        "max_pair_rules": max_rules,
        "code_length_convention": rows[0]["code_length_schema"] if rows else None,
        "fixed_schema_overhead_included": False,
        "snapshots": rows,
        "interpretation_limit": "The grammar is induced by frequency-pair substitution and is not selected by minimizing this MDL score; scores evaluate its code length, not causal policy explanation or optimal grammar induction.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rules", type=int, default=16)
    args = parser.parse_args()
    result = analyze_run(args.run_dir, args.output, max_rules=args.max_rules)
    print(json.dumps({
        "run_dir": result["run_dir"],
        "snapshots": result["snapshots"],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
