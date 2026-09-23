"""Independently replay a private-target AIM trace using only Python's stdlib."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ZERO_HASH = "0" * 64


def canonical_json(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def validate_trace(trace_path: Path, K: int) -> dict:
    previous_hash = ZERO_HASH
    reward_sum = 0
    correct = 0
    rows = 0
    with trace_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number}: record must be a JSON object")
            claimed_hash = row.pop("record_hash", None)
            if row.get("previous_hash") != previous_hash:
                raise ValueError(f"line {line_number}: previous hash mismatch")
            calculated_hash = hashlib.sha256(
                (previous_hash + canonical_json(row)).encode("utf-8")
            ).hexdigest()
            if claimed_hash != calculated_hash:
                raise ValueError(f"line {line_number}: record hash mismatch")

            try:
                episode = int(row["episode"])
                label = int(row["private_label"])
                message_a = row["A_aim"]
                message_b = row["B_aim"]
                target_action = row["target_action"]
                action = row["B_action"]
                reward = int(row["reward"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"line {line_number}: required field missing or invalid") from exc
            if episode != line_number - 1:
                raise ValueError(f"line {line_number}: episode index is {episode}")
            if not 0 <= label <= 9:
                raise ValueError(f"line {line_number}: private label outside 0..9")
            if not isinstance(message_a, list) or not isinstance(message_b, list):
                raise ValueError(f"line {line_number}: messages must be lists")
            if not message_b or len(message_a) != len(message_b):
                raise ValueError(f"line {line_number}: empty or mismatched message lengths")
            if any(type(symbol) is not int or not 0 <= symbol < K for symbol in message_a + message_b):
                raise ValueError(f"line {line_number}: symbol outside codebook range")

            expected_target = "C" if label % 2 == 0 else "D"
            expected_action = "C" if message_b[0] < K // 2 else "D"
            expected_reward = 1 if expected_action == expected_target else -1
            if target_action != expected_target:
                raise ValueError(f"line {line_number}: target action mismatch")
            if action != expected_action:
                raise ValueError(f"line {line_number}: receiver action mismatch")
            if reward != expected_reward:
                raise ValueError(f"line {line_number}: reward mismatch")

            reward_sum += reward
            correct += int(reward > 0)
            previous_hash = claimed_hash
            rows = line_number

    return {
        "valid": True,
        "record_count": rows,
        "correct_count": correct,
        "accuracy": correct / rows if rows else None,
        "mean_reward": reward_sum / rows if rows else None,
        "final_record_hash": previous_hash,
        "validator": "standalone-python-stdlib",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--K", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(validate_trace(args.trace, args.K), indent=2))


if __name__ == "__main__":
    main()
