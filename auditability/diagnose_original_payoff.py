"""Exhaustively audit the original AIM action-payoff table without importing ML dependencies."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "vqvae_agents_AIM.py"
OUTPUT = PROJECT_ROOT / "runs" / "objective_diagnostic.json"
ACTION_PAIRS = (("C", "C"), ("C", "D"), ("D", "C"), ("D", "D"))


def load_payoff_function(source_path: Path):
    """Compile only the pure payoff function, avoiding imports and side effects."""
    source_text = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source_text, filename=str(source_path))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "payoff"
    ]
    if len(nodes) != 1:
        raise RuntimeError(f"Expected one payoff function, found {len(nodes)}")
    isolated_module = ast.Module(body=nodes, type_ignores=[])
    isolated_module = ast.fix_missing_locations(isolated_module)
    namespace: dict[str, object] = {}
    exec(compile(isolated_module, str(source_path), "exec"), {"__builtins__": {}}, namespace)
    return namespace["payoff"], source_text


def main() -> None:
    payoff, source_text = load_payoff_function(SOURCE)
    table = {}
    for parity, label in (("even", 0), ("odd", 1)):
        rows = []
        for action_a, action_b in ACTION_PAIRS:
            reward_a, reward_b = payoff(action_a, action_b, label, 1)
            rows.append(
                {
                    "action_A": action_a,
                    "action_B": action_b,
                    "reward_A": reward_a,
                    "reward_B": reward_b,
                    "joint_reward": reward_a + reward_b,
                }
            )
        maximum = max(row["joint_reward"] for row in rows)
        maximizers = [
            [row["action_A"], row["action_B"]]
            for row in rows
            if row["joint_reward"] == maximum
        ]
        table[parity] = {"label_example": label, "rows": rows, "joint_optimum": maximizers}

    expected = {"even": [["C", "C"]], "odd": [["C", "C"]]}
    observed = {parity: data["joint_optimum"] for parity, data in table.items()}
    if observed != expected:
        raise AssertionError(f"Unexpected joint optima: {observed}")

    result = {
        "source_file": str(SOURCE),
        "source_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
        "objective_used_by_both_policies": "joint_reward = reward_A + reward_B",
        "reward_table_by_label_parity": table,
        "conclusion": (
            "C/C is the unique joint-reward optimum for both even and odd labels; "
            "the parity term does not make the optimal joint action label-dependent."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
