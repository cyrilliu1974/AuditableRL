"""Matched-seed uniform-random-action control for Acrobot fusion ablations."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

import gymnasium as gym
import numpy as np


def bootstrap_ci(deltas: list[float], seed: int) -> list[float]:
    rng = random.Random(seed)
    draws = sorted(statistics.mean(deltas[rng.randrange(len(deltas))] for _ in deltas) for _ in range(5000))
    return [draws[124], draws[4874]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--reset-base", type=int, default=20_000_000)
    args = parser.parse_args()
    task_root = args.task_root.resolve()
    summary = json.loads((task_root / "shared_multiseed_gpi_summary.json").read_text(encoding="utf-8"))
    fusion_returns = summary["composition_evaluation"]["policies"]["grammar_fusion"]["return_by_episode"][:args.episodes]
    env_name = json.loads((task_root / "shared_calibration.json").read_text(encoding="utf-8"))["environment"]
    env = gym.make(env_name)
    output = []
    for replica in range(args.replicates):
        returns = []
        action_counts = {str(action): 0 for action in range(int(env.action_space.n))}
        for episode in range(args.episodes):
            observation, _ = env.reset(seed=args.reset_base + episode)
            rng = random.Random(80_000_000 + replica * 100_000 + episode)
            total = 0.0
            done = False
            while not done:
                action = rng.randrange(int(env.action_space.n))
                action_counts[str(action)] += 1
                observation, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                done = terminated or truncated
            returns.append(total)
        delta = [left - right for left, right in zip(returns, fusion_returns)]
        output.append({
            "replicate": replica,
            "mean_return": statistics.mean(returns),
            "median_return": statistics.median(returns),
            "episodes_terminating_before_500": sum(value > -500 for value in returns),
            "paired_delta_vs_grammar_fusion": statistics.mean(delta),
            "paired_delta_bootstrap_ci95": bootstrap_ci(delta, 80_202_609 + replica),
            "action_counts": action_counts,
        })
    env.close()
    result = {
        "environment": env_name,
        "episodes": args.episodes,
        "replicates": args.replicates,
        "reset_seed_formula": f"{args.reset_base} + episode_index; paired with primary grammar-fusion evaluation",
        "control": "Independent uniformly sampled discrete action at every time step, without consulting source rules or actor logits.",
        "results": output,
        "mean_across_random_replicates": statistics.mean(row["mean_return"] for row in output),
        "interpretation_limit": "This is a matched environment control for this Acrobot setup, not a universal lower bound or a comparison with a tuned stochastic policy.",
    }
    path = task_root / "random_action_control.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(path), "results": output,
                      "mean_across_random_replicates": result["mean_across_random_replicates"]}, indent=2))


if __name__ == "__main__":
    main()
