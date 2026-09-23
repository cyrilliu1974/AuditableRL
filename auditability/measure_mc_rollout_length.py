"""Measure the true rollout-length distribution of the Monte-Carlo arbitration probe.

The earlier A2 cost estimate assumed every MC rollout runs the full 500-step cap.
That assumption was never measured, and if it is wrong the cost estimate is wrong
by the same factor.  This script runs the real ``mc_return_from_action`` on the
real probe states and records how long each rollout actually is.

Output: ``results/mc_rollout_length.json``
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from auditability.diagnose_acrobot_q_action_agreement import collect_candidate_states
from auditability.run_shared_multienv_gpi import load_actor


def rollout_with_length(env, observation: np.ndarray, action: int, actors: list,
                        max_steps: int = 500) -> tuple[float, int]:
    """Same body as mc_return_from_action, but also returns the step count."""
    cos1, sin1, cos2, sin2, vel1, vel2 = map(float, observation)
    env.reset(seed=0)
    env.unwrapped.state = np.asarray([math.atan2(sin1, cos1), math.atan2(sin2, cos2), vel1, vel2],
                                     dtype=np.float64)
    current = np.asarray(env.unwrapped._get_ob(), dtype=np.float32)
    total = 0.0
    steps = 0
    for step in range(max_steps):
        use_action = action if step == 0 else None
        if use_action is None:
            with torch.inference_mode():
                logits = torch.stack([actor(torch.as_tensor(current, dtype=torch.float32))
                                      for actor in actors]).mean(0)
            use_action = int(logits.argmax().item())
        current, reward, terminated, truncated, _ = env.step(use_action)
        total += float(reward)
        steps += 1
        if terminated or truncated:
            break
    return total, steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, default=Path("runs/shared_multienv_gpi/Acrobot-v1"))
    parser.add_argument("--seeds", type=int, nargs=2, default=[11, 29])
    parser.add_argument("--actor-episodes", type=int, default=10)
    parser.add_argument("--sample-count", type=int, default=40)
    parser.add_argument("--reset-start", type=int, default=31_000_000)
    parser.add_argument("--output", type=Path, default=Path("results/mc_rollout_length.json"))
    args = parser.parse_args()

    task_root = args.task_root.resolve()
    calibration = json.loads((task_root / "shared_calibration.json").read_text(encoding="utf-8"))
    env_name = calibration["environment"]
    state_dim = len(calibration["state_bins"]["edges"])
    probe = gym.make(env_name)
    action_dim = int(probe.action_space.n)
    probe.close()

    runs = []
    for seed in args.seeds:
        matches = [p for p in (task_root / "grammar_audit_only").glob(f"seed-{seed}_*")
                   if (p / "metadata.json").is_file()]
        runs.append(matches[0])
    actors = [load_actor(path, state_dim, action_dim) for path in runs]

    states = collect_candidate_states(env_name, actors, args.seeds, args.actor_episodes,
                                      args.sample_count, args.reset_start)
    env = gym.make(env_name)
    rows = []
    for index, (_origin, observation) in enumerate(states):
        for action in range(action_dim):
            start = time.perf_counter()
            value, steps = rollout_with_length(env, observation, action, actors)
            rows.append({"state_index": index, "action": action, "steps": steps,
                         "return": value, "seconds": time.perf_counter() - start})
    env.close()

    lengths = [r["steps"] for r in rows]
    seconds = [r["seconds"] for r in rows]
    total_steps = sum(lengths)
    at_cap = sum(1 for s in lengths if s >= 500)

    result = {
        "environment": env_name,
        "rollouts": len(rows),
        "rollout_length": {
            "mean": statistics.mean(lengths),
            "median": statistics.median(lengths),
            "min": min(lengths),
            "max": max(lengths),
            "stdev": statistics.stdev(lengths),
            "fraction_at_500_step_cap": at_cap / len(lengths),
            "histogram": {str(b): sum(1 for s in lengths if lo <= s < lo + b)
                          for lo, b in [(0, 50), (50, 100), (100, 200), (200, 400), (400, 500), (500, 501)]},
        },
        "timing": {
            "seconds_per_rollout_mean": statistics.mean(seconds),
            "seconds_per_step_mean": sum(seconds) / total_steps,
        },
        "mean_continuation_return": statistics.mean([r["return"] for r in rows]),
        "corrects_earlier_claim": (
            "The A2 cost estimate assumed every rollout runs the full 500 steps. "
            "The measured fraction at the cap is reported above; the cost estimate must be "
            "scaled by rollout_length.mean / 500 if that fraction is below 1.0."
        ),
        "raw": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "raw"}, indent=2))


if __name__ == "__main__":
    main()
