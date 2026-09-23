"""Is the Monte-Carlo action-value estimator degenerate on the states it is asked about?

A2's mc_ranked mode collapsed to "always action 0" on its first episode, with
mean_mc_margin exactly 0.0 - i.e. the estimator returned the same value for all
three forced first actions at every one of 500 decision points.  Two readings are
possible and they have opposite implications:

  (a) a bug in the harness, or
  (b) a genuine property of the estimator: when the continuation policy never
      terminates the episode inside the horizon, every forced first action
      returns exactly -1 x horizon, so the estimator carries no information.

This script separates them by evaluating the same estimator on two state
populations: states drawn from actor rollouts (the design used by the original
diagnostic) and states drawn from an always-action-0 rollout (the population
mc_ranked actually visits).

Output: ``results/mc_degeneracy_probe.json``
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from auditability.diagnose_acrobot_q_action_agreement import (
    collect_candidate_states,
    mc_return_from_action,
)
from auditability.run_shared_multienv_gpi import load_actor


def always_zero_states(env_name: str, count: int, reset_start: int) -> list[np.ndarray]:
    env = gym.make(env_name)
    states: list[np.ndarray] = []
    for episode in range(count):
        observation, _ = env.reset(seed=reset_start + episode)
        done = False
        while not done:
            states.append(np.asarray(observation, dtype=np.float32).copy())
            observation, _, terminated, truncated, _ = env.step(0)
            done = terminated or truncated
    env.close()
    return states


def evaluate_states(env, states: list[np.ndarray], actors: list, action_dim: int,
                    horizon: int) -> list[dict]:
    rows = []
    for observation in states:
        values = [mc_return_from_action(env, observation, action, actors, max_steps=horizon)
                  for action in range(action_dim)]
        rows.append({"mc_values": values,
                     "spread": max(values) - min(values),
                     "argmax": int(np.argmax(values)),
                     "all_equal": len(set(round(v, 9) for v in values)) == 1,
                     "all_at_floor": all(abs(v + horizon) < 1e-9 for v in values)})
    return rows


def summarise(rows: list[dict], action_dim: int, horizon: int) -> dict:
    spreads = [r["spread"] for r in rows]
    return {
        "states": len(rows),
        "fraction_all_three_values_equal": sum(r["all_equal"] for r in rows) / len(rows),
        "fraction_all_values_at_floor": sum(r["all_at_floor"] for r in rows) / len(rows),
        "mean_spread": statistics.mean(spreads),
        "median_spread": statistics.median(spreads),
        "fraction_spread_zero": sum(1 for s in spreads if s == 0.0) / len(spreads),
        "argmax_histogram": {str(a): sum(r["argmax"] == a for r in rows)
                             for a in range(action_dim)},
        "mean_value_by_action": {str(a): statistics.mean([r["mc_values"][a] for r in rows])
                                 for a in range(action_dim)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, default=Path("runs/shared_multienv_gpi/Acrobot-v1"))
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[11, 29, 43, 71, 101, 149, 211, 307])
    parser.add_argument("--actor-episodes", type=int, default=2)
    parser.add_argument("--sample-count", type=int, default=80)
    parser.add_argument("--zero-episodes", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--output", type=Path, default=Path("results/mc_degeneracy_probe.json"))
    args = parser.parse_args()

    torch.set_num_threads(1)
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

    env = gym.make(env_name)
    actor_states = [s for _origin, s in collect_candidate_states(
        env_name, actors, args.seeds, args.actor_episodes, args.sample_count, 31_000_000)]
    zero_states = always_zero_states(env_name, args.zero_episodes, 32_000_000)

    actor_rows = evaluate_states(env, actor_states, actors, action_dim, args.horizon)
    zero_rows = evaluate_states(env, zero_states, actors, action_dim, args.horizon)
    env.close()

    result = {
        "environment": env_name,
        "horizon": args.horizon,
        "n_actors_in_continuation_ensemble": len(actors),
        "seeds": args.seeds,
        "actor_rollout_states": summarise(actor_rows, action_dim, args.horizon),
        "always_action_0_states": summarise(zero_rows, action_dim, args.horizon),
        "interpretation": (
            "If the always-action-0 population shows a high fraction of zero-spread states while "
            "the actor-rollout population does not, the collapse of mc_ranked is a property of the "
            "estimator on the visited distribution, not a harness bug."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
