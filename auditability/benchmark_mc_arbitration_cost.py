"""Measure the wall-clock cost of one Monte-Carlo arbitration decision.

A2 (the rankability test) proposes replacing confidence arbitration with an
MC-based ranker.  The cost of that proposal is entirely determined by two
quantities:

  1. ``cost_per_rollout``  - seconds to evaluate one candidate first action,
     i.e. one forced-action rollout followed by the mean-logit ensemble.
  2. ``N_decisions``       - how many decision points the ranker must be
     queried at when re-running the matched episodes.

This script measures (1) directly, on the real Acrobot task root, using the
same actor checkpoints and the same rollout routine as
``diagnose_acrobot_q_action_agreement.py``.  (2) is read from the recorded
fusion diagnostics so that the extrapolation is not itself an assumption.

Output: ``results/mc_arbitration_cost_benchmark.json``
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from auditability.diagnose_acrobot_q_action_agreement import (
    collect_candidate_states,
    load_symbolizer,
    mc_return_from_action,
)
from auditability.run_shared_multienv_gpi import load_actor


def measure(args) -> dict:
    task_root = args.task_root.resolve()
    calibration = json.loads((task_root / "shared_calibration.json").read_text(encoding="utf-8"))
    env_name = calibration["environment"]
    state_dim = len(calibration["state_bins"]["edges"])
    probe = gym.make(env_name)
    action_dim = int(probe.action_space.n)
    probe.close()

    actor_runs = []
    for seed in args.source_seeds:
        matches = [p for p in (task_root / "grammar_audit_only").glob(f"seed-{seed}_*")
                   if (p / "metadata.json").is_file()]
        if len(matches) != 1:
            raise ValueError(f"expected one complete source run for seed {seed}")
        actor_runs.append(matches[0])
    actors = [load_actor(path, state_dim, action_dim) for path in actor_runs]

    # Real states, drawn exactly the way the existing probe draws them.
    states = collect_candidate_states(env_name, actors, args.source_seeds,
                                      args.actor_episodes, args.sample_count, args.reset_start)
    symbolizer = load_symbolizer(task_root)  # kept so the harness matches the real caller
    del symbolizer

    env = gym.make(env_name)
    rollouts: list[dict] = []
    for index, (_origin, observation) in enumerate(states[: args.timed_states]):
        for action in range(action_dim):
            start = time.perf_counter()
            value = mc_return_from_action(env, observation, action, actors)
            elapsed = time.perf_counter() - start
            rollouts.append({"state_index": index, "action": action,
                             "seconds": elapsed, "return": value})
    env.close()

    per_rollout = [r["seconds"] for r in rollouts]
    mean_rollout = float(np.mean(per_rollout))
    median_rollout = float(np.median(per_rollout))

    diagnostics = json.loads((task_root / "grammar_fusion_diagnostics.json").read_text(encoding="utf-8"))
    primary = diagnostics["primary_reported_grammar_fusion"]
    counts = diagnostics["ablations"]["confidence_first"]["decision_counts"]
    total_steps = int(primary["total_steps"])
    covered = int(counts["covered"])
    conflicts = int(counts["conflicts"])
    # A successful ranker terminates episodes earlier, so the decision count is
    # not fixed: the recorded random-arbitration run only issued 18,802 steps.
    steps_if_ranker_succeeds = int(diagnostics["ablations"]["random_candidate"]["decision_counts"]["steps"])

    def decision_cost(n_decisions: int, horizon: int | None = None) -> dict:
        # A rollout that terminates early is cheaper; the measured mean already
        # includes early termination, so scaling by horizon is an upper bound.
        scale = 1.0 if horizon is None else horizon / args.horizon_reference
        seconds = n_decisions * action_dim * mean_rollout * scale
        return {"n_decisions": n_decisions, "horizon_scale": scale,
                "rollouts": n_decisions * action_dim,
                "seconds": seconds, "hours": seconds / 3600.0,
                "hours_8_way_parallel": seconds / 3600.0 / args.parallel}

    return {
        "environment": env_name,
        "task_root": str(task_root),
        "source_seeds": args.source_seeds,
        "timed_states": args.timed_states,
        "timed_rollouts": len(rollouts),
        "cost_per_rollout_seconds": {
            "mean": mean_rollout,
            "median": median_rollout,
            "p90": float(np.quantile(per_rollout, 0.9)),
            "min": float(np.min(per_rollout)),
            "max": float(np.max(per_rollout)),
        },
        "recorded_decision_points": {
            "total_steps": total_steps,
            "covered_steps": covered,
            "conflict_steps": conflicts,
            "blind_spot_steps": total_steps - covered,
            "steps_if_ranker_succeeds_like_random": steps_if_ranker_succeeds,
        },
        "extrapolation": {
            "full_trace_conflict_only": decision_cost(conflicts),
            "full_trace_conflict_only_horizon_100": decision_cost(conflicts, horizon=100),
            "full_trace_conflict_only_horizon_50": decision_cost(conflicts, horizon=50),
            "full_trace_all_covered": decision_cost(covered),
            "full_trace_all_covered_horizon_100": decision_cost(covered, horizon=100),
            "diagnostic_subsample_2000": decision_cost(2000),
            "ranker_succeeds_conflict_scaled": decision_cost(
                int(round(conflicts * steps_if_ranker_succeeds / total_steps))),
        },
        "parallel_factor_assumed": args.parallel,
        "note": ("Seconds are wall-clock on this machine, single process. The horizon-scaled "
                 "rows assume rollout cost is linear in horizon, which over-estimates because "
                 "early termination is already priced into the measured mean."),
        "raw_rollouts": rollouts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--source-seeds", type=int, nargs=2, default=[11, 29])
    parser.add_argument("--actor-episodes", type=int, default=10)
    parser.add_argument("--sample-count", type=int, default=40)
    parser.add_argument("--timed-states", type=int, default=40)
    parser.add_argument("--reset-start", type=int, default=31_000_000)
    parser.add_argument("--horizon-reference", type=int, default=500)
    parser.add_argument("--parallel", type=float, default=8.0)
    parser.add_argument("--output", type=Path,
                        default=Path("results/mc_arbitration_cost_benchmark.json"))
    args = parser.parse_args()
    result = measure(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = {k: v for k, v in result.items() if k != "raw_rollouts"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
