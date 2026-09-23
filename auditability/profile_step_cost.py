"""Decompose the per-step cost of MC arbitration, to test whether a GPU can help.

A2's inner loop costs ~30 microseconds per environment step.  The two candidate
explanations are (a) the actor network, which is where a GPU would help, and
(b) the Python interpreter plus the gymnasium environment, which a GPU cannot
touch.  This script times each component separately, then measures the
throughput gain from batching the network forward pass, which is the only
thing a GPU accelerates.

Output: ``results/step_cost_decomposition.json``
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from auditability.run_shared_multienv_gpi import load_actor

STEPS = 500


def time_env_only(env_name: str, repeats: int) -> dict:
    """Step the environment with a constant action, no network at all.

    Mirrors the rollout structure of ``mc_return_from_action`` (reset, set the
    unwrapped state, step until done or the 500-step cap) but replaces the
    ensemble argmax with a fixed action, so the difference against the full
    routine isolates the network's contribution.
    """
    env = gym.make(env_name)
    initial = np.asarray([0.1, 0.1, 0.0, 0.0], dtype=np.float64)
    total_steps = 0
    start = time.perf_counter()
    for _ in range(repeats):
        env.reset(seed=0)
        env.unwrapped.state = initial.copy()
        for _ in range(STEPS):
            _, _, terminated, truncated, _ = env.step(0)
            total_steps += 1
            if terminated or truncated:
                break
    elapsed = time.perf_counter() - start
    env.close()
    return {"seconds_per_step": elapsed / total_steps, "steps": total_steps}


def time_full_rollout(env_name: str, actors: list, repeats: int) -> dict:
    """The real thing: forced first action, then the ensemble, 500-step cap."""
    env = gym.make(env_name)
    initial = np.asarray([0.1, 0.1, 0.0, 0.0], dtype=np.float64)
    total_steps = 0
    start = time.perf_counter()
    for _ in range(repeats):
        env.reset(seed=0)
        env.unwrapped.state = initial.copy()
        current = np.asarray(env.unwrapped._get_ob(), dtype=np.float32)
        for _ in range(STEPS):
            with torch.inference_mode():
                logits = torch.stack([actor(torch.as_tensor(current, dtype=torch.float32))
                                      for actor in actors]).mean(0)
            action = int(logits.argmax().item())
            current, _, terminated, truncated, _ = env.step(action)
            total_steps += 1
            if terminated or truncated:
                break
    elapsed = time.perf_counter() - start
    env.close()
    return {"seconds_per_step": elapsed / total_steps, "steps": total_steps}


def time_forward_only(actors: list, state_dim: int, repeats: int) -> float:
    state = torch.zeros(state_dim, dtype=torch.float32)
    with torch.inference_mode():
        for actor in actors:
            actor(state)
    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(repeats):
            for _ in range(STEPS):
                torch.stack([actor(state) for actor in actors]).mean(dim=0)
    elapsed = time.perf_counter() - start
    return elapsed / (repeats * STEPS)


def time_forward_batched(actors: list, state_dim: int, batch: int, repeats: int) -> float:
    states = torch.zeros(batch, state_dim, dtype=torch.float32)
    # warm up
    with torch.inference_mode():
        for actor in actors:
            actor(states)
    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(repeats):
            torch.stack([actor(states) for actor in actors]).mean(dim=0)
    elapsed = time.perf_counter() - start
    return elapsed / repeats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path,
                        default=Path("runs/shared_multienv_gpi/Acrobot-v1"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 29])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("results/step_cost_decomposition.json"))
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

    # Parameter and FLOP accounting for the actor, so the comparison is not hand-waved.
    params = sum(p.numel() for p in actors[0].parameters())
    flops = 2 * (state_dim * 64 + 64 * 64 + 64 * action_dim)

    env_only = time_env_only(env_name, args.repeats)
    full = time_full_rollout(env_name, actors, args.repeats)
    fwd_only = time_forward_only(actors, state_dim, args.repeats)
    batched = {b: time_forward_batched(actors, state_dim, b, args.repeats)
               for b in (1, 256, 4096, 118_806)}
    per_sample = {b: t / b for b, t in batched.items()}
    single_sample = per_sample[1]

    result = {
        "environment": env_name,
        "torch_version": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_threads": torch.get_num_threads(),
        "actor": {"architecture": f"{state_dim}-64-64-{action_dim}",
                  "parameters": params, "flops_per_forward": flops,
                  "actors_in_ensemble": len(actors)},
        "per_step_seconds": {
            "env_step_only_constant_action": env_only["seconds_per_step"],
            "network_only_isolated": fwd_only,
            "full_rollout_with_network": full["seconds_per_step"],
            "network_contribution_implied": full["seconds_per_step"] - env_only["seconds_per_step"],
        },
        "step_counts": {"env_only": env_only["steps"], "full_rollout": full["steps"]},
        "batched_forward_seconds_per_call": batched,
        "batched_forward_per_sample_seconds": per_sample,
        "per_sample_speedup_vs_batch_of_1": {str(b): single_sample / t for b, t in per_sample.items()},
        "interpretation": (
            "env_step_only_constant_action is pure gymnasium/NumPy with no network, so a GPU "
            "cannot reduce it. batched_forward_* shows how much of the network cost is "
            "kernel-launch and Python-loop latency rather than arithmetic: the per-sample "
            "forward cost collapses as the batch grows, which is the only lever a GPU pulls."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
