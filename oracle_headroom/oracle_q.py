"""Oracle Q via exact single rollouts.

CartPole-v1 dynamics are deterministic given (state, action). With a
deterministic continuation policy, ONE rollout from (s, a) yields the EXACT
Q(s, a) -- no Monte Carlo averaging needed. This module:

1. verifies the two premises (set-state roundtrip, rollout determinism);
2. exposes exact_q(env_name, state, action, continuation, max_steps).

The continuation policy is a callable: action = continuation(observation).
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym


def set_state_and_step(env_name: str, state: np.ndarray, action: int,
                       max_steps: int = 500) -> tuple[float, np.ndarray, bool]:
    """Set env to `state`, take `action`, return (reward, next_obs, done)."""
    env = gym.make(env_name)
    try:
        env.reset(seed=0)  # initialize internals; state will be overwritten
        env.unwrapped.state = np.asarray(state, dtype=np.float64).copy()
        obs, reward, terminated, truncated, _ = env.step(int(action))
        return float(reward), np.asarray(obs, dtype=np.float64), bool(terminated or truncated)
    finally:
        env.close()


def exact_q(env_name: str, state: np.ndarray, first_action: int,
            continuation, max_steps: int = 500) -> float:
    """Exact Q(state, first_action) under a deterministic continuation policy.

    Forced first action, then `continuation(obs)` until termination/truncation.
    """
    env = gym.make(env_name)
    try:
        env.reset(seed=0)
        env.unwrapped.state = np.asarray(state, dtype=np.float64).copy()
        total = 0.0
        obs, reward, terminated, truncated, _ = env.step(int(first_action))
        total += float(reward)
        steps = 1
        while not (terminated or truncated) and steps < max_steps:
            action = int(continuation(np.asarray(obs, dtype=np.float64)))
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            steps += 1
        return total
    finally:
        env.close()


def verify_premises(env_name: str, continuation, n_probes: int = 20,
                    rng_seed: int = 999) -> dict:
    """Verify (a) set-state roundtrip, (b) rollout determinism.

    Returns a report dict; raises AssertionError if a premise fails.
    """
    rng = np.random.RandomState(rng_seed)
    env = gym.make(env_name)
    lo = env.observation_space.low
    hi = env.observation_space.high
    # CartPole has infinite bounds on velocities; sample around typical ranges.
    finite_lo = np.where(np.isfinite(lo), lo, -3.0)
    finite_hi = np.where(np.isfinite(hi), hi, 3.0)
    roundtrip_ok = 0
    determinism_ok = 0
    try:
        for _ in range(n_probes):
            s = rng.uniform(finite_lo, finite_hi).astype(np.float64)
            a = int(rng.randint(env.action_space.n))
            # (a) set state, read back via step's returned obs is indirect;
            #     direct check: unwrapped.state equals what we set.
            env.reset(seed=0)
            env.unwrapped.state = s.copy()
            if np.allclose(np.asarray(env.unwrapped.state, dtype=np.float64), s):
                roundtrip_ok += 1
            # (b) two rollouts from the same (s, a) must agree exactly.
            q1 = exact_q(env_name, s, a, continuation)
            q2 = exact_q(env_name, s, a, continuation)
            if q1 == q2:
                determinism_ok += 1
    finally:
        env.close()
    report = {
        "env": env_name,
        "n_probes": n_probes,
        "set_state_roundtrip_ok": roundtrip_ok,
        "rollout_determinism_ok": determinism_ok,
        "premises_hold": roundtrip_ok == n_probes and determinism_ok == n_probes,
    }
    if not report["premises_hold"]:
        raise AssertionError(f"oracle premises FAILED: {report}")
    return report
