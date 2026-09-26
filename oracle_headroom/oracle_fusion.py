"""Oracle fusion policies + common-reset evaluation harness.

Policies evaluated on identical held-out episodes (reset seeds 20_000_000 + i):
  best_single      - argmax of the best-training-mean actor (repo convention)
  actor_mean_logits - 8-actor mean-logit ensemble argmax
  grammar_fusion   - repo's confidence-first grammar_fusion_decision
  oracle_conflict  - at conflict states: argmax over CANDIDATE actions by exact
                     rollout Q; everywhere else identical to grammar_fusion
  oracle_full      - at every covered state: argmax over the FULL action space by
                     exact rollout Q; blind spots use the ensemble fallback

Pre-registered continuation policy for exact Q: argmax of the best-training-mean
actor. Continuation sanity gate (> 400 mean on eval seeds) is checked before the
oracle policies run; if it fails the oracle measurement is declared VOID.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SNAPSHOT = Path.home() / "workspace" / "RL" / "RL_snapshot"
sys.path.insert(0, str(SNAPSHOT))
sys.path.insert(0, str(HERE))

from auditability.grammar_audit_experiment import StateSymbolizer  # noqa: E402
from auditability.run_shared_multienv_gpi import (  # noqa: E402
    actor_action,
    grammar_fusion_decision,
    load_actor,
    load_rules,
)
from oracle_q import exact_q, verify_premises  # noqa: E402

ENV = "CartPole-v1"
SEEDS = [11, 29, 43, 71, 101, 149, 211, 307]
EPISODES = 100
HELDOUT_BASE = 20_000_000
CONTINUATION_GATE = 400.0
LOG = HERE / "logs" / "oracle_fusion.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def load_artifacts():
    task_root = HERE / "runs" / ENV
    calib = json.loads((task_root / "shared_calibration.json").read_text(encoding="utf-8"))
    sym = StateSymbolizer(n_bins=calib["state_bins"]["n_bins"])
    sym.edges = torch.as_tensor(calib["state_bins"]["edges"], dtype=torch.float64)
    run_dirs = []
    for seed in SEEDS:
        matches = sorted((task_root / "grammar_audit_only").glob(f"seed-{seed}_*"))
        assert len(matches) == 1, f"expected one run dir for seed {seed}, found {matches}"
        run_dirs.append(matches[0])
    env_probe = gym.make(ENV)
    state_dim = int(np.prod(env_probe.observation_space.shape))
    action_dim = int(env_probe.action_space.n)
    env_probe.close()
    actors = [load_actor(d, state_dim, action_dim) for d in run_dirs]
    banks = [load_rules(d) for d in run_dirs]
    train_means = [json.loads((d / "metadata.json").read_text(encoding="utf-8"))["episode_return_mean"]
                   for d in run_dirs]
    best_index = max(range(len(SEEDS)), key=lambda i: (train_means[i], -SEEDS[i]))
    return sym, actors, banks, action_dim, best_index, train_means


def evaluate_policy(policy_fn, episodes: int) -> dict:
    returns, counts = [], {"steps": 0, "covered": 0, "conflicts": 0, "blind": 0}
    spreads = []
    for ep in range(episodes):
        env = gym.make(ENV)
        obs, _ = env.reset(seed=HELDOUT_BASE + ep)
        total, done = 0.0, False
        while not done:
            action, detail = policy_fn(np.asarray(obs, dtype=np.float64))
            counts["steps"] += 1
            counts["covered"] += int(detail.get("covered", False))
            counts["conflicts"] += int(detail.get("conflict", False))
            counts["blind"] += int(not detail.get("covered", True))
            if "q_spread" in detail:
                spreads.append(detail["q_spread"])
            obs, reward, terminated, truncated, _ = env.step(int(action))
            total += float(reward)
            done = terminated or truncated
        env.close()
        returns.append(total)
    return {
        "mean_return": statistics.mean(returns),
        "median_return": statistics.median(returns),
        "std_return": statistics.stdev(returns) if len(returns) > 1 else 0.0,
        "return_by_episode": returns,
        "decision_counts": counts,
        "mean_oracle_q_spread": statistics.mean(spreads) if spreads else None,
        "oracle_calls": len(spreads),
    }


def main() -> None:
    t_all = time.perf_counter()
    log("loading artifacts")
    sym, actors, banks, action_dim, best_index, train_means = load_artifacts()
    log(f"best_single seed = {SEEDS[best_index]} (train_mean={train_means[best_index]:.1f})")
    best_actor = actors[best_index]

    def continuation(obs: np.ndarray) -> int:
        return actor_action(best_actor, obs)

    # --- premise verification + continuation sanity gate (pre-registered) ---
    log("verifying oracle premises (set-state roundtrip, rollout determinism)")
    report = verify_premises(ENV, continuation)
    log(f"premises hold: {report['premises_hold']}")

    log("continuation sanity gate: best-actor argmax on eval seeds")
    gate_returns = evaluate_policy(
        lambda obs: (continuation(obs), {"covered": True}), EPISODES)["return_by_episode"]
    gate_mean = statistics.mean(gate_returns)
    log(f"continuation mean return = {gate_mean:.2f} (gate: > {CONTINUATION_GATE})")
    if gate_mean <= CONTINUATION_GATE:
        verdict = {"oracle_measurement": "VOID",
                   "reason": "degenerate continuation policy",
                   "continuation_mean": gate_mean}
        out = HERE / "results" / "oracle_headroom_cartpole.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
        log("VOID: continuation failed sanity gate; stopping.")
        return

    q_cache: dict[tuple[bytes, int], float] = {}

    def q_cached(state: np.ndarray, action: int) -> float:
        key = (np.ascontiguousarray(state, dtype=np.float64).tobytes(), int(action))
        if key not in q_cache:
            q_cache[key] = exact_q(ENV, state, int(action), continuation)
        return q_cache[key]

    def ensemble_action(obs: np.ndarray) -> int:
        state = torch.as_tensor(obs, dtype=torch.float32)
        with torch.inference_mode():
            return int(torch.stack([a(state) for a in actors]).mean(0).argmax().item())

    def oracle_pick(state: np.ndarray, actions: list[int]) -> tuple[int, float]:
        qs = {a: q_cached(state, a) for a in actions}
        vals = sorted(qs.values(), reverse=True)
        best = max(actions, key=lambda a: (qs[a], -a))
        return best, (vals[0] - vals[-1] if len(vals) > 1 else 0.0)

    def make_policy(mode: str):
        def policy(obs: np.ndarray):
            if mode in ("best_single", "actor_mean_logits"):
                if mode == "best_single":
                    return continuation(obs), {"covered": True}
                return ensemble_action(obs), {"covered": True}
            # fusion-family: share the repo's decision structure
            condition = sym.encode(obs)
            candidates = [b[condition] for b in banks if condition in b]
            if not candidates:
                return ensemble_action(obs), {"covered": False, "conflict": False,
                                              "candidate_count": 0}
            cand_actions = sorted({r["action"] for r in candidates})
            is_conflict = len(cand_actions) > 1
            detail = {"covered": True, "conflict": is_conflict,
                      "candidate_count": len(candidates), "candidate_actions": cand_actions}
            if mode == "grammar_fusion":
                return grammar_fusion_decision(obs, sym, banks, actors)
            if mode == "oracle_conflict":
                if not is_conflict:
                    return grammar_fusion_decision(obs, sym, banks, actors)
                best, spread = oracle_pick(obs, cand_actions)
                detail["q_spread"] = spread
                return best, detail
            if mode == "oracle_full":
                best, spread = oracle_pick(obs, list(range(action_dim)))
                detail["q_spread"] = spread
                detail["oracle_over_full_space"] = True
                return best, detail
            raise ValueError(mode)
        return policy

    results: dict = {
        "environment": ENV,
        "seeds": SEEDS,
        "episodes": EPISODES,
        "reset_seed_formula": "20000000 + episode_index; identical for every policy",
        "continuation": {"best_seed": SEEDS[best_index],
                        "sanity_gate_mean": gate_mean,
                        "sanity_gate_threshold": CONTINUATION_GATE,
                        "premises": report},
        "policies": {},
    }
    for mode in ["best_single", "actor_mean_logits", "grammar_fusion",
                 "oracle_conflict", "oracle_full"]:
        t0 = time.perf_counter()
        log(f"evaluating {mode} ...")
        res = evaluate_policy(make_policy(mode), EPISODES)
        log(f"{mode}: mean={res['mean_return']:.2f} median={res['median_return']:.2f} "
            f"in {time.perf_counter() - t0:.1f}s; counts={res['decision_counts']}")
        results["policies"][mode] = res
    results["q_cache_size"] = len(q_cache)
    out = HERE / "results" / "oracle_headroom_cartpole.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    log(f"wrote {out} in {time.perf_counter() - t_all:.1f}s total")


if __name__ == "__main__":
    main()
