"""Compare Acrobot fusion choices, source-critic greedy actions, and MC action returns."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from auditability.grammar_audit_experiment import StateSymbolizer
from auditability.run_shared_multienv_gpi import QRegressor, grammar_fusion_decision, load_actor, load_rules


def bootstrap_mean_ci(values: list[float], seed: int) -> list[float]:
    rng = random.Random(seed)
    draws = sorted(statistics.mean(values[rng.randrange(len(values))] for _ in values) for _ in range(5000))
    return [draws[124], draws[4874]]


def load_symbolizer(task_root: Path):
    calibration = json.loads((task_root / "shared_calibration.json").read_text(encoding="utf-8"))
    symbolizer = StateSymbolizer(calibration["state_bins"]["n_bins"])
    symbolizer.edges = torch.tensor(calibration["state_bins"]["edges"], dtype=torch.float64)
    return symbolizer


def collect_candidate_states(env_name: str, actors: list, source_seeds: list[int], actor_episodes: int,
                             sample_count: int, reset_start: int):
    env = gym.make(env_name)
    rows = []
    reset_seed = reset_start
    for source_seed, actor in zip(source_seeds, actors):
        for _ in range(actor_episodes):
            state, _ = env.reset(seed=reset_seed)
            episode_states = []
            done = False
            while not done:
                episode_states.append(np.asarray(state, dtype=np.float32).copy())
                with torch.inference_mode():
                    action = int(actor(torch.as_tensor(state, dtype=torch.float32)).argmax().item())
                state, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            rows.extend((source_seed, row) for row in episode_states)
            reset_seed += 1
    env.close()
    groups = {seed: [row for origin, row in rows if origin == seed] for seed in source_seeds}
    rng = random.Random(20260922)
    samples = []
    per_origin = max(1, sample_count // len(source_seeds))
    for seed in source_seeds:
        candidates = groups[seed]
        selected = rng.sample(candidates, min(per_origin, len(candidates)))
        samples.extend((seed, state) for state in selected)
    return samples


def mc_return_from_action(env, observation: np.ndarray, action: int, actors: list, max_steps: int = 500) -> float:
    cos1, sin1, cos2, sin2, vel1, vel2 = map(float, observation)
    env.reset(seed=0)
    env.unwrapped.state = np.asarray([math.atan2(sin1, cos1), math.atan2(sin2, cos2), vel1, vel2], dtype=np.float64)
    current = np.asarray(env.unwrapped._get_ob(), dtype=np.float32)
    total = 0.0
    done = False
    for step in range(max_steps):
        use_action = action if step == 0 else None
        if use_action is None:
            with torch.inference_mode():
                logits = torch.stack([actor(torch.as_tensor(current, dtype=torch.float32)) for actor in actors]).mean(0)
            use_action = int(logits.argmax().item())
        current, reward, terminated, truncated, _ = env.step(use_action)
        total += float(reward)
        done = terminated or truncated
        if done:
            break
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--source-seeds", type=int, nargs=2, default=[11, 29])
    parser.add_argument("--actor-episodes", type=int, default=30)
    parser.add_argument("--sample-count", type=int, default=80)
    parser.add_argument("--reset-start", type=int, default=31_000_000)
    args = parser.parse_args()
    task_root = args.task_root.resolve()
    calibration = json.loads((task_root / "shared_calibration.json").read_text(encoding="utf-8"))
    env_name = calibration["environment"]
    state_dim = len(calibration["state_bins"]["edges"])
    env_probe = gym.make(env_name)
    action_dim = int(env_probe.action_space.n)
    env_probe.close()
    actor_runs = []
    for seed in args.source_seeds:
        matches = [p for p in (task_root / "grammar_audit_only").glob(f"seed-{seed}_*") if (p / "metadata.json").is_file()]
        if len(matches) != 1:
            raise ValueError(f"expected one complete source run for seed {seed}")
        actor_runs.append(matches[0])
    actors = [load_actor(path, state_dim, action_dim) for path in actor_runs]
    rules = [load_rules(path) for path in actor_runs]
    critics = []
    for seed in args.source_seeds:
        model = QRegressor(state_dim, action_dim)
        model.load_state_dict(torch.load(task_root / "gpi_fqe" / f"fqe_q_seed_{seed}.pt", map_location="cpu", weights_only=True))
        model.eval()
        critics.append(model)
    states = collect_candidate_states(env_name, actors, args.source_seeds, args.actor_episodes,
                                      args.sample_count, args.reset_start)
    symbolizer = load_symbolizer(task_root)
    env = gym.make(env_name)
    detailed = []
    q_values_all = []
    for index, (origin_seed, observation) in enumerate(states):
        fusion_action, fusion_detail = grammar_fusion_decision(observation, symbolizer, rules, actors)
        state_t = torch.as_tensor(observation, dtype=torch.float32)
        with torch.inference_mode():
            per_source_q = [model(state_t).numpy().tolist() for model in critics]
        mc_values = [mc_return_from_action(env, observation, action, actors) for action in range(action_dim)]
        record = {
            "sample_index": index,
            "state_origin_actor_seed": origin_seed,
            "state": observation.tolist(),
            "fusion_action": fusion_action,
            "fusion_covered": fusion_detail["covered"],
            "fusion_conflict": fusion_detail["conflict"],
            "source_q_values_by_action": {str(seed): values for seed, values in zip(args.source_seeds, per_source_q)},
            "source_q_argmax": {str(seed): int(np.argmax(values)) for seed, values in zip(args.source_seeds, per_source_q)},
            "source_q_maximizing_action_values": {str(seed): float(max(values)) for seed, values in zip(args.source_seeds, per_source_q)},
            "ensemble_gpi_action": int(np.max(np.asarray(per_source_q), axis=0).argmax()),
            "monte_carlo_continuation_return_by_first_action": mc_values,
            "monte_carlo_best_first_action": int(np.argmax(mc_values)),
        }
        detailed.append(record)
        q_values_all.append(per_source_q)
    env.close()
    n = len(detailed)
    agreement_11 = [float(r["fusion_action"] == r["source_q_argmax"][str(args.source_seeds[0])]) for r in detailed]
    agreement_29 = [float(r["fusion_action"] == r["source_q_argmax"][str(args.source_seeds[1])]) for r in detailed]
    agreement_gpi = [float(r["fusion_action"] == r["ensemble_gpi_action"]) for r in detailed]
    agreement_mc = [float(r["fusion_action"] == r["monte_carlo_best_first_action"]) for r in detailed]
    mc_regret = [max(r["monte_carlo_continuation_return_by_first_action"]) -
                 r["monte_carlo_continuation_return_by_first_action"][r["fusion_action"]] for r in detailed]
    q_action_histograms = {}
    for seed in args.source_seeds:
        q_action_histograms[str(seed)] = {
            str(action): int(sum(r["source_q_argmax"][str(seed)] == action for r in detailed))
            for action in range(action_dim)
        }
    mc_values = np.asarray([r["monte_carlo_continuation_return_by_first_action"] for r in detailed], dtype=np.float64)
    metrics = {
        "heldout_sample_count": n,
        "fusion_rule_coverage": float(np.mean([r["fusion_covered"] for r in detailed])),
        "fusion_conflict_rate": float(np.mean([r["fusion_conflict"] for r in detailed])),
        "fusion_action_agreement_with_each_source_q_argmax": {
            str(seed): {
                "rate": float(np.mean(values)),
                "bootstrap_ci95": bootstrap_mean_ci(values, 20260922 + seed),
            }
            for seed, values in zip(args.source_seeds, (agreement_11, agreement_29))
        },
        "fusion_action_agreement_with_any_source_q_argmax": float(np.mean([
            any(r["fusion_action"] == r["source_q_argmax"][str(seed)] for seed in args.source_seeds)
            for r in detailed
        ])),
        "fusion_action_histogram": {str(action): int(sum(r["fusion_action"] == action for r in detailed)) for action in range(action_dim)},
        "source_q_greedy_action_histograms": q_action_histograms,
        "fusion_action_agreement_with_ensemble_gpi_argmax": {
            "rate": float(np.mean(agreement_gpi)), "bootstrap_ci95": bootstrap_mean_ci(agreement_gpi, 20260930),
        },
        "fusion_action_agreement_with_mc_best_first_action": {
            "rate": float(np.mean(agreement_mc)), "bootstrap_ci95": bootstrap_mean_ci(agreement_mc, 20260931),
        },
        "source_q_distribution_pearson_seed_11_vs_29": float(np.corrcoef(
            np.asarray(q_values_all)[:, 0, :].ravel(), np.asarray(q_values_all)[:, 1, :].ravel()
        )[0, 1]),
        "q_action_distribution": {
            str(seed): {
                "mean": float(np.asarray(q_values_all)[:, i, :].mean()),
                "std": float(np.asarray(q_values_all)[:, i, :].std(ddof=1)),
                "p10": float(np.quantile(np.asarray(q_values_all)[:, i, :], 0.1)),
                "median": float(np.quantile(np.asarray(q_values_all)[:, i, :], 0.5)),
                "p90": float(np.quantile(np.asarray(q_values_all)[:, i, :], 0.9)),
            }
            for i, seed in enumerate(args.source_seeds)
        },
        "mc_semantics": "Counterfactual return after forcing the candidate first action, then following the deterministic mean-logit ensemble; finite-horizon environment rollout, not optimal Q* truth.",
        "mc_best_action_margin": {"mean": float(np.mean(mc_regret)), "bootstrap_ci95": bootstrap_mean_ci(mc_regret, 20260932)},
        "monte_carlo_mean_continuation_return_by_first_action": {str(action): float(mc_values[:, action].mean()) for action in range(action_dim)},
        "monte_carlo_best_first_action_histogram": {str(action): int(sum(r["monte_carlo_best_first_action"] == action for r in detailed)) for action in range(action_dim)},
    }
    out = {
        "environment": env_name,
        "source_seeds": args.source_seeds,
        "sample_design": {"actor_episodes_per_seed": args.actor_episodes, "requested_state_count": args.sample_count,
                          "state_source": "new deterministic actor rollouts under non-overlapping reset seeds"},
        "metrics": metrics,
        "heldout_state_details": detailed,
        "limitations": "The MC action target evaluates a forced first action followed by the actor-logit ensemble, not the optimal action value; state count is a sampled diagnostic, and Acrobot deterministic dynamics make one continuation per action reproducible but not a general stochastic expectation estimate.",
    }
    output = task_root / "gpi_fqe" / "acrobot_q_action_agreement.json"
    output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
