"""Counterfactual held-out ablations for Acrobot grammar-fusion decisions."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from auditability.grammar_audit_experiment import StateSymbolizer, canonical_json, sha256_file
from auditability.run_shared_multienv_gpi import actor_action, load_actor, load_rules


def load_shared_symbolizer(task_root: Path) -> StateSymbolizer:
    calibration = json.loads((task_root / "shared_calibration.json").read_text(encoding="utf-8"))
    symbolizer = StateSymbolizer(calibration["state_bins"]["n_bins"])
    symbolizer.edges = torch.tensor(calibration["state_bins"]["edges"], dtype=torch.float64)
    return symbolizer


def decide(observation, symbolizer, rule_banks, actors, fallback_actor, mode, rng=None):
    condition = symbolizer.encode(observation)
    candidates = [bank[condition] for bank in rule_banks if condition in bank]
    conflict = len({rule["action"] for rule in candidates}) > 1
    if not candidates:
        if mode == "best_single_fallback":
            return actor_action(fallback_actor, observation), False, False
        with torch.inference_mode():
            logits = torch.stack([actor(torch.as_tensor(observation, dtype=torch.float32)) for actor in actors]).mean(0)
        return int(logits.argmax().item()), False, False
    if mode == "defer_conflicts_to_ensemble" and conflict:
        with torch.inference_mode():
            logits = torch.stack([actor(torch.as_tensor(observation, dtype=torch.float32)) for actor in actors]).mean(0)
        return int(logits.argmax().item()), True, True
    if mode == "unanimous_only" and conflict:
        with torch.inference_mode():
            logits = torch.stack([actor(torch.as_tensor(observation, dtype=torch.float32)) for actor in actors]).mean(0)
        return int(logits.argmax().item()), True, True
    if mode == "support_first":
        key = lambda rule: (rule["support"], rule["confidence"], rule["mean_reward"], -rule["source_seed"])
        chosen = max(candidates, key=key)
    elif mode == "confidence_only":
        chosen = max(candidates, key=lambda rule: rule["confidence"])
    elif mode == "mean_reward_only":
        chosen = max(candidates, key=lambda rule: rule["mean_reward"])
    elif mode == "random_candidate":
        if rng is None:
            raise ValueError("random arbitration requires a seeded RNG")
        chosen = rng.choice(candidates)
    elif mode.startswith("source_seed_") and conflict:
        preferred = int(mode.rsplit("_", 1)[1])
        matching = [rule for rule in candidates if rule["source_seed"] == preferred]
        chosen = max(matching or candidates, key=lambda rule: (rule["confidence"], rule["support"], rule["mean_reward"]))
    else:
        chosen = max(candidates, key=lambda rule: (rule["confidence"], rule["support"], rule["mean_reward"], -rule["source_seed"]))
    return int(chosen["action"]), True, conflict


def run(task_root: Path, seeds: list[int], episodes: int, reset_base: int) -> dict:
    summary = json.loads((task_root / "shared_multiseed_gpi_summary.json").read_text(encoding="utf-8"))
    calibration = json.loads((task_root / "shared_calibration.json").read_text(encoding="utf-8"))
    environment = calibration["environment"]
    state_dim = len(calibration["state_bins"]["edges"])
    env = gym.make(environment)
    action_dim = int(env.action_space.n)
    env.close()
    agent_runs = []
    for seed in seeds:
        matches = [p for p in (task_root / "grammar_audit_only").glob(f"seed-{seed}_*") if (p / "metadata.json").is_file()]
        if len(matches) != 1:
            raise ValueError(f"expected one complete audit-only run for seed {seed}; got {len(matches)}")
        agent_runs.append(matches[0])
    actors = [load_actor(path, state_dim, action_dim) for path in agent_runs]
    rule_banks = [load_rules(path) for path in agent_runs]
    fallback_seed = int(summary["composition_evaluation"]["fallback_best_single_seed"])
    fallback_actor = actors[seeds.index(fallback_seed)]
    symbolizer = load_shared_symbolizer(task_root)
    modes = ["confidence_first", "confidence_only", "mean_reward_only", "random_candidate", "support_first", "defer_conflicts_to_ensemble", "unanimous_only", f"source_seed_{seeds[0]}", f"source_seed_{seeds[1]}", "best_single_fallback"]
    returns = {mode: [] for mode in modes}
    counts = {mode: {"steps": 0, "covered": 0, "conflicts": 0, "selected_action_differs_from_conflict_ensemble": 0} for mode in modes}
    for episode in range(episodes):
        reset_seed = reset_base + episode
        for mode in modes:
            policy_rng = random.Random(20260922 + episode)
            env = gym.make(environment)
            observation, _ = env.reset(seed=reset_seed)
            total = 0.0
            done = False
            while not done:
                action, covered, conflict = decide(observation, symbolizer, rule_banks, actors, fallback_actor, mode, policy_rng)
                counts[mode]["steps"] += 1
                counts[mode]["covered"] += int(covered)
                counts[mode]["conflicts"] += int(conflict)
                if covered and conflict and mode not in {"defer_conflicts_to_ensemble", "unanimous_only"}:
                    ensemble_action = decide(observation, symbolizer, rule_banks, actors, fallback_actor, "defer_conflicts_to_ensemble", policy_rng)[0]
                    counts[mode]["selected_action_differs_from_conflict_ensemble"] += int(action != ensemble_action)
                observation, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                done = terminated or truncated
            env.close()
            returns[mode].append(total)
    baseline = returns["confidence_first"]
    output = {}
    for mode in modes:
        deltas = [a - b for a, b in zip(returns[mode], baseline)]
        boot_rng = random.Random(70_202_609 + modes.index(mode))
        boots = sorted(statistics.mean(deltas[boot_rng.randrange(len(deltas))] for _ in deltas) for _ in range(5000))
        output[mode] = {
            "mean_return": statistics.mean(returns[mode]),
            "median_return": statistics.median(returns[mode]),
            "std_return": statistics.stdev(returns[mode]) if len(returns[mode]) > 1 else 0.0,
            "paired_mean_delta_vs_current": statistics.mean(deltas),
            "paired_mean_delta_bootstrap_ci95": [boots[124], boots[4874]],
            "return_by_episode": returns[mode],
            "decision_counts": counts[mode],
        }
    main_result = summary["composition_evaluation"]["policies"]["grammar_fusion"]
    random_repetitions = []
    for replica in range(5):
        replica_returns = []
        replica_counts = {"steps": 0, "covered": 0, "conflicts": 0}
        for episode in range(episodes):
            policy_rng = random.Random(70_000_000 + replica * 100_000 + episode)
            env = gym.make(environment)
            observation, _ = env.reset(seed=reset_base + episode)
            total = 0.0
            done = False
            while not done:
                action, covered, conflict = decide(observation, symbolizer, rule_banks, actors, fallback_actor, "random_candidate", policy_rng)
                replica_counts["steps"] += 1
                replica_counts["covered"] += int(covered)
                replica_counts["conflicts"] += int(conflict)
                observation, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                done = terminated or truncated
            env.close()
            replica_returns.append(total)
        deltas = [value - reference for value, reference in zip(replica_returns, baseline)]
        boot_rng = random.Random(71_000_000 + replica)
        boots = sorted(statistics.mean(deltas[boot_rng.randrange(len(deltas))] for _ in deltas) for _ in range(5000))
        random_repetitions.append({
            "rng_replica": replica,
            "mean_return": statistics.mean(replica_returns),
            "paired_mean_delta_vs_current": statistics.mean(deltas),
            "paired_mean_delta_bootstrap_ci95": [boots[124], boots[4874]],
            "decision_counts": replica_counts,
        })
    return {
        "environment": environment,
        "seeds": seeds,
        "episodes": episodes,
        "reset_seed_formula": f"{reset_base} + episode_index, shared by every ablation",
        "fallback_seed_for_blind_spots": fallback_seed,
        "primary_reported_grammar_fusion": {"mean_return": main_result["mean_return"], "blind_spots": summary["composition_evaluation"]["grammar_fusion_decision_mix"]["grammar_fusion_blind_spots"], "total_steps": summary["composition_evaluation"]["grammar_fusion_decision_mix"]["grammar_fusion_total_steps"]},
        "ablations": output,
        "random_arbitration_replicates": random_repetitions,
        "random_arbitration_rng_design": "five independent Python RNG streams per the same 100 environment reset seeds; each random draw uniformly selects among rules available at that state",
        "limitations": "These are policy-level matched-seed ablations, not single-step causal effects. Sequential state-distribution changes mean deltas cannot be attributed by simply multiplying uncovered/conflict fractions by reward.",
        "shared_symbolizer_sha256": calibration["state_bins_canonical_sha256"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 29, 43, 71, 101, 149, 211, 307])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--reset-base", type=int, default=20_000_000)
    args = parser.parse_args()
    result = run(args.task_root.resolve(), args.seeds, args.episodes, args.reset_base)
    path = args.task_root.resolve() / "grammar_fusion_diagnostics.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(path), "ablations": {key: {"mean_return": row["mean_return"], "paired_mean_delta_vs_current": row["paired_mean_delta_vs_current"], "decision_counts": row["decision_counts"]} for key, row in result["ablations"].items()}}, indent=2))


if __name__ == "__main__":
    main()
