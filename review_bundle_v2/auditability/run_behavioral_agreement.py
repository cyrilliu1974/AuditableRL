"""Experiment B: raw-state held-out behavioral agreement for two CartPole actors."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from auditability.run_shared_multienv_gpi import load_actor


def collect_heldout_episode_states(env_name: str, actor, source_seed: int, episodes: int, reset_base: int):
    env = gym.make(env_name)
    records = []
    for episode in range(episodes):
        reset_seed = reset_base + episode
        observation, _ = env.reset(seed=reset_seed)
        states = []
        total = 0.0
        done = False
        while not done:
            states.append(np.asarray(observation, dtype=np.float32).copy())
            with torch.inference_mode():
                action = int(actor(torch.as_tensor(observation, dtype=torch.float32)).argmax().item())
            observation, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            done = terminated or truncated
        records.append({"source_seed": source_seed, "episode": episode, "reset_seed": reset_seed,
                        "episode_return": total, "states": states})
    env.close()
    return records


def bootstrap_episode_cluster_ci(records: list[dict], seed: int = 20260922, replicates: int = 5000):
    episode_rates = []
    for record in records:
        matched = record["agreement"]
        if matched:
            episode_rates.append(sum(matched) / len(matched))
    rng = random.Random(seed)
    samples = sorted(statistics.mean(episode_rates[rng.randrange(len(episode_rates))] for _ in episode_rates)
                     for _ in range(replicates))
    return [samples[int(0.025 * replicates)], samples[int(0.975 * replicates) - 1]]


def summarize(records: list[dict], state_count: int, seed: int, actors: list) -> dict:
    rng = np.random.default_rng(seed)
    pools = {}
    for source_seed in (11, 29):
        pool = [record for record in records if record["source_seed"] == source_seed]
        n_available = sum(len(record["states"]) for record in pool)
        if n_available < state_count:
            raise ValueError(f"seed-{source_seed} held-out pool has {n_available} states, fewer than requested {state_count}")
        flattened = [(record, step, state) for record in pool for step, state in enumerate(record["states"])]
        selected_indexes = rng.choice(len(flattened), size=state_count, replace=False)
        selected = []
        for index in selected_indexes:
            record, step, state = flattened[int(index)]
            selected.append({"source_seed": source_seed, "episode": record["episode"], "step": step,
                             "reset_seed": record["reset_seed"], "state": state})
        pools[source_seed] = selected

    combined = pools[11] + pools[29]
    all_states = torch.as_tensor(np.asarray([row["state"] for row in combined]), dtype=torch.float32)
    actor_11, actor_29 = actors
    with torch.inference_mode():
        actions_11 = actor_11(all_states).argmax(dim=1).numpy()
        actions_29 = actor_29(all_states).argmax(dim=1).numpy()
    agree = actions_11 == actions_29
    for index, row in enumerate(combined):
        row["action_11"] = int(actions_11[index])
        row["action_29"] = int(actions_29[index])
        row["agree"] = bool(agree[index])
    source_results = {}
    episode_rate_pools = {}
    for source_seed in (11, 29):
        rows = [row for row in combined if row["source_seed"] == source_seed]
        episode_groups = {}
        for row in rows:
            episode_groups.setdefault(row["episode"], []).append(row["agree"])
        episode_records = [{"agreement": values} for _, values in sorted(episode_groups.items())]
        episode_rate_pools[source_seed] = [sum(values) / len(values) for values in episode_groups.values()]
        source_results[str(source_seed)] = {
            "n_states": len(rows),
            "n_agree": sum(row["agree"] for row in rows),
            "pooled_state_agreement": float(np.mean([row["agree"] for row in rows])),
            "episode_cluster_mean_agreement": float(statistics.mean(sum(v) / len(v) for v in episode_groups.values())),
            "episode_cluster_bootstrap_ci95": bootstrap_episode_cluster_ci(episode_records, seed + source_seed),
            "action_histogram_seed11": {str(a): sum(row["action_11"] == a for row in rows) for a in (0, 1)},
            "action_histogram_seed29": {str(a): sum(row["action_29"] == a for row in rows) for a in (0, 1)},
        }
    angle_groups = {"near_zero": [], "mid": [], "large": []}
    for index, row in enumerate(combined):
        angle = abs(float(row["state"][2]))
        label = "near_zero" if angle < 0.05 else "mid" if angle < 0.15 else "large"
        angle_groups[label].append(bool(agree[index]))
    strata = {name: {"n_states": len(values), "agreement_rate": float(np.mean(values)) if values else None}
              for name, values in angle_groups.items()}
    cluster_rng = random.Random(seed + 99)
    cluster_boots = []
    for _ in range(5000):
        per_source_means = [
            statistics.mean(rates[cluster_rng.randrange(len(rates))] for _ in rates)
            for rates in episode_rate_pools.values()
        ]
        cluster_boots.append(statistics.mean(per_source_means))
    cluster_boots.sort()
    return {
        "n_states": len(combined),
        "states_per_source_occupancy": state_count,
        "n_agree": int(agree.sum()),
        "n_disagree": int((~agree).sum()),
        "pooled_state_agreement": float(agree.mean()),
        "episode_cluster_balanced_agreement": float(statistics.mean(v["episode_cluster_mean_agreement"] for v in source_results.values())),
        "episode_cluster_balanced_bootstrap_ci95": [cluster_boots[124], cluster_boots[4874]],
        "source_occupancy_results": source_results,
        "stratified_by_absolute_pole_angle": strata,
        "state_feature_ranges": {f"dim_{i}": {"min": float(all_states[:, i].min()), "max": float(all_states[:, i].max())}
                                 for i in range(all_states.shape[1])},
        "state_array_sha256": __import__("hashlib").sha256(all_states.numpy().tobytes()).hexdigest(),
        "actor_action_arrays_sha256": {
            "seed_11": __import__("hashlib").sha256(actions_11.tobytes()).hexdigest(),
            "seed_29": __import__("hashlib").sha256(actions_29.tobytes()).hexdigest(),
        },
        "claim_boundary": "Deterministic argmax action agreement over equally sized fresh held-out state pools from each actor's occupancy; does not establish causal rationale, value agreement, training-process agreement, or action agreement outside sampled support.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--source-seeds", type=int, nargs=2, default=[11, 29])
    parser.add_argument("--episodes-per-actor", type=int, default=100)
    parser.add_argument("--states-per-source", type=int, default=5000)
    parser.add_argument("--reset-base", type=int, default=40_000_000)
    parser.add_argument("--sample-seed", type=int, default=20260922)
    args = parser.parse_args()
    task_root = args.task_root.resolve()
    summary = json.loads((task_root / "shared_multiseed_gpi_summary.json").read_text(encoding="utf-8"))
    calibration = json.loads((task_root / "shared_calibration.json").read_text(encoding="utf-8"))
    environment = calibration["environment"]
    if environment != "CartPole-v1":
        raise ValueError("pole-angle strata are defined only for CartPole-v1")
    state_dim = len(calibration["state_bins"]["edges"])
    import gymnasium as gym
    env = gym.make(environment)
    action_dim = int(env.action_space.n)
    env.close()
    actors, actor_paths, actor_hashes = [], [], []
    for seed in args.source_seeds:
        matches = [p for p in (task_root / "grammar_audit_only").glob(f"seed-{seed}_*") if (p / "metadata.json").is_file()]
        if len(matches) != 1:
            raise ValueError(f"expected one complete audit-only source run for seed {seed}; got {len(matches)}")
        actor_path = matches[0] / "actor.pt"
        actor_paths.append(str(actor_path))
        actors.append(load_actor(matches[0], state_dim, action_dim))
        actor_hashes.append(json.loads((matches[0] / "metadata.json").read_text(encoding="utf-8"))["actor_sha256"])
    heldout = []
    for index, (seed, actor) in enumerate(zip(args.source_seeds, actors)):
        heldout.extend(collect_heldout_episode_states(environment, actor, seed, args.episodes_per_actor,
                                                       args.reset_base + index * 1_000_000))
    overlap_path = task_root / "shared_symbolizer_rule_overlap.json"
    overlap = json.loads(overlap_path.read_text(encoding="utf-8"))
    jaccard = overlap["seed_11_vs_29"]["state_action_jaccard"]
    result = {
        "experiment": "B_behavioral_agreement_fresh_heldout_states",
        "environment": environment,
        "source_seeds": args.source_seeds,
        "actor_architecture": "The project's actual Actor class from grammar_audit_experiment.py (64 hidden units); no duplicate network definition.",
        "actor_checkpoint_paths": actor_paths,
        "actor_checkpoint_sha256": actor_hashes,
        "shared_symbolizer_sha256": calibration["state_bins_canonical_sha256"],
        "grammar_rule_jaccard_seed_11_vs_29": jaccard,
        "heldout_design": {
            "episodes_per_actor": args.episodes_per_actor,
            "fresh_reset_seed_ranges": {str(seed): [args.reset_base + index * 1_000_000,
                                                      args.reset_base + index * 1_000_000 + args.episodes_per_actor - 1]
                                         for index, seed in enumerate(args.source_seeds)},
            "resets_disjoint_from_primary_composition_eval": True,
            "sampled_states_per_actor_occupancy": args.states_per_source,
            "sampling": "Uniform without replacement from each actor's fresh held-out deterministic rollouts; actor occupancies receive equal sample weight.",
        },
        "results": summarize(heldout, args.states_per_source, args.sample_seed, actors),
        "source_episode_returns": {
            str(seed): {"mean": float(np.mean([row["episode_return"] for row in heldout if row["source_seed"] == seed])),
                        "median": float(np.median([row["episode_return"] for row in heldout if row["source_seed"] == seed])),
                        "episodes": sum(row["source_seed"] == seed for row in heldout),
                        "candidate_state_count": sum(len(row["states"]) for row in heldout if row["source_seed"] == seed)}
            for seed in args.source_seeds
        },
    }
    output = task_root / "behavioral_agreement_experiment_B.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "results": result["results"]}, indent=2))


if __name__ == "__main__":
    main()
