"""Run held-out diagnostics for saved FQE critics and their GPI policy."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from auditability.grammar_audit_experiment import Actor, canonical_json, sha256_file
from auditability.run_shared_multienv_gpi import QRegressor, load_actor


def quantiles(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()) if array.size else None,
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "p10": float(np.quantile(array, 0.10)) if array.size else None,
        "median": float(np.quantile(array, 0.50)) if array.size else None,
        "p90": float(np.quantile(array, 0.90)) if array.size else None,
    }


def heldout_states(env_name: str, actor_runs: list[Path], seeds: list[int], state_dim: int,
                   action_dim: int, episodes_per_actor: int, first_reset_seed: int):
    env = gym.make(env_name)
    records = []
    reset_seed = first_reset_seed
    for source_seed, actor_run in zip(seeds, actor_runs):
        actor = load_actor(actor_run, state_dim, action_dim)
        for episode_idx in range(episodes_per_actor):
            state, _ = env.reset(seed=reset_seed)
            episode_states = []
            total = 0.0
            done = False
            while not done:
                episode_states.append(np.asarray(state, dtype=np.float32).copy())
                with torch.inference_mode():
                    action = int(actor(torch.as_tensor(state, dtype=torch.float32)).argmax().item())
                state, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                done = terminated or truncated
            records.append({"source_seed": source_seed, "episode": episode_idx,
                            "reset_seed": reset_seed, "return": total, "states": episode_states})
            reset_seed += 1
    env.close()
    return records


def load_critics(fqe_dir: Path, seeds: list[int], state_dim: int, action_dim: int):
    models = []
    for seed in seeds:
        model = QRegressor(state_dim, action_dim)
        state_dict = torch.load(fqe_dir / f"fqe_q_seed_{seed}.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)
    return models


def evaluate_states(records: list[dict], actors: list[Actor], critics: list[QRegressor], seeds: list[int]):
    state_rows = []
    origins = []
    ep_returns = []
    for record in records:
        for state in record["states"]:
            state_rows.append(state)
            origins.append(record["source_seed"])
            ep_returns.append(record["return"])
    x = torch.as_tensor(np.asarray(state_rows), dtype=torch.float32)
    with torch.inference_mode():
        logits = torch.stack([actor(x) for actor in actors])
        ensemble_actions = logits.mean(dim=0).argmax(dim=1).numpy()
        q_by_seed = torch.stack([critic(x) for critic in critics]).numpy()
    gpi_actions = np.max(q_by_seed, axis=0).argmax(axis=1)
    q_agreement = {}
    for i, seed_i in enumerate(seeds):
        for j, seed_j in enumerate(seeds):
            if i >= j:
                continue
            a = q_by_seed[i]
            b = q_by_seed[j]
            q_agreement[f"{seed_i}_vs_{seed_j}"] = {
                "pearson_flat_q": float(np.corrcoef(a.ravel(), b.ravel())[0, 1]),
                "mean_absolute_q_difference": float(np.abs(a - b).mean()),
                "greedy_action_agreement": float((a.argmax(axis=1) == b.argmax(axis=1)).mean()),
                "q_i_all_actions": quantiles(a.ravel().tolist()),
                "q_j_all_actions": quantiles(b.ravel().tolist()),
            }
    by_origin = {}
    origin_array = np.asarray(origins)
    for seed in seeds:
        mask = origin_array == seed
        by_origin[str(seed)] = {
            "states": int(mask.sum()),
            "actor_ensemble_gpi_action_agreement": float((ensemble_actions[mask] == gpi_actions[mask]).mean()),
            "critic_q_distributions": {
                str(critic_seed): quantiles(q_by_seed[index, mask].ravel().tolist())
                for index, critic_seed in enumerate(seeds)
            },
        }
    low_cut, high_cut = np.quantile([r["return"] for r in records], [0.25, 0.75])
    episode_group = []
    if float(low_cut) != float(high_cut):
        for record in records:
            episode_group.extend(["low_return_q1" if record["return"] <= low_cut else
                                  "high_return_q4" if record["return"] >= high_cut else "middle"] * len(record["states"]))
    direction = {"discriminative": bool(float(low_cut) != float(high_cut))}
    if not episode_group:
        direction["reason"] = "held-out episode returns have identical 25th and 75th percentile cutoffs"
        direction["all_episode_returns"] = quantiles([float(r["return"]) for r in records])
    group_array = np.asarray(episode_group)
    for group_name in (("low_return_q1", "high_return_q4") if episode_group else ()):
        mask = group_array == group_name
        group = {"states": int(mask.sum()), "source_policy_values": {}}
        for index, seed in enumerate(seeds):
            # Compare the critic's value at its own deterministic actor action.
            actor = actors[index]
            with torch.inference_mode():
                action = actor(x[mask]).argmax(dim=1)
                q = torch.as_tensor(q_by_seed[index, mask])
                values = q.gather(1, action[:, None]).squeeze(1).numpy()
            group["source_policy_values"][str(seed)] = quantiles(values.tolist())
        direction[group_name] = group
    if len(state_rows[0]) == 4:
        anchors = {
            "balanced_origin": [0.0, 0.0, 0.0, 0.0],
            "perturbed_low_return_candidate": [0.0, 0.0, 0.25, 2.0],
        }
        anchor_values = {}
        for name, values in anchors.items():
            state = torch.tensor(values, dtype=torch.float32)
            with torch.inference_mode():
                actor_logits = torch.stack([actor(state) for actor in actors]).mean(0)
                qs = torch.stack([critic(state) for critic in critics]).numpy()
            anchor_values[name] = {
                "state": values,
                "actor_ensemble_argmax": int(actor_logits.argmax().item()),
                "gpi_argmax": int(qs.max(axis=0).argmax()),
                "per_critic_q_all_actions": {str(seed): qs[i].tolist() for i, seed in enumerate(seeds)},
            }
        anchor_values["value_direction_check"] = {
            str(seed): {
                "balanced_q_max": float(max(anchor_values["balanced_origin"]["per_critic_q_all_actions"][str(seed)])),
                "perturbed_q_max": float(max(anchor_values["perturbed_low_return_candidate"]["per_critic_q_all_actions"][str(seed)])),
                "balanced_exceeds_perturbed": bool(max(anchor_values["balanced_origin"]["per_critic_q_all_actions"][str(seed)]) > max(anchor_values["perturbed_low_return_candidate"]["per_critic_q_all_actions"][str(seed)])),
            }
            for seed in seeds
        }
    else:
        anchor_values = None
    return {
        "heldout_state_count": len(state_rows),
        "episode_return_quartile_cutoffs": {"q25": float(low_cut), "q75": float(high_cut)},
        "actor_logit_ensemble_vs_fqe_gpi_action_agreement": float((ensemble_actions == gpi_actions).mean()),
        "actor_logit_ensemble_action_histogram": {str(int(a)): int((ensemble_actions == a).sum()) for a in np.unique(ensemble_actions)},
        "fqe_gpi_action_histogram": {str(int(a)): int((gpi_actions == a).sum()) for a in np.unique(gpi_actions)},
        "per_heldout_state_origin": by_origin,
        "q_function_cross_seed_agreement": q_agreement,
        "q_direction_by_heldout_episode_return": direction,
        "cartpole_balanced_vs_perturbed_anchor_states": anchor_values,
        "raw_state_array_sha256": __import__("hashlib").sha256(x.numpy().tobytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 29])
    parser.add_argument("--episodes-per-actor", type=int, default=50)
    parser.add_argument("--first-reset-seed", type=int, default=30_000_000)
    args = parser.parse_args()
    task_root = args.task_root.resolve()
    calibration = json.loads((task_root / "shared_calibration.json").read_text(encoding="utf-8"))
    env_name = calibration["environment"]
    state_dim = len(calibration["state_bins"]["edges"])
    env = gym.make(env_name)
    action_dim = int(env.action_space.n)
    env.close()
    actor_runs = []
    for seed in args.seeds:
        matches = sorted((task_root / "grammar_audit_only").glob(f"seed-{seed}_*"))
        matches = [path for path in matches if (path / "metadata.json").is_file()]
        if len(matches) != 1:
            raise ValueError(f"expected one complete audit-only run for seed {seed}, found {len(matches)}")
        actor_runs.append(matches[0])
    actors = [load_actor(path, state_dim, action_dim) for path in actor_runs]
    critics = load_critics(task_root / "gpi_fqe", args.seeds, state_dim, action_dim)
    records = heldout_states(env_name, actor_runs, args.seeds, state_dim, action_dim,
                             args.episodes_per_actor, args.first_reset_seed)
    report = {
        "environment": env_name,
        "seeds": args.seeds,
        "episodes_per_actor": args.episodes_per_actor,
        "reset_seeds": "disjoint deterministic range, separate from train and primary eval",
        "source_actor_checkpoint_sha256": [json.loads((path / "metadata.json").read_text(encoding="utf-8"))["actor_sha256"] for path in actor_runs],
        "fqe_critic_file_sha256": [sha256_file(task_root / "gpi_fqe" / f"fqe_q_seed_{seed}.pt") for seed in args.seeds],
        "diagnostics": evaluate_states(records, actors, critics, args.seeds),
        "interpretation_limit": "These are held-out state-distribution diagnostics; action agreement and Q direction do not establish policy value accuracy or a classical GPI performance guarantee.",
    }
    path = task_root / "gpi_fqe" / "gpi_fqe_diagnostics.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(path), "diagnostics": report["diagnostics"]}, indent=2))


if __name__ == "__main__":
    main()
