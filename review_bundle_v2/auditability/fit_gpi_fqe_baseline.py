"""Fit off-policy FQE critics for a classical approximate GPI policy baseline."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from auditability.grammar_audit_experiment import canonical_json, sha256_file, seed_everything
from auditability.run_shared_multienv_gpi import (
    QRegressor,
    evaluate_composition,
    load_actor,
    load_rules,
)
from auditability.grammar_audit_experiment import StateSymbolizer


def _reservoir_add(pool: list, item: tuple, seen: int, capacity: int, rng: random.Random) -> None:
    if len(pool) < capacity:
        pool.append(item)
    else:
        index = rng.randrange(seen)
        if index < capacity:
            pool[index] = item


def collect_pooled_transitions(task_root: Path, seeds: list[int], train_cap_per_seed: int, validation_cap_per_seed: int):
    rng = random.Random(611_2026)
    train_rows: list[tuple] = []
    validation_rows: list[tuple] = []
    train_seen = validation_seen = 0
    provenance = []
    for seed in seeds:
        run_dirs = [path for path in (task_root / "grammar_audit_only").glob(f"seed-{seed}_*") if (path / "metadata.json").is_file()]
        if len(run_dirs) != 1:
            raise ValueError(f"expected one complete audit-only run for seed {seed}; found {len(run_dirs)}")
        run_dir = run_dirs[0]
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        episode_count = int(metadata["config"]["episodes"])
        train_cutoff = max(1, int(episode_count * 0.8))
        per_seed_train: list[tuple] = []
        per_seed_validation: list[tuple] = []
        seen_train = seen_validation = 0
        local_rng = random.Random(800_000 + seed)
        with (run_dir / "trajectory.jsonl").open("r", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                item = (
                    tuple(float(x) for x in row["state"]),
                    int(row["action"]),
                    float(row["reward"]),
                    tuple(float(x) for x in row["next_state"]),
                    bool(row["terminated"] or row["truncated"]),
                )
                if int(row["episode"]) < train_cutoff:
                    seen_train += 1
                    _reservoir_add(per_seed_train, item, seen_train, train_cap_per_seed, local_rng)
                else:
                    seen_validation += 1
                    _reservoir_add(per_seed_validation, item, seen_validation, validation_cap_per_seed, local_rng)
        train_rows.extend(per_seed_train)
        validation_rows.extend(per_seed_validation)
        train_seen += seen_train
        validation_seen += seen_validation
        provenance.append({
            "seed": seed,
            "run_dir": str(run_dir),
            "trace_sha256": metadata["trace_sha256"],
            "training_candidates_seen": seen_train,
            "training_samples_retained": len(per_seed_train),
            "validation_candidates_seen": seen_validation,
            "validation_samples_retained": len(per_seed_validation),
            "episode_split": f"episodes < {train_cutoff} train; remaining episodes validation",
        })
    return train_rows, validation_rows, provenance


def rows_to_tensors(rows: list[tuple]) -> tuple[torch.Tensor, ...]:
    return (
        torch.tensor([row[0] for row in rows], dtype=torch.float32),
        torch.tensor([row[1] for row in rows], dtype=torch.long),
        torch.tensor([row[2] for row in rows], dtype=torch.float32),
        torch.tensor([row[3] for row in rows], dtype=torch.float32),
        torch.tensor([row[4] for row in rows], dtype=torch.bool),
    )


def fqe_target(model: QRegressor, actor, next_state: torch.Tensor, reward: torch.Tensor, done: torch.Tensor, gamma: float) -> torch.Tensor:
    with torch.inference_mode():
        next_action = actor(next_state).argmax(dim=1)
        next_q = model(next_state).gather(1, next_action[:, None]).squeeze(1)
        return reward + gamma * (~done).float() * next_q


def fit_fqe_for_policy(
    actor,
    train: tuple[torch.Tensor, ...],
    validation: tuple[torch.Tensor, ...],
    state_dim: int,
    action_dim: int,
    seed: int,
    gamma: float = 0.99,
    updates: int = 1200,
    batch_size: int = 1024,
) -> tuple[QRegressor, dict]:
    seed_everything(seed)
    model = QRegressor(state_dim, action_dim)
    target_model = QRegressor(state_dim, action_dim)
    target_model.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    states, actions, rewards, next_states, dones = train
    generator = torch.Generator().manual_seed(seed)
    model.train()
    losses = []
    for update in range(updates):
        indices = torch.randint(0, len(states), (batch_size,), generator=generator)
        predicted = model(states[indices]).gather(1, actions[indices, None]).squeeze(1)
        with torch.inference_mode():
            next_action = actor(next_states[indices]).argmax(dim=1)
            next_q = target_model(next_states[indices]).gather(1, next_action[:, None]).squeeze(1)
            targets = rewards[indices] + gamma * (~dones[indices]).float() * next_q
        loss = torch.mean((predicted - targets) ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if (update + 1) % 100 == 0:
            target_model.load_state_dict(model.state_dict())
    model.eval()
    vx, va, vr, vnext, vd = validation
    with torch.inference_mode():
        value = model(vx).gather(1, va[:, None]).squeeze(1)
        target = fqe_target(model, actor, vnext, vr, vd, gamma)
        residual = (value - target).abs().numpy()
    return model, {
        "policy_seed": seed,
        "fqe_updates": updates,
        "batch_size": batch_size,
        "gamma": gamma,
        "heldout_bellman_residual_mae": float(residual.mean()),
        "heldout_bellman_residual_rmse": float(np.sqrt(np.mean(residual ** 2))),
        "final_100_update_mean_td_loss": float(np.mean(losses[-100:])),
        "q_function_target": "Q for deterministic argmax source actor, estimated by fitted Q evaluation over the pooled seed replay buffer",
    }


def run_fqe(task_root: Path, seeds: list[int], evaluation_episodes: int, updates: int, output_dir: Path) -> dict:
    calibration = json.loads((task_root / "shared_calibration.json").read_text(encoding="utf-8"))
    environment_name = calibration["environment"]
    state_dim = len(calibration["state_bins"]["edges"])
    import gymnasium as gym
    probe = gym.make(environment_name)
    action_dim = int(probe.action_space.n)
    probe.close()
    agent_runs = []
    for seed in seeds:
        candidates = [path for path in (task_root / "grammar_audit_only").glob(f"seed-{seed}_*") if (path / "metadata.json").is_file()]
        if len(candidates) != 1:
            raise ValueError(f"expected one complete audit-only run for seed {seed}; found {len(candidates)}")
        agent_runs.append(candidates[0])
    actors = [load_actor(path, state_dim, action_dim) for path in agent_runs]
    rule_banks = [load_rules(path) for path in agent_runs]
    existing_summary = json.loads((task_root / "shared_multiseed_gpi_summary.json").read_text(encoding="utf-8"))
    mc_diagnostics = existing_summary.get("gpi_critic_fit_diagnostics", [])
    train_rows, validation_rows, data_provenance = collect_pooled_transitions(
        task_root, seeds, train_cap_per_seed=25_000, validation_cap_per_seed=5_000
    )
    train_tensors, validation_tensors = rows_to_tensors(train_rows), rows_to_tensors(validation_rows)
    fqe_models, fqe_diagnostics = [], []
    for index, (seed, actor) in enumerate(zip(seeds, actors)):
        model, report = fit_fqe_for_policy(
            actor, train_tensors, validation_tensors, state_dim, action_dim,
            seed=900_000 + seed, updates=updates,
        )
        fqe_models.append(model)
        report["seed"] = seed
        fqe_diagnostics.append(report)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_dir / f"fqe_q_seed_{seed}.pt")

    symbolizer = StateSymbolizer(calibration["state_bins"]["n_bins"])
    symbolizer.edges = torch.tensor(calibration["state_bins"]["edges"], dtype=torch.float64)
    composition = evaluate_composition(
        environment_name, task_root, seeds, evaluation_episodes,
        symbolizer,
        agent_runs, actors, rule_banks, fqe_models,
    )
    composition["policies"]["gpi_fqe"] = composition["policies"].pop("gpi_mc_q")
    composition["paired_bootstrap_difference_vs_best_single"]["gpi_fqe"] = composition[
        "paired_bootstrap_difference_vs_best_single"
    ].pop("gpi_mc_q")
    result = {
        "environment": environment_name,
        "seeds": seeds,
        "data_provenance": data_provenance,
        "pooled_replay_buffer": {
            "training_rows_retained": len(train_rows),
            "validation_rows_retained": len(validation_rows),
            "training_row_sampling": "reservoir sample, cap 25,000 per source seed from first 80 percent training episodes",
            "validation_row_sampling": "reservoir sample, cap 5,000 per source seed from last 20 percent training episodes",
        },
        "mc_gpi_diagnostics_for_comparison": mc_diagnostics,
        "fqe_critic_diagnostics": fqe_diagnostics,
        "composition_evaluation": composition,
        "fqe_method": {
            "definition": "For each frozen source actor pi_i, train Q_i(s,a)=r+gamma*Q_i(s',argmax pi_i(s')) by fitted Q evaluation on pooled replay transitions; GPI selects argmax_a max_i Q_i(s,a).",
            "limitations": "Off-policy coverage and neural approximation remain finite; heldout Bellman residual is a diagnostic, not proof of exact Q values or a theoretical GPI guarantee.",
            "critic_state_dicts": [f"fqe_q_seed_{seed}.pt" for seed in seeds],
        },
        "actor_checkpoint_sha256": [json.loads((run / "metadata.json").read_text(encoding="utf-8"))["actor_sha256"] for run in agent_runs],
        "fqe_code_sha256": sha256_file(Path(__file__).resolve()),
    }
    result_path = output_dir / "gpi_fqe_comparison.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 29, 43, 71, 101, 149, 211, 307])
    parser.add_argument("--evaluation-episodes", type=int, default=100)
    parser.add_argument("--updates", type=int, default=1200)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.task_root / "gpi_fqe"
    result = run_fqe(args.task_root, args.seeds, args.evaluation_episodes, args.updates, output_dir)
    print(json.dumps({
        "environment": result["environment"],
        "gpi_fqe_mean_return": result["composition_evaluation"]["policies"]["gpi_fqe"]["mean_return"],
        "fqe_diagnostics": result["fqe_critic_diagnostics"],
        "output": str(output_dir / "gpi_fqe_comparison.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
