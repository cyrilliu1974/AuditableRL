"""Shared-symbolizer multi-seed RL grammar fusion and approximate GPI experiment."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import statistics
from pathlib import Path

import numpy as np
import torch
from torch import nn

from auditability.grammar_audit_experiment import (
    Actor,
    StateSymbolizer,
    canonical_json,
    replay_environment_trace,
    seed_everything,
    sha256_file,
    shared_calibration_symbolizer,
    train_one,
    verify_trace,
    verify_update_trace,
    training_update_core_sha256,
)


DEFAULT_SEEDS = [11, 29, 43, 71, 101, 149, 211, 307]
ARMS = ("baseline", "grammar_audit_only", "grammar_constrained")
POLICIES = ("best_single", "actor_mean_logits", "grammar_fusion", "gpi_mc_q")


def load_actor(run_dir: Path, state_dim: int, action_dim: int) -> Actor:
    checkpoint = torch.load(run_dir / "actor.pt", map_location="cpu", weights_only=True)
    actor = Actor(state_dim=state_dim, action_dim=action_dim)
    actor.load_state_dict(checkpoint["state_dict"])
    actor.eval()
    return actor


def load_rules(run_dir: Path) -> dict[tuple[int, ...], dict]:
    grammar = json.loads((run_dir / "induced_grammar.json").read_text(encoding="utf-8"))
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    grammar_sha = sha256_file(run_dir / "induced_grammar.json")
    rules = {}
    for production in grammar.get("productions", []):
        state_symbol, action_symbol = production["rhs"]
        condition = tuple(int(part) for part in state_symbol.removeprefix("STATE_").split("_"))
        rules[condition] = {
            "action": int(action_symbol.removeprefix("ACTION_")),
            "support": int(production["support"]),
            "confidence": float(production["confidence"]),
            "mean_reward": float(production["mean_reward"]),
            "source_seed": int(metadata["config"]["seed"]),
            "source_rule_id": f"{production['lhs']}:{state_symbol}->{action_symbol}",
            "source_run": str(run_dir),
            "source_grammar_sha256": grammar_sha,
            "source_step_ids": production.get("source_step_ids", []),
        }
    return rules


def verify_constraint_snapshots(run_dir: Path) -> dict:
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    snapshot_path = run_dir / metadata.get("constraint_rule_snapshots_file", "constraint_rule_snapshots.json")
    if not snapshot_path.is_file():
        raise ValueError(f"missing rule snapshot registry: {snapshot_path}")
    expected_registry_hash = metadata.get("constraint_rule_snapshots_sha256")
    if expected_registry_hash and sha256_file(snapshot_path) != expected_registry_hash:
        raise ValueError(f"rule snapshot registry file hash mismatch: {snapshot_path}")
    registry = json.loads(snapshot_path.read_text(encoding="utf-8"))
    for snapshot_id, snapshot in registry.items():
        actual_id = hashlib.sha256(canonical_json(snapshot).encode("utf-8")).hexdigest()
        if actual_id != snapshot_id:
            raise ValueError(f"rule snapshot content hash mismatch: {snapshot_id}")
    referenced = collections.Counter()
    with (run_dir / "trajectory.jsonl").open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            row = json.loads(line)
            reference = row.get("constraint_rule")
            if reference is None:
                continue
            snapshot_id = reference.get("snapshot_id")
            if not snapshot_id or snapshot_id not in registry:
                raise ValueError(f"trace line {line_no}: unknown rule snapshot reference")
            snapshot = registry[snapshot_id]
            expected_state = "STATE_" + "_".join(map(str, row["state_symbol"]))
            source_ids = snapshot["source_step_ids"]
            source_digest = hashlib.sha256(canonical_json(source_ids).encode("utf-8")).hexdigest()
            if snapshot["state_symbol"] != expected_state:
                raise ValueError(f"trace line {line_no}: referenced rule condition does not match state symbol")
            for field in ("action", "support", "condition_count", "confidence", "mean_reward"):
                if reference.get(field) != snapshot.get(field):
                    raise ValueError(f"trace line {line_no}: rule snapshot mismatch for {field}")
            if int(reference.get("source_step_count", -1)) != len(source_ids):
                raise ValueError(f"trace line {line_no}: rule source-step count mismatch")
            if reference.get("source_step_ids_sha256") != source_digest:
                raise ValueError(f"trace line {line_no}: rule source-step hash mismatch")
            referenced[snapshot_id] += 1
    return {
        "valid": True,
        "registry_sha256": sha256_file(snapshot_path),
        "snapshot_count": len(registry),
        "referenced_snapshot_count": len(referenced),
        "decision_references": sum(referenced.values()),
        "all_reference_hashes_and_rule_state_actions_verified": True,
    }


def grammar_fusion_decision(
    observation: np.ndarray,
    symbolizer: StateSymbolizer,
    rule_banks: list[dict[tuple[int, ...], dict]],
    actors: list[Actor],
) -> tuple[int, dict]:
    condition = symbolizer.encode(observation)
    candidates = [bank[condition] for bank in rule_banks if condition in bank]
    state = torch.as_tensor(observation, dtype=torch.float32)
    with torch.inference_mode():
        ensemble_logits = torch.stack([actor(state) for actor in actors]).mean(dim=0)
    fallback_action = int(ensemble_logits.argmax().item())
    if not candidates:
        return fallback_action, {"covered": False, "conflict": False, "candidate_count": 0}
    ranked = sorted(
        candidates,
        key=lambda rule: (
            rule["confidence"], rule["support"], rule["mean_reward"], -rule["source_seed"]
        ),
        reverse=True,
    )
    selected = ranked[0]
    return int(selected["action"]), {
        "covered": True,
        "conflict": len({rule["action"] for rule in candidates}) > 1,
        "candidate_count": len(candidates),
        "selected_rule": {
            key: selected[key]
            for key in ("action", "confidence", "support", "mean_reward", "source_seed", "source_rule_id", "source_run", "source_grammar_sha256")
        },
    }


def episode_mc_rows(trace_path: Path, gamma: float = 0.99) -> tuple[list[dict], int]:
    episodes: dict[int, list[dict]] = collections.defaultdict(list)
    with trace_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            episodes[int(row["episode"])].append(row)
    rows = []
    for episode, records in sorted(episodes.items()):
        target = 0.0
        for record in reversed(records):
            target = float(record["reward"]) + gamma * target
            rows.append({
                "episode": episode,
                "state": record["state"],
                "action": int(record["action"]),
                "return": target,
            })
    return rows, len(episodes)


class QRegressor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden),
            nn.Tanh(), nn.Linear(hidden, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


def fit_mc_q(run_dir: Path, state_dim: int, action_dim: int, fit_seed: int) -> tuple[QRegressor, dict]:
    rows, episode_count = episode_mc_rows(run_dir / "trajectory.jsonl")
    train_cutoff = max(1, int(episode_count * 0.8))
    train_rows = [row for row in rows if row["episode"] < train_cutoff]
    validation_rows = [row for row in rows if row["episode"] >= train_cutoff]
    train_action_counts = collections.Counter(row["action"] for row in train_rows)
    validation_action_counts = collections.Counter(row["action"] for row in validation_rows)
    seed_everything(fit_seed)
    model = QRegressor(state_dim, action_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.tensor([row["state"] for row in train_rows], dtype=torch.float32)
    a = torch.tensor([row["action"] for row in train_rows], dtype=torch.long)
    y = torch.tensor([row["return"] for row in train_rows], dtype=torch.float32)
    generator = torch.Generator().manual_seed(fit_seed)
    batch_size = 1024
    epochs = 8
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(train_rows), generator=generator)
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size]
            prediction = model(x[index]).gather(1, a[index, None]).squeeze(1)
            loss = torch.mean((prediction - y[index]) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
    model.eval()
    with torch.inference_mode():
        vx = torch.tensor([row["state"] for row in validation_rows], dtype=torch.float32)
        va = torch.tensor([row["action"] for row in validation_rows], dtype=torch.long)
        vy = torch.tensor([row["return"] for row in validation_rows], dtype=torch.float32)
        vp = model(vx).gather(1, va[:, None]).squeeze(1)
        errors = (vp - vy).abs().numpy()
    return model, {
        "training_transitions": len(train_rows),
        "heldout_validation_transitions": len(validation_rows),
        "episodes_held_out_by_time": episode_count - train_cutoff,
        "training_action_counts": {str(action): train_action_counts[action] for action in range(action_dim)},
        "validation_action_counts": {str(action): validation_action_counts[action] for action in range(action_dim)},
        "actions_unseen_in_training": [action for action in range(action_dim) if train_action_counts[action] == 0],
        "mc_return_mae": float(errors.mean()) if len(errors) else None,
        "mc_return_rmse": float(np.sqrt(np.mean(errors ** 2))) if len(errors) else None,
        "epochs": epochs,
        "target": "discounted Monte Carlo return for the source policy's sampled state-action pairs",
        "support_limit": "Q values for actions absent from the source-policy training data rely on neural extrapolation",
    }


def collect_policy_artifacts(environment_root: Path, seeds: list[int], state_dim: int, action_dim: int):
    agent_runs = []
    for seed in seeds:
        matches = sorted((environment_root / "grammar_audit_only").glob(f"seed-{seed}_*"))
        if len(matches) != 1:
            raise ValueError(f"expected one audit-only run for seed {seed}; found {len(matches)}")
        agent_runs.append(matches[0])
    actors = [load_actor(path, state_dim, action_dim) for path in agent_runs]
    rules = [load_rules(path) for path in agent_runs]
    critic_models = []
    critic_fit = []
    for index, run_dir in enumerate(agent_runs):
        model, fit_report = fit_mc_q(run_dir, state_dim, action_dim, fit_seed=700_000 + index)
        critic_models.append(model)
        critic_fit.append({"seed": seeds[index], **fit_report})
    return agent_runs, actors, rules, critic_models, critic_fit


def actor_action(actor: Actor, observation: np.ndarray) -> int:
    with torch.inference_mode():
        return int(actor(torch.as_tensor(observation, dtype=torch.float32)).argmax().item())


def gpi_action(models: list[QRegressor], observation: np.ndarray) -> tuple[int, list[float]]:
    state = torch.as_tensor(observation, dtype=torch.float32)
    with torch.inference_mode():
        values = torch.stack([model(state) for model in models])
        maximum_over_policies = values.max(dim=0).values
    return int(maximum_over_policies.argmax().item()), maximum_over_policies.tolist()


def evaluate_composition(
    environment_name: str,
    output_root: Path,
    seeds: list[int],
    episodes: int,
    symbolizer: StateSymbolizer,
    agent_runs: list[Path],
    actors: list[Actor],
    rule_banks: list[dict[tuple[int, ...], dict]],
    critic_models: list[QRegressor],
) -> dict:
    import gymnasium as gym

    env = gym.make(environment_name)
    training_means = [json.loads((path / "metadata.json").read_text(encoding="utf-8"))["episode_return_mean"] for path in agent_runs]
    best_index = max(range(len(seeds)), key=lambda index: (training_means[index], -seeds[index]))
    returns = {policy: [] for policy in POLICIES}
    decision_counts = collections.Counter()
    reset_seeds = [20_000_000 + index for index in range(episodes)]
    action_log = []
    for episode_index, reset_seed in enumerate(reset_seeds):
        for policy in POLICIES:
            observation, _ = env.reset(seed=reset_seed)
            total = 0.0
            terminated = truncated = False
            length = 0
            while not (terminated or truncated):
                detail = {}
                if policy == "best_single":
                    action = actor_action(actors[best_index], observation)
                elif policy == "actor_mean_logits":
                    state = torch.as_tensor(observation, dtype=torch.float32)
                    with torch.inference_mode():
                        action = int(torch.stack([actor(state) for actor in actors]).mean(0).argmax().item())
                elif policy == "grammar_fusion":
                    action, detail = grammar_fusion_decision(observation, symbolizer, rule_banks, actors)
                    decision_counts["grammar_fusion_total_steps"] += 1
                    decision_counts["grammar_fusion_covered_steps"] += int(detail["covered"])
                    decision_counts["grammar_fusion_conflicts"] += int(detail["conflict"])
                    decision_counts["grammar_fusion_blind_spots"] += int(not detail["covered"])
                else:
                    action, q_values = gpi_action(critic_models, observation)
                    detail = {"max_over_policy_action_values": q_values}
                observation, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                length += 1
                if episode_index == 0 and length <= 8 and policy in {"grammar_fusion", "gpi_mc_q"}:
                    action_log.append({
                        "policy": policy, "episode": episode_index, "reset_seed": reset_seed,
                        "step": length - 1, "action": int(action), **detail,
                    })
            returns[policy].append(total)
    env.close()
    result = {
        "heldout_episodes": episodes,
        "heldout_reset_seed_formula": "20000000 + episode_index; same reset seeds for every policy",
        "fallback_best_single_seed": seeds[best_index],
        "fallback_selection_basis": "training episode_return_mean only",
        "policies": {
            name: {
                "mean_return": statistics.mean(values),
                "median_return": statistics.median(values),
                "sample_std_return": statistics.stdev(values) if len(values) > 1 else 0.0,
                "return_by_episode": values,
            }
            for name, values in returns.items()
        },
        "paired_bootstrap_difference_vs_best_single": {},
        "grammar_fusion_decision_mix": dict(decision_counts),
        "illustrative_composition_actions_first_3_episodes": action_log,
    }
    rng = random.Random(20260922)
    for policy in POLICIES:
        if policy == "best_single":
            continue
        deltas = [left - right for left, right in zip(returns[policy], returns["best_single"])]
        bootstrap = sorted(
            statistics.mean(deltas[rng.randrange(len(deltas))] for _ in deltas)
            for _ in range(5000)
        )
        result["paired_bootstrap_difference_vs_best_single"][policy] = {
            "mean_delta": statistics.mean(deltas),
            "ci95_percentile": [bootstrap[124], bootstrap[4874]],
        }
    return result


def shared_rule_core(rule_banks: list[dict[tuple[int, ...], dict]], seeds: list[int]) -> dict:
    appearances = collections.Counter(condition for bank in rule_banks for condition in bank)
    actions = collections.defaultdict(collections.Counter)
    for bank in rule_banks:
        for condition, rule in bank.items():
            actions[condition][rule["action"]] += 1
    shared = [condition for condition, count in appearances.items() if count >= 2]
    core = [condition for condition, count in appearances.items() if count == len(rule_banks)]
    stable_core = [condition for condition in core if len(actions[condition]) == 1]
    return {
        "seed_count": len(seeds),
        "seeds": seeds,
        "shared_rule_conditions_present_in_at_least_two_seeds": len(shared),
        "all_seed_rule_condition_intersection": len(core),
        "all_seed_same_action_core": len(stable_core),
        "all_seed_core_rules": [
            {"condition": list(condition), "action": next(iter(actions[condition]))}
            for condition in sorted(stable_core)
        ],
        "per_seed_rule_counts": [len(bank) for bank in rule_banks],
        "conflicting_conditions_in_all_seed_intersection": len(core) - len(stable_core),
    }


def run_task(
    environment_name: str,
    output_root: Path,
    seeds: list[int],
    episodes: int,
    calibration_episodes: int,
    n_bins: int,
    min_support: int,
    min_confidence: float,
    evaluation_episodes: int,
) -> dict:
    import gymnasium as gym

    env = gym.make(environment_name)
    state_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(env.action_space.n)
    env.close()
    task_root = output_root / environment_name
    task_root.mkdir(parents=True, exist_ok=True)
    symbolizer = shared_calibration_symbolizer(seeds, calibration_episodes, n_bins, environment_name)
    calibration_record = {
        "method": "pooled empirical quantiles from separate random-policy roll-ins; frozen for every seed and arm",
        "environment": environment_name,
        "calibration_seeds": seeds,
        "calibration_episodes_per_seed": calibration_episodes,
        "state_bins": symbolizer.to_json(),
        "state_bins_canonical_sha256": hashlib.sha256(canonical_json(symbolizer.to_json()).encode("utf-8")).hexdigest(),
    }
    (task_root / "shared_calibration.json").write_text(json.dumps(calibration_record, indent=2) + "\n", encoding="utf-8")

    run_dirs: dict[int, dict[str, Path]] = {}
    for seed in seeds:
        run_dirs[seed] = {}
        for arm in ARMS:
            arm_root = task_root / arm
            completed = [
                path for path in sorted(arm_root.glob(f"seed-{seed}_*"))
                if (path / "metadata.json").is_file()
                and (path / "trajectory.jsonl").is_file()
                and (path / "training_updates.jsonl").is_file()
            ]
            if len(completed) > 1:
                raise ValueError(f"multiple completed folders exist for {environment_name} seed={seed} arm={arm}: {completed}")
            if completed:
                run_dir = completed[0]
            else:
                run_dir = train_one(
                    seed, arm, episodes, task_root, symbolizer,
                    min_support=min_support, min_confidence=min_confidence,
                    environment_name=environment_name, symbolizer_scope="pooled_across_all_experiment_seeds",
                )
            run_dirs[seed][arm] = run_dir

    audit_rows = []
    all_traces_exact = True
    all_updates_exact = True
    total_transitions = 0
    total_updates = 0
    unique_codec = None
    for seed in seeds:
        run_info = {}
        for arm in ARMS:
            run_dir = run_dirs[seed][arm]
            metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
            chain = verify_trace(run_dir / "trajectory.jsonl")
            updates = verify_update_trace(run_dir / "training_updates.jsonl")
            run_info[arm] = {
                "actor_sha256": metadata["actor_sha256"],
                "trace_sha256": metadata["trace_sha256"],
                "update_core_sha256": metadata["training_update_core_sha256"],
                "trace_valid": chain["valid"],
                "trace_records": chain["records"],
                "update_valid": updates["valid"],
                "update_records": updates["records"],
                "training_mean_return": metadata["episode_return_mean"],
                "heldout_mean_return": metadata["evaluation"]["mean_return"],
                "grammar_rule_count": len(json.loads((run_dir / "induced_grammar.json").read_text(encoding="utf-8")).get("productions", [])),
            }
            if arm == "grammar_constrained":
                run_info[arm]["constraint_snapshot_audit"] = verify_constraint_snapshots(run_dir)
            total_transitions += chain["records"]
            total_updates += updates["records"]
            all_traces_exact = all_traces_exact and chain["valid"]
            all_updates_exact = all_updates_exact and updates["valid"]
        pair = run_info["baseline"]["actor_sha256"] == run_info["grammar_audit_only"]["actor_sha256"] and run_info["baseline"]["trace_sha256"] == run_info["grammar_audit_only"]["trace_sha256"] and run_info["baseline"]["update_core_sha256"] == run_info["grammar_audit_only"]["update_core_sha256"]
        all_traces_exact = all_traces_exact and pair
        all_updates_exact = all_updates_exact and pair
        audit_rows.append({"seed": seed, "runs": run_info, "baseline_audit_only_exact_training_equivalence": pair})
        baseline_trace = run_dirs[seed]["baseline"] / "trajectory.jsonl"
        replay = replay_environment_trace(baseline_trace, environment_name)
        if not replay["valid"]:
            raise AssertionError(f"environment replay failed for {environment_name} seed {seed}")
        audit_rows[-1]["environment_replay"] = replay
        if unique_codec is None:
            from auditability.grammar_audit_experiment import replay_induced_trace
            codec_output = run_dirs[seed]["baseline"] / "lossless_trace_grammar.json.gz"
            encoded = replay_induced_trace(baseline_trace, codec_output, max_rules=16)
            unique_codec = {key: encoded[key] for key in (
                "token_count", "production_count", "roundtrip_exact", "input_sha256", "decoded_sha256",
                "raw_trace_bytes", "raw_trace_gzip_bytes", "grammar_gzip_bytes",
            ) if key in encoded}
    if not all(row["baseline_audit_only_exact_training_equivalence"] for row in audit_rows):
        raise AssertionError(f"observer changed training for {environment_name}")

    agent_runs, actors, rule_banks, critics, critic_reports = collect_policy_artifacts(task_root, seeds, state_dim, action_dim)
    composition = evaluate_composition(
        environment_name, task_root, seeds, evaluation_episodes, symbolizer,
        agent_runs, actors, rule_banks, critics,
    )
    result = {
        "environment": environment_name,
        "environment_spaces": {"state_dim": state_dim, "action_dim": action_dim},
        "experiment_design": {
            "seeds": seeds,
            "episodes_per_seed_arm": episodes,
            "arms": list(ARMS),
            "shared_calibration": calibration_record,
            "grammar_min_support": min_support,
            "grammar_min_confidence": min_confidence,
            "evaluation_episodes": evaluation_episodes,
        },
        "per_seed": audit_rows,
        "grammar_core": shared_rule_core(rule_banks, seeds),
        "composition_evaluation": composition,
        "gpi_critic_fit_diagnostics": critic_reports,
        "audit_integrity": {
            "hash_chains_valid": all_traces_exact,
            "update_ledgers_valid_and_observer_equivalent": all_updates_exact,
            "baseline_traces_environment_replayed": True,
            "baseline_audit_only_pairs_exactly_equal": all(row["baseline_audit_only_exact_training_equivalence"] for row in audit_rows),
            "baseline_trace_records_across_three_arms": total_transitions,
            "optimizer_update_records_across_three_arms": total_updates,
            "one_lossless_trace_codec": unique_codec,
        },
        "gpi_method": {
            "definition": "approximate sampled-policy GPI: fit one action-value regressor per frozen source policy from its logged Monte Carlo returns, then select argmax_a max_i Q_i(s,a)",
            "theory_scope": "the classical GPI bound depends on value-estimation error; these fitted neural Q estimates are approximate and do not inherit an exact guarantee",
            "data_split": "last 20 percent of each source agent's episodes held out by time for critic prediction diagnostics; all policy evaluation uses disjoint reset seeds",
        },
        "claim_scope": "shared-symbolizer, fixed discrete Gymnasium tasks; composition comparison is empirical, with approximate fitted-MC GPI and no universal auditability claim",
    }
    (task_root / "shared_multiseed_gpi_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1] / "runs" / "shared_multienv_gpi")
    parser.add_argument("--environments", nargs="+", default=["CartPole-v1", "Acrobot-v1"])
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--calibration-episodes", type=int, default=24)
    parser.add_argument("--evaluation-episodes", type=int, default=100)
    parser.add_argument("--state-bins", type=int, default=4)
    parser.add_argument("--min-support", type=int, default=8)
    parser.add_argument("--min-confidence", type=float, default=0.70)
    args = parser.parse_args()
    all_results = []
    for environment_name in args.environments:
        all_results.append(run_task(
            environment_name, args.output_root, args.seeds, args.episodes,
            args.calibration_episodes, args.state_bins, args.min_support,
            args.min_confidence, args.evaluation_episodes,
        ))
    compact = [{
        "environment": result["environment"],
        "grammar_core": {
            key: value for key, value in result["grammar_core"].items()
            if key not in {"all_seed_core_rules", "per_seed_rule_counts", "seeds"}
        },
        "composition_mean_returns": {
            policy: details["mean_return"]
            for policy, details in result["composition_evaluation"]["policies"].items()
        },
        "paired_differences_vs_best_single": result["composition_evaluation"]["paired_bootstrap_difference_vs_best_single"],
        "audit_integrity": {
            key: value for key, value in result["audit_integrity"].items()
            if key not in {"one_lossless_trace_codec"}
        },
        "summary_path": str(args.output_root / result["environment"] / "shared_multiseed_gpi_summary.json"),
    } for result in all_results]
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
