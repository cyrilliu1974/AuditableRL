"""Fuse two seed-specific CartPole rule grammars with explicit provenance and blind spots."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import statistics
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from auditability.grammar_audit_experiment import (
    Actor,
    StateSymbolizer,
    ZERO_HASH,
    canonical_json,
    chain_record,
    sha256_file,
)


def load_agent(run_dir: Path) -> dict:
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    grammar_path = run_dir / "induced_grammar.json"
    grammar = json.loads(grammar_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(run_dir / "actor.pt", map_location="cpu", weights_only=True)
    actor = Actor(state_dim=4)
    actor.load_state_dict(checkpoint["state_dict"])
    actor.eval()
    symbolizer = StateSymbolizer(metadata["config"]["state_bins"]["n_bins"])
    symbolizer.edges = torch.tensor(metadata["config"]["state_bins"]["edges"], dtype=torch.float64)
    rules = {}
    for production in grammar.get("productions", []):
        state_symbol, action_symbol = production["rhs"]
        rules[state_symbol] = {
            "action": int(action_symbol.removeprefix("ACTION_")),
            "support": int(production["support"]),
            "confidence": float(production["confidence"]),
            "mean_reward": float(production["mean_reward"]),
            "source_step_ids": production.get("source_step_ids", []),
            "source_step_ids_sha256": hashlib.sha256(
                canonical_json(production.get("source_step_ids", [])).encode("utf-8")
            ).hexdigest(),
            "source_run": str(run_dir),
            "seed": int(metadata["config"]["seed"]),
            "state_symbol": state_symbol,
            "rule_id": f"seed-{metadata['config']['seed']}:{state_symbol}->{action_symbol}",
        }
    return {
        "run_dir": run_dir,
        "seed": int(metadata["config"]["seed"]),
        "metadata": metadata,
        "grammar_path": grammar_path,
        "grammar_sha256": sha256_file(grammar_path),
        "actor": actor,
        "actor_sha256": metadata["actor_sha256"],
        "symbolizer": symbolizer,
        "rules": rules,
    }


def neural_action(actor: Actor, observation: np.ndarray) -> int:
    with torch.inference_mode():
        return int(actor(torch.as_tensor(observation, dtype=torch.float32)).argmax().item())


def source_rule_reference(agent: dict, rule: dict) -> dict:
    return {
        "rule_id": rule["rule_id"],
        "source_run": rule["source_run"],
        "source_seed": rule["seed"],
        "grammar_sha256": agent["grammar_sha256"],
        "state_symbol": rule["state_symbol"],
        "action": rule["action"],
        "source_step_count": len(rule["source_step_ids"]),
        "source_step_ids_sha256": rule["source_step_ids_sha256"],
        "support": rule["support"],
        "confidence": rule["confidence"],
        "mean_reward": rule["mean_reward"],
    }


def select_merged_action(observation: np.ndarray, left: dict, right: dict, fallback: dict) -> tuple[int, dict]:
    left_state = "STATE_" + "_".join(map(str, left["symbolizer"].encode(observation)))
    right_state = "STATE_" + "_".join(map(str, right["symbolizer"].encode(observation)))
    left_rule = left["rules"].get(left_state)
    right_rule = right["rules"].get(right_state)
    event = {
        "left_state_symbol": left_state,
        "right_state_symbol": right_state,
        "left_rule_id": None if left_rule is None else left_rule["rule_id"],
        "right_rule_id": None if right_rule is None else right_rule["rule_id"],
        "conflict": False,
        "blind_spot": False,
        "source_agent_seeds": [],
        "source_rule_ids": [],
        "source_rule_provenance": [],
        "candidate_rule_provenance": [
            source_rule_reference(agent, rule)
            for agent, rule in ((left, left_rule), (right, right_rule))
            if rule is not None
        ],
        "fallback_actor_seed": None,
    }
    candidates = [(agent, rule) for agent, rule in ((left, left_rule), (right, right_rule)) if rule is not None]
    if len(candidates) == 2 and left_rule["action"] == right_rule["action"]:
        chosen_action = left_rule["action"]
        chosen = candidates
        event["decision_source"] = "both_agents_agree"
    elif len(candidates) == 2:
        event["conflict"] = True
        priority = lambda pair: (
            pair[1]["confidence"], pair[1]["support"], pair[1]["mean_reward"]
        )
        chosen = sorted(candidates, key=priority, reverse=True)[:1]
        chosen_action = chosen[0][1]["action"]
        if priority(candidates[0]) == priority(candidates[1]):
            chosen_action = neural_action(fallback["actor"], observation)
            chosen = []
            event["decision_source"] = "conflict_tie_fallback_actor"
            event["fallback_actor_seed"] = fallback["seed"]
        else:
            event["decision_source"] = "conflict_arbitrated_by_confidence_support_reward"
    elif len(candidates) == 1:
        chosen = candidates
        chosen_action = candidates[0][1]["action"]
        event["decision_source"] = "single_agent_rule"
    else:
        event["blind_spot"] = True
        chosen = []
        chosen_action = neural_action(fallback["actor"], observation)
        event["decision_source"] = "blind_spot_fallback_actor"
        event["fallback_actor_seed"] = fallback["seed"]

    event["source_agent_seeds"] = [agent["seed"] for agent, _ in chosen]
    event["source_rule_ids"] = [rule["rule_id"] for _, rule in chosen]
    event["source_rule_provenance"] = [source_rule_reference(agent, rule) for agent, rule in chosen]
    return int(chosen_action), event


def probe_source_traces(left: dict, right: dict, fallback: dict, max_examples: int = 8) -> dict:
    """Count merge decisions on each agent's logged training states and retain real conflicts."""
    summary = {}
    for source in (left, right):
        counts = collections.Counter()
        conflict_examples = []
        trace_path = source["run_dir"] / "trajectory.jsonl"
        with trace_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                action, decision = select_merged_action(np.asarray(row["state"], dtype=np.float32), left, right, fallback)
                counts["steps"] += 1
                counts[decision["decision_source"]] += 1
                counts["conflicts"] += decision["conflict"]
                counts["blind_spots"] += decision["blind_spot"]
                if decision["conflict"] and len(conflict_examples) < max_examples:
                    conflict_examples.append({
                        "source_trace_seed": source["seed"],
                        "episode": row["episode"],
                        "step": row["step"],
                        "raw_state": row["state"],
                        "left_rule_id": decision["left_rule_id"],
                        "right_rule_id": decision["right_rule_id"],
                        "chosen_action": action,
                        "resolution": decision["decision_source"],
                        "chosen_source_rule_ids": decision["source_rule_ids"],
                    })
        summary[f"seed_{source['seed']}_training_states"] = {
            "trace_path": str(trace_path),
            "counts": dict(counts),
            "conflict_examples": conflict_examples,
        }
    return summary


def replay_merged_trace(trace_path: Path) -> dict:
    env = gym.make("CartPole-v1")
    transitions = 0
    previous_hash = ZERO_HASH
    current_episode = -1
    expected_step = 0
    ended = True
    with trace_path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            row = json.loads(line)
            claimed_hash = row.pop("record_hash")
            if row.get("previous_hash") != previous_hash:
                raise ValueError(f"line {line_no}: hash-chain previous-hash mismatch")
            actual_hash = hashlib.sha256((previous_hash + canonical_json(row)).encode("utf-8")).hexdigest()
            if actual_hash != claimed_hash:
                raise ValueError(f"line {line_no}: hash-chain record-hash mismatch")
            row["record_hash"] = claimed_hash
            previous_hash = claimed_hash
            episode = int(row["episode"])
            if episode != current_episode:
                if current_episode >= 0 and not ended:
                    raise ValueError(f"line {line_no}: previous episode did not terminate")
                if int(row["step"]) != 0:
                    raise ValueError(f"line {line_no}: first episode row has nonzero step")
                observation, _ = env.reset(seed=int(row["reset_seed"]))
                current_episode = episode
                expected_step = 0
            if int(row["step"]) != expected_step:
                raise ValueError(f"line {line_no}: step discontinuity")
            if np.asarray(observation, dtype=np.float32).tolist() != row["state"]:
                raise ValueError(f"line {line_no}: state mismatch")
            observation, reward, terminated, truncated, _ = env.step(int(row["action"]))
            if np.asarray(observation, dtype=np.float32).tolist() != row["next_state"]:
                raise ValueError(f"line {line_no}: next state mismatch")
            if float(reward) != float(row["reward"]):
                raise ValueError(f"line {line_no}: reward mismatch")
            if bool(terminated) != row["terminated"] or bool(truncated) != row["truncated"]:
                raise ValueError(f"line {line_no}: termination/truncation mismatch")
            ended = bool(terminated or truncated)
            expected_step += 1
            transitions += 1
    env.close()
    return {
        "valid": True,
        "transitions": transitions,
        "episodes": current_episode + 1,
        "hash_chain_final": previous_hash,
    }


def verify_rule_provenance(trace_path: Path, left: dict, right: dict, fallback: dict) -> dict:
    agents = {agent["seed"]: agent for agent in (left, right)}
    decisions = rule_references = fallback_references = 0
    with trace_path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            row = json.loads(line)
            decisions += 1
            refs = row["source_rule_provenance"]
            candidates = row["candidate_rule_provenance"]
            if len(refs) != len(row["source_rule_ids"]):
                raise ValueError(f"line {line_no}: rule IDs and provenance entries differ in length")
            if row["blind_spot"] and refs:
                raise ValueError(f"line {line_no}: blind-spot decision has a claimed rule source")
            if row["blind_spot"] != (len(candidates) == 0):
                raise ValueError(f"line {line_no}: candidate rules disagree with blind-spot flag")
            if row["fallback_actor_seed"] is not None:
                if row["fallback_actor_seed"] != fallback["seed"]:
                    raise ValueError(f"line {line_no}: fallback actor seed mismatch")
                fallback_references += 1
            actions_by_rule = {}
            for candidate in candidates:
                seed = int(candidate["source_seed"])
                agent = agents.get(seed)
                if agent is None or candidate["grammar_sha256"] != agent["grammar_sha256"]:
                    raise ValueError(f"line {line_no}: unknown candidate agent or grammar hash")
                rule = next((item for item in agent["rules"].values() if item["rule_id"] == candidate["rule_id"]), None)
                if rule is None or candidate["source_step_ids_sha256"] != rule["source_step_ids_sha256"]:
                    raise ValueError(f"line {line_no}: candidate rule provenance does not resolve")
                if candidate["support"] != rule["support"] or candidate["confidence"] != rule["confidence"]:
                    raise ValueError(f"line {line_no}: candidate rule statistics mismatch")
                actions_by_rule[candidate["rule_id"]] = rule["action"]
            if row["conflict"] != (len(actions_by_rule) == 2 and len(set(actions_by_rule.values())) > 1):
                raise ValueError(f"line {line_no}: candidate actions disagree with conflict flag")
            actions = []
            for ref, rule_id in zip(refs, row["source_rule_ids"]):
                seed = int(ref["source_seed"])
                agent = agents.get(seed)
                if agent is None or ref["grammar_sha256"] != agent["grammar_sha256"]:
                    raise ValueError(f"line {line_no}: unknown source agent or grammar hash")
                rule = next((item for item in agent["rules"].values() if item["rule_id"] == rule_id), None)
                if rule is None:
                    raise ValueError(f"line {line_no}: source rule ID does not resolve")
                if ref["source_step_ids_sha256"] != rule["source_step_ids_sha256"]:
                    raise ValueError(f"line {line_no}: source-step provenance hash mismatch")
                if int(ref["source_step_count"]) != len(rule["source_step_ids"]):
                    raise ValueError(f"line {line_no}: source-step provenance count mismatch")
                if int(ref["support"]) != rule["support"] or float(ref["confidence"]) != rule["confidence"]:
                    raise ValueError(f"line {line_no}: copied rule statistics mismatch")
                actions.append(rule["action"])
                rule_references += 1
                if rule_id not in actions_by_rule:
                    raise ValueError(f"line {line_no}: selected rule was not among the candidates")
            if actions and any(action != row["action"] for action in actions):
                raise ValueError(f"line {line_no}: selected source rule does not support emitted action")
    return {
        "valid": True,
        "decisions_checked": decisions,
        "source_rule_references_resolved": rule_references,
        "fallback_agent_references_checked": fallback_references,
    }


def evaluate_merge(left: dict, right: dict, episodes: int, output_root: Path, reset_seed_base: int) -> dict:
    # Use training return only to preselect a deterministic fallback; evaluation returns do not affect arbitration.
    fallback = max(
        (left, right),
        key=lambda agent: agent["metadata"]["episode_return_mean"],
    )
    policies = ("agent_left", "agent_right", "merged")
    returns = {policy: [] for policy in policies}
    mix = collections.Counter()
    trace_path = output_root / "merged_policy_trace.jsonl"
    previous_hash = ZERO_HASH
    env = gym.make("CartPole-v1")

    with trace_path.open("w", encoding="utf-8", newline="\n") as trace:
        for episode in range(episodes):
            seed = reset_seed_base + episode
            for policy in policies:
                observation, _ = env.reset(seed=seed)
                total_return = 0.0
                step = 0
                terminated = truncated = False
                while not (terminated or truncated):
                    observation = np.asarray(observation, dtype=np.float32)
                    event: dict = {}
                    if policy == "agent_left":
                        action = neural_action(left["actor"], observation)
                    elif policy == "agent_right":
                        action = neural_action(right["actor"], observation)
                    else:
                        action, event = select_merged_action(observation, left, right, fallback)
                        mix["total_steps"] += 1
                        mix["conflicts"] += event["conflict"]
                        mix["blind_spots"] += event["blind_spot"]
                        mix[event["decision_source"]] += 1
                        for source_seed in event["source_agent_seeds"]:
                            mix[f"rule_decisions_seed_{source_seed}"] += 1
                    next_observation, reward, terminated, truncated, _ = env.step(action)
                    total_return += float(reward)
                    if policy == "merged":
                        record = {
                            "episode": episode,
                            "step": step,
                            "global_step": mix["total_steps"] - 1,
                            "reset_seed": seed if step == 0 else None,
                            "state": observation.tolist(),
                            "state_symbol_left": event["left_state_symbol"],
                            "state_symbol_right": event["right_state_symbol"],
                            "action": int(action),
                            "next_state": np.asarray(next_observation, dtype=np.float32).tolist(),
                            "reward": float(reward),
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                            **event,
                        }
                        chained = chain_record(record, previous_hash)
                        trace.write(canonical_json(chained) + "\n")
                        previous_hash = chained["record_hash"]
                    observation = next_observation
                    step += 1
                returns[policy].append(total_return)
    env.close()

    audit = replay_merged_trace(trace_path)
    provenance_audit = verify_rule_provenance(trace_path, left, right, fallback)
    source_probe = probe_source_traces(left, right, fallback)
    policies_summary = {
        policy: {
            "mean_return": statistics.mean(values),
            "median_return": statistics.median(values),
            "return_by_episode": values,
        }
        for policy, values in returns.items()
    }
    result = {
        "claim_scope": "deterministic CartPole-v1 rule-fusion prototype; same-host evaluation, not evidence of general transfer or improved RL training",
        "source_agents": [
            {"seed": agent["seed"], "run_dir": str(agent["run_dir"]), "actor_sha256": agent["actor_sha256"],
             "grammar_sha256": agent["grammar_sha256"], "rule_count": len(agent["rules"])}
            for agent in (left, right)
        ],
        "fallback_agent_seed_selected_by_training_mean_return": fallback["seed"],
        "fallback_selection_metric": "episode_return_mean from training metadata; no held-out return used for arbitration",
        "grammar_arbitration": "same-action agreement; conflicts ordered by confidence then support then mean reward, exact ties fall back; no-rule states are explicit blind spots and use the preselected neural fallback",
        "evaluation_seed_formula": "reset_seed_base + episode_index",
        "evaluation_episodes": episodes,
        "evaluation_policies": policies_summary,
        "paired_bootstrap_differences_vs_left": {},
        "merged_decision_mix": dict(mix),
        "source_training_state_merge_probe": source_probe,
        "fusion_code_sha256": sha256_file(Path(__file__).resolve()),
        "merged_trace_sha256": sha256_file(trace_path),
        "merged_trace_hash_chain_final": previous_hash,
        "merged_trace_environment_replay": audit,
        "merged_rule_provenance_audit": provenance_audit,
        "merged_trace_path": str(trace_path),
    }
    rng = random.Random(20260922)
    for policy in ("agent_right", "merged"):
        deltas = [a - b for a, b in zip(returns[policy], returns["agent_left"])]
        bootstrap = []
        for _ in range(5000):
            sample = [deltas[rng.randrange(len(deltas))] for _ in deltas]
            bootstrap.append(statistics.mean(sample))
        bootstrap.sort()
        result["paired_bootstrap_differences_vs_left"][policy] = {
            "mean_delta": statistics.mean(deltas),
            "ci95_percentile": [bootstrap[125], bootstrap[4874]],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-run", type=Path, required=True)
    parser.add_argument("--right-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--reset-seed-base", type=int, default=9_000_000)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    left = load_agent(args.left_run)
    right = load_agent(args.right_run)
    result = evaluate_merge(left, right, args.episodes, args.output_root, args.reset_seed_base)
    result_path = args.output_root / "merge_evaluation.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
