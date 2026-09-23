"""A1 - deploy-protocol check.

Two questions, both answerable from artifacts already on disk.

Part 1 (returns).  Training selects actions by sampling from the categorical
distribution; evaluation and deployment use argmax.  If the two disagree, every
reported number in the paper describes a policy that was never trained.  This
part evaluates the same eight saved actors on the same 100 held-out reset seeds
under both protocols and reports both.

Part 2 (rule banks).  The induced bank is a table of majority actions observed
during training.  If training sampled while deployment is greedy, the bank
summarises a behaviour the deployed policy does not exhibit.  This part
re-induces each bank twice from the *same* recorded trace - once from the
sampled action, once from the greedy action recovered from the logged
action probabilities - so the only thing that changes is the action label.

Output: ``results/a1_deploy_protocol.json``
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch


from auditability.grammar_audit_experiment import StateActionGrammar, StateSymbolizer
from auditability.run_shared_multienv_gpi import load_actor

SEEDS = [11, 29, 43, 71, 101, 149, 211, 307]
HELDOUT_BASE = 20_000_000
HELDOUT_EPISODES = 100


def load_symbolizer(task_root: Path) -> StateSymbolizer:
    calibration = json.loads((task_root / "shared_calibration.json").read_text(encoding="utf-8"))
    symbolizer = StateSymbolizer(calibration["state_bins"]["n_bins"])
    symbolizer.edges = torch.tensor(calibration["state_bins"]["edges"], dtype=torch.float64)
    return symbolizer


def find_run(task_root: Path, seed: int) -> Path:
    matches = [p for p in (task_root / "grammar_audit_only").glob(f"seed-{seed}_*")
               if (p / "metadata.json").is_file()]
    if len(matches) != 1:
        raise ValueError(f"expected one complete audit-only run for seed {seed}; got {len(matches)}")
    return matches[0]


def evaluate(actor, env_name: str, protocol: str, seed: int, episodes: int,
             reset_base: int) -> list[float]:
    """One actor, one protocol, the same reset seeds for every protocol."""
    env = gym.make(env_name)
    returns: list[float] = []
    generator = torch.Generator().manual_seed(900_000 + seed)
    for episode in range(episodes):
        observation, _ = env.reset(seed=reset_base + episode)
        total = 0.0
        done = False
        while not done:
            state_t = torch.as_tensor(observation, dtype=torch.float32)
            with torch.inference_mode():
                logits = actor(state_t)
            if protocol == "argmax":
                action = int(logits.argmax().item())
            elif protocol == "sampled":
                # Categorical(logits).sample() has no generator argument; multinomial
                # on the softmax is the same draw and is seedable.
                action = int(torch.multinomial(torch.softmax(logits, dim=-1), 1,
                                               generator=generator).item())
            else:
                raise ValueError(protocol)
            observation, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            done = terminated or truncated
        returns.append(total)
    env.close()
    return returns


def induce_from_trace(trace_path: Path, symbolizer: StateSymbolizer, label: str,
                      min_support: int, min_confidence: float) -> dict:
    """Re-induce the bank from a recorded trace using one action-label rule.

    ``label="sampled"`` uses the action the run actually took.  ``label="greedy"``
    uses argmax of the logged post-constraint action probabilities, i.e. what the
    same network would have done at deployment.
    """
    grammar = StateActionGrammar(min_support=min_support, min_confidence=min_confidence)
    visited: collections.Counter = collections.Counter()
    greedy_differs = 0
    records = 0
    with trace_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            condition = tuple(int(part) for part in row["state_symbol"])
            visited[condition] += 1
            sampled_action = int(row["action"])
            probs = row["action_probabilities_after_constraint"]
            greedy_action = int(np.argmax(probs))
            if greedy_action != sampled_action:
                greedy_differs += 1
            action = sampled_action if label == "sampled" else greedy_action
            grammar.observe(condition, action, float(row["reward"]), records)
            records += 1
    rules = grammar.induce()
    return {
        "records": records,
        "distinct_conditions_visited": len(visited),
        "rules": {condition: {"action": rule["action"], "support": rule["support"],
                              "confidence": rule["confidence"]}
                  for condition, rule in rules.items()},
        "n_rules": len(rules),
        "greedy_differs_from_sampled": greedy_differs,
        "greedy_disagreement_rate": greedy_differs / records if records else 0.0,
    }


def cross_seed_conflict(banks: list[dict]) -> dict:
    """How often do the eight banks disagree about the action for a condition?"""
    table: dict[tuple, set] = collections.defaultdict(set)
    present: collections.Counter = collections.Counter()
    for bank in banks:
        for condition, rule in bank.items():
            table[condition].add(rule["action"])
            present[condition] += 1
    shared = {c: actions for c, actions in table.items() if present[c] >= 2}
    conflicting = {c: actions for c, actions in shared.items() if len(actions) > 1}
    return {
        "conditions_in_any_bank": len(table),
        "conditions_in_two_or_more_banks": len(shared),
        "conflicting_conditions": len(conflicting),
        "conflict_rate": len(conflicting) / len(shared) if shared else 0.0,
    }


def analyse_environment(task_root: Path, env_name: str, args) -> dict:
    symbolizer = load_symbolizer(task_root)
    metadata = json.loads((find_run(task_root, SEEDS[0]) / "metadata.json").read_text(encoding="utf-8"))
    config = metadata["config"]
    state_dim = config["state_dim"]
    action_dim = config["action_dim"]

    per_seed = {}
    for seed in SEEDS:
        run_dir = find_run(task_root, seed)
        actor = load_actor(run_dir, state_dim, action_dim)
        argmax_returns = evaluate(actor, env_name, "argmax", seed, HELDOUT_EPISODES, HELDOUT_BASE)
        sampled_returns = evaluate(actor, env_name, "sampled", seed, HELDOUT_EPISODES, HELDOUT_BASE)
        deltas = [a - b for a, b in zip(sampled_returns, argmax_returns)]
        per_seed[seed] = {
            "argmax_mean": statistics.mean(argmax_returns),
            "argmax_median": statistics.median(argmax_returns),
            "sampled_mean": statistics.mean(sampled_returns),
            "sampled_median": statistics.median(sampled_returns),
            "paired_sampled_minus_argmax": statistics.mean(deltas),
            "argmax_return_by_episode": argmax_returns,
            "sampled_return_by_episode": sampled_returns,
        }

    argmax_means = [row["argmax_mean"] for row in per_seed.values()]
    sampled_means = [row["sampled_mean"] for row in per_seed.values()]

    bank_comparison = {}
    sampled_banks, greedy_banks = [], []
    for seed in SEEDS:
        run_dir = find_run(task_root, seed)
        trace = run_dir / "trajectory.jsonl"
        sampled = induce_from_trace(trace, symbolizer, "sampled",
                                    config["grammar_min_support"], config["grammar_min_confidence"])
        greedy = induce_from_trace(trace, symbolizer, "greedy",
                                   config["grammar_min_support"], config["grammar_min_confidence"])
        recorded = json.loads((run_dir / "induced_grammar.json").read_text(encoding="utf-8"))
        recorded_rules = {}
        for production in recorded.get("productions", []):
            state_symbol, action_symbol = production["rhs"]
            condition = tuple(int(p) for p in state_symbol.removeprefix("STATE_").split("_"))
            recorded_rules[condition] = {"action": int(action_symbol.removeprefix("ACTION_")),
                                         "support": int(production["support"]),
                                         "confidence": float(production["confidence"])}
        sampled_banks.append(sampled["rules"])
        greedy_banks.append(greedy["rules"])
        bank_comparison[seed] = {
            "recorded_n_rules": len(recorded_rules),
            "resampled_n_rules": sampled["n_rules"],
            "regreedy_n_rules": greedy["n_rules"],
            "resampled_matches_recorded": recorded_rules == sampled["rules"],
            "records": sampled["records"],
            "distinct_conditions_visited": sampled["distinct_conditions_visited"],
            "greedy_disagreement_rate": sampled["greedy_disagreement_rate"],
            "recorded_coverage": len(recorded_rules) / sampled["distinct_conditions_visited"],
            "resampled_coverage": sampled["n_rules"] / sampled["distinct_conditions_visited"],
            "regreedy_coverage": greedy["n_rules"] / sampled["distinct_conditions_visited"],
        }

    return {
        "environment": env_name,
        "heldout_reset_seed_formula": f"{HELDOUT_BASE} + episode_index, identical for both protocols",
        "heldout_episodes": HELDOUT_EPISODES,
        "per_seed": per_seed,
        "protocol_summary": {
            "argmax_mean_across_seeds": statistics.mean(argmax_means),
            "sampled_mean_across_seeds": statistics.mean(sampled_means),
            "mean_paired_sampled_minus_argmax": statistics.mean(
                [row["paired_sampled_minus_argmax"] for row in per_seed.values()]),
            "seeds_where_sampled_better": sum(1 for row in per_seed.values()
                                              if row["paired_sampled_minus_argmax"] > 0),
        },
        "bank_comparison": bank_comparison,
        "bank_summary": {
            "resampled_total_rules": sum(b["resampled_n_rules"] for b in bank_comparison.values()),
            "regreedy_total_rules": sum(b["regreedy_n_rules"] for b in bank_comparison.values()),
            "recorded_total_rules": sum(b["recorded_n_rules"] for b in bank_comparison.values()),
            "mean_resampled_coverage": statistics.mean([b["resampled_coverage"] for b in bank_comparison.values()]),
            "mean_regreedy_coverage": statistics.mean([b["regreedy_coverage"] for b in bank_comparison.values()]),
            "mean_greedy_disagreement_rate": statistics.mean(
                [b["greedy_disagreement_rate"] for b in bank_comparison.values()]),
            "conflict_sampled_banks": cross_seed_conflict(sampled_banks),
            "conflict_greedy_banks": cross_seed_conflict(greedy_banks),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs/shared_multienv_gpi"))
    parser.add_argument("--output", type=Path, default=Path("results/a1_deploy_protocol.json"))
    parser.add_argument("--environments", nargs="+", default=["Acrobot-v1", "CartPole-v1"])
    args = parser.parse_args()

    torch.set_num_threads(1)
    output = {"environments": {}}
    for env_name in args.environments:
        task_root = (args.runs_root / env_name).resolve()
        output["environments"][env_name] = analyse_environment(task_root, env_name, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    for env_name, block in output["environments"].items():
        summary = block["protocol_summary"]
        banks = block["bank_summary"]
        print(f"\n=== {env_name} ===")
        print(f"  argmax mean across 8 seeds : {summary['argmax_mean_across_seeds']:.2f}")
        print(f"  sampled mean across 8 seeds: {summary['sampled_mean_across_seeds']:.2f}")
        print(f"  mean paired (sampled-argmax): {summary['mean_paired_sampled_minus_argmax']:.2f}")
        print(f"  seeds where sampled better : {summary['seeds_where_sampled_better']}/8")
        print(f"  rules recorded/resampled/regreedy: {banks['recorded_total_rules']}"
              f" / {banks['resampled_total_rules']} / {banks['regreedy_total_rules']}")
        print(f"  coverage resampled/regreedy: {banks['mean_resampled_coverage']:.4f}"
              f" / {banks['mean_regreedy_coverage']:.4f}")
        print(f"  greedy-vs-sampled disagreement rate: {banks['mean_greedy_disagreement_rate']:.4f}")
        print(f"  cross-seed conflict (sampled banks): {banks['conflict_sampled_banks']}")
        print(f"  cross-seed conflict (greedy banks) : {banks['conflict_greedy_banks']}")
        print(f"  resampled reproduces recorded for every seed: "
              f"{all(b['resampled_matches_recorded'] for b in block['bank_comparison'].values())}")


if __name__ == "__main__":
    main()
