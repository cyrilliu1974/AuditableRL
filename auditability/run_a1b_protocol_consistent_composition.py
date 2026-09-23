"""A1b - protocol-consistent composition.

A1 established two facts about the recorded pipeline:

  * training selects actions by sampling, evaluation and deployment use argmax;
  * on Acrobot the gap is catastrophic - argmax returns exactly -500.00 for all
    eight saved actors, while sampling averages -446.12.

Every recorded composition number for Acrobot (-443.37 fusion, -187.02 random
arbitration, -500.00 GPI) was therefore measured under argmax, in a regime where
all eight base actors already sit at the floor - **while the rule banks those
compositions consume were induced from sampled rollouts**.  The banks and the
evaluation disagree about the action rule.

This script closes that gap with a 2x2: {bank induced from sampled actions, bank
induced from greedy actions} x {evaluation under argmax, evaluation under
sampling}.  The corner (sampled bank, argmax evaluation) is the recorded
configuration and acts as a harness self-test.

Also reports the base policies under both evaluation protocols, so the
composition can be compared against the regime it is actually being run in.

Output: ``results/a1b_protocol_consistent_composition.json``
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import statistics
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from auditability.run_shared_multienv_gpi import load_actor

SEEDS = [11, 29, 43, 71, 101, 149, 211, 307]
HELDOUT_BASE = 20_000_000
HELDOUT_EPISODES = 100


def load_symbolizer(task_root: Path):
    from auditability.grammar_audit_experiment import StateSymbolizer
    calibration = json.loads((task_root / "shared_calibration.json").read_text(encoding="utf-8"))
    symbolizer = StateSymbolizer(calibration["state_bins"]["n_bins"])
    symbolizer.edges = torch.tensor(calibration["state_bins"]["edges"], dtype=torch.float64)
    return symbolizer


def induce_bank(trace_path: Path, label: str, min_support: int, min_confidence: float,
                source_seed: int, run_dir: Path) -> dict[tuple, dict]:
    """Re-induce a rule bank from a recorded trace under one action-label rule."""
    counts: dict[tuple, collections.Counter] = collections.defaultdict(collections.Counter)
    rewards: dict[tuple, float] = collections.defaultdict(float)
    with trace_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            condition = tuple(int(part) for part in row["state_symbol"])
            sampled = int(row["action"])
            action = sampled if label == "sampled" else int(np.argmax(
                row["action_probabilities_after_constraint"]))
            counts[condition][action] += 1
            rewards[(condition, action)] += float(row["reward"])
    bank: dict[tuple, dict] = {}
    for condition, action_counts in counts.items():
        total = sum(action_counts.values())
        action, count = min(action_counts.items(), key=lambda item: (-item[1], item[0]))
        confidence = count / total if total else 0.0
        if count >= min_support and confidence >= min_confidence:
            bank[condition] = {
                "action": action, "support": count, "confidence": confidence,
                "mean_reward": rewards[(condition, action)] / count,
                "source_seed": source_seed, "source_run": str(run_dir),
            }
    return bank


def ensemble_logits(actors, observation):
    state = torch.as_tensor(observation, dtype=torch.float32)
    with torch.inference_mode():
        return torch.stack([actor(state) for actor in actors]).mean(dim=0)


def pick(logits, protocol: str, generator) -> int:
    if protocol == "argmax":
        return int(logits.argmax().item())
    return int(torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=generator).item())


def decide(policy: str, observation, actors, best_index, bank, protocol, generator, rng,
           fallback_actors):
    """Return (action, detail). ``bank`` is None for base policies."""
    if policy == "best_single":
        with torch.inference_mode():
            logits = actors[best_index](torch.as_tensor(observation, dtype=torch.float32))
        return pick(logits, protocol, generator), {"covered": False, "conflict": False}
    if policy == "actor_mean_logits":
        return pick(ensemble_logits(actors, observation), protocol, generator), {
            "covered": False, "conflict": False}
    # Composition policies: the bank arbitrates, the base protocol handles blind spots.
    condition = None
    if bank is not None:
        from auditability.grammar_audit_experiment import StateSymbolizer  # noqa: F401
    condition = bank_key(observation)
    candidates = [b[condition] for b in bank if condition in b] if bank else []
    if not candidates:
        return pick(ensemble_logits(fallback_actors, observation), protocol, generator), {
            "covered": False, "conflict": False}
    actions = sorted({rule["action"] for rule in candidates})
    detail = {"covered": True, "conflict": len(actions) > 1}
    if policy == "random_candidate":
        return int(rng.choice(candidates)["action"]), detail
    if policy == "grammar_fusion":
        chosen = max(candidates, key=lambda rule: (rule["confidence"], rule["support"],
                                                   rule["mean_reward"], -rule["source_seed"]))
        return int(chosen["action"]), detail
    raise ValueError(policy)


_BANK_KEY_FN = None


def bank_key(observation):
    return _BANK_KEY_FN(observation)


def run_policy(policy: str, protocol: str, episodes: int, env_name: str, actors, best_index,
               bank, action_dim) -> dict:
    returns = []
    counts = {"steps": 0, "covered": 0, "conflicts": 0}
    histogram = {str(a): 0 for a in range(action_dim)}
    for episode in range(episodes):
        env = gym.make(env_name)
        generator = torch.Generator().manual_seed(700_000 + episode * 31 + len(policy))
        rng = random.Random(20260922 + episode)
        observation, _ = env.reset(seed=HELDOUT_BASE + episode)
        total = 0.0
        done = False
        while not done:
            action, detail = decide(policy, observation, actors, best_index, bank, protocol,
                                    generator, rng, actors)
            counts["steps"] += 1
            counts["covered"] += int(detail["covered"])
            counts["conflicts"] += int(detail["conflict"])
            histogram[str(action)] += 1
            observation, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            done = terminated or truncated
        env.close()
        returns.append(total)
    return {"policy": policy, "protocol": protocol,
            "mean_return": statistics.mean(returns),
            "median_return": statistics.median(returns),
            "std_return": statistics.stdev(returns) if len(returns) > 1 else 0.0,
            "return_by_episode": returns, "decision_counts": counts,
            "action_histogram": histogram}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs/shared_multienv_gpi"))
    parser.add_argument("--environments", nargs="+", default=["Acrobot-v1", "CartPole-v1"])
    parser.add_argument("--episodes", type=int, default=HELDOUT_EPISODES)
    parser.add_argument("--output", type=Path,
                        default=Path("results/a1b_protocol_consistent_composition.json"))
    args = parser.parse_args()

    torch.set_num_threads(1)
    global _BANK_KEY_FN
    results = {"episodes": args.episodes,
               "reset_seed_formula": f"{HELDOUT_BASE} + episode_index", "environments": {}}

    for env_name in args.environments:
        task_root = (args.runs_root / env_name).resolve()
        calibration = json.loads((task_root / "shared_calibration.json").read_text(encoding="utf-8"))
        state_dim = len(calibration["state_bins"]["edges"])
        probe = gym.make(env_name)
        action_dim = int(probe.action_space.n)
        probe.close()
        symbolizer = load_symbolizer(task_root)
        _BANK_KEY_FN = symbolizer.encode

        summary = json.loads((task_root / "shared_multiseed_gpi_summary.json").read_text(encoding="utf-8"))
        best_seed = int(summary["composition_evaluation"]["fallback_best_single_seed"])
        best_index = SEEDS.index(best_seed)

        runs = []
        for seed in SEEDS:
            matches = [p for p in (task_root / "grammar_audit_only").glob(f"seed-{seed}_*")
                       if (p / "metadata.json").is_file()]
            if len(matches) != 1:
                raise ValueError(f"expected one complete run for seed {seed}")
            runs.append(matches[0])
        actors = [load_actor(path, state_dim, action_dim) for path in runs]

        metadata = json.loads((runs[0] / "metadata.json").read_text(encoding="utf-8"))
        min_support = metadata["config"]["grammar_min_support"]
        min_confidence = metadata["config"]["grammar_min_confidence"]

        print(f"=== {env_name} (best_single seed {best_seed}) ===", flush=True)
        print("  inducing banks from recorded traces ...", flush=True)
        start = time.perf_counter()
        banks = {"sampled": [], "greedy": []}
        for seed, run_dir in zip(SEEDS, runs):
            trace = run_dir / "trajectory.jsonl"
            for label in ("sampled", "greedy"):
                banks[label].append(induce_bank(trace, label, min_support, min_confidence,
                                                seed, run_dir))
        print(f"  banks induced in {time.perf_counter() - start:.0f}s; "
              f"sampled {sum(len(b) for b in banks['sampled'])} rules, "
              f"greedy {sum(len(b) for b in banks['greedy'])} rules", flush=True)

        block = {"best_single_seed": best_seed,
                 "bank_rule_totals": {label: sum(len(b) for b in blist)
                                      for label, blist in banks.items()},
                 "base_policies": {}, "composition": {}}
        for policy in ("best_single", "actor_mean_logits"):
            for protocol in ("argmax", "sampled"):
                row = run_policy(policy, protocol, args.episodes, env_name, actors, best_index,
                                 None, action_dim)
                block["base_policies"][f"{policy}__{protocol}"] = row
                print(f"  {policy:18s} {protocol:8s} mean {row['mean_return']:9.2f}", flush=True)
        for label, bank in banks.items():
            for policy in ("grammar_fusion", "random_candidate"):
                for protocol in ("argmax", "sampled"):
                    row = run_policy(policy, protocol, args.episodes, env_name, actors, best_index,
                                     bank, action_dim)
                    block["composition"][f"{policy}__{label}_bank__{protocol}"] = row
                    print(f"  {policy:18s} {label:6s} bank {protocol:8s} "
                          f"mean {row['mean_return']:9.2f} "
                          f"steps {row['decision_counts']['steps']}", flush=True)
        results["environments"][env_name] = block

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    recorded = {}
    for env_name in args.environments:
        task_root = (args.runs_root / env_name).resolve()
        summ = json.loads((task_root / "shared_multiseed_gpi_summary.json").read_text(encoding="utf-8"))
        entry = {
            "best_single": summ["composition_evaluation"]["policies"]["best_single"]["mean_return"],
            "actor_mean_logits": summ["composition_evaluation"]["policies"]["actor_mean_logits"]["mean_return"],
            "grammar_fusion": None,
            "random_candidate": None,
        }
        diag_path = task_root / "grammar_fusion_diagnostics.json"
        if diag_path.is_file():
            diag = json.loads(diag_path.read_text(encoding="utf-8"))
            entry["grammar_fusion"] = diag["ablations"]["confidence_first"]["mean_return"]
            entry["random_candidate"] = diag["ablations"]["random_candidate"]["mean_return"]
        recorded[env_name] = entry
    print("\n=== self-test: (sampled bank, argmax) must reproduce the recorded deterministic numbers ===")
    ok = True
    for env_name in args.environments:
        block = results["environments"][env_name]
        got = {
            "best_single": block["base_policies"]["best_single__argmax"]["mean_return"],
            "actor_mean_logits": block["base_policies"]["actor_mean_logits__argmax"]["mean_return"],
            "grammar_fusion": block["composition"]["grammar_fusion__sampled_bank__argmax"]["mean_return"],
        }
        for key in ("best_single", "actor_mean_logits", "grammar_fusion"):
            want = recorded[env_name].get(key)
            if want is None:
                print(f"  {env_name:12s} {key:18s} recorded diagnostics unavailable for this env; skipping assertion")
                continue
            match = abs(got[key] - want) < 1e-6
            ok = ok and match
            print(f"  {env_name:12s} {key:18s} rerun {got[key]:9.2f} recorded {want:9.2f} {match}")
        # random arbitration depends on RNG draw order; the orchestrated composition
        # run interleaves all four policies inside one episode loop, whereas this
        # re-run steps random_candidate independently, so the value cannot match
        # exactly. Report it for reference but do NOT assert it.
        rc_recorded = recorded[env_name].get("random_candidate")
        rc_rerun = block["composition"]["random_candidate__sampled_bank__argmax"]["mean_return"]
        if rc_recorded is None:
            print(f"  {env_name:12s} {'random_candidate':18s} rerun {rc_rerun:9.2f} recorded diagnostics unavailable; skipped")
        else:
            print(f"  {env_name:12s} {'random_candidate':18s} rerun {rc_rerun:9.2f} "
                  f"recorded {rc_recorded:9.2f} (RNG order differs, not asserted)")
    results["harness_self_test"] = {"recorded": recorded, "passed": ok}
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nself-test passed: {ok}")


if __name__ == "__main__":
    main()
