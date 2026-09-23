"""A2 - rankability test.

Question.  The induced rule bank is uninformative under every *ranking* rule tried
so far (confidence, support, mean_reward all sit at or near the -500 floor), while
ranking-free random arbitration reaches -187.02.  Random wins because it
occasionally lands on the best action; the rankers never do.  Does the bank
contain rankable information, or only state-conditioned action diversity?

Method.  Replace the ranking function with a Monte-Carlo estimate of each
candidate action's value, on the same 100 held-out reset seeds as the recorded
ablations.

Result (measured before the policy-level runs, see ``results/mc_degeneracy_probe_8actor.json``).
The MC estimator is **degenerate** under the continuation policy the pipeline
actually uses.  ``mc_return_from_action`` values a candidate first action by
forcing it and then following the 8-actor mean-logit ensemble.  That ensemble
never terminates Acrobot within the 500-step horizon, so every forced first action
returns exactly -1 x 500 = -500 and the estimator cannot separate the actions:

    actor-rollout states   : 80/80 states tied at -500, spread 0.0
    always-action-0 states : 500/500 states tied at -500, spread 0.0

Consequences, both proved rather than assumed:

  * ``mc_ranked``  reduces to "take the smallest action any bank proposes"
    (``max`` over tied values returns the first element of the sorted candidate
    action list).
  * ``mc_oracle``  reduces to "always take action 0" (``max`` over range(3) with
    all values tied returns index 0).

So the expensive MC policy runs are unnecessary: the two degenerate modes are
evaluated here as their cheap deterministic equivalents, and the equivalence is
re-verified on a sample of live decision points by ``--verify-equivalence``.

This is why the earlier diagnostic's apparent MC signal does not transfer: the
recorded probe in ``acrobot_q_action_agreement.json`` used a **2-actor** ensemble
as its continuation, and a 2-actor ensemble does terminate early (margins 1-22,
mean spread 91.9).  The reported fusion uses all 8 actors.  The probe and the
policy it claims to explain therefore have different continuation policies.

Output: ``results/a2_rankability.json``
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from auditability.run_shared_multienv_gpi import (
    grammar_fusion_decision,
    load_actor,
    load_rules,
)

SEEDS = [11, 29, 43, 71, 101, 149, 211, 307]
HELDOUT_BASE = 20_000_000
CONFIDENCE_KEY = lambda rule: (rule["confidence"], rule["support"], rule["mean_reward"],
                               -rule["source_seed"])  # noqa: E731


def bootstrap_ci(values: list[float], seed: int, draws: int = 5000) -> list[float]:
    rng = random.Random(seed)
    boots = sorted(statistics.mean(values[rng.randrange(len(values))] for _ in values)
                   for _ in range(draws))
    return [boots[124], boots[4874]]


def ensemble_argmax(observation, actors) -> int:
    state = torch.as_tensor(observation, dtype=torch.float32)
    with torch.inference_mode():
        logits = torch.stack([actor(state) for actor in actors]).mean(dim=0)
    return int(logits.argmax().item())


def decide(mode: str, observation, symbolizer, rule_banks, actors, policy_rng):
    """One arbitration decision. Returns (action, detail)."""
    condition = symbolizer.encode(observation)
    candidates = [bank[condition] for bank in rule_banks if condition in bank]
    if not candidates:
        return ensemble_argmax(observation, actors), {
            "covered": False, "conflict": False, "candidate_count": 0,
            "candidate_actions": [], "degenerate_mode": None}
    actions = sorted({rule["action"] for rule in candidates})
    detail = {"covered": True, "conflict": len(actions) > 1,
              "candidate_count": len(candidates), "candidate_actions": actions,
              "degenerate_mode": None}
    if mode == "confidence_first":
        return int(max(candidates, key=CONFIDENCE_KEY)["action"]), detail
    if mode == "random_candidate":
        return int(policy_rng.choice(candidates)["action"]), detail
    if mode == "mc_ranked_as_smallest_candidate":
        detail["degenerate_mode"] = "mc_ranked reduces to min candidate action"
        return int(actions[0]), detail
    if mode == "mc_oracle_as_constant_0":
        detail["degenerate_mode"] = "mc_oracle reduces to always action 0"
        return 0, detail
    raise ValueError(mode)


def run_mode(mode: str, episodes: int, env_name: str, symbolizer, rule_banks, actors,
             action_dim, progress) -> dict:
    returns = []
    counts = {"steps": 0, "covered": 0, "conflicts": 0, "blind_spots": 0}
    action_histogram = {str(a): 0 for a in range(action_dim)}
    action0_when_covered = 0
    action0_when_conflict = 0
    start = time.perf_counter()
    for episode in range(episodes):
        env = gym.make(env_name)
        policy_rng = random.Random(20260922 + episode)
        observation, _ = env.reset(seed=HELDOUT_BASE + episode)
        total = 0.0
        done = False
        while not done:
            action, detail = decide(mode, observation, symbolizer, rule_banks, actors, policy_rng)
            counts["steps"] += 1
            action_histogram[str(action)] += 1
            counts["covered"] += int(detail["covered"])
            counts["conflicts"] += int(detail["conflict"])
            counts["blind_spots"] += int(not detail["covered"])
            if detail["covered"]:
                action0_when_covered += int(action == 0)
                if detail["conflict"]:
                    action0_when_conflict += int(action == 0)
            observation, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            done = terminated or truncated
        env.close()
        returns.append(total)
        progress(mode, episode, total, counts["steps"])
    elapsed = time.perf_counter() - start
    return {
        "mode": mode,
        "mean_return": statistics.mean(returns),
        "median_return": statistics.median(returns),
        "std_return": statistics.stdev(returns) if len(returns) > 1 else 0.0,
        "return_by_episode": returns,
        "decision_counts": counts,
        "action_histogram": action_histogram,
        "action_0_selection_rate": action_histogram["0"] / counts["steps"] if counts["steps"] else 0.0,
        "action_0_rate_when_covered": action0_when_covered / counts["covered"] if counts["covered"] else 0.0,
        "action_0_rate_when_conflict": (action0_when_conflict / counts["conflicts"]
                                        if counts["conflicts"] else 0.0),
        "wall_clock_seconds": elapsed,
    }


def verify_equivalence(episodes: int, env_name: str, symbolizer, rule_banks, actors,
                       action_dim, max_checks: int) -> dict:
    """Confirm on live decision points that the degenerate reductions hold.

    At each sampled decision point we run the real Monte-Carlo estimator and check
    (a) that all candidate action values tie, and (b) that the MC-driven choice
    equals the cheap deterministic rule the policy run uses.
    """
    from auditability.diagnose_acrobot_q_action_agreement import mc_return_from_action
    mc_env = gym.make(env_name)
    checks = []
    for episode in range(episodes):
        if len(checks) >= max_checks:
            break
        env = gym.make(env_name)
        policy_rng = random.Random(20260922 + episode)
        observation, _ = env.reset(seed=HELDOUT_BASE + episode)
        done = False
        while not done and len(checks) < max_checks:
            condition = symbolizer.encode(observation)
            candidates = [bank[condition] for bank in rule_banks if condition in bank]
            if candidates:
                actions = sorted({rule["action"] for rule in candidates})
                values = {a: mc_return_from_action(mc_env, observation, a, actors)
                          for a in range(action_dim)}
                ordered = sorted(values.values(), reverse=True)
                mc_ranked_choice = max(actions, key=lambda a: values[a])
                mc_oracle_choice = max(range(action_dim), key=lambda a: values[a])
                checks.append({
                    "episode": episode,
                    "values": {str(k): v for k, v in values.items()},
                    "spread": ordered[0] - ordered[-1],
                    "all_tied": len(set(round(v, 9) for v in values.values())) == 1,
                    "mc_ranked_choice": mc_ranked_choice,
                    "cheap_min_candidate": actions[0],
                    "mc_ranked_matches_cheap": mc_ranked_choice == actions[0],
                    "mc_oracle_choice": mc_oracle_choice,
                    "mc_oracle_matches_constant_0": mc_oracle_choice == 0,
                })
            action, _ = decide("mc_ranked_as_smallest_candidate", observation, symbolizer,
                               rule_banks, actors, policy_rng)
            observation, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        env.close()
    mc_env.close()
    return {
        "checks": len(checks),
        "all_tied_rate": sum(c["all_tied"] for c in checks) / len(checks) if checks else None,
        "mc_ranked_matches_cheap_rate": (sum(c["mc_ranked_matches_cheap"] for c in checks) / len(checks)
                                         if checks else None),
        "mc_oracle_matches_constant_0_rate": (sum(c["mc_oracle_matches_constant_0"] for c in checks)
                                              / len(checks) if checks else None),
        "mean_spread": statistics.mean([c["spread"] for c in checks]) if checks else None,
        "sample": checks[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, default=Path("runs/shared_multienv_gpi/Acrobot-v1"))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--modes", nargs="+",
                        default=["confidence_first", "random_candidate",
                                 "mc_ranked_as_smallest_candidate", "mc_oracle_as_constant_0"])
    parser.add_argument("--verify-equivalence", type=int, default=0,
                        help="number of live decision points to check the degeneracy reduction on")
    parser.add_argument("--output", type=Path, default=Path("results/a2_rankability.json"))
    args = parser.parse_args()

    torch.set_num_threads(1)
    task_root = args.task_root.resolve()
    calibration = json.loads((task_root / "shared_calibration.json").read_text(encoding="utf-8"))
    env_name = calibration["environment"]
    state_dim = len(calibration["state_bins"]["edges"])
    probe = gym.make(env_name)
    action_dim = int(probe.action_space.n)
    probe.close()

    from auditability.diagnose_acrobot_fusion import load_shared_symbolizer
    symbolizer = load_shared_symbolizer(task_root)
    runs = []
    for seed in SEEDS:
        matches = [p for p in (task_root / "grammar_audit_only").glob(f"seed-{seed}_*")
                   if (p / "metadata.json").is_file()]
        if len(matches) != 1:
            raise ValueError(f"expected one complete run for seed {seed}")
        runs.append(matches[0])
    actors = [load_actor(path, state_dim, action_dim) for path in runs]
    rule_banks = [load_rules(path) for path in runs]

    def progress(mode, episode, total, steps):
        if episode % 20 == 0 or episode == args.episodes - 1:
            print(f"  [{mode}] episode {episode + 1}/{args.episodes} return={total:.1f} "
                  f"cumulative_steps={steps}", flush=True)

    print(f"A2 rankability test on {env_name}, {args.episodes} episodes, "
          f"reset seeds {HELDOUT_BASE}+i", flush=True)
    results = {}
    for mode in args.modes:
        print(f"running mode {mode} ...", flush=True)
        results[mode] = run_mode(mode, args.episodes, env_name, symbolizer, rule_banks, actors,
                                 action_dim, progress)
        row = results[mode]
        print(f"  -> {mode}: mean {row['mean_return']:.2f} median {row['median_return']:.1f} "
              f"steps {row['decision_counts']['steps']} "
              f"action0_rate {row['action_0_selection_rate']:.3f} "
              f"({row['wall_clock_seconds']:.0f}s)", flush=True)

    baseline = results.get("confidence_first", {}).get("return_by_episode")
    if baseline:
        for mode, row in results.items():
            if mode == "confidence_first":
                continue
            deltas = [a - b for a, b in zip(row["return_by_episode"], baseline)]
            row["paired_mean_delta_vs_confidence_first"] = statistics.mean(deltas)
            row["paired_delta_bootstrap_ci95"] = bootstrap_ci(deltas, 80_000_000 + len(mode))

    equivalence = None
    if args.verify_equivalence:
        print(f"verifying degeneracy reduction on {args.verify_equivalence} live decision points ...",
              flush=True)
        equivalence = verify_equivalence(args.episodes, env_name, symbolizer, rule_banks, actors,
                                         action_dim, args.verify_equivalence)
        print(f"  all-tied rate {equivalence['all_tied_rate']}, "
              f"mc_ranked==min_candidate {equivalence['mc_ranked_matches_cheap_rate']}, "
              f"mc_oracle==constant0 {equivalence['mc_oracle_matches_constant_0_rate']}", flush=True)

    recorded = json.loads((task_root / "grammar_fusion_diagnostics.json").read_text(encoding="utf-8"))
    recorded_conf = recorded["ablations"]["confidence_first"]
    recorded_rand = recorded["ablations"]["random_candidate"]
    self_test = {
        "confidence_first_recorded_mean": recorded_conf["mean_return"],
        "confidence_first_rerun_mean": results.get("confidence_first", {}).get("mean_return"),
        "random_candidate_recorded_mean": recorded_rand["mean_return"],
        "random_candidate_rerun_mean": results.get("random_candidate", {}).get("mean_return"),
    }
    for key in ("confidence_first", "random_candidate"):
        got = self_test[f"{key}_rerun_mean"]
        want = self_test[f"{key}_recorded_mean"]
        self_test[f"{key}_matches_recorded"] = got is not None and abs(got - want) < 1e-6

    output = {
        "environment": env_name,
        "episodes": args.episodes,
        "reset_seed_formula": f"{HELDOUT_BASE} + episode_index, same as the recorded ablations",
        "seeds": SEEDS,
        "modes": results,
        "degeneracy_equivalence_check": equivalence,
        "harness_self_test": self_test,
        "estimator_caveat": (
            "mc_return_from_action values a candidate first action by forcing it and then "
            "following the deterministic 8-actor mean-logit ensemble, which never terminates "
            "Acrobot inside the 500-step horizon. Every forced action therefore returns exactly "
            "-500 and the estimator carries no information. The mode names "
            "mc_ranked_as_smallest_candidate and mc_oracle_as_constant_0 state the reduction that "
            "follows; --verify-equivalence re-checks it against the real estimator on live states."
        ),
        "prior_probe_discrepancy": (
            "results/../gpi_fqe/acrobot_q_action_agreement.json reports discriminating MC values "
            "(action 0 -132.92, action 1 -202.47, action 2 -157.11). That probe used a 2-actor "
            "ensemble as its continuation policy, which does terminate early. The reported fusion "
            "uses 8 actors. The probe and the policy it is used to explain therefore have "
            "different continuation policies, and the probe's signal does not transfer."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print("\n=== summary ===")
    for mode, row in results.items():
        delta = row.get("paired_mean_delta_vs_confidence_first")
        ci = row.get("paired_delta_bootstrap_ci95")
        extra = f"  delta_vs_confidence {delta:+.2f} CI{[round(v, 2) for v in ci]}" if delta is not None else ""
        print(f"{mode:34s} mean {row['mean_return']:9.2f}  median {row['median_return']:7.1f}  "
              f"steps {row['decision_counts']['steps']:6d}  "
              f"action0 {row['action_0_selection_rate']:.3f}{extra}")
    print(f"self-test: {self_test}")


if __name__ == "__main__":
    main()
