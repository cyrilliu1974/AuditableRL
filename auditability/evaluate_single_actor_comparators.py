"""Evaluate every frozen source actor on the matched composition evaluation set.

Motivation
----------
The primary composition comparison in the paper selects its single-actor
comparator by *training* return mean. That is a post-hoc choice, so the
headline ``fusion beats the best single actor'' is only as strong as the
selection rule. This script re-evaluates every frozen source actor (and the
actor-logit ensemble) on the *same* 100 common reset seeds that the
composition evaluation uses, so that several single-actor comparators can be
reported on one matched footing.

It also re-derives the recorded ``best single actor'' and ``actor-logit
ensemble'' values, which is used as a self-check: if the harness is faithful,
those two rows must reproduce the recorded numbers exactly.

Output: runs/shared_multienv_gpi/<env>/single_actor_comparators.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

import gymnasium as gym
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from auditability.run_shared_multienv_gpi import load_actor  # noqa: E402

SEEDS = [11, 29, 43, 71, 101, 149, 211, 307]
ENVIRONMENTS = {
    "CartPole-v1": {"state_dim": 4, "action_dim": 2},
    "Acrobot-v1": {"state_dim": 6, "action_dim": 3},
}
EPISODES = 100
BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 20260922


def actor_run_dir(environment: str, seed: int) -> Path:
    root = PROJECT_ROOT / "runs" / "shared_multienv_gpi" / environment / "grammar_audit_only"
    matches = sorted(root.glob(f"seed-{seed}_*"))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one audit-only run for seed {seed}, found {len(matches)}")
    return matches[0]


def rollout(env, policy) -> float:
    """Run one episode under ``policy`` and return the undiscounted return."""
    observation, _ = env.reset(seed=policy["reset_seed"])
    total = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        action = policy["act"](observation)
        observation, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)
    return total


def evaluate(environment: str) -> dict:
    spec = ENVIRONMENTS[environment]
    env = gym.make(environment)
    actors, training_means = {}, {}
    for seed in SEEDS:
        run_dir = actor_run_dir(environment, seed)
        actors[seed] = load_actor(run_dir, spec["state_dim"], spec["action_dim"])
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        training_means[seed] = metadata["episode_return_mean"]

    def greedy(actor):
        def act(observation):
            with torch.inference_mode():
                return int(
                    actor(torch.as_tensor(observation, dtype=torch.float32)).argmax().item()
                )

        return act

    def ensemble(observation):
        state = torch.as_tensor(observation, dtype=torch.float32)
        with torch.inference_mode():
            logits = torch.stack([actors[seed](state) for seed in SEEDS]).mean(0)
        return int(logits.argmax().item())

    per_seed_returns = {seed: [] for seed in SEEDS}
    ensemble_returns = []
    for index in range(EPISODES):
        reset_seed = 20_000_000 + index
        for seed in SEEDS:
            per_seed_returns[seed].append(
                rollout(env, {"reset_seed": reset_seed, "act": greedy(actors[seed])})
            )
        ensemble_returns.append(rollout(env, {"reset_seed": reset_seed, "act": ensemble}))
    env.close()

    matched = {seed: statistics.mean(per_seed_returns[seed]) for seed in SEEDS}
    values = [matched[seed] for seed in SEEDS]
    train_best = max(SEEDS, key=lambda seed: (training_means[seed], -seed))
    heldout_best = max(SEEDS, key=lambda seed: (matched[seed], -seed))

    fusion = json.loads(
        (
            PROJECT_ROOT
            / "results"
            / "shared_multienv_gpi"
            / environment
            / "gpi_fqe"
            / "gpi_fqe_comparison.json"
        ).read_text(encoding="utf-8")
    )["composition_evaluation"]["policies"]["grammar_fusion"]["return_by_episode"]
    rng = random.Random(BOOTSTRAP_SEED)

    def paired_vs_fusion(returns):
        deltas = [f - r for f, r in zip(fusion, returns)]
        bootstrap = sorted(
            statistics.mean(deltas[rng.randrange(len(deltas))] for _ in deltas)
            for _ in range(BOOTSTRAP_RESAMPLES)
        )
        return {
            "mean_delta": statistics.mean(deltas),
            "ci95_percentile": [bootstrap[124], bootstrap[4874]],
        }

    return {
        "environment": environment,
        "episodes": EPISODES,
        "reset_seed_formula": "20000000 + episode_index; identical to the composition evaluation",
        "per_seed": {
            seed: {
                "training_mean_return": training_means[seed],
                "matched_mean_return": matched[seed],
                "matched_median_return": statistics.median(per_seed_returns[seed]),
            }
            for seed in SEEDS
        },
        "return_by_episode": per_seed_returns,
        "ensemble_return_by_episode": ensemble_returns,
        "best_training_return_actor": {
            "seed": train_best,
            "matched_mean_return": matched[train_best],
        },
        "best_heldout_actor_on_matched_set": {
            "seed": heldout_best,
            "matched_mean_return": matched[heldout_best],
        },
        "mean_actor_matched_mean_return": statistics.mean(values),
        "median_actor_matched_mean_return": statistics.median(values),
        "min_actor_matched_mean_return": min(values),
        "max_actor_matched_mean_return": max(values),
        "actor_logit_ensemble_matched_mean_return": statistics.mean(ensemble_returns),
        "rule_fusion_matched_mean_return": statistics.mean(fusion),
        "paired_fusion_vs_best_heldout_actor": paired_vs_fusion(per_seed_returns[heldout_best]),
        "paired_fusion_vs_best_training_actor": paired_vs_fusion(per_seed_returns[train_best]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", default="CartPole-v1", choices=sorted(ENVIRONMENTS))
    args = parser.parse_args()
    result = evaluate(args.environment)
    out = (
        PROJECT_ROOT
        / "runs"
        / "shared_multienv_gpi"
        / args.environment
        / "single_actor_comparators.json"
    )
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print(
        "training-argmax actor: seed {seed} -> {matched_mean_return:.2f}".format(
            **result["best_training_return_actor"]
        )
    )
    print(
        "best held-out actor:   seed {seed} -> {matched_mean_return:.2f}".format(
            **result["best_heldout_actor_on_matched_set"]
        )
    )
    print(f"mean actor: {result['mean_actor_matched_mean_return']:.2f}")
    print(f"median actor: {result['median_actor_matched_mean_return']:.2f}")
    print(f"ensemble: {result['actor_logit_ensemble_matched_mean_return']:.2f}")
    print(f"fusion: {result['rule_fusion_matched_mean_return']:.2f}")


if __name__ == "__main__":
    main()
