# -*- coding: utf-8 -*-
"""Numeric regression for the claims in Section 5.3 of merged_main.tex.

Every number asserted in the rewritten ``sec:evolution:constraint`` subsection, in
``tab:perseed`` and in ``tab:attribution`` is recomputed here from the run artifacts on disk and
compared against the value that appears in the manuscript. No training is performed.

The state-occupancy metric has no saved artifact, so it is recomputed by instrumenting the held-out
evaluation loop. The instrumented loop must first reproduce the recorded return of every run
byte-for-byte; otherwise the occupancy numbers it produces cannot be trusted.

Three occupancy configurations are measured per seed:

* ``unconstrained``      - the audit-only actor, which is bit-identical to the baseline actor;
* ``hybrid``             - the constrained checkpoint evaluated with rule substitution ON, i.e. the
                           behavior whose return the constrained arm reports;
* ``network_only``       - the same constrained checkpoint with substitution OFF, i.e. the learned
                           policy on its own.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import gymnasium as gym
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auditability.grammar_audit_experiment import (  # noqa: E402
    Actor,
    StateSymbolizer,
    evaluate_policy,
    grammar_predict,
)
from auditability.run_shared_multienv_gpi import load_actor, load_rules  # noqa: E402

ENVIRONMENT = "CartPole-v1"
SEEDS = [11, 29, 43, 71, 101, 149, 211, 307]
EPISODES = 20
TASK_ROOT = ROOT / "runs" / "shared_multienv_gpi" / ENVIRONMENT
PILOT_ROOT = ROOT / "runs" / "grammar_audit_pilot"
OUTPUT = ROOT / "results" / "constraint_claim_regression.json"


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------
def symbolizer_from_metadata(metadata: dict) -> StateSymbolizer:
    bins = metadata["config"]["state_bins"]
    symbolizer = StateSymbolizer(n_bins=int(bins["n_bins"]))
    symbolizer.edges = torch.tensor(bins["edges"], dtype=torch.float64)
    return symbolizer


def latest_run(root: Path, arm: str, seed: int) -> Path:
    matches = sorted((root / arm).glob(f"seed-{seed}_*"))
    completed = [p for p in matches if (p / "metadata.json").exists()]
    if not completed:
        raise FileNotFoundError(f"no completed {arm} run for seed {seed} under {root}")
    return completed[-1]


def perplexity(counts: dict) -> tuple[float, float, int]:
    """Return (2**H, H in bits, number of distinct symbols) for a symbol-count table."""
    total = sum(counts.values())
    if total == 0:
        return 0.0, 0.0, 0
    entropy = 0.0
    for n in counts.values():
        p = n / total
        entropy -= p * math.log2(p)
    return 2.0**entropy, entropy, len(counts)


def top_k_mass(counts: dict, k: int = 10) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return sum(sorted(counts.values(), reverse=True)[:k]) / total


def training_trace_occupancy(run_dir: Path) -> dict:
    """Stream a run's training trace and summarise the state-symbol distribution.

    This is the measurement the manuscript reports for state occupancy. It is deliberately kept
    separate from the held-out evaluation occupancy above: the two disagree in direction on four of
    the eight seeds, because a policy that fails early visits more of the state space before it dies.
    """
    counts: dict[tuple, int] = {}
    steps = 0
    with (run_dir / "trajectory.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            symbol = tuple(record["state_symbol"])
            counts[symbol] = counts.get(symbol, 0) + 1
            steps += 1
    ppl, entropy, distinct = perplexity(counts)
    return {
        "training_steps": steps,
        "distinct_state_symbols": distinct,
        "entropy_bits": entropy,
        "perplexity_2_pow_H": ppl,
        "top10_symbol_mass": top_k_mass(counts),
    }


def instrumented_eval(
    actor: Actor,
    symbolizer: StateSymbolizer,
    rules: dict,
    arm: str,
    seed: int,
    n_episodes: int = EPISODES,
) -> dict:
    """Replicate evaluate_policy, additionally recording the visited state-symbol table."""
    env = gym.make(ENVIRONMENT)
    counts: dict[tuple, int] = {}
    returns, lengths = [], []
    covered = correct = total = 0
    actor.eval()
    with torch.inference_mode():
        for episode in range(n_episodes):
            observation, _ = env.reset(seed=seed * 10_000 + 900_000 + episode)
            total_reward = 0.0
            length = 0
            terminated = truncated = False
            while not (terminated or truncated):
                condition = symbolizer.encode(observation)
                counts[tuple(condition)] = counts.get(tuple(condition), 0) + 1
                logits = actor(torch.as_tensor(observation, dtype=torch.float32))
                neural_action = int(logits.argmax().item())
                grammar_action = grammar_predict(rules, condition)
                total += 1
                if grammar_action is not None:
                    covered += 1
                    correct += int(grammar_action == neural_action)
                action = neural_action
                if arm == "grammar_constrained" and grammar_action is not None:
                    action = grammar_action
                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                length += 1
            returns.append(total_reward)
            lengths.append(length)
    env.close()
    actor.train()
    ppl, entropy, distinct = perplexity(counts)
    return {
        "mean_return": sum(returns) / len(returns),
        "grammar_state_coverage": covered / max(total, 1),
        "grammar_action_agreement_when_covered": correct / max(covered, 1),
        "covered_steps": covered,
        "total_steps": total,
        "distinct_state_symbols": distinct,
        "entropy_bits": entropy,
        "perplexity_2_pow_H": ppl,
        "top10_symbol_mass": top_k_mass(counts),
    }


# --------------------------------------------------------------------------------------------
# claim ledger
# --------------------------------------------------------------------------------------------
class Ledger:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def check(self, claim: str, expected, actual, tol: float = 0.05) -> None:
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            ok = abs(float(expected) - float(actual)) <= tol
        else:
            ok = expected == actual
        self.rows.append({
            "claim": claim, "in_manuscript": expected, "recomputed": actual, "ok": bool(ok),
        })

    def report(self) -> bool:
        width = max(len(r["claim"]) for r in self.rows)
        failures = 0
        for r in self.rows:
            mark = "PASS" if r["ok"] else "FAIL"
            if not r["ok"]:
                failures += 1
            print("%-4s %-*s  manuscript=%-14s recomputed=%s"
                  % (mark, width, r["claim"], r["in_manuscript"], r["recomputed"]))
        print()
        print("%d claims checked, %d failures" % (len(self.rows), failures))
        return failures == 0


def main() -> int:
    ledger = Ledger()
    per_seed = []
    occ_rows = []
    train_rows = []

    for seed in SEEDS:
        constrained_dir = latest_run(TASK_ROOT, "grammar_constrained", seed)
        audit_dir = latest_run(TASK_ROOT, "grammar_audit_only", seed)
        baseline_dir = latest_run(TASK_ROOT, "baseline", seed)

        meta_c = json.loads((constrained_dir / "metadata.json").read_text(encoding="utf-8"))
        meta_a = json.loads((audit_dir / "metadata.json").read_text(encoding="utf-8"))
        meta_b = json.loads((baseline_dir / "metadata.json").read_text(encoding="utf-8"))

        state_dim = int(meta_c["config"]["state_dim"])
        action_dim = int(meta_c["config"]["action_dim"])
        symbolizer = symbolizer_from_metadata(meta_c)
        rules = load_rules(constrained_dir)

        actor_c: Actor = load_actor(constrained_dir, state_dim, action_dim)
        actor_a: Actor = load_actor(audit_dir, state_dim, action_dim)

        hybrid = instrumented_eval(actor_c, symbolizer, rules, "grammar_constrained", seed)
        network_only = instrumented_eval(actor_c, symbolizer, rules, "baseline", seed)
        unconstrained = instrumented_eval(actor_a, symbolizer, rules, "baseline", seed)

        recorded = float(meta_c["evaluation"]["mean_return"])
        recorded_uncon = float(meta_b["evaluation"]["mean_return"])

        # Harness check: the instrumented loop must reproduce the recorded numbers exactly.
        ledger.check(
            f"seed {seed}: instrumented loop reproduces recorded constrained return",
            recorded, round(hybrid["mean_return"], 6), tol=1e-6,
        )
        ledger.check(
            f"seed {seed}: baseline metadata equals audit-only evaluation",
            float(meta_a["evaluation"]["mean_return"]), recorded_uncon, tol=1e-6,
        )

        row = {
            "seed": seed,
            "baseline_return": recorded_uncon,
            "hybrid_return": hybrid["mean_return"],
            "network_only_return": network_only["mean_return"],
            "substitution_gain": hybrid["mean_return"] - network_only["mean_return"],
            "network_deficit_vs_unconstrained": (
                unconstrained["mean_return"] - network_only["mean_return"]
            ),
            "coverage_constrained": hybrid["grammar_state_coverage"],
            "coverage_audit_only": float(meta_a["evaluation"]["grammar_state_coverage"]),
            "agreement_constrained": hybrid["grammar_action_agreement_when_covered"],
            "agreement_audit_only": float(
                meta_a["evaluation"]["grammar_action_agreement_when_covered"]
            ),
            "covered_fraction_constrained": (
                hybrid["covered_steps"] / max(hybrid["total_steps"], 1)
            ),
        }
        per_seed.append(row)
        occ_rows.append({
            "seed": seed,
            "unconstrained": {
                "distinct_state_symbols": unconstrained["distinct_state_symbols"],
                "entropy_bits": unconstrained["entropy_bits"],
                "perplexity_2_pow_H": unconstrained["perplexity_2_pow_H"],
                "top10_symbol_mass": unconstrained["top10_symbol_mass"],
            },
            "hybrid": {
                "distinct_state_symbols": hybrid["distinct_state_symbols"],
                "entropy_bits": hybrid["entropy_bits"],
                "perplexity_2_pow_H": hybrid["perplexity_2_pow_H"],
                "top10_symbol_mass": hybrid["top10_symbol_mass"],
            },
            "network_only": {
                "distinct_state_symbols": network_only["distinct_state_symbols"],
                "entropy_bits": network_only["entropy_bits"],
                "perplexity_2_pow_H": network_only["perplexity_2_pow_H"],
                "top10_symbol_mass": network_only["top10_symbol_mass"],
            },
        })
        train_rows.append({
            "seed": seed,
            "audit_only": training_trace_occupancy(audit_dir),
            "constrained": training_trace_occupancy(constrained_dir),
        })
        train_occ = train_rows[-1]
        print("seed %-4d baseline=%8.2f hybrid=%8.2f network=%8.2f gain=%+8.2f "
              "ppl_heldout(uncon)=%6.1f ppl_heldout(hybrid)=%6.1f "
              "ppl_train(audit)=%6.1f ppl_train(cons)=%6.1f"
              % (seed, recorded_uncon, hybrid["mean_return"], network_only["mean_return"],
                 row["substitution_gain"], unconstrained["perplexity_2_pow_H"],
                 hybrid["perplexity_2_pow_H"],
                 train_occ["audit_only"]["perplexity_2_pow_H"],
                 train_occ["constrained"]["perplexity_2_pow_H"]))

    by_seed = {r["seed"]: r for r in per_seed}
    occ_by_seed = {r["seed"]: r for r in occ_rows}

    print()
    print("=" * 78)
    print("CLAIM REGRESSION")
    print("=" * 78)

    # ---- tab:perseed -------------------------------------------------------------------------
    perseed_expected = {
        11: (500.0, 182.3), 29: (131.0, 293.5), 43: (153.75, 500.0), 71: (130.15, 387.15),
        101: (201.3, 149.15), 149: (500.0, 96.1), 211: (500.0, 403.2), 307: (500.0, 161.2),
    }
    for seed, (base, cons) in perseed_expected.items():
        ledger.check(f"tab:perseed seed {seed} baseline", base, by_seed[seed]["baseline_return"])
        ledger.check(f"tab:perseed seed {seed} constrained", cons, by_seed[seed]["hybrid_return"])

    # ---- narrative seed pairs ----------------------------------------------------------------
    ledger.check("narrative: seed 29 131.0 -> 293.5",
                 293.5 - 131.0, by_seed[29]["hybrid_return"] - by_seed[29]["baseline_return"])
    ledger.check("narrative: seed 71 130.2 -> 387.2",
                 387.2 - 130.2, by_seed[71]["hybrid_return"] - by_seed[71]["baseline_return"])
    ledger.check("narrative: seed 43 153.75 -> 500.0",
                 500.0 - 153.75, by_seed[43]["hybrid_return"] - by_seed[43]["baseline_return"])
    ledger.check("narrative: seed 11 500.0 -> 182.3",
                 182.3 - 500.0, by_seed[11]["hybrid_return"] - by_seed[11]["baseline_return"])
    ledger.check("narrative: seed 149 500.0 -> 96.1",
                 96.1 - 500.0, by_seed[149]["hybrid_return"] - by_seed[149]["baseline_return"])
    ledger.check("narrative: seed 307 500.0 -> 161.2",
                 161.2 - 500.0, by_seed[307]["hybrid_return"] - by_seed[307]["baseline_return"])

    # ---- coverage / agreement aggregate ------------------------------------------------------
    cov_a = sum(r["coverage_audit_only"] for r in per_seed) / len(per_seed)
    cov_c = sum(r["coverage_constrained"] for r in per_seed) / len(per_seed)
    agr_a = sum(r["agreement_audit_only"] for r in per_seed) / len(per_seed)
    agr_c = sum(r["agreement_constrained"] for r in per_seed) / len(per_seed)
    ledger.check("coverage mean audit-only = 0.477", 0.477, cov_a, tol=0.0005)
    ledger.check("coverage mean constrained = 0.761", 0.761, cov_c, tol=0.0005)
    ledger.check("agreement mean audit-only = 0.887", 0.887, agr_a, tol=0.0005)
    ledger.check("agreement mean constrained = 0.676", 0.676, agr_c, tol=0.0005)
    ledger.check("coverage rises in 8/8 seeds",
                 8, sum(1 for r in per_seed if r["coverage_constrained"] > r["coverage_audit_only"]))
    ledger.check("agreement falls in 8/8 seeds",
                 8, sum(1 for r in per_seed
                        if r["agreement_constrained"] < r["agreement_audit_only"]))

    # ---- effective state occupancy (2**H), training trace -------------------------------------
    train_by_seed = {r["seed"]: r for r in train_rows}
    train_expected = {
        11: (79.3, 58.4), 149: (91.1, 45.4), 307: (70.5, 37.5), 43: (83.8, 63.5),
        211: (53.1, 60.4),
    }
    for seed, (before, after) in train_expected.items():
        ledger.check(f"train-trace occupancy seed {seed} audit-only 2^H = {before}",
                     before, train_by_seed[seed]["audit_only"]["perplexity_2_pow_H"], tol=0.06)
        ledger.check(f"train-trace occupancy seed {seed} constrained 2^H = {after}",
                     after, train_by_seed[seed]["constrained"]["perplexity_2_pow_H"], tol=0.06)

    contracts_train = sum(
        1 for r in train_rows
        if r["constrained"]["perplexity_2_pow_H"] < r["audit_only"]["perplexity_2_pow_H"]
    )
    ledger.check("manuscript: training-trace occupancy falls in 7 of 8 seeds", 7, contracts_train)

    contracts_heldout = sum(
        1 for r in occ_rows
        if r["hybrid"]["perplexity_2_pow_H"] < r["unconstrained"]["perplexity_2_pow_H"]
    )
    ledger.check("manuscript: held-out occupancy contracts in only 4 of 8 seeds",
                 4, contracts_heldout)

    # ---- substitution share ------------------------------------------------------------------
    fracs = [r["covered_fraction_constrained"] for r in per_seed]
    ledger.check("substitution fires on 61%-95% of held-out steps (min)",
                 0.61, min(fracs), tol=0.005)
    ledger.check("substitution fires on 61%-95% of held-out steps (max)",
                 0.95, max(fracs), tol=0.005)

    # ---- tab:attribution ---------------------------------------------------------------------
    attribution_expected = {
        11: (182.3, 127.8, 500.0, 54.5), 29: (293.5, 365.1, 131.0, -71.6),
        43: (500.0, 217.9, 153.8, 282.1), 71: (387.1, 237.6, 130.2, 149.6),
        101: (149.2, 80.0, 201.3, 69.2), 149: (96.1, 486.4, 500.0, -390.2),
        211: (403.2, 120.8, 500.0, 282.4), 307: (161.2, 165.4, 500.0, -4.2),
    }
    for seed, (hyb, net, unc, gain) in attribution_expected.items():
        ledger.check(f"tab:attribution seed {seed} hybrid", hyb, by_seed[seed]["hybrid_return"], tol=0.06)
        ledger.check(f"tab:attribution seed {seed} network only", net,
                     by_seed[seed]["network_only_return"], tol=0.06)
        ledger.check(f"tab:attribution seed {seed} unconstrained", unc,
                     by_seed[seed]["baseline_return"], tol=0.06)
        ledger.check(f"tab:attribution seed {seed} substitution gain", gain,
                     by_seed[seed]["substitution_gain"], tol=0.06)

    mean_hybrid = sum(r["hybrid_return"] for r in per_seed) / len(per_seed)
    mean_net = sum(r["network_only_return"] for r in per_seed) / len(per_seed)
    mean_unc = sum(r["baseline_return"] for r in per_seed) / len(per_seed)
    mean_gain = sum(r["substitution_gain"] for r in per_seed) / len(per_seed)
    ledger.check("tab:attribution mean hybrid = 271.6", 271.6, mean_hybrid, tol=0.06)
    ledger.check("tab:attribution mean network only = 225.1", 225.1, mean_net, tol=0.06)
    ledger.check("tab:attribution mean unconstrained = 327.0", 327.0, mean_unc, tol=0.06)
    ledger.check("tab:attribution mean substitution gain = +46.4", 46.4, mean_gain, tol=0.06)
    ledger.check("prose: network mean 225.13 vs unconstrained 327.02",
                 225.13, mean_net, tol=0.02)
    ledger.check("prose: headline hybrid difference = -55.45",
                 -55.45, mean_hybrid - mean_unc, tol=0.02)
    ledger.check("prose: four seeds lose more than 120 points to the unconstrained actor",
                 4, sum(1 for r in per_seed if r["network_deficit_vs_unconstrained"] > 120))
    ledger.check("prose: seed 149 network-only 486.4 vs hybrid 96.1",
                 486.4, by_seed[149]["network_only_return"], tol=0.06)
    ledger.check("prose: seed 149 hybrid 96.1",
                 96.1, by_seed[149]["hybrid_return"], tol=0.06)
    ledger.check("prose: substitution gain positive on 5/8 seeds",
                 5, sum(1 for r in per_seed if r["substitution_gain"] > 0))
    ledger.check("prose: network-only exceeds unconstrained on 3/8 seeds",
                 3, sum(1 for r in per_seed if r["network_deficit_vs_unconstrained"] < 0))

    # ---- masked step rate and observer self-support ------------------------------------------
    self_support = json.loads(
        (ROOT / "results" / "constraint_observer_self_support.json").read_text(encoding="utf-8")
    )
    masked = [r["masked_step_fraction_all_states"] for r in self_support["per_seed"]]
    ledger.check("masked step rate min = 36.7%", 0.367, min(masked), tol=0.001)
    ledger.check("masked step rate max = 74.4%", 0.744, max(masked), tol=0.001)
    ledger.check("observer self-support mean = 68%",
                 0.68, self_support["aggregate"]["mean_self_generated_support_fraction"], tol=0.005)
    ledger.check("observer self-support range 45%-85% (min)",
                 0.45, self_support["aggregate"]["min_self_generated_support_fraction"], tol=0.005)
    ledger.check("observer self-support range 45%-85% (max)",
                 0.85, self_support["aggregate"]["max_self_generated_support_fraction"], tol=0.005)

    # ---- monotone bank: no retraction, no action flip ----------------------------------------
    retraction = json.loads(
        (ROOT / "results" / "constraint_rule_retraction.json").read_text(encoding="utf-8")
    )
    ledger.check("monotone bank: 0 retractions of an admitted rule",
                 0, retraction["total_retraction_events"])
    ledger.check("monotone bank: 0 changes to a rule's forced action",
                 0, retraction["total_action_flip_events"])
    ledger.check("monotone bank: verdict holds", True, retraction["bank_is_monotone"])
    ledger.check("manuscript: six re-induction generations per run",
                 [6] * 8, [len(r["generations_observed"]) for r in retraction["per_seed"]])
    ledger.check("manuscript: the bank is monotone in all 8 seeds",
                 8, sum(1 for r in retraction["per_seed"] if r["retraction_events"] == 0
                        and r["action_flip_events"] == 0))

    # ---- pilot (tau_c = 0.90) ----------------------------------------------------------------
    pilot = {}
    for arm in ("baseline", "grammar_constrained"):
        for seed in (11, 29):
            d = latest_run(PILOT_ROOT, arm, seed)
            m = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
            pilot[(arm, seed)] = m["evaluation"]
    ledger.check("pilot seed 11 baseline return = 500", 500.0, pilot[("baseline", 11)]["mean_return"])
    ledger.check("pilot seed 11 constrained return = 121.3",
                 121.3, pilot[("grammar_constrained", 11)]["mean_return"], tol=0.06)
    ledger.check("pilot seed 11 delta = -378.7",
                 -378.7, pilot[("grammar_constrained", 11)]["mean_return"]
                 - pilot[("baseline", 11)]["mean_return"], tol=0.06)
    ledger.check("pilot seed 29 baseline return = 131.0", 131.0, pilot[("baseline", 29)]["mean_return"])
    ledger.check("pilot seed 29 constrained return = 162.5",
                 162.5, pilot[("grammar_constrained", 29)]["mean_return"], tol=0.06)
    ledger.check("pilot seed 29 delta = +31.5",
                 31.5, pilot[("grammar_constrained", 29)]["mean_return"]
                 - pilot[("baseline", 29)]["mean_return"], tol=0.06)
    ledger.check("pilot seed 11 held-out coverage = 0.41%",
                 0.0041, pilot[("grammar_constrained", 11)]["grammar_state_coverage"], tol=0.00005)
    ledger.check("pilot seed 29 held-out coverage = 6.98%",
                 0.0698, pilot[("grammar_constrained", 29)]["grammar_state_coverage"], tol=0.00005)

    # ---- Acrobot null result -----------------------------------------------------------------
    acrobot = {}
    for arm in ("baseline", "grammar_constrained"):
        vals = []
        for seed in SEEDS:
            d = latest_run(TASK_ROOT.parent / "Acrobot-v1", arm, seed)
            m = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
            vals.append(m["evaluation"]["mean_return"])
        acrobot[arm] = vals
    ledger.check("Acrobot: all 8 baseline seeds = -500",
                 [-500.0] * 8, acrobot["baseline"])
    ledger.check("Acrobot: all 8 constrained seeds = -500",
                 [-500.0] * 8, acrobot["grammar_constrained"])

    ok = ledger.report()

    summary = {
        "purpose": "Recompute every numeric claim in Section 5.3 of merged_main.tex from run artifacts.",
        "environment": ENVIRONMENT,
        "episodes_per_policy": EPISODES,
        "per_seed": per_seed,
        "state_occupancy_held_out": occ_rows,
        "state_occupancy_training_trace": train_rows,
        "pilot_tau_090": {f"{arm}_seed_{seed}": ev for (arm, seed), ev in pilot.items()},
        "acrobot_returns": acrobot,
        "claims": ledger.rows,
        "all_claims_pass": ok,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print()
    print("WROTE", OUTPUT)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
