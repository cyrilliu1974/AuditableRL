# Oracle Headroom Experiment — Pre-registered Protocol

**Status**: PRE-REGISTERED before any training or measurement for this experiment.
**Date**: 2026-09-26
**Author**: Muse (assistant), authorized by user
**Codebase**: `~/workspace/RL/RL_snapshot` (commit 057e173deb12a686f44f958606b27064dfde5405)

## 0. Question

Does the rule bank contain rankable/selectable signal, or is the candidate set itself
devoid of good actions? Formally: decompose the fusion-vs-best gap into an
*arbitration gap* (fixable by a better arbiter) and a *representation gap*
(not fixable by any arbiter).

## 1. Estimand

On CartPole-v1, with 8 retrained baseline actors (seeds 11, 29, 43, 71, 101, 149,
211, 307; same config as the paper: 300 episodes, shared 4-bin calibration
symbolizer, min_support=8, min_confidence=0.70):

- `R_conf`     = mean return of confidence-first grammar fusion (repo's `grammar_fusion_decision`)
- `R_oc`       = mean return of **oracle-conflict**: at conflict states pick
                 argmax over *candidate* actions by exact rollout Q; elsewhere identical to confidence fusion
- `R_of`       = mean return of **oracle-full**: at every *covered* state pick argmax over the
                 *full* action space by exact rollout Q; blind spots still use the ensemble fallback
- `R_best`     = mean return of best single actor (selected by **training** episode_return_mean,
                 same convention as the repo's `evaluate_composition`)
- `R_ens`      = mean return of 8-actor mean-logit ensemble (reference only)

All five evaluated on the **same** 100 held-out episodes with common reset seeds
`20_000_000 + episode_index` (repo convention).

Decomposition identity:

```
R_best − R_conf = (R_best − R_of) + (R_of − R_oc) + (R_oc − R_conf)
                = coverage/fallback gap + agreed-action gap + arbitration gap
```

Interpretation ladder:
- `R_oc >> R_conf` → the arbiter is the bottleneck (worth improving arbitration).
- `R_of >> R_oc`  → banks agree on wrong actions (representation issue: agreed actions are bad).
- `R_of ≈ R_best` → representation is fine; only arbitration was broken.
- `R_of << R_best` → even perfect per-state choices on covered states cannot reach best;
  the remainder is blind spots / fallback.

## 2. Oracle construction (fixed before running)

- **Q estimation**: exact single rollout (NOT Monte Carlo averaging). CartPole-v1 dynamics
  are deterministic given (state, action); the continuation policy is deterministic.
  One rollout = exact Q. This will be *verified* by a determinism check
  (two rollouts from the same (s,a) must return bit-identical totals) before the main run.
- **State setting**: `env.unwrapped.state = np.array(s, dtype=np.float64)` then `env.step(a)`.
  Verified by a set-state roundtrip check before the main run.
- **Continuation policy**: argmax of the best-training-mean actor (`R_best`'s actor), fixed.
- **Continuation sanity gate** (pre-registered): the continuation policy must achieve
  mean return > 400 on the 100 evaluation reset seeds. If not, the oracle measurement
  is declared VOID (degenerate continuation, cf. A2's Acrobot finding) and the experiment stops.
- **Conflict definition**: same as repo — covered state where candidate actions from
  ≥2 banks disagree.
- **Candidate actions**: from the *sampled-action* banks produced by `train_one`
  (the paper's main-line artifact). Greedy-bank oracle is a follow-up, not this run.

## 3. Decision rule (pre-registered stop rule)

Let `H = R_oc − R_conf` (arbitration headroom), `T = R_best − R_conf` (total gap).

> **STOP** (rule banks lack usable signal; do not proceed to Stage 1–4 on these banks):
> `H / T < 1/3`, i.e. even a perfect arbiter recovers less than one third of the
> fusion-vs-best gap.
>
> **GO** (arbitration is the bottleneck; improving the arbiter is worthwhile):
> `H / T ≥ 1/3`.

Uncertainty: 95% bootstrap CIs (5,000 draws, repo's `bootstrap_ci`) on all means and on
`H/T`. The rule is applied to point estimates; CIs are reported alongside.

Note on CartPole action space: with 2 actions, a conflict state's candidate set is
always {0, 1} = the full action space, so "candidate recall" is trivially 1.0 at
conflicts. The agreed-action gap (`R_of − R_oc`) plays the role of the representation
diagnostic here instead.

## 4. What this run does NOT do

- No Acrobot in this run (sparse rewards → high-variance oracle; CartPole first as the
  cheap informative probe, per the run-first principle).
- No greedy-bank variant in this run (follow-up if sampled-bank oracle is informative).
- No retraining of the paper's original actors: this run retrains 8 CartPole baseline
  seeds with identical seeds/config; all comparisons are *within* the retrained set,
  so exact reproduction of the original weights is not required for internal validity.
- No changes to `~/workspace/RL/RL_snapshot` (the canonical repo): all new code lives
  under `~/workspace/RL/oracle_headroom/`.

## 5. Deliverables

- `PROTOCOL.md` (this file — written before any run)
- `stage0_train.py` — retrain 8 CartPole baseline seeds via repo's `train_one`
- `oracle_q.py` — exact-rollout Q + determinism/set-state verification
- `oracle_fusion.py` — oracle-conflict / oracle-full policies + common-reset evaluation
- `oracle_diagnostics.py` — decomposition metrics, bootstrap CIs, stop-rule verdict
- `EXPERIMENT_LOG.md` — chronological log of runs, failures, fixes
- `REPORT.md` — final verdict with numbers
- `results/oracle_headroom_cartpole.json` — machine-readable results
- Full stdout/stderr logs under `logs/`
