# A1 and A2 execution report

Date: 2026-09-23. Zero training. All results reproducible from saved artifacts.

Scope: A1 (deploy-protocol check) and A2 (rankability test) from the Project A work
plan in `AI_Grammar_Induction_Experiment_Log.md`.

---

## 0. Harness validation

Every claim below rests on a harness that first had to reproduce recorded numbers.
Three independent self-tests pass exactly.

| Self-test | Result |
|---|---|
| Re-induce rule banks from the recorded training trace | reproduces `induced_grammar.json` **exactly, 8/8 seeds, both environments** |
| Argmax evaluation at reset seeds `seed*10000 + 900000 + i`, 20 episodes | reproduces `metadata.evaluation.mean_return` **exactly, 16/16 (8 seeds x 2 envs)** |
| Re-run the recorded fusion ablations on the same 100 reset seeds | `confidence_first` **-443.37** (recorded -443.37); `random_candidate` **-187.02** (recorded -187.02) |

The A2 self-test is the important one: it means the A2 numbers are directly
comparable to the recorded ablations, not merely "in the same spirit".

---

## 1. A1 - deploy protocol

Training selects actions by sampling from `Categorical(masked_logits)`. Evaluation
and deployment use `argmax`. A1 asks whether that mismatch matters, and whether the
induced bank describes the policy that is actually deployed.

### 1.1 Returns under both protocols

Same eight saved actors, same 100 held-out reset seeds (`20000000 + i`), both
protocols.

**Acrobot-v1**

| seed | argmax mean | sampled mean | delta (sampled - argmax) |
|---|---|---|---|
| 11 | -500.00 | -434.61 | +65.39 |
| 29 | -500.00 | -500.00 | 0.00 |
| 43 | -500.00 | -311.06 | +188.94 |
| 71 | -500.00 | -500.00 | 0.00 |
| 101 | -500.00 | -500.00 | 0.00 |
| 149 | -500.00 | -500.00 | 0.00 |
| 211 | -500.00 | -326.02 | +173.98 |
| 307 | -500.00 | -497.23 | +2.77 |
| **mean** | **-500.00** | **-446.12** | **+53.88** |

**CartPole-v1**

| seed | argmax mean | sampled mean | delta (sampled - argmax) |
|---|---|---|---|
| 11 | 500.00 | 476.05 | -23.95 |
| 29 | 131.11 | 125.12 | -5.99 |
| 43 | 163.65 | 148.61 | -15.04 |
| 71 | 134.51 | 131.90 | -2.61 |
| 101 | 217.62 | 121.10 | -96.52 |
| 149 | 500.00 | 457.03 | -42.97 |
| 211 | 500.00 | 492.41 | -7.59 |
| 307 | 500.00 | 481.35 | -18.65 |
| **mean** | **330.86** | **304.20** | **-26.66** |

### 1.2 Rule banks re-induced from the same trace

Only the action label changes: the sampled action versus the greedy action recovered
from the logged `action_probabilities_after_constraint`. States, visits and rewards
are identical.

| metric | Acrobot-v1 | CartPole-v1 |
|---|---|---|
| rules recorded / resampled / regreedy | 1694 / **1694** / 2434 | 921 / **921** / 1218 |
| coverage resampled -> regreedy | 0.2010 -> 0.2869 | 0.5601 -> 0.7394 |
| greedy-vs-sampled action disagreement | 15.06% | 23.66% |
| cross-seed conflict, sampled banks | 346 / 445 = **77.75%** | 1 / 165 = **0.61%** |
| cross-seed conflict, greedy banks | 447 / 531 = **84.18%** | 3 / 191 = 1.57% |

### 1.3 A1 findings

**F1. The train/deploy gap flips sign between environments.** On CartPole the greedy
protocol is better on 8 of 8 seeds (mean +26.66 for argmax). On Acrobot the greedy
protocol is worse: argmax returns exactly -500.00 for every one of the eight seeds,
while sampling averages -446.12. The action-selection rule is not a neutral
implementation detail; its sign depends on the environment.

**F2. On Acrobot the `best_single` selection is uninformative under the protocol it is
deployed with.** `best_single` is chosen by `training episode_return_mean` - a
*sampled-protocol* quantity. Seed 211 won that contest with -192.28. Deployed
greedily, seed 211 scores **-500.00, exactly like all seven other seeds**. The
selection variable and the deployment variable are measured under different action
rules, and under the deployment rule the selection variable is constant across
candidates. Any statement that compares fusion against "the best single actor" on
Acrobot is therefore comparing against an arbitrary member of an eight-way tie.

**F3. Cross-seed bank conflict tracks the fusion outcome.** CartPole banks agree
almost everywhere (0.61% conflict) and fusion works there. Acrobot banks disagree on
77.75% of shared conditions and fusion fails there. This is the association the
reframed claim rests on, and A1 measures it directly on the banks rather than
inferring it.

**F4. The action-selection confound does not explain away the Acrobot conflict.**
Re-inducing from greedy actions raises conflict from 77.75% to 84.18% in Acrobot. If
anything the deployment protocol makes the banks *more* inconsistent, so the confound
is not the source of the Acrobot failure. It also raises rule count and coverage in
both environments (Acrobot 1694 -> 2434, coverage 0.201 -> 0.287; CartPole 921 ->
1218, 0.560 -> 0.739), i.e. the greedy policy visits a richer set of conditions.

**F5. A follow-up experiment is now obvious and cheap.** Every recorded Acrobot
composition number (-443.37 fusion, -187.02 random, -500.00 GPI) was measured under
`argmax`, in a regime where all eight base actors are already at the floor. The rule
banks, however, were induced from *sampled* rollouts. Re-running the composition under
the sampled protocol would ask whether the composition machinery helps at all in the
regime its rules came from. This is zero-training and not yet done.

---

## 2. A2 - rankability test

Question: the bank is uninformative under every ranking rule tried so far, while
ranking-free random arbitration reaches -187.02. Does the bank contain rankable
information, or only state-conditioned action diversity?

### 2.1 The estimator is degenerate under the pipeline's own continuation policy

`mc_return_from_action` values a candidate first action by forcing it and then
following the **8-actor mean-logit ensemble**. That ensemble never terminates Acrobot
inside the 500-step horizon, so every forced first action returns exactly
-1 x 500 = -500. The estimator cannot separate the actions.

| state population | states | all three values equal | mean spread |
|---|---|---|---|
| actor-rollout states, 8-actor continuation | 80 | **100%** | **0.0** |
| always-action-0 states, 8-actor continuation | 500 | **100%** | **0.0** |
| actor-rollout states, 2-actor continuation | 30 | 0% | 91.87 |

The third row is the control that rules out a harness bug: with a 2-actor continuation
the same code discriminates fine (margins 1-22, argmax histogram `{0: 23, 1: 4, 2: 3}`).
The degeneracy is a property of the 8-actor ensemble, not of the estimator code.

Two reductions follow, and both were re-verified against the real estimator on live
decision points:

- `mc_ranked` = "take the smallest action any bank proposes" (`max` over tied values
  returns the first element of the sorted candidate-action list).
- `mc_oracle` = "always take action 0" (`max` over `range(3)` with all values tied
  returns index 0).

### 2.2 Policy-level results

100 episodes, reset seeds `20000000 + i`, identical to the recorded ablations.

| mode | mean return | median | steps | action-0 rate | paired delta vs confidence |
|---|---|---|---|---|---|
| `confidence_first` (recorded baseline) | -443.37 | -500.0 | 44,376 | 0.239 | - |
| `random_candidate` | **-187.02** | -165.0 | 18,802 | 0.294 | **+256.35** [+232.33, +278.99] |
| `mc_ranked` (value-based ranker) | **-500.00** | -500.0 | 50,000 | 1.000 | **-56.63** [-75.11, -40.14] |
| `mc_oracle` (ceiling) | **-500.00** | -500.0 | 50,000 | 1.000 | **-56.63** [-75.29, -39.59] |

Equivalence check on 12 live decision points: all-tied rate **12/12**,
`mc_ranked` equals the cheap minimum-candidate rule **12/12**, `mc_oracle` equals
constant action 0 **12/12**.

### 2.3 A2 findings

**G1. The Monte-Carlo ranker is a constant function, not merely a weak ranker.** It
does not rank badly; it does not rank at all. The failure is in the estimator's
interaction with the continuation policy, not in the bank.

**G2. Value-based ranking is the worst option of the three tested.** -500.00 against
-443.37 for confidence and -187.02 for random, with a paired delta of -56.63
[-75.11, -40.14] against confidence. Substituting a value estimate for confidence does
not help; it is significantly worse than both alternatives.

**G3. The oracle equals the floor, so A3 should not be built.** Design constraint #1
in the log said: if the oracle is only marginally better than random, perfect ranking
is worth very little and no production ranker should be built. Here the ceiling
(-500.00) is **313 points worse than random arbitration** (-187.02). There is no
headroom for any ranker restricted to this candidate set. A3 is closed, and A5 is
triggered.

**G4. The recorded MC probe has a continuation-policy mismatch and its signal does not
transfer.** `gpi_fqe/acrobot_q_action_agreement.json` reports discriminating values
(action 0 -132.92, action 1 -202.47, action 2 -157.11) and the derived statements
"MC-best first action is action 0 on 43.5% of states" and "fusion agrees with MC-best
on 22.5%". Those were computed with a **2-actor** continuation. The reported fusion
uses **8 actors**. The probe and the policy it is used to explain therefore have
different continuation policies. The probe's apparent rankable signal is an artefact
of the smaller ensemble and should not be cited as evidence that the bank contains
rankable information.

**G5. The reframed claim survives, with a stronger warrant.** The honest claim is "the
induced rule bank supplies state-conditioned action diversity, not rankable
preference". A2 upgrades this from "untested" to "the value-based route is closed":
confidence, support, mean_reward and Monte-Carlo value have now all been tried, and
the ranking-free method beats every one of them.

---

## 3. What changes

- **A3 (production ranker): do not build.** The oracle ceiling is the floor.
- **A5 (claim rewrite): triggered.** Use the G5 wording.
- **A1 F5: new zero-training experiment.** Re-run the Acrobot composition under the
  sampled protocol, since the banks were induced under sampling while every recorded
  composition number was measured under argmax.
- **Project B (complexity hypothesis): deprioritised, with a new competing
  explanation on the table.** A1 shows the deploy protocol alone destroys every
  Acrobot actor (-500.00 across the board) and that bank conflict is already 77.75%
  before any complexity argument. Task complexity is no longer the most parsimonious
  explanation for the Acrobot failure, and A1 F5 is a cheaper test of the alternative.
- **Paper text affected.** The `best_single` comparison on Acrobot (F2), and the
  MC-probe-derived statements in the diagnostics (G4), both need review.

---

## 4. Problems encountered and how they were resolved

| # | Problem | Resolution |
|---|---|---|
| 1 | Background runs were killed with SIGTERM and produced no output | Switched to foreground runs; A1 fits in 2m06s and A2 in 20s, so chunking was not needed |
| 2 | `mc_return_from_action` calls `env.reset()` and overwrites `env.unwrapped.state`, so passing the episode's own environment corrupted the episode in progress (episodes "ended" after 1 step) | Caught by a 3-episode smoke test; fixed with a dedicated MC environment, documented in the docstring |
| 3 | `KeyError: 'n_rules'` in the A1 summary block | Key name mismatch (`resampled_n_rules`); fixed |
| 4 | A claim from the previous turn ("MC rollouts run 499.5 of the 500-step cap") was circular - derived by dividing by an assumed per-step constant | Measured properly: mean 134.15 steps, median 102, only 7.5% at the cap. The A2 cost estimate survived because it was built from a directly measured per-rollout time; only the explanation was wrong |
| 5 | A2's first real run reported `mean_mc_margin = 0.0` | Investigated rather than accepted: turned out to be the substantive finding (G1), confirmed with a 2-actor control |
| 6 | `auditability/` has no `__init__.py`, so `from auditability.x import y` fails when running a file path | Launch as `python -m auditability.<script>` from `C:\AI\RL` |
| 7 | A1's 100-episode argmax did not match the recorded per-seed `heldout_mean_return` | Not a bug: the recorded per-seed evaluation uses **20 episodes** at reset seeds `seed*10000 + 900000 + i`, while the composition evaluation uses **100** at `20000000 + i`. Reproduced the 20-episode protocol exactly, 16/16 |

---

## 5. Artifacts

| Script | Output |
|---|---|
| `auditability/run_a1_deploy_protocol.py` | `results/a1_deploy_protocol.json` |
| `auditability/run_a2_rankability.py` | `results/a2_rankability.json` |
| `auditability/diagnose_mc_degeneracy.py` | `results/mc_degeneracy_probe.json`, `results/mc_degeneracy_probe_8actor.json` |
| `auditability/measure_mc_rollout_length.py` | `results/mc_rollout_length.json` |
| `auditability/benchmark_mc_arbitration_cost.py` | `results/mc_arbitration_cost_benchmark.json` |
| `auditability/profile_step_cost.py` | `results/step_cost_decomposition.json` |

All scripts are run as `python -m auditability.<name>` from `C:\AI\RL` with the
project interpreter `aim_auditability/.venv/Scripts/python.exe`.
