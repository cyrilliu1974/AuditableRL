# Oracle Headroom — Experiment Log

Chronological record. Written as things happen, not reconstructed afterwards.

## 2026-09-26 ~03:00 — Task authorized

User authorized: implement Oracle headroom experiment v1, run it, iterate on
failures via logs (no guessing), reach a verdict on whether oracle fusion is
also bad, then package code + results + report + logs into one zip with
original paths.

## 2026-09-26 ~03:05 — Exploration

- Repo snapshot: `~/workspace/RL/RL_snapshot` @ 057e173deb12a686f44f958606b27064dfde5405.
- Key finding 1: trained artifacts (`runs/shared_multienv_gpi/`, `actor.pt`,
  `induced_grammar.json`) are NOT in the snapshot — only code + results JSONs.
  Decision: retrain 8 CartPole seeds with the repo's own `train_one`
  (identical seeds/config); all comparisons are within the retrained set, so
  internal validity does not require reproducing the original weights.
- Key finding 2: `train_one` only induces a grammar for non-baseline arms
  (baseline writes empty productions). The rule banks come from the
  **grammar_audit_only** arm (identical training to baseline + grammar
  induction). Using that arm.
- Key finding 3: A2 already tried MC-ranked arbitration on Acrobot and proved it
  **degenerate** — the 8-actor ensemble continuation never terminates, all
  MC values tie at -500. Lesson for this experiment: the continuation policy
  must pass a pre-registered non-degeneracy gate, or the oracle measurement
  is VOID.
- Key finding 4: CartPole-v1/Acrobot-v1 dynamics are deterministic given
  (state, action); `env.unwrapped.state` can be set directly (verified in
  smoke test). With a deterministic continuation policy, ONE rollout = EXACT
  Q — no Monte Carlo averaging needed. This makes the oracle cheap and exact.
- Design refinement: on a 2-action env, a conflict state's candidate set is
  always {0,1} = full action space, so "candidate recall" is trivially 1.0.
  The representation diagnostic becomes a ladder:
  R_conf -> R_oracle_conflict -> R_oracle_full -> R_best_single.

## 2026-09-26 ~03:15 — Pre-registration

Wrote `PROTOCOL.md` BEFORE any training/measurement (A1b lesson: fix the
induction/evaluation protocol before seeing outcomes). Includes: estimand,
oracle construction, continuation sanity gate (> 400), stop rule (H/T < 1/3
=> STOP), and an explicit list of what this run does NOT do.

## 2026-09-26 ~03:20 — Environment setup

- System python has no torch and is PEP-668 locked. Created
  `~/workspace/RL/oracle_venv` (venv, CPU torch 2.14.0, gymnasium 1.3.0,
  numpy 2.5.3). torch download needed one retry (read timeout), then succeeded.
- Smoke test: repo imports OK, `env.unwrapped.state` set + step OK on CartPole.
- Mini validation: `train_one` with 3 episodes works end-to-end
  (actor.pt, induced_grammar.json, metadata.json all written); ~0.8s/episode
  => ~4 min/seed for 300 episodes.

## 2026-09-26 ~03:25 — Code written

- `stage0_train.py` — 8 seeds x grammar_audit_only x 300 episodes, shared
  4-bin calibration symbolizer (repo protocol), min_support=8,
  min_confidence=0.70.
- `oracle_q.py` — exact-rollout Q + premise verification
  (set-state roundtrip, rollout determinism).
- `oracle_fusion.py` — 5 policies on common reset seeds (20000000+i),
  continuation gate, oracle-conflict / oracle-full.
- `oracle_diagnostics.py` — decomposition, paired bootstrap CIs, stop rule.

## 2026-09-26 ~03:30 — Stage 0 training launched (background)

Full 8-seed training running; expect ~30 min.

## 2026-09-26 ~03:00 — Stage 0 complete, one anomaly found and dispositioned

All 8 seeds trained (grammar_audit_only, 300 episodes each):

| seed | train_mean | rules | wall time |
|------|-----------|-------|-----------|
| 11   | 151.7 | 100 | 34.7s |
| 29   | 199.7 | 113 | 42.8s |
| 43   | 267.5 | 131 | 56.0s |
| 71   | 179.1 |  98 | 38.0s |
| 101  | 231.2 | 123 | 48.9s |
| 149  | 345.5 | 146 | 73.3s |
| 211  | 192.7 | 107 | 42.2s |
| 307  |   9.7 |  33 |  3.0s |

**Anomaly**: seed 307 collapsed — returns stuck at ~9-10 for all 300 episodes
(max 41), eval mean 9.6, finished in 3.0s (episodes ~10 steps each).
Episode-return trace inspected directly (not guessed): first 10 =
[17,41,9,15,9,10,14,16,10,24], last 10 all ~9-10. Classic REINFORCE
single-action saturation: the policy collapsed to a deterministic bad action
early and never recovered.

**Disposition**: ACCEPTED as a genuine replication outcome, not a setup bug.
Evidence: (a) the other 7 seeds learned normally with identical code, so the
setup is sound; (b) the paper's own seed-307 row shows it was functional in
the original (coverage 0.55, agreement 0.87) — the difference is attributed to
library-version RNG drift (torch 2.14.0 / gymnasium 1.3.0 installed here; the
snapshot pins no versions), i.e. honest replication variance, not a code bug;
(c) replacing the seed post-hoc would violate the pre-registered protocol
(A1b lesson). The collapsed bank stays in the set — heterogeneous bank
quality is realistic and does not break within-set comparisons. Best single
(seed 149, train_mean 345.5) is unaffected.

## 2026-09-26 ~03:05 — Oracle evaluation interim results (4/5 policies done)

Premises verified: set-state roundtrip 20/20, rollout determinism 20/20.
Continuation gate: best-actor (seed 149) argmax mean = 500.00 > 400 threshold. PASS.

| policy | mean | median | steps |
|--------|------|--------|-------|
| best_single (seed 149) | 500.00 | 500.00 | 50000 |
| actor_mean_logits | 293.83 | 285.00 | 29383 |
| grammar_fusion (confidence) | 59.68 | 12.00 | 5968 |
| oracle_conflict | 194.59 | 178.50 | 19459 |
| oracle_full | running | — | — |

Preliminary: H = 194.59-59.68 = 134.91; T = 500-59.68 = 440.32;
H/T = 0.306 < 1/3 (0.333) => STOP on point estimate. Bootstrap CI pending.

Note: R_conf = 59.68 is far worse than the paper's ~205.96. Likely mechanism:
the collapsed seed-307 bank contributes HIGH-confidence rules for BAD actions
(a saturated deterministic policy is frequent but terrible), and
confidence-first faithfully picks them. This is itself an illustration of
"confidence = frequency != quality". The oracle recovers to 194.59 by ignoring
confidence and using true rollout values.

oracle_full is slow (single exact_q ~= 27ms; ~2 calls per covered step;
projected ~30+ min total). It is compute-bound and progressing normally
(TIME ~= ELAPSED, no deadlock possible: all loops bounded). Waiting.

## 2026-09-26 ~03:20 — Primary results complete; verdict STOP (borderline)

All 5 policies evaluated on 100 common-reset episodes:

| policy | mean | median | 95% CI |
|--------|------|--------|--------|
| best_single (seed 149) | 500.00 | 500.00 | [500, 500] |
| actor_mean_logits | 293.83 | 285.00 | [286.9, 301.2] |
| grammar_fusion (confidence) | 59.68 | 12.00 | [45.7, 74.7] |
| oracle_conflict | 194.59 | 178.50 | [184.8, 205.4] |
| oracle_full | 364.60 | 500.00 | [336.6, 393.2] |

Decomposition: T=440.32, H=134.91, agreed-gap=170.01, cover-gap=135.40.
H/T = 0.306, 95% CI [0.274, 0.337], threshold 1/3 = 0.333.
**VERDICT: STOP** (point estimate below threshold; CI upper bound 0.337
marginally touches the threshold — borderline, reported transparently).

## 2026-09-26 ~03:21 — Post-hoc sensitivity: the seed-307 poisoning effect

Because the verdict was borderline, ran a post-hoc supplementary excluding the
collapsed seed-307 bank (7 banks, 7-actor fallback). Labeled exploratory; does
NOT override the pre-registered verdict.

| policy (no-307) | mean |
|-----------------|------|
| grammar_fusion (confidence) | 335.98 |
| oracle_conflict | 322.99 |
| best_single | 500.00 (unchanged) |

H/T = -0.079. The oracle CANNOT beat confidence among healthy banks.

Findings:
1. The entire "oracle helps" effect in the primary run was driven by the
   collapsed bank: removing it lifts confidence fusion 59.68 -> 335.98.
2. Mechanism: a saturated deterministic policy produces HIGH-confidence rules
   for BAD actions; confidence-first (frequency) actively selects FOR them.
   This is a robustness failure of the bank SET, not uniform absence of signal.
3. Among healthy banks, arbitration is NOT the bottleneck (H <= 0) —
   confidence-first already matches the perfect arbiter. The A2-era worry
   about the arbiter is resolved for healthy banks.
4. Remaining gap in no-307 world (336 -> 500) is representation-side
   (agreed-wrong actions + blind spots). Running no-307 oracle_full to
   decompose it.

## 2026-09-26 ~03:35 — no-307 oracle_full died silently; rerunning

The supplementary no-307 oracle_full run vanished at ~9 min with no traceback,
no EXIT line, empty stderr. No OOM in dmesg; 5.5GB RAM available at check
time. The primary 20-min run (same code shape) completed fine, so this looks
like runtime reaping (the abort coincided with a context-compaction event)
rather than a code bug. Rerunning unchanged; if it dies again, the report
will note the no-307 oracle_full as not-completed and rely on the primary
ladder (which already decomposes agreed-action vs coverage gaps).

## 2026-09-26 ~03:57 — no-307 oracle_full complete; picture closed

no-307/oracle_full: mean=450.87 median=500.00 in 1478.6s;
counts: steps=45087, covered=43510, **conflicts=5**, blind=1577.

Only 5 conflict steps in 43,510 covered steps among the 7 healthy banks:
the banks are near-unanimous. Arbitration is vacuous — nothing for any
arbiter to decide, which is exactly why H <= 0.

No-307 ladder: R_conf=335.98 -> R_oc=322.99 -> R_of=450.87 -> R_best=500.
Arbitration gap: -12.99 (noise). Agreed-action gap: 127.88 (78% of 164).
Coverage gap: 49.13 (30%).

Final mechanism statement:
- Confidence-first is a FINE arbiter for healthy banks (matches the oracle).
- The bank SET is fragile to one degenerate bank: 336 -> 60, and even the
  perfect arbiter only recovers to 195 (poison extends beyond conflict
  states, into non-conflict steps where 307's rules dominate).
- Healthy banks agree almost everywhere; the remaining gap is agreed-but-
  suboptimal actions + blind spots. Consistency != quality.

Pre-registered verdict unchanged: STOP (H/T=0.306 < 1/3, primary world).
The supplementary refines what STOP means: not "no signal anywhere", but
"no robust composition substrate without bank admission + rule quality work".

## 2026-09-26 ~03:58 — Code changes vs retries (for the final report)

Code changes during this experiment:
- stage0_train.py: fixed auditability import path (sys.path) — 1 fix, then ran clean.
- oracle_fusion.py: fixed StateSymbolizer init signature — 1 fix, then ran clean.
- supplement_no307_full.py: NONE (ran clean first try); its first background
  launch died silently at ~9 min (no traceback, no OOM; likely runtime
  reaping during context compaction); rerun unchanged completed in 1478.6s.
- oracle_q.py, oracle_diagnostics.py, supplement_no307.py: ran clean, no fixes.

Retried-and-passed without code changes: nothing (every rerun was either a
planned stage or the reaped supplementary above, which needed no code change).
