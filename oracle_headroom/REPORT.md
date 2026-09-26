# Oracle Headroom Experiment — Report

**Date**: 2026-09-26
**Question**: does the rule bank contain selectable signal, or is the candidate
set itself devoid of good actions? (I.e., is arbitration the bottleneck, or is
the representation the bottleneck?)
**Protocol**: pre-registered in `PROTOCOL.md` BEFORE any training/measurement.
**Environment**: CartPole-v1, 8 retrained actors (repo's `train_one`,
grammar_audit_only arm, seeds 11/29/43/71/101/149/211/307), shared 4-bin
calibration symbolizer, 100 common-reset held-out episodes.

## Verdict

**STOP — the rule-bank set, as constituted, lacks usable signal for composition.
Do not build Stage 1–4 (generative composition) on these banks.**

Pre-registered stop rule: STOP if H/T < 1/3, where H = R_oracle_conflict −
R_confidence and T = R_best − R_confidence.
Measured: **H/T = 0.306** (95% paired-bootstrap CI [0.274, 0.337]) < 0.333.
The CI upper bound marginally touches the threshold; this is reported as a
borderline STOP, not a crushing one.

## Numbers (primary, pre-registered)

| policy | mean | median | 95% CI |
|--------|------|--------|--------|
| best_single (seed 149) | 500.00 | 500.00 | [500.0, 500.0] |
| actor_mean_logits (8) | 293.83 | 285.00 | [286.9, 301.2] |
| grammar_fusion (confidence-first) | 59.68 | 12.00 | [45.7, 74.7] |
| oracle_conflict (perfect arbiter over candidates) | 194.59 | 178.50 | [184.8, 205.4] |
| oracle_full (perfect action at every covered state) | 364.60 | 500.00 | [336.6, 393.2] |

Decomposition of the fusion-vs-best gap (T = 440.32):

```
R_best − R_conf = (R_best − R_of) + (R_of − R_oc) + (R_oc − R_conf)
     440.32     =      135.40     +     170.01     +     134.91
                =  coverage/fallback + agreed-wrong-actions + arbitration
                =       31%          +        39%          +      31%
```

Reading: a perfect arbiter recovers only 31% of the gap. 69% is
representation-side — banks agreeing on wrong actions (39%) and blind spots
falling back to the ensemble (31%).

## Oracle construction (why these numbers are exact, not estimates)

CartPole-v1 dynamics are deterministic given (state, action); the continuation
policy (argmax of the best actor) is deterministic. Therefore ONE rollout from
(s, a) yields the EXACT Q(s, a) — no Monte Carlo averaging, no estimation
noise. Premises verified before the main run: set-state roundtrip 20/20,
rollout determinism 20/20. Continuation sanity gate (pre-registered):
best-actor argmax mean = 500.00 > 400 → PASS (contrast with A2's Acrobot
finding, where the ensemble continuation was degenerate and the MC oracle
was void).

## Post-hoc supplementary: the seed-307 poisoning effect (exploratory)

Seed 307 collapsed during retraining (train mean 9.7; saturated deterministic
policy; 33 rules). Excluding its bank post-hoc (7 banks, 7-actor fallback):

| policy (no-307) | mean |
|-----------------|------|
| grammar_fusion (confidence) | 335.98 |
| oracle_conflict | 322.99 |
| oracle_full | 450.87 |
| best_single | 500.00 |

H/T = −0.079: **the oracle cannot beat confidence among healthy banks.**
The no-307 ladder: arbitration gap −12.99 (noise), agreed-action gap 127.88
(78% of the 164-point gap), coverage gap 49.13 (30%).

The smoking gun: among the 7 healthy banks there were only **5 conflict steps
out of 43,510 covered steps** — the banks are near-unanimous. Arbitration is
vacuous; there is nothing for any arbiter to decide. The remaining gap is
purely that the banks *agree on suboptimal actions* (78%) plus blind spots
(30%). Consistency ≠ quality.

Three refined findings (exploratory, clearly labeled):

1. **The "oracle helps" effect was driven by the collapsed bank.** Removing it
   lifts confidence fusion 59.68 → 335.98. Mechanism: a saturated deterministic
   policy emits HIGH-confidence rules for BAD actions, and confidence-first
   (= frequency) actively selects FOR them. This is a **robustness failure of
   the bank set**, not a uniform absence of signal.
2. **Arbitration is not the bottleneck for healthy banks.** Confidence-first
   already matches the perfect arbiter (H ≤ 0). The A2-era worry about the
   arbiter is resolved in this regime.
3. **The remaining gap (336 → 500) is representation-side**: the 7 healthy
   banks are near-unanimous (5 conflicts / 43,510 covered steps), so 78% of the
   remaining gap is agreed-but-suboptimal actions and 30% is blind spots.
   Consistency does not imply quality.

## Unified reading

The banks are highly self-consistent — they agree with each other almost
everywhere. That consistency is the double-edged finding: a collapsed bank is
*confidently* wrong (frequency ≠ quality), and healthy banks agree on actions
that are good-not-great. The oracle separates the two questions cleanly:
"do the banks disagree?" (arbitration — answered: almost never, and
confidence already matches the oracle) vs "are the agreed actions good?"
(representation — answered: no, 128 of 164 points lost there).

Note on look-ahead: seed 307 was identified as collapsed from training
outcomes, then excluded — this is exploratory. A confirmatory version would
pre-register a bank-admission health check (e.g., source-actor eval mean
threshold) before seeing fusion outcomes.

## What this means for the paper

- The pre-registered STOP stands: do not build Stage 1–4 on the banks as-is.
- The refined reading redirects the research, rather than merely blocking it:
  the next question is not "a better arbiter" (answered: unnecessary for
  healthy banks) but **bank admission robustness** (one degenerate bank
  poisons the set: 336 → 60, and even the oracle only recovers to 195) and
  **agreed-action quality / coverage** (the remaining representation gap).
- For the paper's narrative: this adds a fourth "不代表" — candidate sets do
  not guarantee selectable good actions — with a mechanism (frequency≠quality
  under policy collapse) rather than just a number.

## Artifacts

- `PROTOCOL.md` — pre-registered protocol (written before any run)
- `stage0_train.py`, `oracle_q.py`, `oracle_fusion.py`, `oracle_diagnostics.py`
- `supplement_no307.py`, `supplement_no307_full.py` (post-hoc, labeled)
- `results/oracle_headroom_cartpole.json` — primary machine-readable results
- `results/oracle_diagnostics.json` — decomposition + verdict
- `results/supplement_no307.json`, `results/supplement_no307_full.json`
- `runs/CartPole-v1/` — retrained actors, rule banks, symbolizer, manifests
- `logs/` — complete stdout/stderr + timestamped logs for every stage
- `EXPERIMENT_LOG.md` — chronological log
