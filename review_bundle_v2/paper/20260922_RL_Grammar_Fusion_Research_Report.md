# Seed-Specific RL Grammar Fusion: Research Report

**Date:** 2026-09-22  
**Workspace:** `C:\AI\RL\`  
**Purpose:** Record the first executable test of the rule-level fusion proposal in `20260922.claude.md` and define a defensible paper claim.

## Executive result

The report's proposed rule-level fusion is now implemented and tested using the trained CartPole agents from seeds 11 and 29. The prototype makes cross-agent rule agreement, action conflicts, uncovered states, rule provenance, fallback decisions, and the merged policy's complete decision trace visible and checkable.

At the permissive rule threshold (minimum support 8, confidence >=0.70), the two learned grammars shared 73 of 134 distinct state-symbol/action productions (Jaccard 0.545). On a common 200-episode evaluation, the fused policy used at least one rule on 58,272 of 100,000 decisions, marked 41,728 decisions (41.7%) as blind spots, and preserved the stronger agent's mean return of 500. The latter is a ceiling result: the stronger source agent also scored 500, so fusion did not improve return. The merged trace passed hash-chain validation, exact environment replay, and source-rule provenance checks.

At confidence >=0.90, only 3 productions were shared (Jaccard 0.088), and none of the 100,000 held-out merged-policy decisions matched a rule. The system therefore used the neural fallback for every decision. This is direct evidence that rule overlap alone does not establish usable fusion coverage.

## Research question and scope

The supplied proposal asks whether an auditable symbolic rule layer can support transparent RL-agent fusion by making three things explicit: conflicts between agents, states with no reliable rule, and the source of every fused decision. This report tests that operational claim on two already-trained CartPole agents. It does not assume that grammar fusion improves RL return.

The CartPole state-action grammars use data-fitted quantile bins and induced state-symbol-to-action productions. This is a controlled rule-learning representation; it is **not** the AIM model's emergent message language. Evidence that AIM symbols causally affect decisions is reported separately below.

## Methods

### Source agents and grammar overlap

Both source policies were trained for 300 REINFORCE episodes. Grammar observation ran in the `grammar_audit_only` arm, whose actor weights and environment trace were independently verified to match the no-observer baseline. Each seed fitted its own four-bin-per-dimension quantile symbolizer from 24 calibration episodes. Rule support was at least 8; confidence thresholds were 0.90 and 0.70.

Exact rule overlap uses production identity `(state-symbol, action-symbol)`. The report also maps each raw observation through both agents' own frozen quantizers and compares their rule actions on the same recorded states. This second measure avoids assuming that same-numbered bins from independently fitted quantizers have identical raw-state boundaries.

### Fusion policy

At each raw state, the merger queries both agents' state symbolizers and rule banks:

- If both agents provide the same action, the merger emits the shared action and records both sources.
- If only one provides a rule, it emits that action and records the supplying agent and rule.
- If both provide different actions, it records a conflict and chooses by confidence, then support, then mean rule reward. An exact tie falls back to the neural policy.
- If neither provides a rule, it marks a blind spot and uses a neural fallback selected by training mean return. No held-out evaluation return is used to choose the fallback.

Each merged decision records both candidate rule IDs, the candidates' grammar hashes and source-step provenance hashes, the selected rule source(s), or the fallback agent. The verifier resolves each reference against the original grammar and checks the emitted action. A second pass verifies the hash chain and replays every action in CartPole from its recorded reset seed, comparing state, next state, reward, termination, and truncation.

### Evaluation

The fusion experiment evaluates both source actors and the fused policy on the same 200 reset seeds (`9,000,000 + episode_index`). The resulting 100,000 fused decisions are saved in a hash-chained JSONL trace. We report return, rule-use categories, conflicts, blind spots, source provenance checks, and exact replay. This is a two-agent, one-environment prototype rather than a benchmark of general policy composition.

## Results

### Cross-seed rule core

| Confidence threshold | Seed 11 rules | Seed 29 rules | Shared | Union | Jaccard |
|---|---:|---:|---:|---:|---:|
| >=0.90 | 26 | 11 | 3 | 34 | 0.088 |
| >=0.70 | 113 | 94 | 73 | 134 | 0.545 |

There were no action disagreements among conditions with the same symbolic condition at either threshold. However, the calibration edges differed between seeds: the largest absolute edge difference was 0.1569 CartPole observation units. Therefore symbolic-set Jaccard is a fast syntactic estimate, not a semantic equivalence proof.

On the raw training states, the >=0.90 grammars jointly covered only 0.25% and 0.26% of the seed-11 and seed-29 traces, respectively; their selected actions agreed on every jointly covered step. At >=0.70, joint coverage was 24.6% on seed 11's trace and 28.9% on seed 29's trace. Actions agreed on 99.992% and 99.991% of jointly covered steps, with one disagreement in each trace. These are occupancy-weighted training-state agreement statistics, not independent held-out causal tests.

### Fused-policy evaluation

| Rule confidence | Agent 11 mean return | Agent 29 mean return | Fused mean return | Held-out blind spots | Fused trace replay |
|---|---:|---:|---:|---:|---|
| >=0.90 | 500 | 131.085 | 500 | 100,000 / 100,000 (100%) | PASS, 100,000 decisions |
| >=0.70 | 500 | 131.085 | 500 | 41,728 / 100,000 (41.7%) | PASS, 100,000 decisions |

At >=0.70, both agents' rules agreed on 10,137 held-out decisions; exactly one source rule applied on 48,135; and neither agent had a rule on 41,728 decisions. No conflict occurred in these held-out episodes. Three state-level conflicts did occur in the two source training traces (two on seed 11's trace, one on seed 29's trace); the merger exposed the opposing candidate rules and resolved them using its declared priority order.

On the 100,000-decision >=0.70 merged trace, 68,409 chosen rule references resolved to source grammars and source-step hashes; all 41,728 fallback references resolved to the selected seed-11 actor. Environment replay and chain verification both passed. A reward-tampering probe on a copy of the trace was rejected at line 1.

The fused policy's return equaled seed 11's 500-point ceiling on every evaluation episode. The paired mean difference and percentile bootstrap interval versus seed 11 were all zero. This supports a claim of transparent, auditable composition in this controlled case; it says nothing about return improvement.

### Trajectory-grammar MDL by induction round

For the seed-11, confidence >=0.70 audit-only trace, a lossless pair-substitution grammar was re-induced on the complete raw trace prefix at each 50-episode round, with 16 pair rules. All six prefixes round-tripped exactly. The declared two-part code measures `L(G)` (terminal dictionary and productions) plus `L(trace|G)` (encoded start sequence). It uses Elias-gamma lengths and fixed-width symbol references; fixed schema overhead is excluded and provenance storage is reported separately.

| Completed episodes | Productions | MDL bits | Raw UTF-8 trace bits | MDL/raw ratio |
|---:|---:|---:|---:|---:|
| 50 | 16 | 5,431,716 | 10,596,512 | 0.513 |
| 100 | 16 | 22,216,132 | 41,464,272 | 0.536 |
| 150 | 16 | 79,681,129 | 145,056,288 | 0.549 |
| 200 | 16 | 149,020,329 | 265,595,528 | 0.561 |
| 250 | 16 | 218,044,849 | 379,408,112 | 0.575 |
| 300 | 16 | 281,958,036 | 490,363,440 | 0.575 |

This score measures the lossless trajectory pair grammar, not the state-action policy rule bank. The pair-substitution inducer does not optimize MDL, and ordinary gzip remains smaller than the serialized grammar archive in the existing byte comparison. The MDL table is descriptive evidence, not a compression-superiority claim.

## Separate AIM-symbol causal evidence already available

The existing private-target coordination experiment contains six independent 100,000-decision AIM runs (600,000 logged decisions total). The receiver's sender-sampled accuracy averaged 0.998923, versus 0.503292 for shuffled messages and 0.500 for the no-message control. The matched-support intervention on the information-bearing AIM token position changed the receiver's action probability by a mean 0.998366 (per-seed range 0.996986–0.999685; common-support overlap mean 0.94308). Which token position carried the signal varied across seeds.

This supports the bounded claim that learned AIM symbols caused a receiver decision in that private-parity game. It does not give every symbol a human-readable natural-language meaning, and the AIM traces do not contain all optimizer, gradient, or RNG states needed to claim complete training-process auditability. The CartPole cold reruns separately show exact same-host reproducibility of all logged transitions, per-episode update fingerprints, final actor/optimizer/RNG checkpoint, and actor weights for seeds 11 and 29; across those two seeds 135,427 distinct transitions were each replayed twice.

## Proposed paper positioning

### Defensible title

**From Replayable Trajectories to Rule-Level RL Fusion: Provenance, Shared Cores, and Explicit Blind Spots**

### Candidate abstract

Reinforcement-learning audit claims span distinct questions: whether logged decisions can be replayed, whether learned symbols causally affect decisions, and whether rules inferred from separate agents can be combined transparently. We evaluate these questions separately in controlled experiments. Across six seeds of an AIM private-target coordination task, matched-support interventions on the information-bearing learned symbol changed receiver action probability by 0.996986–0.999685. For CartPole, two 300-episode agents produced state-action grammars with three shared productions at 0.90 confidence and 73 at 0.70 confidence (Jaccard 0.088 and 0.545). A provenance-aware merger marked conflicts and uncovered states explicitly, selected rules by declared confidence/support/reward priorities, and emitted a hash-chained trace. In a 200-episode evaluation, the permissive merger used rules for 58.3% of 100,000 decisions and fell back on 41.7%; every selected source rule resolved to its originating grammar and every transition replayed exactly. Return matched the stronger agent's 500-point ceiling but did not improve it. These results support causal symbol use in one designed task and auditable rule composition in one deterministic benchmark, while showing that replayability, causal explanation, cross-seed stability, and performance are separate properties.

### Main claim to defend

In these controlled settings, symbolic RL artifacts can support **bounded causal testing and provenance-preserving rule-level composition**: an auditor can see which source rule selected a decision, which conflicts were detected, and where the available rule banks had blind spots; the resulting decision trace can be checked and replayed exactly.

### Claims not supported

- “AI mother tongue makes arbitrary RL training fully auditable.”
- “Grammar constraints reliably improve RL performance.”
- “The learned rules are natural-language explanations of all model decisions.”
- “The current two-seed rule intersection establishes a universal, semantically identical policy core.”
- “Rule-level merging outperforms weight averaging, GPI, or other policy-composition methods.” No such baseline comparison was run.

## Related work and positioning

- PIRL learns policies in a human-readable domain-specific program language and supports symbolic verification; it is a relevant comparison for interpretable, executable policy rules. [Verma et al., ICML 2018](https://proceedings.mlr.press/v80/verma18a.html).
- Successor features and Generalized Policy Improvement combine knowledge across policies using value-based action selection; geometric policy composition studies another explicit policy-composition route. Our prototype focuses on rule provenance and visibility, and does not claim better returns than these methods. [Barreto et al., NeurIPS 2017](https://papers.nips.cc/paper/6994-successor-features-for-transfer-in-reinforcement-learning), [Thakoor et al., ICML 2022](https://proceedings.mlr.press/v162/thakoor22a.html).
- The SEQUITUR paper defines a sequence grammar-compression method and cautions that its generated rules are not automatically a generalized grammar. Our pair-substitution trace codec is Re-Pair-like and is not canonical SEQUITUR; exact trace reconstruction is kept separate from policy-rule generalization. [Nevill-Manning and Witten, 1997](https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume7/nevill97a.pdf).
- Emergent-communication evaluation work distinguishes message/action correlation from causal influence and motivates interventions. [Lowe et al., AAMAS 2019](https://arxiv.org/abs/1903.05168).
- Deterministic deep-RL reproducibility requires explicit control and documentation of environment, seed, software, and hardware sources of nondeterminism. Our exact rerun result is bounded to the recorded CPU/software configuration. [Nagarajan et al., 2018](https://arxiv.org/abs/1809.05676).

## Remaining work before external submission

1. Repeat rule overlap and fusion on more independent seeds and use one shared frozen state symbolizer, or a principled raw-state alignment, for the primary shared-core statistic.
2. Evaluate on a task without a 500-point return ceiling so the fusion policy can show meaningful improvement or degradation relative to each source agent and explicit policy-composition baselines.
3. Run the planned return-weighted rule-bank ablation with a declared normalization rule. Keep it outside the current paper's supported claims until measured.
4. Decide whether to present the AIM causal-symbol study and CartPole fusion study as one evidence-layer paper or two focused papers after coauthor review. They support related but distinct claims.
5. Choose a target venue and check its current call, format, and artifact requirements before submission. This local report is not an external submission.

## Reproducibility artifacts

- Grammar overlap: `aim_auditability/runs/grammar_intersections/confidence_090_audit_only.json`, `confidence_070_audit_only.json`.
- Fused evaluation and decision trace: `aim_auditability/runs/grammar_merge_070_final/merge_evaluation.json`, `merged_policy_trace.jsonl`.
- Strict-threshold fallback case: `aim_auditability/runs/grammar_merge_090_final/merge_evaluation.json`.
- Per-round MDL: `aim_auditability/runs/grammar_mdl_seed11_070.json`.
- Training repeatability: `aim_auditability/runs/repeatability_full/`.
- AIM six-seed causal aggregate: `aim_auditability/runs/replication_summary.json`.
- Detailed append-only experiment history: `AI_Grammar_Induction_Experiment_Log.md` and `AIM_Auditability_Research_Log.md`.

## 2026-09-22 update: shared symbolizer, held-out behavior, Acrobot arbitration, and GPI checks

This update supersedes the earlier report's two-seed/seed-specific-symbolizer fusion snapshot for the shared-core results. The original results above are retained as historical evidence from the earlier experiment.

### Eight-seed shared-symbolizer results

Eight CartPole and eight Acrobot source agents (seeds 11, 29, 43, 71, 101, 149, 211, 307) were trained for 300 episodes with one pooled frozen quantizer per task, fitted on separate random-policy roll-ins. With the same symbolizer fixed across source seeds, exact seed-11/29 state-action Jaccard is 0.5833 on CartPole (77 shared rules, 132 union) and 0.0496 on Acrobot (21 shared, 423 union); 77 of 98 Acrobot shared conditions have conflicting actions. Mean all-pairs Jaccard across the eight seeds is 0.5572 on CartPole and 0.0956 on Acrobot.

### Experiment B: independent raw-state agreement

Fresh held-out deterministic rollouts supplied 5,000 raw states from each actor's occupancy. Both CartPole actors were evaluated on the same 10,000 states. Seed 11 and seed 29 agreed on 51.38% of argmax actions; the episode-cluster balanced estimate is 51.36% (95% bootstrap interval 50.48%–52.26%). The action agreement is near chance for this two-action task despite exact grammar Jaccard 0.5833. This is direct evidence that rule-set overlap alone does not support a high behavioral-consensus claim.

### Acrobot: high coverage, high conflict, and arbitration sensitivity

The Acrobot fusion policy averaged -443.37 versus -500 for the best single source actor (paired delta +56.63; 95% interval [+39.32, +75.15]), with median -500 and 39/100 episodes terminating before the 500-step cap. Its decisions were 95.70% rule-covered, 89.24% conflicting, and 4.30% blind spots. Therefore a high blind-spot rate does not account for this result.

In matched-seed arbiter comparisons, confidence-only scored -444.49, mean-reward-only and support-first scored -500, and conflict fallback to the actor ensemble scored -500. Random choice among available source rules scored -187.02 on one selector stream and between -206.17 and -194.10 on five independent selector streams (mean -198.82). A separate uniform-random-action policy averaged -498.71 across five streams. The random-rule result is not reproduced by unconditioned action noise alone; however, the rule-availability state-conditioning and source-selection mechanism require more experiments before attribution or generalization.

### GPI separation and quality limits

FQE-GPI was reported separately from the grammar fusion result. On CartPole, it achieved mean return 9.34, versus 163.65 for the best actor; on 31,563 held-out states its greedy action agreed with the actor-logit ensemble on 35.27%. On Acrobot, FQE-GPI achieved -499.57 and selected action 0 on all 50,000 states in its held-out diagnostic. Acrobot's all--500 source episodes do not provide high-/low-return strata to test value direction.

On 200 Acrobot states, seed-11 and seed-29 FQE output vectors had Pearson correlation 0.652, but their greedy action choices were highly divergent; the fused action matched seed-11 Q argmax on 64.5%, seed-29 on 0%, multi-critic GPI on 0%, and the highest action return under the fixed actor-ensemble continuation on 22.5% (95% interval 16.5%–28.5%). The last comparison uses exact deterministic finite-horizon Acrobot rollouts after each forced first action, followed by the fixed mean-logit actor. It estimates action values for that continuation policy, not optimal Q-star. Current critic checks reject FQE-GPI as a credible policy-composition baseline for this paper version.

### Updated claim boundary and artifact pointers

These results demonstrate exact logged-trace integrity/replay under the audited software setup and provenance-visible rule-level fusion in the tested tasks. They do not show that shared grammar rules reliably capture common neural actions, that current confidence arbitration is optimal, or that the learned rules identify causal decision reasons. The random-rule result and near-chance Experiment B agreement are findings to report, not evidence to hide.

The canonical code now lives in `auditability/`, full traces/checkpoints in local `runs/shared_multienv_gpi/`, and compact review copies in `results/shared_multienv_gpi/`. The full diagnostic outputs include `CartPole-v1/behavioral_agreement_experiment_B.json`, `Acrobot-v1/grammar_fusion_diagnostics.json`, `Acrobot-v1/random_action_control.json`, and the `gpi_fqe/` reports. The root `README.md` describes the experiment settings, commands, package layout, and release boundaries.
