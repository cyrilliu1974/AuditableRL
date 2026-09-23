# From Replayable Traces to Provenance-Aware RL Rule Fusion

> Manuscript status: evidence-aligned draft, 2026-09-22. Claims are limited to the evaluated prototype, tasks, seeds, and execution setup. The study demonstrates existence of an end-to-end auditable workflow; it does not establish general or complete auditability for arbitrary RL training.

## Abstract

Replayable trajectories, causal symbol use, shared symbolic rules, and policy composition are distinct properties. We report a bounded end-to-end study that measures each separately. In a designed AIM private-target coordination task, matched-support interventions on the information-bearing learned symbol changed receiver action probability by 0.996986–0.999685 across six seeds. In CartPole-v1 and Acrobot-v1, eight 300-episode agents per task used a shared frozen state symbolizer. Seed 11/29 exact rule Jaccard was 0.5833 for CartPole and 0.0496 for Acrobot, but fresh-state CartPole action agreement was only 51.36% (95% episode-cluster bootstrap interval 50.48%–52.26%). This counterexample shows that grammar overlap is not behavioral consensus. A provenance-aware fusion system records source rules, arbitration, blind spots, and fallbacks; its hash-linked decisions can be checked against rule sources and replayed against the environment. CartPole fusion reached 205.96 mean return versus 163.65 for the best single actor, while some policies hit the 500-step ceiling. Acrobot fusion reached -443.37 versus -500 for the best actor, with 95.70% coverage, 89.24% conflict, and 4.30% blind spots. Ablations exposed strong sensitivity to arbitration, including an unexpected advantage for random selection among available source rules. FQE-GPI diagnostics failed quality checks and are reported separately. The evidence supports existence of auditable trace and rule-composition mechanisms in these settings; generality, universal causal explanation, and complete auditability of arbitrary RL training remain unproven.

## 1. Introduction

“RL auditability” can refer to different questions: whether a recorded execution is intact and replayable; whether a learned symbol causally changes a decision; whether symbolic rules capture behavior shared across agents; and whether those rules can be composed with visible provenance. Passing one test does not imply passing the others. We therefore treat them as separate claims and report counterevidence where a tempting inference fails.

This study evaluates a bounded existence claim: an RL workflow can expose replayable execution records, test a learned symbol's causal influence in a designed communication task, and compose learned state-action rules while identifying their sources, conflicts, and uncovered states. It does not claim that these mechanisms reveal every internal cause of a neural policy or make arbitrary RL training fully auditable.

Contributions:

1. A provenance-aware rule-fusion prototype with explicit source attribution, conflict arbitration, blind-spot detection, and fallback logging.
2. Shared-symbolizer, multi-seed measurements of rule overlap on CartPole-v1 and Acrobot-v1, paired with a fresh raw-state behavioral-agreement test.
3. End-to-end artifact checks covering hash-linked records, source-rule resolution, training/replay evidence, and environment transition replay under the recorded setup.
4. A causal intervention result for learned AIM symbols in a designed task, kept distinct from state-action rule fusion and from general natural-language interpretability.
5. Arbitration and FQE-GPI diagnostics that bound the policy-composition conclusions and identify unresolved mechanisms.

## 2. Methods

### 2.1 Shared-symbolizer RL experiments

We trained REINFORCE agents on CartPole-v1 and Acrobot-v1 with seeds 11, 29, 43, 71, 101, 149, 211, and 307, for 300 episodes per seed and arm. Within each environment, a pooled calibration procedure produced one frozen four-bin quantile symbolizer shared across agents. Rule admission required support of at least 8 and confidence of at least 0.70. Baseline and audit-only arms were checked for training equivalence. The state-action rule bank is distinct from the lossless sequence grammar used to encode trajectories: neither representation should be treated as the other.

Exact rule overlap is Jaccard similarity over `(state-symbol, action-symbol)` productions. This is a symbolic-set statistic. It is not a semantic equivalence test. Experiment B separately compares two actors' deterministic argmax actions on the same fresh raw states, sampled equally from each actor's held-out occupancy.

### 2.2 Rule fusion and evaluation

At each state, the merger queries both source rule banks. Agreement, single-source coverage, conflict, or no coverage is recorded. Conflicts use the declared confidence, support, and mean-reward priority; ties use the actor fallback. Blind spots use a source actor selected from training-return metadata, without evaluation returns. Each decision links selected rules to source grammars and source-step provenance, records the fallback or arbitration outcome, and participates in a hash chain. Validation checks record integrity, resolves rule references, and replays logged actions in the environment against state, next-state, reward, termination, and truncation fields.

Primary policy evaluation uses 100 matched reset seeds per task. Acrobot arbiter ablations use the same reset seeds; random source-rule selection is repeated with five independent selector streams. A separate five-stream uniform-random-action control measures unconditioned action noise. All return comparisons are descriptive of these policies and test conditions, not claims about optimal arbitration.

### 2.3 AIM symbol intervention and value baseline

The AIM causal-symbol experiment is a separate six-seed private-target coordination task. Matched-support interventions vary the information-bearing learned token while controlling receiver context and the other token. It tests whether that symbol changes a receiver decision in the designed game; it does not test whether every AIM symbol has a human-readable meaning or explains a general RL policy.

FQE-GPI is treated as a diagnostic baseline. We inspect held-out action agreement, value direction where identifiable, cross-seed Q estimates, and finite-horizon counterfactual action comparisons. Because the value estimates failed key checks, their policy scores are not merged into the primary grammar-fusion conclusion.

## 3. Results

### 3.1 Shared symbolic rules and behavioral agreement

Across eight agents, mean pairwise exact grammar Jaccard was 0.5572 on CartPole and 0.0956 on Acrobot. For seeds 11 and 29, CartPole had 77 shared productions out of 132 in the union (Jaccard 0.5833), with no action conflicts among those exact shared conditions. Acrobot had 21 shared productions out of 423 in the union (Jaccard 0.0496); 77 of 98 shared conditions had conflicting actions.

On 10,000 fresh CartPole raw states, the seed-11 and seed-29 actors agreed on 51.38% of actions. The episode-cluster balanced estimate was 51.36% (95% interval 50.48%–52.26%). Thus, a substantial exact grammar intersection did not imply strong behavioral consensus. This is a central negative result and rules out using Jaccard alone as evidence that agents share a common policy core.

### 3.2 Policy returns and Acrobot arbitration diagnosis

| Task and policy | Mean return | Paired difference vs best single | 95% bootstrap interval |
|---|---:|---:|---:|
| CartPole best single actor | 163.65 | — | — |
| CartPole actor-logit ensemble | 500.00 | — | — |
| CartPole grammar fusion | 205.96 | +42.31 | [23.98, 61.93] |
| Acrobot best single actor | -500.00 | — | — |
| Acrobot actor-logit ensemble | -500.00 | — | — |
| Acrobot grammar fusion | -443.37 | +56.63 | [39.32, 75.15] |

CartPole contains a 500-step return ceiling for some policies, so ceiling results mask policy differences. Acrobot fusion had 95.70% rule coverage, 89.24% conflict, and 4.30% blind spots. Its return distribution had median -500; 39 of 100 episodes ended above -500, indicating early goal termination. The low blind-spot rate means the mean return difference is not explained by frequent fallback alone.

The current confidence/support/reward arbiter scored -443.37. Confidence-only scored -444.49 (paired difference -1.12; interval [-3.16, +0.06]); mean-reward-only, support-first, and conflict deferral to the actor ensemble each scored -500. Random choice among available source rules scored -187.02 on one selector stream and -194.10 to -206.17 on five independent streams (mean -198.82). By comparison, five uniform-random-action streams averaged -498.71. This separates the surprising result from unconditioned random action noise, but does not identify why state-conditioned random rule selection performs well or establish transfer. It also means the current confidence-ranked arbiter is not shown to be a strong conflict resolver.

### 3.3 Causal AIM symbol evidence and trace auditability

Across six AIM seeds and 600,000 logged decisions, sender-sampled receiver accuracy averaged 0.998923, compared with 0.503292 for shuffled messages and 0.500 for the no-message control. The matched-support intervention effect on the active learned symbol averaged 0.998366 (per-seed range 0.996986–0.999685; mean common-support overlap 0.94308). The active token position varied across seeds. These data support a causal role for a learned symbol in the tested private-target game, not a general semantic decoding of the AIM language.

For the recorded RL artifacts, hash-chain, update-ledger, provenance, and environment replay checks passed under the audited setup. Across the eight-seed runs, audit records cover 1,725,655 CartPole rows and 2,544,513 Acrobot rows; replay checks cover 1,122,403 and 1,742,570 transitions, respectively. Exact replay is evidence about the recorded executions and specified software/hardware conditions. It cannot by itself establish that the logger captured every training cause, prove logger honesty, or guarantee cross-hardware equivalence.

### 3.4 FQE-GPI diagnostics kept separate

CartPole FQE-GPI achieved mean return 9.34 versus 163.65 for the best actor and matched the actor-logit ensemble on 35.27% of 31,563 held-out states. Acrobot FQE-GPI achieved -499.57 and selected action 0 on all 50,000 diagnostic states. On 200 Acrobot states, the seed-specific Q vectors had Pearson correlation 0.652 but their greedy actions diverged; fusion matched seed-11 Q argmax on 64.5%, seed-29 Q argmax on 0%, multi-critic GPI on 0%, and the best first action under a fixed actor-ensemble continuation on 22.5% (95% interval 16.5%–28.5%). This last measure evaluates that stated continuation, not optimal Q-star. These failures make FQE-GPI an unvalidated comparator in this study, not evidence against the grammar-fusion return itself.

## 4. Discussion

The end-to-end evidence closes a bounded existence argument: the workflow records traces and provenance, supports exact replay checks under controlled conditions, exposes source rules and blind spots during composition, and includes an intervention showing causal influence of a learned symbol in one designed task. These are concrete audit capabilities with inspectable artifacts and falsification attempts. The conclusion is deliberately narrower than “AI mother tongue makes general RL training completely auditable.”

Several results define the boundary. First, the 0.5833 CartPole grammar Jaccard coexists with only 51.36% fresh-state behavioral agreement. Second, Acrobot shared grammar conditions often conflict. Third, the current arbiter is highly sensitive to its selection rule, and randomized source-rule selection unexpectedly outperformed the tested deterministic variants. Fourth, FQE-GPI failed its value/action quality checks. These findings should be reported alongside positive results; they prevent symbolic overlap, return deltas, or a single comparator from carrying claims they do not support.

The AIM causal intervention and rule-fusion experiments answer related but distinct questions. The former tests symbol influence in a designed communication game. The latter tests visible provenance and composability of state-action rules. Neither makes the induced grammar a natural-language explanation of all neural decisions. Nor does replaying observed trajectories establish counterfactual explanations for unobserved decisions.

## 5. Limitations and Future Work

The evidence covers two benchmark tasks, eight seeds per task, one shared quantizer per task, and a particular software/hardware setup. CartPole has a return ceiling; many Acrobot episodes remain at the -500 cap. Fresh-state behavior was measured directly for the seed-11/29 CartPole pair, not every agent pair or task. The Acrobot random-rule result needs mechanism analysis; current evidence cannot attribute its advantage to a general arbitration principle. FQE-GPI was not reliable enough to support a policy-composition comparison. The AIM intervention task is designed and does not establish human-readable semantics for the full symbol sequence.

Four directions remain for future work:

1. Decompose the random-arbitration result by source rule, action frequency, and state occupancy.
2. Test return-weighted rule-bank induction using a preregistered normalization and held-out evaluation.
3. Use native AIM sequence symbols directly as the control representation and test their decision-level audit value.
4. Evaluate broader behavior on nonsaturated tasks and across hardware configurations.

## 6. Conclusion

In the evaluated prototype, end-to-end evidence supports replayable trace inspection, provenance-aware rule composition, explicit conflict and blind-spot reporting, and causal influence of learned symbols in a designed task. At the same time, grammar overlap did not predict strong actor agreement, arbitration behavior remains unexplained, and FQE-GPI failed quality checks. We therefore claim bounded existence, not universality: these mechanisms make specified RL artifacts more auditable under tested conditions, while general auditability and complete causal explanation remain unproven.

## References

1. Verma et al. “Programmatically Interpretable Reinforcement Learning.” ICML 2018. [PMLR](https://proceedings.mlr.press/v80/verma18a.html).
2. Barreto et al. “Successor Features for Transfer in Reinforcement Learning.” NeurIPS 2017. [NeurIPS](https://papers.nips.cc/paper/6994-successor-features-for-transfer-in-reinforcement-learning).
3. Thakoor et al. “Generalised Policy Improvement with Geometric Policy Composition.” ICML 2022. [PMLR](https://proceedings.mlr.press/v162/thakoor22a.html).
4. Nevill-Manning and Witten. “Identifying Hierarchical Structure in Sequences: A linear-time algorithm.” 1997. [JAIR paper](https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume7/nevill97a.pdf).
5. Lowe et al. “On the Pitfalls of Measuring Emergent Communication.” AAMAS 2019. [arXiv](https://arxiv.org/abs/1903.05168).
6. Nagarajan et al. “Deterministic Implementations for Reproducibility in Deep Reinforcement Learning.” 2018. [arXiv](https://arxiv.org/abs/1809.05676).

## Reproducibility Artifacts

All paths are relative to `C:\AI\RL\`.

- `auditability/` — experiment runners, validators, and diagnostics.
- `results/shared_multienv_gpi/CartPole-v1/behavioral_agreement_experiment_B.json` — fresh raw-state actor agreement.
- `results/shared_multienv_gpi/CartPole-v1/shared_symbolizer_rule_overlap.json` and `Acrobot-v1/shared_symbolizer_rule_overlap.json` — shared-symbolizer overlap.
- `results/shared_multienv_gpi/Acrobot-v1/grammar_fusion_diagnostics.json` and `random_action_control.json` — arbitration and random-action controls.
- `results/shared_multienv_gpi/CartPole-v1/artifact_audit.json` and `Acrobot-v1/artifact_audit.json` — audit and replay summaries.
- `results/shared_multienv_gpi/CartPole-v1/gpi_fqe/` and `Acrobot-v1/gpi_fqe/` — diagnostic-only FQE-GPI artifacts.
- `paper/AI_Grammar_Induction_Experiment_Log.md` — detailed experiment chronology, corrections, and negative results.
- `paper/20260922_RL_Grammar_Fusion_Research_Report.md` — expanded evidence report and historical pilot.
- `results/RESULTS_MANIFEST.json` — hashes and sizes for the compact review package.
