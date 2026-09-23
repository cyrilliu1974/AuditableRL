# From Replayable Trajectories to Rule-Level RL Fusion

## Abstract

Reinforcement-learning audit claims often conflate replaying a logged trajectory, explaining a decision, and composing policies. We evaluate these properties separately in two controlled settings. In a private-target coordination game, six AIM-trained seeds produced 600,000 logged decisions; matched-support interventions on the information-bearing learned symbol changed the receiver's action probability by 0.996986–0.999685 across seeds. In CartPole, state-action grammars induced from two 300-episode agents shared 3 productions at a 0.90 confidence threshold and 73 at 0.70 (Jaccard 0.088 and 0.545). We implement a provenance-aware rule merger that records candidate rules, arbitration, and uncovered-state fallbacks. In a common 200-episode evaluation, the permissive merger used rules on 58.3% of 100,000 decisions and marked 41.7% as blind spots. Its hash-chained trace replayed exactly, and 68,409 selected rule references resolved to source grammars and source-step hashes. The fused policy matched the stronger source agent's 500-point CartPole ceiling but did not improve its return. These results support causal symbol use in one designed task and auditable rule composition in one deterministic benchmark; they do not establish general RL auditability or performance gains.

## 1. Introduction

An RL system can produce a replayable action trace without explaining why its policy selected those actions. A symbolic rule can be readable without generalizing beyond observed states. Two agents can agree on a rule label even when their state discretizers assign different meanings to that label. We therefore treat trajectory replay, causal decision evidence, cross-seed rule stability, and policy composition as separate empirical questions.

This paper reports a controlled rule-level fusion prototype. Given two independently trained agents, the merger makes agreement, conflict, provenance, and uncovered states explicit. It emits a fused decision trace that can be checked against both source grammars and the environment. The objective is audit visibility, not higher return.

Our primary contributions are:

1. A provenance-preserving merger for learned state-action rule banks, with explicit conflict arbitration and blind-spot fallback.
2. A two-seed estimate of shared grammar rules, accompanied by a same-raw-state action comparison that does not assume independently fitted bins are identical.
3. An end-to-end validation of the fused decision trace through rule-provenance resolution, hash-chain verification, environment replay, and a tamper probe.
4. A distinction between causal symbol evidence, trajectory auditability, and RL performance, with a six-seed AIM causal intervention result reported separately from the CartPole fusion result.

## 2. Related Work

Programmatic RL methods express policies in a domain-specific language and may support symbolic verification. PIRL is a relevant example of readable program policies [1]. Successor features and Generalized Policy Improvement compose policy knowledge through value-based action selection [2]; geometric policy composition provides another policy-composition method [3]. Our prototype asks a different question: can the provenance and conflicts of discrete rules from separate agents be made explicit at decision time? We do not claim a return advantage over these methods.

Sequence-grammar compression must also be distinguished from policy-rule generalization. SEQUITUR induces repeated sequence structure, but a compression grammar is not automatically a policy explanation or generalized decision rule [4]. Work on emergent communication similarly warns that message/action correlation alone does not establish causal influence; intervention is needed [5]. Finally, exact RL reproducibility depends on seeds and software/environment conditions [6].

## 3. Methods

### 3.1 Agents and induced grammars

We use the existing CartPole-v1 REINFORCE runs for seeds 11 and 29. Each run has 300 episodes. An audit-only arm observes state/action pairs without changing the actor update; for both seeds, baseline and audit-only actor hashes and complete transition traces match exactly. Each seed fits a four-bin quantile symbolizer on 24 separate random-policy calibration episodes. The induced rule bank maps a discretized state condition to its majority action. We compare minimum confidence thresholds of 0.90 and 0.70; minimum support is 8.

The quantile boundaries are fitted separately by seed. As a result, exact token-set overlap is only a syntactic statistic. We also apply each grammar to the same raw observations in both recorded training traces, with each grammar's own frozen quantizer, and compare covered-state actions.

### 3.2 Rule-level fusion

For each raw state, both source quantizers produce their own condition token and each grammar is queried. If both rules select the same action, the merger records both sources. If only one rule exists, it selects that rule. If rules disagree, it records both candidates and selects by confidence, support, and mean reward, in that order; exact ties use the fallback actor. If neither grammar covers the state, the merger records a blind spot and uses a neural fallback selected by training mean return, without consulting evaluation returns.

Each decision record includes source rule identifiers, grammar hashes, support/confidence statistics, hashes and counts of source step IDs, fallback identity when used, transition data, and a chained record hash. A verifier resolves source rules against their source grammars and checks that the emitted action matches the selected rule. An independent replay pass resets CartPole with the logged seed and replays each action, comparing the state, next state, reward, termination, and truncation fields.

### 3.3 Evaluation

The two source actors and the fused policy are evaluated on the same 200 reset seeds, from 9,000,000 through 9,000,199. We report episode return, rule-use categories, conflicts, blind spots, and paired bootstrap differences. The fused trace contains 100,000 decisions because all fused episodes reach CartPole-v1's 500-step limit.

### 3.4 AIM causal-symbol evidence

As a separate experiment, six seeds of a private-parity coordination game train AIM senders and receivers. Matched-support interventions vary the information-bearing AIM token position while controlling the other token and receiver context. The active position is selected per seed rather than assumed to be fixed. This experiment tests whether AIM symbols affect a receiver decision; it does not test CartPole grammar fusion.

## 4. Results

### 4.1 Cross-seed grammar overlap

| Minimum confidence | Seed 11 productions | Seed 29 productions | Intersection | Union | Jaccard |
|---|---:|---:|---:|---:|---:|
| 0.90 | 26 | 11 | 3 | 34 | 0.088 |
| 0.70 | 113 | 94 | 73 | 134 | 0.545 |

All shared symbolic conditions chose the same action. However, the maximum absolute difference between the independently fitted bin edges was 0.1569 observation units. On the same recorded raw states, the 0.90 grammars jointly covered only 0.25% and 0.26% of the seed-11 and seed-29 traces, respectively, with complete action agreement on the jointly covered subset. The 0.70 grammars jointly covered 24.6% and 28.9%; their actions agreed on 99.992% and 99.991% of jointly covered steps, with one disagreement on each trace.

### 4.2 Fused-policy evaluation

| Confidence | Agent 11 mean return | Agent 29 mean return | Fused mean return | Blind-spot decisions |
|---|---:|---:|---:|---:|
| 0.90 | 500 | 131.085 | 500 | 100,000 / 100,000 |
| 0.70 | 500 | 131.085 | 500 | 41,728 / 100,000 |

At confidence 0.70, 10,137 held-out decisions used rules on which both agents agreed, 48,135 used exactly one source rule, and 41,728 used the explicit Agent-11 fallback. No conflict occurred in these held-out episodes. Three actual conflicts occurred in the source training traces (two on the seed-11 trace, one on the seed-29 trace); each was exposed with both candidate rule IDs and resolved by the predeclared priority order.

The merged trace passed hash-chain validation and exact environment replay for all 100,000 decisions. The provenance audit resolved all 68,409 selected rule references to their source grammars and step hashes and checked all 41,728 fallback references. Changing a reward in the first row of a copied trace caused immediate hash verification failure.

The fused score equaled the stronger agent's 500-point environment ceiling. Its paired mean difference and percentile bootstrap interval versus Agent 11 were [0, 0]. This establishes neither an improvement nor a meaningful non-inferiority margin under a nonsaturated task. At confidence 0.90, the merger used fallback for every held-out decision; its return therefore reflects the fallback agent, not successful rule transfer.

### 4.3 AIM symbol intervention

Across six seeds and 600,000 logged decisions, sender-sampled receiver accuracy averaged 0.998923; shuffled-message accuracy averaged 0.503292; and the no-message baseline was 0.500. The matched-support intervention effect on the active symbol position averaged 0.998366 (range 0.996986–0.999685), with mean shared-support overlap 0.94308. The active token position varied across seeds. These results support a causal role for learned symbols in this designed private-target game, not a general natural-language interpretation of AIM symbols.

### 4.4 Trajectory-grammar description length

For the seed-11 audit-only trace, the lossless pair grammar was re-induced on each 50-episode prefix. All six prefixes reconstructed exactly. Under the declared code-length scheme, MDL/raw UTF-8 bit ratios were 0.513, 0.536, 0.549, 0.561, 0.575, and 0.575 from rounds 1–6. This is a descriptive score for the trajectory pair grammar; the algorithm does not optimize MDL, and the state-action rule bank is evaluated separately. Ordinary gzip remains smaller than the serialized grammar archive.

## 5. Discussion

The prototype demonstrates the report's central transparency mechanisms in a bounded setting: shared-rule counts can be measured; conflicts can be surfaced and arbitrated; uncovered states can be named rather than silently treated as known; source rules can be resolved back to their original records; and the fused policy's event trace can be replayed. At the permissive threshold, the merger uses rules on a majority of evaluated decisions and exposes a substantial remaining blind spot. At the strict threshold, almost all test states are uncovered, despite a nonempty training grammar.

This does not show that rule fusion is more accurate or more useful than neural policy selection. CartPole's score ceiling masks behavioral differences: the fused policy and Agent 11 both score 500, and the paired difference is identically zero. The low-threshold grammars individually agree with their own neural actors on held-out states only 59.9% and 74.2% of the time. High cross-agent agreement among overlapping rules is therefore not equivalent to faithful explanations of either full policy.

The AIM intervention and CartPole fusion results answer different questions. AIM interventions support symbol-mediated receiver decisions in a designed task. CartPole demonstrates provenance-preserving rule composition and exact replay under deterministic environment conditions. Neither result proves complete auditability of arbitrary RL training, and the AIM trace alone does not record all weights, gradients, optimizer tensors, and RNG state throughout learning.

## 6. Limitations and Next Experiments

The grammar-fusion experiment uses two seeds, seed-specific state quantizers, one deterministic environment, and a reward-saturated stronger policy. Before making a broad claim, future work should: (i) use a shared frozen symbolizer across agents; (ii) increase the number of source agents; (iii) use a task with a nonsaturated return scale; (iv) compare against an explicit policy-composition baseline such as GPI; and (v) evaluate return-weighted rule induction using a preregistered normalization formula. The present evidence supports audit visibility, not superior RL performance.

## 7. Conclusion

In controlled experiments, learned symbolic rules can expose a measurable shared core between RL agents and support a provenance-preserving fused policy whose decisions, conflicts, blind spots, and fallback sources are recorded and checked. The fused CartPole trace replayed exactly, while confidence-threshold ablations showed that symbolic overlap and state coverage trade off against individual policy fidelity. AIM symbols also showed large matched-support causal effects on receiver actions across six seeds in a private-target game. The appropriate conclusion is bounded: these artifacts make selected trajectories and decisions more auditable in the tested setups; they do not make general RL training fully auditable or improve return by themselves.

## References

1. Verma et al. “Programmatically Interpretable Reinforcement Learning.” ICML 2018. [PMLR](https://proceedings.mlr.press/v80/verma18a.html).
2. Barreto et al. “Successor Features for Transfer in Reinforcement Learning.” NeurIPS 2017. [NeurIPS](https://papers.nips.cc/paper/6994-successor-features-for-transfer-in-reinforcement-learning).
3. Thakoor et al. “Generalised Policy Improvement with Geometric Policy Composition.” ICML 2022. [PMLR](https://proceedings.mlr.press/v162/thakoor22a.html).
4. Nevill-Manning and Witten. “Identifying Hierarchical Structure in Sequences: A linear-time algorithm.” 1997. [JAIR paper](https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume7/nevill97a.pdf).
5. Lowe et al. “On the Pitfalls of Measuring Emergent Communication.” AAMAS 2019. [arXiv](https://arxiv.org/abs/1903.05168).
6. Nagarajan et al. “Deterministic Implementations for Reproducibility in Deep Reinforcement Learning.” 2018. [arXiv](https://arxiv.org/abs/1809.05676).

## Reproducibility Artifacts

All paths are relative to `C:\AI\RL\`.

- `aim_auditability/auditability/merge_seed_grammars.py`
- `aim_auditability/auditability/compare_seed_grammars.py`
- `aim_auditability/auditability/analyze_trace_mdl_by_round.py`
- `aim_auditability/runs/grammar_merge_070_final/merge_evaluation.json`
- `aim_auditability/runs/grammar_merge_070_final/merged_policy_trace.jsonl`
- `aim_auditability/runs/grammar_merge_090_final/merge_evaluation.json`
- `aim_auditability/runs/grammar_intersections/`
- `aim_auditability/runs/grammar_mdl_seed11_070.json`
- `aim_auditability/runs/repeatability_full/`
- `aim_auditability/runs/replication_summary.json`
