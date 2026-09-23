
# AI Grammar Induction and RL Audit Experiment Log

## Initial source and literature diagnosis

- User-supplied design sketch is a proposal to assess, not an instruction source. The target claim is whether automatically induced AIM/RL state-action rules and grammar improve held-out replay/auditability in general RL, beyond recording observed trajectories.
- Source inspected: `aim_auditability/ai_grammar_induction.py`; Python compilation succeeds. Its direct example currently prints a dependency message because Gymnasium is absent from the project venv. The first shell invocation looked for the venv at the workspace root; the actual environment is `aim_auditability/.venv`, and the corrected run confirmed torch 2.2.0+cpu, numpy 1.26.4, Gymnasium absent.
- The CartPole example passes `verbose=True` to `AIMotherTongueRL.__init__`, which has no such parameter; when reached, it will raise `TypeError`. The train loop calls `env.reset()` inside the `hasattr` expression and again in the selected branch, resetting twice and not fixing a seed. Gymnasium `step` returns `(observation, reward, terminated, truncated, info)`, but the implementation binds only `terminated` as `done` and ignores `truncated`; current CartPole-v1 ends on either termination or the 500-step time limit. These API mismatches make the stated example not a valid run as written.
- Rule observation conflates one-step `(state, action)` counts with flattened three-action-window counts in one Counter, while confidence divides both kinds by one total. Reward is not used. The window clears on episode end, but one-step rules have no episode/next-state/reward linkage.
- `SequiturInducer` is not the canonical SEQUITUR algorithm: it greedily replaces the most frequent repeated bigram, but does not enforce digram uniqueness/rule utility. It flattens each rule into repeated symbols, concatenates sequences with separators, and keeps only `compressed[:50]` as the start production. `source_rules` is inferred via substring tests, not occurrence provenance. This is lossy and cannot certify trajectory replay.
- `FormalGrammar.is_stable` checks only equal production counts. Each induction starts a new grammar at generation 1, so the `max_grammar_generations` early-stop condition cannot progress as intended. `ReplayAuditor` searches the first two RHS symbols of productions for exact condition/action pairs, skips uninitialized abstractions without counting them as failures, and never reruns the policy or environment; its coverage is not replayability. Grammar masking is a global action set, not condition-specific; it can permanently prune an action merely because that action was absent in observed frequent rules. This may make training worse and is not a safe constraint.
- Primary-source reading changed the experimental design: the 1997 SEQUITUR paper explicitly states its output is not strictly a generalized grammar and generates only one observed string, while defining digram uniqueness and rule utility. Therefore grammar compression (exact reconstruction of a seen trace) and a learned state-conditioned policy rule (generalization to unseen states) need separate tests. The paper also says SEQUITUR structure alone is not evidence that extracted rules explain agent decisions. See [primary paper](https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume7/nevill97a.pdf).
- PIRL represents policies in a domain-specific program language and uses symbolic verification; unlike that design, grammar induction from logged traces must prove both provenance and predictive/generalization performance. This distinction is recorded for comparison: [PIRL, ICML/PMLR 2018](https://proceedings.mlr.press/v80/verma18a.html).
- Exact RL replay requires controlling and recording sources of nondeterminism, not just a deterministic policy representation. Nagarajan et al. analyze seed/environment/software/hardware sources and distinguish deterministic implementation from same-condition replicability: [Deterministic Implementations for Reproducibility in Deep RL](https://arxiv.org/abs/1809.05676).
- Gymnasium's official API returns separate `terminated` and `truncated` flags and recommends recording reset seeds; its CartPole-v1 limit is 500 steps. References: [Env API](https://gymnasium.farama.org/main/api/env/) and [CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/).

## Experimental hypothesis and planned comparisons

- H1 (lossless replay): a grammar induced solely from training-trajectory symbols can losslessly encode/decode those trajectories, with byte/canonical-token equality. Held-out episodes cannot be counted as replayed merely because every token is in the alphabet; evaluate exact transition-record reconstruction and independently verify episode boundaries/rewards.
- H2 (policy-rule generalization): an automatically learned state-symbol -> action rulebook should predict an independent holdout trajectory above an action-frequency baseline, with coverage, exact action accuracy, reward return, and confidence reported separately. Use chronological/train-seed separation and freeze quantizer/grammar before holdout.
- H3 (causal usefulness of constraints): compare unconstrained RL versus grammar-constrained RL over paired seeds. Constraints may be applied only to a state condition with adequate training support and must have an explicit fallback for unknown/low-confidence states. Track return and forbidden-action interventions; do not treat training coverage as evidence of generalization.
- H4 (full training auditability): reconstructing the sampled episode trace alone is insufficient. Capture run configuration, code and environment hashes, all observations/actions/rewards/termination flags, RNG states, optimizer/update boundary and checkpoint hashes; compare deterministic rerun parameter/checkpoint hashes and per-update/episode trace. Grammar may compress records, but must round-trip exactly and preserve provenance. The first experiments will measure bounded trajectory replay and rule generalization before claiming whole-training auditability.
- Baseline-entry run: `python -m py_compile` passed. The first invocation from workspace root used the wrong venv path; rerun from the actual `aim_auditability/.venv` succeeded and demonstrated only that Gymnasium is missing. No source files changed yet.

## Prototype run and first implementation correction

- Installed Gymnasium 1.3.0 in the isolated `aim_auditability/.venv` and pinned it in `aim_auditability/requirements.txt`; it was previously absent. Running the original file then reached the CartPole example and failed exactly as predicted: `TypeError: AIMotherTongueRL.__init__() got an unexpected keyword argument 'verbose'`.
- Added `auditability/grammar_audit_experiment.py` as the validated execution path and changed `ai_grammar_induction.py` to label its older classes as a legacy prototype and route its CLI to the experiment. Updated workspace README to document the real experiment paths and scope.
- The experiment separates (a) state/action production induction, (b) an exact pair-substitution CFG codec for the complete JSONL transition trace, (c) held-out grammar coverage/action agreement, and (d) effects of applying high-confidence state-conditioned grammar constraints to REINFORCE. CartPole continuous states are quantile-symbolized from separate random-policy calibration episodes; bins are frozen before scored RL training. The grammar receives no manually specified state/action productions; only the type tags and action-space boundary are declared.
- Smoke run 1 (`grammar_audit_smoke`, seed 11, 10 episodes, two calibration episodes, five codec rules) successfully compiled and executed all three arms. Each full trace was exactly decoded with matching SHA-256; five induced pair productions reduced lexical symbols from 21,150 to 17,979 (ratio 0.8501). The 10-episode policy had no eligible rule coverage, as expected under support threshold 8. Baseline and observer-only transition hashes matched, but actor file hashes initially appeared different because the serialized wrapper included the arm label; that was a hash-definition bug, not a weight difference.
- Fixed actor comparison to hash only the actor state dictionary. Second smoke run (`grammar_audit_smoke_v2`) now confirmed baseline and audit-only actor hashes identical and transition-trace hashes identical; all three grammar traces still round-trip exactly. This verifies observer instrumentation did not affect that short run, but 10 episodes are only a plumbing smoke test, not performance evidence.
- Updated the root `README.md` to distinguish the validated runner from the older prototype and added `gymnasium==1.3.0` to requirements. The command-line wrapper was also changed to print the result summary (the initial wrapper smoke saved artifacts but was silent).

## CartPole pilot: first independent seed

- Pilot configuration: seed 11; 300 REINFORCE episodes per arm; 24 separate random-policy calibration episodes; CartPole-v1; 4 quantile bins per observation dimension; state-action productions admitted at support >=8 and confidence >=0.90; three paired arms (baseline, audit-only, grammar-constrained).
- Baseline and audit-only produced exactly identical training trace hashes and actor-weight hashes. Both trained at mean return 325.49 (last-50 mean 442.44) and scored mean 500 on 20 held-out episodes. This shows the instrumentation itself did not perturb this seed's training.
- The grammar-constrained arm scored mean 121.3 on the same held-out reset seeds. The induced rule covered only 0.4% of held-out steps; on that tiny covered subset it agreed with the unconstrained actor 100%. Thus conditional precision is not useful when coverage is this low; constraints damaged return without generalizing broadly. Treat as a negative result for naive state-bin lookup constraints. This does not establish that grammar is generally harmful; seed 29 is still running.
- This observation sharpens the next analysis: stratify by training support/confidence and unseen-state coverage, and compare rule predictions to the unconstrained policy on a disjoint evaluation trajectory. Do not conflate exact compression of a training trace with predictive grammar utility.

## Two-seed CartPole pilot: first complete results and compression failure

- The 300-episode pilot finished for seeds 11 and 29 across baseline, audit-only, and grammar-constrained arms. Every baseline/audit-only pair had identical actor-weight SHA-256 and transition-trace SHA-256, confirming exact instrumentation equivalence for both seeds.
- Hash-chain validation passed for all six runs and covered 366,323 recorded transitions. The seed-11 baseline/audit trace had 97,646 transitions; the constrained trace had 66,663, reflecting a shorter, weaker policy. Seed 29 also produced valid traces.
- Held-out mean returns were strongly seed-dependent: seed 11 baseline/audit = 500.0, constrained = 121.3 (delta -378.7); seed 29 baseline/audit = 131.0, constrained = 162.5 (delta +31.5). The two-seed mean constraint delta was -173.6, but the signs disagree and the baseline itself varied greatly (500 vs 131). This is not a reliable performance benefit; it shows the current online majority-rule constraint is unstable and can sharply hurt a learned policy.
- For seed 11, the grammar had 0.4% held-out state coverage and 100% action agreement on that small covered subset; for seed 29, it covered 7.0% and agreed on 81.1%. High precision on a tiny or seed-dependent subset must not be reported as full trajectory replay.
- The initial lossless pair grammar induced 16 rules and reduced lexical symbols from 9,178,724 to 6,558,026 (ratio 0.7145), with exact token reconstruction and identical source/decoded SHA-256. However, its pretty-printed JSON artifact was 110,549,187 bytes versus 61,295,430 bytes for the original JSONL trace. Thus lexical-symbol reduction did **not** reduce disk size; the readable string-reference encoding made the file 1.80x larger. Reworked serialization to integer terminal IDs, compact gzip grammar archives, and explicit comparisons against gzip-compressed raw traces. Only one complete baseline trace is compressed per experiment because the audit-only trace is already byte-identical.
- A hash chain only protects record integrity; it does not prove the transitions occurred in CartPole. Added an independent Gymnasium environment-replay validator that replays reset seeds/actions and compares every state, next state, reward, termination, and truncation flag exactly. The 10-episode smoke check passed on all three arms (225 transitions each); now apply it to the full pilot before interpreting replay claims.

## Full pilot trace replay and compact archive comparison

- Added the independent environment replay check to the summary. It replayed all 366,323 logged CartPole transitions across the six runs from recorded episode reset seeds and actions. Every raw state/next-state float32 vector, reward, terminated flag, and truncated flag matched exactly. Hash chains also passed for all six runs; baseline and audit-only trace/actor hashes were identical within each seed.
- Re-encoded one complete unique trace (seed-11 baseline; the audit-only trace is byte-identical) using a compact grammar archive with integer symbol IDs, exact token offsets, and gzip serialization. The 61,295,430-byte JSONL trace contained 9,178,724 lexical tokens. A 16-production pair grammar reduced the symbolic start sequence to 6,558,026 IDs (0.7145), and the serialized grammar archive round-tripped to the same SHA-256.
- Byte-size comparison: raw JSONL 61,295,430 bytes; raw JSONL with gzip level 9, 12,337,358 bytes; grammar JSON payload 52,043,713 bytes; grammar archive gzip level 9, 13,369,931 bytes. The grammar archive is 78.2% smaller than the raw JSONL, but 8.4% larger than ordinary gzip of that same trace. Therefore the induced grammar delivers exact reconstruction and explicit production/provenance structure, but this experiment does **not** show a storage-compression advantage over a standard compressor.
- The summary is `aim_auditability/runs/grammar_audit_pilot/experiment_summary.json`; the grammar archive and byte-size manifest are saved beside the seed-11 baseline trace. The first run's over-large pretty JSON artifact is retained as a historical pilot output; the final summary references the compact `.json.gz` archive.

## Update-level process ledger and deterministic cold reruns

- Added `training_updates.jsonl`, a separate hash-chained record with one row after each REINFORCE episode update. Each row commits to the clipped gradient, post-update actor weights, post-update Adam state, Python/NumPy/PyTorch RNG state fingerprints, loss, episode return/length, global transition count, and grammar generation. The run also saves `training_checkpoint.pt` with final actor and optimizer tensors plus RNG states. The checkpoint is additional to, not a replacement for, the step-level trace.
- Added an independent update-ledger verifier and included it in run summaries. Historical pilot runs predate this addition; summaries now label their update-ledger coverage unavailable instead of implying it was recorded. The experiment summary now reads episode count from run metadata rather than assuming 300.
- The 8-episode smoke test passed: baseline and observer-only training update ledgers were byte-identical; all three traces passed chain verification and exact Gymnasium transition replay (651 total transitions); all ledgers had eight valid episode updates. A separate two-run smoke check produced identical actor, transition, update-ledger, and full-checkpoint SHA-256 values.
- Added `auditability/verify_deterministic_repeat.py` for cold same-seed reruns. At 300 episodes, independent reruns for seed 11 matched on actor, transition trace, all 300 per-episode update records, and full training checkpoint; both traces independently replayed 97,646 environment transitions exactly. Seed 29 also matched on all four hashes across 300 episodes; both traces replayed 37,781 transitions exactly. The identical outcomes are bounded to the same CPU host, Python/PyTorch/Gymnasium versions, deterministic CPU settings, and CartPole-v1.
- Across these two seeds there are 135,427 distinct transition records; each was reproduced and compared again in the second run, for 270,854 exact transition comparisons. This is strong reproducibility evidence for this controlled task and implementation. It is not cross-hardware validation, proof against a compromised logger, or evidence that an induced grammar identifies a policy's causal rationale.
- Tightened source provenance by hashing the experiment and legacy prototype source at run start, before training.
- Next test: lower only the rule confidence admission threshold from 0.90 to 0.70 while holding four state bins, support >=8, training seeds and duration constant. This tests whether the earlier low held-out coverage is an adjustable coverage/faithfulness tradeoff and whether it changes constrained-policy return.

## Research steering and next methods (user direction; planned, not yet tested)

- Primary paper claim is trajectory auditability under stated replay conditions, not improved RL return from grammar constraints. Current return evidence is seed-sensitive and does not support a general performance-improvement claim.
- Short-term addition: report Minimum Description Length at every grammar-induction round. Define and log both the encoded production cost in bits and the encoded complete observed trace cost in bits; report their sum and exact round-trip status. Compressed file size alone is not an induction-quality metric. Make the code-length convention explicit and include metadata/provenance overhead separately so the comparison is reproducible.
- Medium-term ablation: add episode-return weighting to rule-bank counts, with a documented normalized-return transform and formula `weight = 1 + normalized_return`. Compare unweighted versus return-weighted rules on held-out state coverage, action agreement, grammar MDL, and constrained-policy return under the same seeds. This is a hypothesis to test, not an assumed improvement.
- The 0.70 confidence-threshold experiment was running when this research steering was recorded and is now complete (results follow below). Next, implement MDL measurements, then the reward-weighted rule-bank variant as a separately identified experimental condition.

## Confidence-threshold ablation (0.70) and audit-ledger correction

- Completed a matched 300-episode CartPole comparison on seeds 11 and 29, changing only rule admission confidence from 0.90 to 0.70 (4 quantile bins, minimum support 8, 24 calibration episodes, same reset/evaluation seeds). All six traces and all 1,800 update ledgers passed hash-chain checks; 344,459 transitions were replayed exactly in the environment.
- Lowering confidence raised held-out rule coverage substantially but reduced agreement with the unconstrained policy: seed 11 coverage 75.6%, agreement 59.9%; seed 29 coverage 56.4%, agreement 74.2%. The constrained-policy evaluation delta was -330.4 return for seed 11 and +17.0 for seed 29 (mean -156.7). This is a coverage/fidelity tradeoff, not evidence of a reliable performance gain. It supports the user-directed paper positioning around trajectory auditability rather than grammar constraints improving RL.
- The first summary marked baseline vs audit-only update hashes unequal even though actor and transition-trace hashes matched. Inspection found the ledger included `grammar_generation`, a grammar sidecar counter that differs between arms but does not affect optimization. This was an audit-schema contamination, not a training difference. Removed the sidecar field from new optimizer ledgers and added a normalized core-update hash that excludes legacy grammar-generation metadata and hash-chain wrapper fields.
- A 60-episode verification crosses the first 50-episode grammar induction boundary: baseline and audit-only now match exactly on actor weights, full trajectory, raw update ledger, and optimizer-update core hash; all 180 update records and 9,120 transitions across three arms passed verification/replay. The constrained arm diverges only after applying learned rules, as intended.
- The original 0.70 full-run update ledger remains valid; its raw file hashes differ because they contain the old sidecar field. The corrected summary reports raw-ledger equality separately from optimizer-core equality. This correction is explicitly retained in the research record rather than rewriting it as if no issue occurred.

## Cross-seed shared-rule-core estimate

- Compared the actual `grammar_audit_only` induced grammars for seeds 11 and 29, so grammar observation did not change either policy's training. At confidence >=0.90 the grammars contain 26 and 11 productions; exact `(state-symbol, action-symbol)` intersection is 3, union 34, Jaccard 0.088. All three shared conditions select the same action.
- At confidence >=0.70 they contain 113 and 94 productions; exact-symbol intersection is 73, union 134, Jaccard 0.545. All 73 shared conditions select the same action. This is a much larger syntactic shared core, but the threshold experiment shows individual held-out policy agreement is lower (59.9% and 74.2%), so shared rules are not automatically faithful explanations.
- The seed-specific calibration quantile edges are not identical (maximum absolute threshold difference 0.1569 in CartPole observation units). To avoid relying only on matching bin labels, applied both grammars to the same raw states in each seed's recorded training trace, using each grammar's own fitted bin boundaries. At >=0.90, both grammars jointly covered only 0.25% and 0.26% of the respective traces, with 100% same-action agreement on jointly covered steps. At >=0.70, joint coverage rose to 24.6% of seed-11 states and 28.9% of seed-29 states; actions agreed on 99.992% and 99.991% of jointly covered steps (one mismatch on each trace).
- Operational estimate: lowering confidence enlarged the shared rule core and substantially increased the state mass on which both seed grammars provide the same action. These are cross-seed agreement statistics on training-state occupancy, not independent held-out evidence or causal explanation. The 0.70 grammar's low action fidelity to the trained neural actor remains a limitation.
- Reusable analysis: `aim_auditability/auditability/compare_seed_grammars.py`; JSON outputs: `aim_auditability/runs/grammar_intersections/confidence_090_audit_only.json` and `.../confidence_070_audit_only.json`. The comparison records raw symbolic Jaccard, supports/confidences, calibration-boundary distances, and same-state action agreement.

## Rule-level fusion prototype and end-to-end audit

- Implemented `auditability/merge_seed_grammars.py` using the seed-11 and seed-29 `grammar_audit_only` agents. The runtime maps the same raw CartPole state through each agent's own frozen quantizer, surfaces both candidate rules, selects a single-source rule or same-action agreement, marks disagreements as conflicts, resolves conflicts by confidence/support/mean reward, and marks no-rule states as blind spots. Blind spots and exact arbitration ties call a neural fallback selected using training return only.
- The output decision trace records both candidate rule IDs and full candidate provenance (source run/seed, grammar hash, support, confidence, reward, and source-step hash/count), then separately records the selected source rule(s) or fallback agent. A separate validator resolves every selected reference back to the source grammar, checks source-step hashes/statistics and emitted action, verifies the trace hash chain, and replays the environment transitions.
- Confidence >=0.70 fusion, evaluated on 200 common reset seeds / 100,000 merged decisions: Agent 11 mean return 500; Agent 29 mean 131.085; fused policy mean 500 (median 500). Paired difference versus Agent 11 was 0, 95% bootstrap interval [0, 0]. This is ceiling preservation in this evaluation, not a performance improvement or meaningful evidence that the rules outperform the stronger agent.
- On those held-out merged-policy decisions, both rules agreed on 10,137 steps, exactly one agent supplied a rule on 48,135, and 41,728 (41.7%) were explicitly marked blind spots and used Agent 11's neural fallback. No conflicts occurred in these 200 held-out episodes. Across the two source training traces, the merger encountered three actual state-level conflicts (two on seed 11's trace, one on seed 29's), made them visible, and resolved by the declared confidence/support/reward order. The candidate rules and selected source were saved in each affected training-state probe example.
- The 100,000-decision merged trace passed both hash-chain verification and exact CartPole environment replay. Provenance audit resolved 68,409 selected rule references to source grammars/step IDs and checked all 41,728 fallback-agent references. A tamper probe modified the first reward in a copy; replay verification rejected it at line 1 with a hash mismatch. The source trace was left intact.
- At confidence >=0.90, the same 200-episode merge had 100% blind spots in its 100,000 held-out decisions, so it reduced entirely to Agent 11 fallback despite having 37 combined training productions and three syntactically shared rules. This demonstrates why a large-looking or high-confidence rule core must also be tested for state coverage in the target distribution.
- Conclusion for this bounded outcome: the report's transparency features are implemented and demonstrated—shared-rule accounting, explicit conflicts, explicit blind spots, selective rule use, source provenance, and a replayable fused-policy trace. The experiment does **not** establish improved RL return, general policy transfer, or general auditability beyond deterministic CartPole and these two agents.
- Artifacts: `aim_auditability/runs/grammar_merge_070_final/merge_evaluation.json` and `merged_policy_trace.jsonl`; strict-threshold comparison in `aim_auditability/runs/grammar_merge_090_final/`; reusable code in `aim_auditability/auditability/merge_seed_grammars.py`.

## Per-round MDL measurements

- Added a trace-prefix analyzer and measured all six 50-episode grammar rounds for seed 11's confidence >=0.70 audit-only run. It re-induced a 16-rule pair-substitution grammar on each complete raw JSONL prefix and verified exact reconstruction every round.
- The two-part code length is explicit: `L(G)` stores the terminal dictionary (UTF-8 literals with Elias-gamma length codes) and pair productions (fixed-width symbol references); `L(trace|G)` stores the start sequence using the same fixed-width references. Fixed schema overhead is excluded, and provenance metadata bytes are reported separately. This is a reproducible description-length score; the pair-substitution algorithm does not optimize the score.
- Across rounds 1-6 (50 to 300 episodes), MDL/uncompressed UTF-8 trace-bit ratios were 0.513, 0.536, 0.549, 0.561, 0.575, and 0.575. Every prefix round-tripped exactly. The increasing ratio after round one means this grammar/code convention's relative savings did not improve monotonically as the trace grew. At 300 episodes MDL was 281,958,036 bits versus 490,363,440 raw UTF-8 bits; the earlier gzip comparison still favors ordinary gzip over the serialized grammar archive, so do not claim best-in-class storage compression.
- Important scope: this is MDL for the **lossless trajectory pair grammar**. The state-action rule bank cannot encode the complete trajectory by itself, so its rule quality remains evaluated separately by coverage, action agreement, and cross-seed merge behavior. Report the distinction in any paper.
- Result file: `aim_auditability/runs/grammar_mdl_seed11_070.json`; reusable analyzer: `aim_auditability/auditability/analyze_trace_mdl_by_round.py`.

## Research report and paper draft completed

- Consolidated the fusion design, cross-seed core, MDL-by-round, six-seed AIM intervention evidence, exact replay/repeatability, and explicit claim limits in `paper/20260922_RL_Grammar_Fusion_Research_Report.md`.
- Created an English workshop-style manuscript draft at `aim_auditability/manuscript/rl_rule_fusion_draft.md` with abstract, related work, method, measured results, discussion, limitations, and primary-source links. Updated the workspace README with the new analysis and manuscript locations.
- Current supportable paper claim: in these controlled settings, learned symbolic rules support measurable cross-seed agreement and provenance-preserving rule composition, including visible conflicts/blind spots and exact replay of the fused trajectory; separate AIM interventions show causal symbol use in the private-target game. Do not claim broad RL auditability, human-readable semantics for every AIM symbol, or improved return.
- The 2026-09-22 Claude note was treated as a proposal for experiments. Its rule-fusion outcome is now demonstrated for the two seed-11/29 agents in CartPole. A shared-symbolizer multi-seed replication, a nonsaturated task, an explicit composition baseline, and return-weighted induction remain future work rather than results.

## Shared-symbolizer multi-seed results and Acrobot fusion diagnosis (2026-09-22)

### Confirmed multi-seed comparison

- Completed matched runs for eight seeds (11, 29, 43, 71, 101, 149, 211, 307), each with 300 training episodes, on CartPole-v1 and Acrobot-v1. Every seed within an environment used the same pooled frozen quantizer, fitted from 24 separate random-policy calibration episodes per seed. Eight agents per task were compared under baseline, audit-only, and grammar-constrained arms; the baseline and audit-only training pairs are exactly equivalent.
- With this shared symbolizer, CartPole seed 11 vs seed 29 has 77 exact state-action rules in the intersection, 132 in the union, and Jaccard 0.5833; no shared-condition action conflict. Acrobot has 21 rules in the intersection, 423 in the union, and Jaccard 0.0496; 77 of the 98 shared conditions have conflicting actions. Mean all-pairs Jaccard across eight seeds is 0.5572 for CartPole and 0.0956 for Acrobot. The measured rule core is task-dependent; shared frozen bin edges remove the seed-specific quantizer artifact for these comparisons, but the statistic remains exact symbolic overlap, not causal equivalence.
- Held-out grammar fusion on Acrobot-v1 used 100 common reset seeds (20,000,000 + episode index). The best single actor and actor-logit ensemble each averaged -500.00. Grammar fusion averaged -443.37 (paired delta +56.63; percentile-bootstrap 95% CI [+39.32, +75.15]), median -500, and maximum -126. There were 39/100 episodes with return greater than -500, indicating early goal termination; most episodes still reached the 500-step limit. The published main result remains separate from all GPI results.

### Acrobot coverage and arbitration diagnostics

- On the original 100-episode grammar-fusion evaluation, 42,467 of 44,376 decisions were rule-covered (95.70%); 1,909 were blind spots (4.30%); 39,602 decisions were conflicts (89.24%). This rules out a large blind-spot fraction as the main explanation of the observed return pattern. Conflict frequency is high, and action selection within conflicts is the primary measured mechanism to investigate.
- Replayed the original confidence -> support -> mean-reward -> source-seed arbitration and exactly reproduced -443.37. On the same environment reset seeds, confidence-only averaged -444.49 (paired delta -1.12; 95% CI [-3.16, +0.06]); mean-reward-only averaged -500.00 (delta -56.63; 95% CI [-75.47, -39.67]); support-first averaged -500.00; and deferring conflicts to the actor-logit ensemble averaged -500.00. Changing only blind-spot fallback to the best single actor gave -452.69 (paired delta -9.32; 95% CI [-19.51, +0.50]). The fallback sensitivity is smaller and uncertain relative to the overall +56.63 gain.
- Uniform random choice among available source rules at each state conflict averaged -187.02 on the initial fixed RNG stream. Five additional independent arbitration RNG streams over the same 100 environment reset seeds averaged -194.10, -195.38, -201.46, -196.97, and -206.17. Their paired deltas against the current arbitration were +249.27, +247.99, +241.91, +246.40, and +237.20; each paired bootstrap 95% interval was wholly positive (range of lower bounds +215.11 to +226.90). This is a surprising policy-composition result: random rule arbitration substantially outperformed the current confidence-ranked arbitration in this fixed Acrobot benchmark. It is not a universal lower-bound result or evidence that random rules are generally preferable; it motivates inspection of action frequencies, state occupancy, and rule-source composition before attributing the gain to grammar quality.
- The mean-reward statistic is weakly discriminative here: 1,575 of 1,694 extracted Acrobot rules have mean reward exactly -1.0. This helps explain why a mean-reward-only ranking collapses to deterministic tie order. The observed outcome also shows that the current confidence/support priority is not a validated optimal conflict resolver: confidence-only is nearly tied with current, while random arbitration is much stronger in these runs.

### FQE-GPI quality checks; keep separate from the fusion finding

- Ran held-out FQE diagnostics on fresh deterministic actor rollouts using reset seeds disjoint from training and primary evaluation. For CartPole, FQE-GPI selected the same action as the actor-logit ensemble on 35.27% of 31,563 states; its policy selected action 0 on 30,758 states. On the separate full evaluation it scored mean return 9.34 versus best-single 163.65 (paired delta -154.31; 95% CI [-164.38, -144.65]). This FQE-GPI implementation is not a credible performance baseline at this stage.
- CartPole value direction was mixed across source critics: seed 11 assigned higher source-policy Q to states from top-quartile-return episodes than bottom-quartile episodes (10.986 vs 9.425), while seed 29 slightly reversed that ordering (10.347 vs 10.518). Both critics assigned a somewhat higher maximum action value to the balanced anchor state [0, 0, 0, 0] than to the perturbed candidate [0, 0, 0.25, 2], but the GPI argmax at the balanced anchor disagreed with the actor ensemble. The anchor comparison is only a local smoke diagnostic, not validation of value accuracy.
- On the 200-state Acrobot sample, FQE-GPI's greedy action disagreed with the grammar fusion action at every state; it selected action 0 on all 200 sampled states. Seed-11 and seed-29 critic Q-values on the same states had Pearson correlation 0.652, yet their greedy actions differed almost completely (seed 11: action 1 on 186/200; seed 29: action 0 on 200/200). A scalar Q-value correlation is therefore not action-value agreement. The complete held-out actor trajectories all scored -500, so high-return vs low-return Q-direction discrimination was not identifiable on this task's current source actors.
- A separate action-level Monte Carlo diagnostic sampled 200 states from new seed-11/29 actor rollouts. It evaluated each possible first action and then followed the fixed deterministic actor-logit ensemble for the rest of the episode. Grammar fusion agreed with the seed-11 critic argmax on 64.5% of states (95% CI [58.0%, 71.0%]), seed-29 critic argmax on 0%, the multi-critic GPI argmax on 0%, and the highest continuation-return MC first action on 22.5% (95% CI [16.5%, 28.5%]). Mean continuation returns after first action 0/1/2 were -132.92/-202.47/-157.11; mean regret relative to the best of these three MC continuations was 108.24 return points (95% CI [86.10, 132.03]). This MC comparison is exact for the deterministic Acrobot rollout and fixed ensemble continuation at the sampled states, but it estimates Q for that continuation policy rather than optimal Q-star.
- The earlier Monte-Carlo regression baseline's seed-11 and seed-29 validation MAEs were 43.22 and 28.82 return units on observed source-policy state-action pairs. Those errors plus the FQE action disagreements warn against blaming conflict arbitration from Q-based counterfactual claims until critic quality is established. Full FQE-GPI evaluation means were -499.57 on Acrobot (best-single paired delta +0.43, 95% CI [0.00, 1.29]) and 9.34 on CartPole; neither should be merged into the clean grammar-fusion result.

### Reproducibility artifacts and checks

- Main per-task results and full audit artifacts: `aim_auditability/runs/shared_multienv_gpi/CartPole-v1/` and `aim_auditability/runs/shared_multienv_gpi/Acrobot-v1/`; exact shared-bin overlap files are `shared_symbolizer_rule_overlap.json` in each task folder.
- Acrobot arbitration counterfactuals: `aim_auditability/runs/shared_multienv_gpi/Acrobot-v1/grammar_fusion_diagnostics.json`.
- FQE action/value diagnostics: `aim_auditability/runs/shared_multienv_gpi/CartPole-v1/gpi_fqe/gpi_fqe_diagnostics.json` and `aim_auditability/runs/shared_multienv_gpi/Acrobot-v1/gpi_fqe/gpi_fqe_diagnostics.json`.
- Acrobot fusion-vs-Q-vs-MC actions: `aim_auditability/runs/shared_multienv_gpi/Acrobot-v1/gpi_fqe/acrobot_q_action_agreement.json`.
- Reusable diagnostics are `aim_auditability/auditability/diagnose_fqe_gpi.py`, `diagnose_acrobot_fusion.py`, and `diagnose_acrobot_q_action_agreement.py`. All three passed `py_compile`; all generated diagnostics were written and reviewed. Source code and all result artifacts remain under `C:\AI\RL`; only plain-text research notes are stored in `paper`.
- Interpretation boundary: these outcomes support a bounded claim that provenance-carrying rule fusion is replayable and highly covered in this evaluation, and that the confidence-ranked Acrobot fusion gains over the failing source actors on 39/100 episodes. They do not show that its arbitration is optimal, that the rules reveal causal agent rationale, or that FQE-GPI is validated. The random-arbitration effect is now a key open empirical finding.

## Experiment B: raw-state held-out behavioral agreement (2026-09-22)

- Implemented `aim_auditability/auditability/run_behavioral_agreement.py`, using the user-provided `behavioral_agreement.py` only as a reference for the question and report structure. The reference defines a separate 128-hidden-unit Actor and samples source training traces. The executed experiment instead loads the actual project `Actor` class/checkpoints (64 hidden units) and gathers new raw observations from fresh deterministic actor rollouts, avoiding an architecture mismatch and training/holdout leakage.
- For each source policy, collected 100 fresh CartPole episodes on non-overlapping reset-seed ranges: seed 11 used 40,000,000–40,000,099; seed 29 used 41,000,000–41,000,099. This yielded 50,000 and 13,085 candidate raw states. Uniformly sampled 5,000 states without replacement from each source occupancy, then applied both actors to the exact same 10,000 raw states. Each occupancy therefore had equal weight. The raw state-array SHA-256 and both actor-action-array hashes are in the result artifact.
- Overall deterministic argmax agreement was 5,138/10,000 = 51.38%; episode-cluster balanced mean was 51.36%, with 95% episode-cluster bootstrap interval [50.48%, 52.26%]. On seed-11 occupancy agreement was 50.56% (5,000 states); on seed-29 occupancy it was 52.20% (5,000 states). Both source policies used a mix of actions over their own occupancy; when cross-applied to the other occupancy their action distributions diverged sharply.
- The earlier shared-symbolizer state-action grammar Jaccard for the same pair was 0.5833. With the frozen shared symbolizer, this syntax overlap is real and not caused by seed-specific bin edges, but the independent raw-state test shows no corresponding high behavioral agreement (roughly chance-level agreement in this two-action task). It directly rejects the inference that Jaccard 0.5833 alone demonstrates a robust shared action core. No 85% behavioral-consensus claim is supported.
- By absolute pole angle, agreement was 50.16% near zero (n=7,807), 55.77% in the middle band (n=2,182), and 45.45% in the large-angle band (n=11); the last stratum is too small to interpret. The two agents' fresh held-out returns were seed 11 mean/median 500/500 and seed 29 mean/median 130.85/131, so the state sample spans materially different policy-quality/occupancy regimes despite equal source weighting.
- The result is an action-agreement test on the sampled deterministic argmax policies, not evidence that either action is optimal, that a symbol caused the action, or that the policies agree on all reachable states. The held-out-state design is stronger than the reference code's training-trace sampling, while the state-distribution and policy-quality asymmetry remain explicit limitations.
- Result file: `aim_auditability/runs/shared_multienv_gpi/CartPole-v1/behavioral_agreement_experiment_B.json`. Implementation was `py_compile` checked and run end-to-end; exact checkpoints, reset ranges, sampling counts, state/action hashes, occupancy-stratified agreement, and episode-cluster bootstrap results are saved in the JSON.

## Random rule selection vs uniform random actions (2026-09-22)

- Added a matched Acrobot control because the five repeated random-rule arbitration runs were unexpectedly much stronger than confidence-ranked fusion. The new control samples one of the three environment actions uniformly and independently at every time step, without looking at a source actor or rule, over the same 100 reset seeds. It was repeated with five independent policy RNG streams.
- Uniform random action mean returns were -499.98, -496.39, -497.76, -499.85, and -499.55 (replicate mean -498.706); medians were all -500. The corresponding random-rule arbitration runs averaged -194.10, -195.38, -201.46, -196.97, and -206.17 (mean across replicates -198.816). Thus the random-rule result is not explained by unconditioned uniform action noise alone. Candidate availability depends on the symbolic state, so the stochastic selector creates a state-conditioned mixture of source rules; that distinction is an interpretation of the tested policies, not proof that the symbolic rules encode optimal actions.
- The random-action control is paired to the primary grammar-fusion reset seeds, and each random-action replicate's paired delta vs confidence-ranked fusion is negative (about -53 to -57 points; all 95% bootstrap intervals below zero). The contrast with random-rule arbitration is robust across five selector streams, while the exact mechanism still needs action/source-frequency analysis and evaluation on more tasks.
- Output: `aim_auditability/runs/shared_multienv_gpi/Acrobot-v1/random_action_control.json`; implementation: `aim_auditability/auditability/diagnose_acrobot_random_action.py`. The package was compiled and executed end-to-end.

## Repository-root organization for review and release preparation

- After the experiments, moved the project source files and experiment tree from the nested `aim_auditability/` folder to the `C:\AI\RL` repository root: `auditability/`, `runs/`, `manuscript/`, `references/`, core Python files, and `requirements.txt`. The `paper/` directory remains reserved for plain-text research notes. The old nested folder now contains only the local Python environment and caches.
- Added a root-level research README and `.gitignore`. Full trace/checkpoint output under `runs/` is retained locally but excluded from ordinary Git commits due its multi-gigabyte size. A compact 16-file, approximately 0.51 MB review package is under `results/shared_multienv_gpi/`; `results/RESULTS_MANIFEST.json` records each result copy's source path, byte count, and SHA-256.
- Updated the manuscript artifact list to the canonical root-level paths. Historical notes keep the paths that were valid when those earlier runs were made.
- Marked the current manuscript body as a pre-multiseed draft so its earlier two-agent CartPole claims cannot be mistaken for the latest results; the README and updated research report carry the current evidence summary. Reconcile the manuscript before submission.
- Rechecked all root and `auditability/` Python files with `py_compile`; ran Experiment B again from the new root package; verified all compact JSON files parse and all 16 manifest hashes match; checked the documented audit CLI usage and `git diff --check`. Full training/replay outputs were moved intact; the code/result move did not modify raw trajectories or checkpoints.

## Evidence gap analysis: what is still needed to state a claim about task complexity (2026-09-23)

### The claim under test and what it currently rests on

- Working hypothesis (user): fusing RL training results requires the training task not be designed too complex. As stated it is not falsifiable, because "complexity" is never operationalized into a quantity. The version the existing data can actually reach is narrower: **fusion success is predicted by cross-seed rule conflict, and in this repo conflict tracks the symbolizer's conditions-per-sample budget; on Acrobot the failure is attributable to the ranking function rather than to the task.**
- Current support, three layers, scored separately. Description layer (fusion gains in the low-conflict environment, fails in the high-conflict one): strong, it is a direct measurement. Association layer (conflict rather than environment identity predicts it): moderate, two environments, direction consistent, no manipulation of the mediator. Causal layer (complexity is the limiting factor): weak and partly contradicted - on Acrobot, swapping only the ranker from confidence to random moves fusion from -443.37 to -187.02 (paired delta +256.35 [231.38, 278.21], replicated on five independent RNG streams), so the same banks on the same task are fusible and the ranking function is what broke.
- The single measurement that bridges "complexity" to a number, computed from all 16 `grammar_audit_only` trajectories as total training steps divided by distinct state symbols visited: CartPole-v1 (dims 4, max abs S 256) 179.1, 269.4, 275.0, 320.6, 450.3, 457.6, 476.3, 514.7, mean **367.9**; Acrobot-v1 (dims 6, max abs S 4096) 51.7, 69.8, 75.7, 79.0, 83.9, 125.0, 126.7, 141.8, mean **94.2**. Ratio **3.91x** and the two distributions do not overlap (CartPole minimum 179.1 exceeds Acrobot maximum 141.8).
- Because abs S = bins^dims, 4^4 = 256 versus 4^6 = 4096 is a 16x difference in condition count producing the observed ~4x difference in samples per condition. That chain alone accounts for the conflict difference (0 of 77 shared conditions conflicting on CartPole versus 77 of 98 on Acrobot) without invoking intrinsic task difficulty.
- Note that "symbolizer resolution" and "samples per condition" are not two separate causes but one knob: raising bins raises the condition count and lowers samples per condition. The claim should be tightened to a single chain, resolution/dims -> samples per condition -> conflict -> outcome.

### Required experiments, in execution order

**P0 - Train/deploy action-selection mismatch (prerequisite; zero training).**
- Question: does "competent" mean the same thing during training and during evaluation?
- Verified facts: training samples from `Categorical(masked_logits)` (`auditability/grammar_audit_experiment.py` L594-596) and the rules are induced from those *sampled* actions (L606); evaluation uses deterministic `.argmax()` (`auditability/run_shared_multienv_gpi.py` L247). Acrobot baseline training means are -263.03, -466.42, -251.30, -488.64, -297.37, -415.04, -192.28, -294.46, while held-out is **exactly -500.00 on all 8 seeds**. With reward -1/step and a 500-step cap, -500 means timed out without reaching the goal and -192 means the goal was reached at step 192, so the agents sometimes solve it while training and never once when deployed.
- Design: re-evaluate the same 8 saved actors under (a) sampled actions and (b) argmax; report both. Then re-derive rule banks from argmax rollouts and compare coverage and conflict against the sampled-rollout banks.
- Cost: zero training. This must precede everything else, because every downstream competence and conflict number is currently measured through a lens that differs between training and deployment.

**P1 - Symbolizer-resolution sweep to the matched-abs-S point (load-bearing test; zero training).**
- Question: does cross-seed conflict rise to Acrobot levels *within CartPole* once samples per condition is matched?
- Key arithmetic: 8^4 = 4096 = 4^6, so CartPole at bins=8 has the same abs S as Acrobot at bins=4. The CartPole-to-Acrobot samples-per-condition gap can therefore be reproduced inside a single task by raising the resolution.
- Design: re-extract rules from the *existing* CartPole traces at bins in {2,4,6,8} and recompute conflict density, pairwise rule overlap, coverage, and fusion outcome. No retraining is required because rule extraction reads the recorded traces.
- Pre-registered decision rules. (a) If conflict at bins=8 reaches Acrobot-like levels, "complexity" reduces to samples per condition and the task-identity explanation is dead. (b) If conflict rises but fusion still succeeds, the claim "conflict predicts the outcome" is refuted and conflict drops out of the explanatory set. (b) is the most informative possible failure and must be written down before running.
- This is currently ranked in the plan as a cheap side item. It should be first among the empirical tests, because it is the only zero-training manipulation of the mediator.

**P2 - Competent-but-incompatible pairs (zero training).**
- Question: does conflict predict failure once source quality is held fixed?
- Verified facts: `runs/shared_multienv_gpi/CartPole-v1/single_actor_comparators.json` shows seeds 11, 149, 211 and 307 all reach matched mean return 500.0 on the shared 100 reset seeds, so competent pairs already exist on disk. The existing Experiment B pair (11 vs 29) is competent-versus-incompetent (500.0 vs 131.11), which is why its 51.38% agreement cannot support a "competent but incompatible" reading.
- Design: rerun Experiment B on a competent pair (11 vs 149), and recompute rule conflict and fusion restricted to those two banks.
- Framework correction to make at the same time: report fusion against the best *held-out* actor, not against the training-return-selected actor. That selection is anti-correlated here - seed 43 has the best training return (335.57) but only 163.65 held-out, while seed 149 has a middling training return (193.10) but 500.0 held-out. Fusion is 205.96, i.e. 294.04 below the best held-out actor.
- Cost: zero training.

**P3 - Arbitration ablation (zero training).**
- Question: is the Acrobot failure the ranking function rather than the task?
- Verified facts: `runs/shared_multienv_gpi/Acrobot-v1/grammar_fusion_diagnostics.json` gives `ablations.random_candidate` = -187.02 against `ablations.confidence_first` = -443.37, with five RNG-stream replicates from -194.10 to -206.17; uniform random *actions* average -498.71, so the random-rule result is not explained by unconditioned noise.
- Design: same banks, same 100 episodes, swap only the ranker: confidence, mean-reward, fitted-Q, MC-based, contextual-reward. Note that the fitted-Q option is not a cheap reuse: the existing Acrobot critic is degenerate (Q-value std ~0.008, greedy action agreement 0.0626, GPI selecting action 0 on all 50,000 probed states).
- Decision rule: if a principled ranker beats random (-187.02), the failure is the ranking function and difficulty leaves the explanatory set. On present evidence this is the most likely outcome.

**P4 - Fusion resampling design (zero training; statistical prerequisite).**
- Question: what is the sampling distribution of the fusion object?
- Problem: the 8 seeds replicate *training*, not fusion. Fusion is a single instance, so any "fusion delta versus X" correlation has no variance source. Raising the seed count does not fix this.
- Design: leave-one-seed-out over the eight rule banks, plus rule-bank bootstrap, to produce a distribution of fusion outcomes.
- Needed before any fusion-level comparison can be tested, including the plan's early-warning probe.

**P5 - Budget axis run on both environments (training required).**
- Question: does the fusion outcome track samples per condition as an independent variable?
- Design: run the plan's budget sweep {300, 600, 1500, 3000} on **both** CartPole and Acrobot rather than Acrobot alone, giving a 2-task by 4-budget grid. This reuses the existing pipeline and gives the difficulty axis the three or more levels the hypothesis needs.
- Blocker to resolve first: the plan's item A pseudocode adds entropy regularisation while item A's own prose requires keeping the algorithm family fixed. Entropy regularisation changes the objective to max-ent RL, so either drop it or declare that this axis moves the algorithm too.

**P6 - Within-task difficulty knobs (training, light).**
- Horizon, action noise and reward sparsification on Acrobot, as the plan proposes. Caveat: raising the time limit above 500 does not make the task harder, only lowering it does, so {200, 500, 1000} is an asymmetric knob and should be used knowingly or replaced by a monotone one.

**P7 - Independent difficulty metric (blocked; needed before any difficulty regression).**
- The plan's three proxies are currently unusable. `1/(1-gamma)`: gamma is 0.99 in *both* environments, so this is 100 for both and carries zero discriminative power. "Symbolized abs S x abs A (CartPole 256 vs Acrobot 4096)": 256 = 4^4 and 4096 = 4^6 are abs S, not abs S x abs A, which would be 512 and 12288. Bellman residual: the critic it comes from is degenerate and is fit on the source policies' logged returns, which violates the independence requirement the plan itself states.
- Until this is fixed, any regression of fusion outcome on a difficulty metric is uninterpretable.

**P8 - Early-warning probe (blocked; requires retraining).**
- Verified blocker: `training_updates.jsonl` stores only hashes (`actor_state_sha256_after_update`, `optimizer_state_sha256_after_update`, `gradient_sha256_after_clipping`, `rng_state_sha256_after_update`) across 300 records, with **no weight tensors**, so mid-training checkpoints cannot be reconstructed. The probe therefore requires retraining, and P4 must land first or it has no denominator.

**Deferred, correctly:** the Procgen calibrated-difficulty surface and the sample-complexity-to-fusability bound. Neither should be started before P1 to P5 produce a monotone curve.

### Pre-registered decision criteria

- If P1 shows conflict rising to Acrobot-like levels at matched abs S: the hypothesis survives only in the reduced form "samples per condition limits fusability", and the task-identity explanation is dead.
- If P1 shows conflict rising while fusion still succeeds: conflict is not the predictor; the reframed claim is falsified and must be withdrawn.
- If P3 shows a principled ranker beating random arbitration: the Acrobot failure is a ranking-function failure and complexity leaves the explanatory set entirely.
- If P5 shows the fusion outcome not tracking samples per condition across the 2-task by 4-budget grid: the reduced form is falsified too, and the honest conclusion is that the current evidence supports only the descriptive layer.
- Only if P1, P3 and P5 all fail to move the outcome would intrinsic task difficulty remain a live variable - and at that point P7 must be repaired before it can be measured.

## Project A: repairing Acrobot fusion, and its split from the complexity-claim programme (2026-09-23)

### Correction to the ordering in the previous section

- The previous section listed P0-P8 as a single execution order. That conflated two different objectives: **Project A**, making Acrobot fusion work, and **Project B**, establishing whether task complexity is a legitimate explanatory variable. Most of the P items serve B, and only P3 and P4 serve A.
- The split matters because **A decides whether B is needed at all**. If a repaired ranker makes Acrobot fusion work, the complexity hypothesis is dead in its original form and P1/P5/P7 become unnecessary. If Acrobot still fails after the ranker is repaired, the case for the hypothesis is much stronger. A therefore goes first: it is both cheaper and decision-bearing for B.
- **P1 is shared by both programmes** (it is zero-training and it manipulates the mediator), so it should not be filed under B alone.

### The Acrobot picture, verified from `shared_multiseed_gpi_summary.json` and `gpi_fqe/acrobot_q_action_agreement.json`

- Held-out mean and median return on the 100 matched reset seeds: `best_single` (seed 211, selected on training return) -500.0 / -500.0; `actor_mean_logits` (ensemble) -500.0 / -500.0; `gpi_mc_q` -500.0 / -500.0; `grammar_fusion` (confidence arbitration) **-443.37 / -500.0**; random arbitration **-187.02 / -165.0**; uniform random actions -498.71 / -500.0.
- Every ranking-based method sits at or near the floor while the one ranking-free method has median -165, i.e. it solves the task in the majority of episodes. **"Defer to the ensemble" is therefore not a remedy** - the ensemble is also at -500.
- Decision mix for grammar fusion: 44,376 total steps, 42,467 covered (95.7%), 39,602 conflicts (89.2% of covered steps), 1,909 blind spots.
- Monte-Carlo action semantics on 200 fresh states: mean continuation return by forced first action is action 0 **-132.92**, action 1 -202.47, action 2 -157.11; the MC-best first action is action 0 on **87/200 = 43.5%** of states; fusion action histogram is **action 0 = 0, action 1 = 129, action 2 = 71**. Fusion agrees with the MC-best first action on 22.5% [0.165, 0.285]. The best-action margin is 108.24 [86.10, 132.03], so the decisions are not near-ties.
- **The ranker never selects action 0**, which is the best first action on 43.5% of states and has the best mean continuation return. Random arbitration selects it roughly one time in three. This is the most likely reason random beats confidence: random occasionally lands on the best action, confidence never does.
- The rule bank carries no value signal in either environment: Acrobot 1694 rules with **1575 (93.0%) having mean_reward exactly -1.0**; CartPole 921 rules with **one distinct mean_reward value**. Acrobot confidence spans 0.700-1.000 (median 0.838) and support spans 8-12953 (median 44). Confidence and support are behavioural statistics, not value signals, which is why mean-reward-first arbitration returns -500.
- Interpretation boundary: random arbitration is *state-conditioned but ranking-free*, because candidate availability depends on the symbolic state. Uniform random actions return -498.71, so the ~300-point gain comes from state-conditioning, not from ranking. **Random winning therefore does not establish that the bank contains rankable information.** The open question is whether any principled ranker can beat -187.02; if none can, the bank's value is state-conditioned action diversity and "ranking" is the wrong frame.

### Project A work items

- **A1 - Deploy-protocol check (= P0).** Re-evaluate the same eight saved actors under sampled actions and under argmax, and report both; then re-derive rule banks from argmax rollouts and compare coverage and conflict. Difficulty **low** (~1 hour), zero training. **Do now.** It defines what "competent" means and fixes the state distribution the ranker would be tuned against. The result is paper-usable either way, since the train/deploy gap is itself a finding.
- **A2 - Rankability test.** Run MC-based arbitration on the same 100 matched episodes and compare against random (-187.02) and confidence (-443.37). Difficulty **medium** (~1 day), zero training. **Do now.** This is the only experiment that can convert the existing negative finding ("random beats confidence") into a positive contribution ("here is a ranker that beats both").
- **A3 - Build the production ranker.** Wire A2's estimator into the fusion decision and evaluate on the matched episodes. Difficulty **medium** (1-2 days), zero training. **Conditional on A2.**
- **A4 - Statistics (= P4).** Leave-one-seed-out over the eight rule banks plus rule-bank bootstrap, so that a statement like "new ranker -150 versus confidence -443" has a sampling distribution. Fusion is currently a single instance and the existing intervals are episode-level only. Difficulty **low** (~2 hours), zero training. **Do now.**
- **A5 - Rewrite the claim.** If A2 fails, the honest claim becomes "the induced rule bank supplies state-conditioned action diversity, not rankable preference". Difficulty low (writing only). **Conditional on A2.**

Total for a complete Project A: roughly two to four days, entirely without training.

### Three design constraints on A2

1. **A2 must produce three numbers, not two.** Random (-187.02), the MC-based ranker, and the **MC oracle ceiling** (per-state best action from the counterfactual rollouts). If the oracle is only marginally better than random, perfect ranking is worth very little and A3 should not be built.
2. **The lever to measure is the action-0 selection rate.** Known values: MC-best 43.5% of states, confidence fusion 0/200, random roughly one third. A2 should report this rate for every ranker it tests.
3. **The estimator is bootstrapped off the policy under repair.** The existing MC estimator values a candidate first action by forcing it and then following the actor-logit ensemble, i.e. it uses the very policy being improved as its continuation. That is one step of policy iteration and may need two or three rounds to stabilise; this must be written into the protocol rather than discovered later.

### Disposition

- **Now:** A1, A2, A4.
- **Conditional on A2:** A3, A5.
- **Future work, and deliberately not now:** Project B (P1, P5, P7). The manuscript makes no complexity claim, so B is not needed for it, and A2's outcome decides whether B is worth running at all.
- **Exception:** P1 (the symbolizer-resolution sweep) needs no training, so it can be run at any time without touching the manuscript, and it remains the cheapest entry point if the complexity hypothesis is pursued separately.

## A1 and A2 executed (2026-09-23)

Full report: `paper/A1_A2_execution_report.md`. Zero training. Harness validated three
ways before any claim was made: bank re-induction reproduces `induced_grammar.json`
exactly (8/8 seeds, both environments), argmax evaluation reproduces
`metadata.evaluation.mean_return` exactly (16/16), and the A2 rerun reproduces the
recorded ablations exactly (`confidence_first` -443.37, `random_candidate` -187.02).

**A1 - deploy protocol.** The train/deploy action-selection gap flips sign between
environments. CartPole: argmax beats sampling on **8/8 seeds** (mean 330.86 vs 304.20).
Acrobot: argmax returns **exactly -500.00 for all 8 seeds** while sampling averages
-446.12. Consequently the Acrobot `best_single` selection is uninformative under the
protocol it is deployed with - seed 211 was chosen for the best *sampled* training mean
(-192.28) and scores -500.00 greedily, exactly like the other seven. Cross-seed bank
conflict tracks the fusion outcome (CartPole 0.61%, Acrobot 77.75%). Re-inducing from
greedy actions raises Acrobot conflict to 84.18%, so the confound does not explain the
Acrobot failure away. New cheap follow-up: every recorded Acrobot composition number
was measured under argmax where all base actors are already at the floor, while the
banks were induced from sampled rollouts - re-run the composition under the sampled
protocol.

**A2 - rankability.** The Monte-Carlo ranker is **degenerate, not weak**. Under the
8-actor continuation the pipeline actually uses, every forced first action returns
exactly -500 at **100% of 580 probed states** (spread 0.0); a 2-actor control
discriminates normally, which rules out a harness bug. `mc_ranked` therefore reduces to
"smallest candidate action" and `mc_oracle` to "always action 0"; both reductions were
re-verified against the real estimator on 12 live decision points (12/12 each).
Policy-level: `mc_ranked` **-500.00** vs `random_candidate` -187.02 vs
`confidence_first` -443.37, paired delta -56.63 [-75.11, -40.14] against confidence.
The oracle ceiling is 313 points **worse** than random, so by the pre-registered
constraint #1 **A3 is closed** and **A5 is triggered**. Separately, the recorded MC
probe in `gpi_fqe/acrobot_q_action_agreement.json` used a 2-actor continuation while
the reported fusion uses 8, so its apparent rankable signal does not transfer.

## A1b executed (2026-09-23) - protocol-consistent composition

**Question:** does Acrobot fuse once the bank's action labels match the deployment
protocol? Every recorded Acrobot composition number was measured under argmax (all 8
base actors at the -500 floor) using banks induced from *sampled* rollouts - a
train/deploy protocol mismatch. A1b runs the full 2x2: {sampled-induced bank,
greedy/argmax-induced bank} x {argmax eval, sampled eval}, plus base policies under
both protocols. Harness self-test passed: the recorded config (sampled bank, argmax
eval) reproduces -443.37 / -187.02 / -500.00 / -500.00 exactly.

**Acrobot-v1 (best_single seed 211):** grammar_fusion sampled-bank argmax = -443.37
(recorded baseline); **greedy-bank argmax = -141.77 (+301.60)**; greedy-bank sampled
eval = -133.21. Versus best argmax base actor (-500.00) the greedy bank is **+358.23**
and is now the best of all 8 tested strategies. Coverage (95.4% vs 95.7%) and conflict
(87.6% vs 89.2%) are essentially unchanged between banks, so the gain is purely
label-protocol consistency. The evaluation protocol barely matters (~8-pt swing); the
induction protocol dominates (~301-pt swing).

**CartPole-v1 (best_single seed 43):** sign flips - the *sampled* bank gives the
better grammar_fusion (205.96 vs 191.41 for greedy, +14.55). Conflict ~0.2-0.5%, so the
environment is robust; no universal "use the greedy bank" rule.

**Findings:**
- Acrobot fusion was a protocol artifact, not an impossibility - it fuses at -141.77.
- Induction protocol dominates; evaluation protocol is near-irrelevant.
- Protocol-consistent grammar_fusion (-141.77) now beats random_candidate (-234.26) by
  +92.49, reversing the recorded ordering and showing the ranker was fed wrong labels,
  not that ranking is useless (consistent with A2 degeneracy).
- Effect is environment-dependent (sign flip) - claims must be per-environment.

**Conclusion for the driving question:** A2 did NOT decide B; it only closed A3 / triggered
A5 and redirected diagnosis to A1b. Acrobot CAN be fused now (-141.77). Therefore the
task-complexity hypothesis B loses its primary motivation - Acrobot's failure was a
methodology bug (train/deploy mismatch), not complexity. B moves to Future Work.
Report: `paper/A1b_execution_report.md`; data: `results/a1b_protocol_consistent_composition.json`.
