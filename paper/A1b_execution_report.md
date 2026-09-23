# A1b — Protocol-Consistent Composition: Is Acrobot Fusion Recoverable?

**Date:** 2026-09-23
**Script:** `auditability/run_a1b_protocol_consistent_composition.py`
**Output:** `results/a1b_protocol_consistent_composition.json`
**Harness self-test:** PASSED (Acrobot `grammar_fusion` sampled-bank/argmax rerun = −443.37 = recorded exactly; `best_single` and `actor_mean_logits` reproduced exactly).

---

## 1. The question this experiment answers

A1 proved that the pipeline selects training actions by **sampling** but evaluates and deploys by **argmax**, and that on Acrobot this gap is catastrophic (argmax returns exactly −500.00 for all eight saved actors; sampling averages −446.12).

A2 proved the value-based **ranker** is degenerate under the pipeline's 8-actor continuation (every forced action returns −500), so a "better ranker" cannot rescue fusion. A2 **closed A3** (no production ranker to build) and **triggered A5** (claim rewrite). But A2 did **not** tell us whether Acrobot can actually be fused — it only ruled out one mechanism.

This left exactly one live hypothesis for the recorded Acrobot fusion numbers:

> Every recorded Acrobot composition number (−443.37 fusion, −187.02 random, −500.00 GPI) was measured under **argmax**, in a regime where all eight base actors already sit at the floor −500 — while the rule banks those compositions consume were induced from **sampled** rollouts. The bank's action labels and the deployment action rule disagree.

A1b tests this directly with a 2×2:

| | Bank induced from **sampled** actions | Bank induced from **greedy** (argmax) actions |
|---|---|---|
| **Evaluated under argmax** | recorded config (self-test) | protocol-consistent |
| **Evaluated under sampling** | protocol-consistent (induction side) | protocol-consistent (both sides) |

For each cell we report `grammar_fusion` (confidence-ranked) and `random_candidate` (random arbitration), plus the base policies (`best_single`, `actor_mean_logits`) under both protocols, so every composition is read against the regime it actually runs in.

---

## 2. Results — Acrobot-v1 (best_single seed 211)

| policy | bank | eval | mean return | Δ vs recorded fusion | coverage | conflict | action-0 share |
|---|---|---|---:|---:|---:|---:|---:|
| best_single | — | argmax | −500.00 | — | — | — | — |
| best_single | — | sampled | −332.14 | — | — | — | — |
| actor_mean_logits | — | argmax | −500.00 | — | — | — | — |
| actor_mean_logits | — | sampled | −229.34 | — | — | — | — |
| grammar_fusion | **sampled** | argmax | **−443.37** | (baseline) | 95.7% | 89.2% | 23.9% |
| grammar_fusion | sampled | sampled | −436.87 | +6.50 | 97.4% | 90.9% | 24.9% |
| random_candidate | sampled | argmax | −187.02 | +256.35 | 96.7% | 81.7% | 29.4% |
| random_candidate | sampled | sampled | −186.69 | +256.68 | 96.8% | 81.8% | 29.6% |
| **grammar_fusion** | **greedy** | **argmax** | **−141.77** | **+301.60** | 95.4% | 87.6% | 31.2% |
| grammar_fusion | greedy | sampled | −133.21 | +310.16 | 97.1% | 89.9% | 31.7% |
| random_candidate | greedy | argmax | −234.26 | +209.11 | 98.3% | 94.0% | 24.3% |
| random_candidate | greedy | sampled | −229.64 | +213.73 | 98.5% | 94.4% | 24.9% |

Rule counts: sampled bank **1694** rules, greedy bank **2434** rules.

### Headline for Acrobot
- **Greedy-bank grammar_fusion under argmax = −141.77**, versus the recorded **−443.37** → **+301.60**.
- Versus the best argmax base actor (−500.00) → **+358.23**. It is now the single best Acrobot composition of all eight strategies tested.
- **The evaluation protocol barely matters** (argmax −141.77 vs sampled −133.21 under the greedy bank: an 8.6-point swing). **The induction protocol dominates** (sampled −443.37 vs greedy −141.77: a 301.6-point swing).
- Coverage (95.4% vs 95.7%) and conflict rate (87.6% vs 89.2%) are essentially identical between the two banks. So the +301.6 gain is **not** more coverage and **not** better conflict resolution — it is purely that the induced **action label** now matches what argmax deployment would actually choose.

---

## 3. Results — CartPole-v1 (best_single seed 43)

| policy | bank | eval | mean return | coverage | conflict | action-0 share |
|---|---|---|---:|---:|---:|---:|
| best_single | — | argmax | 163.65 | — | — | — |
| best_single | — | sampled | 143.90 | — | — | — |
| actor_mean_logits | — | argmax | 500.00 | — | — | — |
| actor_mean_logits | — | sampled | 500.00 | — | — | — |
| grammar_fusion | **sampled** | argmax | **205.96** | 98.3% | 0.4% | 52.4% |
| grammar_fusion | sampled | sampled | 208.01 | 98.6% | 0.4% | 52.4% |
| random_candidate | sampled | argmax | 198.40 | 98.3% | 0.2% | 52.5% |
| random_candidate | sampled | sampled | 199.56 | 98.1% | 0.2% | 52.5% |
| grammar_fusion | **greedy** | argmax | **191.41** | 99.5% | 0.5% | 52.7% |
| grammar_fusion | greedy | sampled | 191.50 | 99.5% | 0.5% | 52.7% |
| random_candidate | greedy | argmax | 193.56 | 99.4% | 0.5% | 52.7% |
| random_candidate | greedy | sampled | 193.97 | 99.5% | 0.5% | 52.7% |

Rule counts: sampled bank **921** rules, greedy bank **1218** rules.

### Headline for CartPole
- Here the sign **flips**: the **sampled** bank (protocol-mismatched w.r.t. argmax deployment) gives the **better** grammar_fusion (205.96 vs 191.41 for greedy, a 14.55-point edge).
- Conflict is near zero (0.2–0.5%), so the bank is internally consistent across seeds; the environment is easy enough that the label source is a second-order effect.
- This confirms A1 finding F1 (the train/deploy gap flips sign between environments) at the composition level: there is **no universal "use the greedy bank" rule**. Acrobot needs protocol consistency; CartPole is robust either way and even prefers the sampled bank.

---

## 4. Findings

- **F-A1b-1 — Acrobot fusion was a protocol artifact, not an impossibility.** Switching only the bank's action-label source from sampled to greedy (deployment-consistent) lifts grammar_fusion from −443.37 to −141.77, a +301.60 improvement, while coverage and conflict are unchanged. Acrobot **can** be fused.
- **F-A1b-2 — Induction protocol dominates, evaluation protocol is nearly irrelevant.** On Acrobot the eval-protocol swing is ~8 points; the induction-protocol swing is ~301 points. The bug is in *how the grammar is built*, not in how it is later rolled out.
- **F-A1b-3 — Protocol-consistent grammar_fusion is now the best Acrobot composition.** Under the greedy bank, grammar_fusion (−141.77) beats random_candidate (−234.26) by +92.49, reversing the recorded ordering where random (−187.02) beat fusion (−443.37). A2 had shown the ranker was degenerate; A1b shows the ranker was being fed the wrong labels, not that ranking is useless.
- **F-A1b-4 — The effect is environment-dependent (sign flip).** CartPole's best fusion uses the *sampled* bank, the opposite of Acrobot. Any claim about "which induction protocol is correct" must be stated per-environment, not globally.
- **F-A1b-5 — Harness is validated.** The recorded config (sampled bank, argmax eval) reproduces the recorded −443.37 / −187.02 / −500.00 / −500.00 exactly, so the re-run is a faithful re-execution, not a re-implementation.

---

## 5. Answer to the driving question

> *"A2 的結果會決定 B 要不要做? 重點是 Acrobot 現在能不能融合?"*

1. **A2 did not decide B.** A2 only ruled out the ranker (closed A3, triggered A5). It redirected the diagnosis — once ranking was exonerated as degenerate, the failure had to live elsewhere. A1b is where "elsewhere" was found.
2. **Acrobot can be fused now.** With a protocol-consistent (greedy) bank, grammar_fusion reaches −141.77 — the best of all eight tested strategies and +358.23 above every argmax base actor. The prior "Acrobot can't fuse / fusion is worse than base" reading was an artifact of measuring a sampled-induced bank under argmax deployment.
3. **Therefore B (the task-complexity hypothesis) loses its primary motivation.** The Acrobot data do **not** support "RL tasks must not be too complex to fuse." Acrobot's failure was a methodology bug (train/deploy protocol mismatch), which is fixable without invoking complexity. B remains a legitimate future research direction, but it is no longer the explanation for the Acrobot result.

---

## 6. What changes for the manuscript (A5 / B reframing)

- The Acrobot composition section must report the protocol-consistent result (−141.77) alongside the recorded −443.37, and state that the gap is explained by induction/deployment protocol mismatch, not by fusion being infeasible.
- The "MC-best 43.5% / agreement 22.5%" probe claim (G4) must be retired: it was measured with a 2-actor continuation but reported against an 8-actor fusion, so it does not transfer. Use A1b's direct, reproducible numbers instead.
- The complexity hypothesis (B) should be moved to Future Work, framed as "protocol consistency is necessary; whether task complexity imposes a further ceiling is open," not as the cause of the Acrobot result.
- **No paper edit applied yet** — awaiting explicit confirmation before modifying `merged_main.tex` (standing rule: align before editing).

---

## 7. Artifacts

- `auditability/run_a1b_protocol_consistent_composition.py` — the 2×2 runner (dead code removed; self-test made file-existence-robust).
- `results/a1b_protocol_consistent_composition.json` — full per-episode returns, decision counts, coverage/conflict, action histograms, and the harness self-test block.
