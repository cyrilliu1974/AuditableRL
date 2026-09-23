# Expert Review Bundle v2 — RL Grammar Fusion: Current Status (2026-09-22)

Supersedes RL_Grammar_Fusion_ExpertReview.zip (v1, two-seed CartPole prototype only).

## What changed since v1
- Shared frozen symbolizer, 8 seeds (11/29/43/71/101/149/211/307) x 300 episodes, two envs.
- CartPole seed11/29 Jaccard 0.583 (77/132), zero conflicts; 8-seed mean 0.557.
- Acrobot seed11/29 Jaccard 0.050 (21/423), 77/98 shared conditions conflict; 8-seed mean 0.096.
- Acrobot grammar fusion (100 held-out episodes): mean -443.37 vs best single -500.00
  (paired +56.63, 95% CI [+39.32,+75.15]); median -500 both; 39/100 early terminations.
- Fusion decision mix on 44,376 steps: 95.7% rule-covered, 89.2% conflicts, 4.3% blind spots.
- Arbitration ablation: confidence-only -444.49 (~tie); mean-reward-only / support-first /
  defer-to-actor all -500.00; uniform random rule choice ~-195 (5 RNG streams), uniformly
  better than confidence ranking — open empirical finding, mechanism not yet decomposed.
- Experiment B (raw-state behavioral agreement): same pair's grammar Jaccard 0.583 but
  deterministic-argmax agreement only 51.38% (95% CI [50.48%, 52.26%]) — rejects the
  inference that syntactic overlap equals a shared action core.
- FQE-GPI is NOT a baseline here: CartPole agreement with actor ensemble 35.27%,
  full-eval 9.34 vs 163.65; Acrobot Q correlation 0.652 yet greedy actions fully diverge.
  Kept in appendix only, separated from the clean fusion result.

## Reading order
1. paper/AI_Grammar_Induction_Experiment_Log.md — newest sections first:
   "Shared-symbolizer multi-seed results and Acrobot fusion diagnosis",
   "Experiment B", "Random rule selection vs uniform random actions".
2. results/shared_multienv_gpi/{CartPole-v1,Acrobot-v1}/shared_multiseed_gpi_summary.json
3. results/.../shared_symbolizer_rule_overlap.json (both envs)
4. results/.../Acrobot-v1/grammar_fusion_diagnostics.json + random_action_control.json
5. results/.../behavioral_agreement_experiment_B.json (CartPole)
6. results/.../gpi_fqe/* (appendix: why GPI is excluded from the main claim)
7. paper/20260922_RL_Grammar_Fusion_Research_Report.md — first-stage report (historical)
8. Legacy two-seed prototype: runs/grammar_merge_*_final/merge_evaluation.json,
   runs/grammar_intersections/, runs/replication_summary.json (AIM 6-seed causality),
   runs/grammar_mdl_seed11_070.json.

## Reproduce (root-level package; full traces stay local, git-ignored)
- auditability/grammar_audit_experiment.py → run_shared_multienv_gpi.py
- auditability/compare_seed_grammars.py / analyze_shared_symbolizer_overlap.py
- auditability/merge_seed_grammars.py → diagnose_acrobot_fusion.py
- auditability/fit_gpi_fqe_baseline.py → diagnose_fqe_gpi.py → diagnose_acrobot_q_action_agreement.py
- auditability/run_behavioral_agreement.py (Experiment B)

## Claim boundary
Existence proof that trajectory replay + provenance-carrying rule fusion is implementable
with hash/env-replay evidence in these controlled settings. NOT general auditability,
NOT semantic universality, NOT optimal arbitration, NOT a validated GPI comparison.
