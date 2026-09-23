# Expert Review Bundle — Seed-Specific RL Grammar Fusion (2026-09-22)

## What this bundle proves
Provenance-preserving rule-level fusion of two CartPole agents: shared-rule core,
explicit conflicts/blind spots, hash-chained + replay-verified merged trace.
Fusion preserves the stronger agent's 500-point ceiling; it does not improve it.

## Suggested reading order
1. paper/20260922_RL_Grammar_Fusion_Research_Report.md — main report, claims + limits
2. aim_auditability/manuscript/rl_rule_fusion_draft.md — workshop-style paper draft
3. paper/AI_Grammar_Induction_Experiment_Log.md — append-only experiment history
4. paper/AIM_Auditability_Research_Log.md — AIM source diagnosis + causal design
5. paper/20260922.claude.md — original fusion proposal (now tested)
6. Reproduce: auditability/grammar_audit_experiment.py -> compare_seed_grammars.py
   -> merge_seed_grammars.py -> verify_deterministic_repeat.py

## Key numbers (all in the JSON evidence)
- 0.70 threshold: 73/134 shared productions (Jaccard 0.545); held-out 100k decisions:
  10,137 agree + 48,135 single + 41,728 blind spots (41.7%); fused return 500 = seed-11 ceiling.
- 0.90 threshold: 3/34 shared (Jaccard 0.088); held-out 100k/100k blind spots (100% fallback).
- AIM private-target: 6 seeds x 100k decisions; matched-support intervention delta ~0.998.
- Trace MDL by round (seed 11, 0.70): ratios 0.513 -> 0.575; gzip still smaller than grammar archive.

## Deliberately excluded
- merged_policy_trace.jsonl (148MB + 84MB full traces): reproduced via merge_seed_grammars.py;
  hashes recorded in the two merge_evaluation.json files.
- Full per-seed training traces/checkpoints under runs/: available on request.
