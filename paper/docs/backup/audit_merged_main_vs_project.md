# Audit: `merged_main.tex` vs. RL Project Experiment Results & Code

**Date:** 2026-09-22
**Target:** `C:\AI\RL\paper\docs\merged_main.tex` (fused version)
**Method:** Each empirical claim was scanned *one experiment at a time* and traced to its
primary source artifact in `results/`, `runs/`, and `auditability/`. No claim was accepted
without a matching number in the data or code.

**Verdict:** 1 divergence found (a mislabeled denominator in the MDL ratio).
All other empirical claims match the project data and code exactly or within rounding.

---

## A. Experiments verified CLEAN (claims match data/code)

| # | Experiment / claim group | Key traced numbers | Source artifact |
|---|--------------------------|--------------------|-----------------|
| 1 | Config: B=4 bins, 3 quantile cuts, τ_s=8, τ_c=0.70, 24 pooled calibration eps/seed, frozen shared symbolizer, max per-seed edge diff 0.157 | edge diff 0.156857…; single `state_bins_canonical_sha256`/env; `n_bins=4`, `torch.linspace(0,1,5)[1:-1]` | `shared_calibration.json`, `review_bundle_v2/runs/confidence_070_audit_only.json`, `grammar_audit_experiment.py:94,100-101,226-248` |
| 2 | Integrity P1–P3: hash-chain, env replay, provenance; codec token counts, 16 productions, lossless round-trip, ≤96 rules, Elias-gamma, bytes | CartPole trace 1,725,655 / updates 7,200 / replay 1,122,403 / refs 290,052; Acrobot 2,544,513 / 7,200 / 1,742,570 / 697,817; tokens 9,178,724 (CP) / 8,549,496 (AC); raw/gzip/codec MB exact; roundtrip_exact=true | `shared_multiseed_gpi_summary.json` (`audit_integrity`, `one_lossless_trace_codec`), `grammar_audit_experiment.py:408,803,363-367` |
| 3 | CartPole composition P4/P6: fusion +42.31, ensemble +336.35, decision mix, structure table | fusion 205.96, ensemble 500.0, best_single 163.65; paired Δ 42.31 [23.98,61.93] / 336.35; decision_mix 20,596 / 20,240 / 75 / 356; per-seed counts, Jaccard 0.557, seed11vs29 77/0 | `shared_multiseed_gpi_summary.json` |
| 4 | Acrobot conflict/arbiter P7/P8: ablations, 5 random replicates, MC-probe table | ablations confidence_first −443.37, confidence_only −444.49, random_candidate −187.02, support_first/defer/unanimous −500.0, source 11/29 −499.48/−500.0, best_single_fallback −452.69; 5 random replicates −194.10…−206.17 (mean −198.82); decision_mix 39,602/1,909 of 44,376; MC-probe 100%/90.5%/64.5%/0.0%/22.5%, margin 108.24, returns −132.92/−202.47/−157.11, hist 87/46/67 | `grammar_fusion_diagnostics.json`, `random_action_control.json` (−498.706), `acrobot_q_action_agreement.json` |
| 5 | Behavioral agreement (Experiment B): 10,000 fresh states, Jaccard, cluster-balanced, 51.4%, CI, own-occupancy, pole-angle strata | n_agree 5138 → 51.4%; Jaccard 0.5833; cluster-balanced 0.5136, CI [0.5048,0.5226]; own-occ 50.6%/52.2%; pole-angle near-zero 50.2%/mid 55.8%; 77 shared conditions all agree | `CartPole-v1/behavioral_agreement_experiment_B.json`, `CartPole-v1/shared_symbolizer_rule_overlap.json` |
| 6 | Causal symbol intervention (private-target comm game): 6 runs × 100k, accuracies, Δ̄, overlap, per-position TV | true 0.998923 (= mean sampled acc across 6 seeds), shuffled 0.503292 (= mean), none 0.500; Δ̄ 0.998366 (range 0.996986–0.999685) = `matched_token_delta_P_B_C` mean/min/max; overlap 0.94308 = mean common-support overlap mass; TV pos0 0.0966797 / pos1 0.9984375 | `runs/replication_summary.json`, `runs/private_target_extended/.../evaluation.json`, `position_intervention_analysis.json`, `paper/AIM_Auditability_Research_Log.md:223` |
| 7 | FQE failure diagnostics (CartPole + Acrobot) | CP: gpi_fqe 9.34, histogram 30,758:805, actor-vs-FQE agreement 0.3527 (35.3%), cross-seed Q mean 10.67/std 0.806 (≈10.7/0.81); AC: gpi_fqe −499.57, action 0 on all 50,000, Q −11.4404/std 0.00824 (−11.44±0.009), cross-seed greedy agreement 0.06262 (6.3%), Bellman residual MAE ≈0.88–0.92 (≈0.92) | `gpi_fqe/gpi_fqe_comparison.json` (mean 9.34/−499.57; `heldout_bellman_residual_mae`), `gpi_fqe_diagnostics.json` |

---

## B. DIVERGENCE (1) — MDL ratio denominator mislabeled

**Location:** `merged_main.tex`, §Lossless coding (around line 1092):

> "The MDL ratio (grammar code length / **gzip** code length) sits at 0.51–0.58 across rounds."

**What the code actually computes** (`auditability/grammar_audit_experiment.py`):
- `mdl_to_uncompressed_ratio` (line 398) = `(grammar_bits + trace_bits) / raw_bits` → this is the
  value in the **0.51–0.58** range (MDL two-part code length vs. **raw/uncompressed** trace).
- `grammar_gzip_to_raw_gzip_bytes_ratio` (line 839) = `grammar_gzip_bytes / raw_trace_gzip_bytes`
  = 13,377,210 / 12,341,225 = **1.084**. This is the ratio the paper already reports two lines earlier
  (line 1091) as "the grammar–gzip archive is 8.4% larger than gzip".

**Why it is wrong:** If the denominator were really *gzip* code length, the ratio would be ~1.08,
not 0.51–0.58. The 0.51–0.58 number is the MDL-vs-**raw-uncompressed** ratio. The label "gzip code
length" is therefore incorrect and internally contradicts the paper's own adjacent "8.4% larger than
gzip" statement.

**Proposed fix (NOT yet applied — awaiting your go-ahead):**
```
Old: The MDL ratio (grammar code length / gzip code length) sits at 0.51--0.58 across rounds.
New: The MDL ratio (two-part grammar code length / raw uncompressed trace size) sits at 0.51--0.58 across rounds.
```
The numeric value 0.51–0.58 is correct as-is; only the denominator description needs correcting.

---

## C. Notes / non-issues

- The paper distinguishes `gpi_fqe` (neural fitted-Q, featured as "Fitted-Q GPI" = 9.34/−499.57)
  from `gpi_mc_q` (Monte-Carlo Q = 9.38/−500.0). The fusion tables use `gpi_fqe`; the 9.34 vs 9.38
  gap is *not* an error (different baselines). Confirmed.
- Code default `min_confidence=0.90` (`grammar_audit_experiment.py:125`) is the unused pilot variant;
  real runs record τ_c=0.70 via CLI, matching the paper. No conflict.
- All integrity flags (hash-chain, replay, actor-equivalence, roundtrip_exact) are `true` in data.
