# Publication Package Manifest

**Package:** AIM Auditability and Rule-Level RL Fusion — verifiable data and code release
**Source workspace:** `C:\AI\RL`
**Prepared:** 2026-09-23
**Scope decision:** "Broad but clean" — every real experimental artifact is included; development
scratch, environment caches, and third-party clones are excluded with reasons recorded below.

This release is delivered on **two tracks**, because the full trace tree cannot be hosted in Git
(it is ~4.7 GB and contains individual files above GitHub's 100 MB per-file limit).

| Track | Contents | Where | Size |
|---|---|---|---|
| **1 — Git repository** | Source code, compact result JSON, paper sources, figures, reviewer bundles | `github.com/cyrilliu1974/AuditableRL` | 6.7 MB / 194 files |
| **2 — archived dataset** | Full `runs/` trace and checkpoint tree | figshare (see §5) | 1020 MB compressed |

Track 1 alone is sufficient to inspect every reported number. Track 2 is required only to replay
traces or re-derive results from raw artifacts.

---

## 1. What is included

### Track 1 — Git repository

| Group | Path | Files | Size | Contents |
|---|---|---:|---:|---|
| Config & guide | `README.md`, `requirements.txt`, `.gitignore`, `PUBLICATION_MANIFEST.md` | 4 | small | Reproduction guide, pinned dependencies, exclusion rules |
| Root source | `ai_grammar_induction.py`, `vqvae_agents_AIM.py`, `aim_dictionary_json.py`, `behavioral_agreement.py` | 4 | ~65 KB | Grammar inducer entry point, AIM source snapshot, dictionary snapshot, Experiment B reference implementation |
| Experiment code | `auditability/` | 34 | ~940 KB | All runners, validators, ablation and diagnostic analyses |
| Compact results | `results/` | 33 | ~930 KB | Multi-seed summaries, overlap tables, audit reports, FQE diagnostics, arbitration ablations, random-action control, Experiment B, A1/A1b/A2 protocols |
| Integrity index | `results/RESULTS_MANIFEST.json` | 1 | 5 KB | Source path, byte count, and SHA-256 for each of the 16 curated review copies |
| Paper | `paper/` | 46 | ~2.7 MB | Research log, execution reports, study summary and report, LaTeX source, bibliography, figures |
| Manuscript | `manuscript/` | 1 | 16 KB | English paper draft |
| Figures & PDF | `output/` | 10 | ~1.6 MB | Compiled paper PDF and QA page renders |
| Reviewer bundles | `review_bundle/`, `review_bundle_v2/` | 62 | ~310 KB | Curated snapshot bundles previously distributed for expert review |

Largest tracked file is 0.53 MB (`paper/docs/merged_main.pdf`), well inside GitHub's limits.

### Track 2 — archived dataset (`runs/`)

| Directory | Size | Role |
|---|---:|---|
| `runs/shared_multienv_gpi/` | 3.1 GB | **Primary.** The 8-seed, 2-environment experiment behind every headline number |
| `runs/grammar_confidence_070/` | 584 MB | Confidence-threshold rule-bank sweep |
| `runs/grammar_audit_pilot/` | 340 MB | Pilot audit run |
| `runs/repeatability_full/` | 164 MB | Deterministic-repeatability verification |
| `runs/private_target_replicates/` | 153 MB | Multi-replicate private-target-game runs |
| `runs/grammar_merge_070_final/`, `_070/`, `_070_v2/` | 364 MB | Seed-grammar merge at confidence 0.70 (final, base, superseded variant retained for audit trail) |
| `runs/grammar_merge_090_final/`, `_090/` | 165 MB | Seed-grammar merge at confidence 0.90 |
| `runs/private_target_extended/`, `_batch8/`, `private_target/` | 44 MB | Private-target-game variants |
| `runs/grammar_intersections/` | 36 KB | Pairwise rule intersections |
| `runs/seed-0_20260922T013053Z/`, `_012930Z/` | 15 MB | Single-seed AIM baseline runs cited in `paper/AIM_Auditability_Research_Log.md` |
| loose files | ~1.6 MB | `replication_summary.json`, `replication_summary_interim.json`, `grammar_mdl_seed11_070.json`, `objective_diagnostic.json`, `seed-0_full_baseline_console.log` |

Total: 584 files, 4.7 GB uncompressed.

Per-directory role notes: `shared_multienv_gpi/` holds per-step trajectory traces, training-update
ledgers, actor and training checkpoints, shared calibration, induced rule banks, and FQE critic
checkpoints. The legacy directories hold the single-seed AIM baseline, the grammar-merge and
calibration sweeps, the private-target-game runs, and the repeatability verification.

**Ten files exceed GitHub's 100 MB per-file limit** (largest 391.5 MB); this is the reason for the
two-track split. The largest offenders are `trajectory.jsonl` traces and
`lossless_trace_grammar.json`, which compress by roughly 6.4x.

---

## 2. What is intentionally excluded, and why

| Excluded | Size | Reason |
|---|---:|---|
| `aim_auditability/` | 1.8 GB | Contains only `.venv/`, `.cache/`, `.tmp/`, `__pycache__/` — a local Python environment, no research source. Already excluded from upstream Git. |
| `runs/*smoke*` (12 dirs) | ~37 MB | Development smoke tests: `shared_multienv_smoke{,_v2}`, `grammar_audit_smoke{,_v2}`, `grammar_audit_entry_smoke`, `grammar_core_hash_smoke`, `grammar_merge_provenance_smoke`, `smoke_private_target{,_v2...v5}`, `acrobot_action_mask_smoke`, `update_audit_smoke`, `repeatability_smoke`. These are pre-flight checks, not evidence; including them would mix non-verifiable scratch with verifiable results. |
| `references/` | 6.3 MB | Third-party public repository clones inspected during method design. **Not** dependencies of the core experiment. Upstream names, licenses, and revisions must be checked before redistribution, so they are deliberately not vendored here. |
| `__pycache__/`, `**/*.pyc` | small | Compiled bytecode caches; non-portable and regenerable. |
| `*.zip` (`RL_Grammar_Fusion_ExpertReview{,_v2}.zip`) | ~300 KB | Redundant archives of `review_bundle/` and `review_bundle_v2/`, which are present here in unpacked form. |
| `.git/` | — | The source repository's history is not copied; this package carries its own repository. |

---

## 3. Integrity verification performed

| Check | Method | Result |
|---|---|---|
| Track-1 file-count parity | `find | wc -l` compared per subtree | `auditability/` 34=34, `results/` 33=33, `paper/` 46=46 |
| Track-2 file-count parity | Included `runs/` dirs vs destination | 584 = 584 |
| Large-binary integrity | SHA-256 on sampled `.pt` checkpoints and `.jsonl` traces | 8/8 matched |
| Compact review package | All 16 entries of `results/RESULTS_MANIFEST.json` re-hashed against shipped files | 16/16 verified, 0 mismatches |
| GitHub size compliance | Scan of every file Git would track | No file above 100 MB; total 6.7 MB |
| Script syntax | `py_compile` on all shipped Python | exit 0 |
| Bytecode cleanup | `__pycache__` count in package | 0 |
| Archive integrity | `tar -tzf` listing count and checksum | see §5 |

---

## 4. How to read this package

1. `README.md` — experimental design, evidence boundaries, reproduction commands, dataset availability.
2. `manuscript/rl_rule_fusion_draft.md` — the paper draft being submitted.
3. `paper/AI_Grammar_Induction_Experiment_Log.md` — hypotheses, negative results, claim limits.
4. `paper/A1_A2_execution_report.md`, `paper/A1b_execution_report.md` — deploy-protocol, rankability, and protocol-consistent composition findings.
5. `results/shared_multienv_gpi/<env>/shared_multiseed_gpi_summary.json` — the headline multi-seed numbers.
6. `results/RESULTS_MANIFEST.json` — hashes for every compact review copy.
7. `runs/shared_multienv_gpi/<env>/<arm>/seed-*/` (Track 2) — full traces, update ledgers, and checkpoints for replay.

## 5. Dataset deposit record

| Field | Value |
|---|---|
| Platform | figshare |
| DOI | `10.6084/m9.figshare.33970177` — resolved at <https://doi.org/10.6084/m9.figshare.33970177> |
| Record | <https://figshare.com/articles/dataset/33970177> |
| Published | 2026-09-23 |
| Title | AIM Auditability and Rule-Level RL Fusion: Full Training Traces and Checkpoints |
| Files | `AuditableRL_runs_full.tar.gz` (1069033552 bytes) and `AuditableRL_runs_full.tar.gz.sha256` (95 bytes) |
| Contents | complete `runs/` tree, 584 files, 4.7 GB uncompressed |
| SHA-256 | `b45124669f4f9467506e36ba9b1490bcc23bade54223ae809bb15884029b4786` |
| License | CC BY 4.0 |
| Unpack | `tar -xzf AuditableRL_runs_full.tar.gz` at the repository root restores `runs/` |

Both files were confirmed present on the published record at the byte sizes listed above.

Verify a download against the published checksum before extraction:

```bash
sha256sum -c AuditableRL_runs_full.tar.gz.sha256
```

The record is versioned (`...33970177.v1`). The unversioned DOI above is the one cited throughout this package, so the reference survives any future corrected deposit.

### Platform history

The dataset was first staged for Zenodo. Browser and command-line uploads both stalled: the TCP send queue to CERN (`137.138.52.235:443`) remained near 500 KB with roughly 3 KB draining per minute, indicating a routing problem between the depositing network and Zenodo rather than a fault in the archive or the depositing host. No files were ever attached to the Zenodo draft, and it was abandoned without publishing. The archive was deposited to figshare instead, where transfer completed normally. The archive itself is unchanged from the one verified in §3.

## 6. Claim boundary

This package is an existence proof that trajectory replay plus provenance-carrying rule fusion is
implementable with hash and environment-replay evidence in the tested controlled settings. It does
**not** establish general auditability, semantic universality, optimal conflict arbitration, or a
validated GPI comparison. See `README.md` §"Evidence boundaries" and §"Limitations and next work".
