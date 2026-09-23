# AIM Auditability and Rule-Level RL Fusion

This repository contains the code, compact review results, and research notes for experiments on replayable RL traces, automatically induced symbolic rules, cross-seed rule overlap, behavioral agreement, and rule-level policy fusion. The project tests bounded claims. It does **not** establish that an AI-native symbol system makes arbitrary RL training fully auditable or that a symbolic rule is a causal explanation of every neural-policy decision.

## Current research status

The current study uses eight REINFORCE agents (seeds 11, 29, 43, 71, 101, 149, 211, and 307) on CartPole-v1 and Acrobot-v1. Agents within each environment share one frozen state symbolizer fitted to separate random-policy calibration rollouts. Training uses 300 episodes per seed and arm; the main composition comparison evaluates 100 common reset seeds per policy.

| Result | CartPole-v1 | Acrobot-v1 |
|---|---:|---:|
| Seed 11 vs 29 exact state-action grammar Jaccard | 0.5833 | 0.0496 |
| Same-condition action conflicts, seed 11 vs 29 | 0 / 77 | 77 / 98 |
| Eight-seed mean pairwise grammar Jaccard | 0.5572 | 0.0956 |
| Best single actor mean held-out return | 163.65 | -500.00 |
| Actor-logit ensemble mean held-out return | 500.00 | -500.00 |
| Grammar-fusion mean held-out return | 205.96 | -443.37 |
| Grammar-fusion paired delta vs best single (95% bootstrap CI) | +42.31 [23.98, 61.93] | +56.63 [39.32, 75.15] |

CartPole reaches a 500-point ceiling for some policies, so its score scale is saturated. Acrobot’s grammar fusion produced 39/100 episodes with return above -500 (early goal termination), while the median remained -500. Mean, median, and episode distribution should be read together.

The Acrobot fusion evaluation was 95.70% rule-covered, 89.24% conflicting, and 4.30% blind spots. On the same 100 reset seeds, current fusion scored -443.37, confidence-only arbitration -444.49, mean-reward-only and support-first variants -500, and deferring conflicts to the actor ensemble -500. Randomly choosing an available rule at each state conflict scored -187.02 on one RNG stream; five independent selector streams scored between -206.17 and -194.10 (mean -198.82). A separate five-stream uniform-random-action control averaged -498.71 on those environment resets. The random-rule effect is therefore not explained by unconditioned random actions alone, although its mechanism and transfer to other tasks remain untested. This is a strong, unexpected result for this task and implementation, not evidence that random arbitration is generally better. It means the current confidence-ranked arbiter is not established as a good conflict resolver.

Experiment B directly compared the seed 11 and 29 CartPole actors on 10,000 **fresh raw held-out states**, sampled equally from each actor’s separate held-out state occupancy. Their deterministic argmax actions agreed on 51.38% of states (episode-cluster balanced 51.36%; 95% bootstrap CI 50.48%–52.26%), despite exact grammar Jaccard 0.5833. Symbolic-set overlap therefore cannot stand in for held-out behavioral consensus.

FQE-GPI remains a diagnostic-only comparison. On CartPole, FQE-GPI achieved 9.34 mean return and agreed with the actor-logit ensemble on 35.27% of held-out states. On Acrobot, it achieved -499.57 and selected action 0 on every state in a 50,000-state diagnostic sample. The critic checks do not support treating those GPI returns as a valid policy-composition conclusion.

## Evidence boundaries

The experiments distinguish several questions that should not be collapsed into one “auditability” score:

1. **Trace integrity and replay:** Are observations, actions, rewards, episode boundaries, and updates recorded with verifiable provenance, and can an independent run reproduce environment transitions?
2. **Lossless trace coding:** Can an induced sequence grammar reconstruct the observed trace exactly? This is a compression/reconstruction question, not policy explanation.
3. **Rule coverage and agreement:** How often does an induced state-action rule apply on held-out states, and does it predict a source actor’s action?
4. **Behavioral agreement:** Do two frozen actors choose the same action on the same fresh raw states? Experiment B found approximately 51% agreement for CartPole seeds 11 and 29.
5. **Composition quality:** What return does a declared rule-fusion policy achieve, how are conflicts resolved, and what happens in uncovered states?
6. **Value-model reliability:** Are fitted Q functions predictive enough for GPI action selection? Current smoke diagnostics say not yet.

Exact trace replay supports a bounded claim about the recorded execution on the tested software/hardware setup. It does not prove the logger was honest, establish cross-hardware equivalence, identify a causal neural-network rationale, or generalize to arbitrary RL environments. Jaccard overlap is a syntax statistic. The held-out action agreement experiment supplies a separate behavioral measurement and, in this run, shows that the syntax statistic overstates policy consensus.

## Experiment design

### Shared symbolizer and rule extraction

- Environments: `CartPole-v1` (4-dimensional observation, 2 actions) and `Acrobot-v1` (6-dimensional observation, 3 actions).
- Seeds: 11, 29, 43, 71, 101, 149, 211, 307.
- Calibration: 24 random-policy episodes per seed pooled within the environment; empirical quantile boundaries use four bins per state dimension and are frozen across all training seeds and arms.
- Arms: baseline, audit-only observer, and grammar-constrained policy. Baseline and audit-only are checked for exact training equivalence.
- Rule admission: minimum support 8 and confidence 0.70.
- Scored training: 300 REINFORCE episodes per arm and seed.
- Primary policy evaluation: 100 identical reset seeds per policy; the fallback best-single actor is selected from training-return metadata only.

The state-action rule bank is distinct from the lossless trace grammar. A rule such as `state-symbol -> action-symbol` is not itself a replay of an entire training process.

### Audit records and replay

The experiment stores per-step trajectory rows, optimizer-update ledgers, actor and training checkpoints, shared calibration metadata, and constraint-rule provenance snapshots. Hash-chain verification detects modification of logged records. Environment replay re-executes the recorded reset seeds and actions and compares state, next state, reward, termination, and truncation fields. Rule snapshot hashes link a constrained decision to the rule statistics and source-step IDs used at that update.

Post-run audit summaries report exact baseline/audit-only equivalence, chain and update-ledger validity, replay status, and source-rule reference validation. The 8-seed audit records cover 1,725,655 trace rows for CartPole and 2,544,513 for Acrobot across their three arms; the independent environment replay checks cover 1,122,403 and 1,742,570 transitions, respectively.

### Experiment B: behavioral agreement

The reference script `behavioral_agreement.py` describes the action-agreement question. The executed implementation is `auditability/run_behavioral_agreement.py`, which loads the project’s actual 64-hidden-unit Actor and the seed 11/29 actor checkpoints. It uses fresh Gymnasium resets, samples 5,000 states from each actor’s state occupancy without replacement, applies both actors to those same 10,000 raw observations, and reports pooled and per-occupancy agreement, episode-cluster bootstrap intervals, action histograms, pole-angle strata, checkpoint hashes, and a raw-state-array hash. It does not use training trace states as held-out observations.

### Acrobot arbitration and GPI diagnostics

`auditability/diagnose_acrobot_fusion.py` evaluates matched-seed policy-level variants: current confidence/support/reward ranking, confidence-only, mean-reward-only, random candidate selection, support-first, ensemble fallback on conflicts, source-seed preference, and best-single blind-spot fallback. Random arbitration is repeated with five independent selector RNG streams over the same 100 environment reset seeds. `auditability/diagnose_acrobot_random_action.py` adds five matched uniform-random-action controls; this helps separate state-conditioned rule selection from unconditioned action noise.

`auditability/diagnose_fqe_gpi.py` checks action agreement between FQE-GPI and the actor-logit ensemble, source-seed Q-function distributions on the same held-out states, value direction across high- and low-return held-out episodes, and CartPole balanced/perturbed anchor states. `auditability/diagnose_acrobot_q_action_agreement.py` compares grammar-fusion actions with seed-specific FQE argmax actions, multi-critic GPI, and counterfactual one-step Monte Carlo actions followed by a fixed deterministic actor ensemble. That MC value is for the stated continuation policy; it is not optimal Q-star.

Current FQE failures are material. CartPole source-critic value direction differs by seed and overall FQE-GPI actions match the actor-logit ensemble on only 35.27% of 31,563 held-out states. On Acrobot, two critics’ scalar Q outputs correlate around 0.65 on shared held-out states while their greedy actions disagree sharply; GPI selects action 0 throughout a 50,000-state diagnostic set. In a 200-state sampled action comparison, grammar fusion matched the highest actor-ensemble-continuation Monte Carlo action on 22.5% of states (95% interval 16.5%–28.5%). The sampled return margin does not estimate the optimal policy value and should not be read as a complete causal analysis of fusion.

## Repository layout

Entries marked `[Git]` ship in this repository; `[archive]` ship in the Zenodo dataset
described under "Dataset availability".

```text
C:\AI\RL\
|-- README.md                         # This experiment and publication guide          [Git]
|-- PUBLICATION_MANIFEST.md           # What is published, what is excluded, and why   [Git]
|-- .gitignore                        # Excludes local environments, reference clones,
|                                     #   and the archived run tree                    [Git]
|-- auditability\                     # Experiment runners, validators, diagnostics    [Git]
|-- requirements.txt                  # Python dependencies for reproduction          [Git]
|-- ai_grammar_induction.py            # Legacy wrapper; routes to the validated path  [Git]
|-- vqvae_agents_AIM.py                # Local AIM source snapshot                      [Git]
|-- aim_dictionary_json.py             # Local AIM dictionary source snapshot          [Git]
|-- behavioral_agreement.py            # Reference implementation for Experiment B     [Git]
|-- manuscript\                       # English paper draft and revisions             [Git]
|-- paper\                             # Research logs, reports, LaTeX source, figures [Git]
|-- output\                            # Compiled paper PDF and QA page renders        [Git]
|-- review_bundle\, review_bundle_v2\ # Curated snapshots for expert review           [Git]
|-- results\                           # Compact review package, suitable for GitHub   [Git]
|-- references\                       # Local research-code clones; not vendored into
|                                     #   Git (third-party licensing)                  [none]
`-- runs\                              # Full traces, checkpoints, and run outputs     [archive]
```

To work from the archived tree, unpack the Zenodo archive into the repository root; it restores
`runs/` exactly as the reproduction commands expect.

The prior nested `aim_auditability/` project content has been moved to the repository root. That directory remains only as a local Python environment/cache location and is excluded from Git. Historical log entries may contain the previous `aim_auditability/...` paths; the canonical current paths are shown above.

The full `runs/` tree is retained for trace replay and provenance inspection, but is not committed to Git: it is multi-gigabyte and contains individual files above GitHub's per-file size limit. It is published as the separate archived dataset described under "Dataset availability". A curated 16-file, approximately 0.51 MB review package under `results/` contains the multi-seed summaries, overlap tables, audit reports, FQE diagnostics, arbitration ablations, the uniform random-action control, Experiment B, and the action-level MC comparison. `results/RESULTS_MANIFEST.json` records each package file’s source path, byte count, and SHA-256. The full run tree remains the authoritative source for that package.

Research logs and report files stay in `paper/`; scripts and generated artifacts belong in the repository’s root-level code and results folders. No files outside `C:\AI\RL` are part of this workspace change.

## Reproducing the primary runs

Use Python 3.11 and install the pinned project dependencies in a local environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

From `C:\AI\RL`, reproduce both main environments with:

```powershell
.\.venv\Scripts\python.exe -m auditability.run_shared_multienv_gpi `
  --output-root .\runs\shared_multienv_gpi `
  --environments CartPole-v1 Acrobot-v1 `
  --seeds 11 29 43 71 101 149 211 307 `
  --episodes 300 `
  --calibration-episodes 24 `
  --evaluation-episodes 100 `
  --state-bins 4 `
  --min-support 8 `
  --min-confidence 0.70
```

The source checkpoints and runs already in `runs/shared_multienv_gpi/` were produced with these settings. Run the primary diagnostics after the main batch:

```powershell
.\.venv\Scripts\python.exe -m auditability.fit_gpi_fqe_baseline `
  --task-root .\runs\shared_multienv_gpi\CartPole-v1 --updates 1200 --evaluation-episodes 100
.\.venv\Scripts\python.exe -m auditability.fit_gpi_fqe_baseline `
  --task-root .\runs\shared_multienv_gpi\Acrobot-v1 --updates 1200 --evaluation-episodes 100

.\.venv\Scripts\python.exe -m auditability.diagnose_acrobot_fusion `
  --task-root .\runs\shared_multienv_gpi\Acrobot-v1 --episodes 100

.\.venv\Scripts\python.exe -m auditability.diagnose_acrobot_random_action `
  --task-root .\runs\shared_multienv_gpi\Acrobot-v1 --episodes 100 --replicates 5

.\.venv\Scripts\python.exe -m auditability.diagnose_fqe_gpi `
  --task-root .\runs\shared_multienv_gpi\CartPole-v1 --episodes-per-actor 50
.\.venv\Scripts\python.exe -m auditability.diagnose_fqe_gpi `
  --task-root .\runs\shared_multienv_gpi\Acrobot-v1 --episodes-per-actor 50

.\.venv\Scripts\python.exe -m auditability.diagnose_acrobot_q_action_agreement `
  --task-root .\runs\shared_multienv_gpi\Acrobot-v1 --actor-episodes 40 --sample-count 200

.\.venv\Scripts\python.exe -m auditability.run_behavioral_agreement `
  --task-root .\runs\shared_multienv_gpi\CartPole-v1 `
  --episodes-per-actor 100 --states-per-source 5000 --reset-base 40000000
```

Run the integrity and replay audit on existing task outputs with:

```powershell
.\.venv\Scripts\python.exe -m auditability.audit_shared_multienv_artifacts `
  .\runs\shared_multienv_gpi\CartPole-v1 .\runs\shared_multienv_gpi\Acrobot-v1
```

Reproducing the full training batch is compute-intensive. A smoke run or a partial directory does not reproduce the reported eight-seed result. Record Python, PyTorch, Gymnasium, operating system, device, and code hashes when creating new runs.

## Compact review artifacts

Start from `results/shared_multienv_gpi/` for a fast review. The most relevant files are:

- `CartPole-v1/shared_symbolizer_rule_overlap.json` and `Acrobot-v1/shared_symbolizer_rule_overlap.json`: pairwise exact-rule intersections and Jaccard values.
- `CartPole-v1/behavioral_agreement_experiment_B.json`: fresh-state actor action agreement, occupancy balance, strata, and hashes.
- `Acrobot-v1/grammar_fusion_diagnostics.json`: matched-seed arbiter and random-arbitration ablations.
- `Acrobot-v1/random_action_control.json`: matched uniform-random-action comparison.
- `CartPole-v1/gpi_fqe/` and `Acrobot-v1/gpi_fqe/`: FQE-GPI comparisons and action/value diagnostics.
- `Acrobot-v1/gpi_fqe/acrobot_q_action_agreement.json`: fusion action, source-Q argmax, GPI action, and MC-continuation comparisons on the same sampled states.
- `CartPole-v1/artifact_audit.json` and `Acrobot-v1/artifact_audit.json`: replay, hash-chain, update-ledger, and provenance-reference audit summaries.
- `results/RESULTS_MANIFEST.json`: size and SHA-256 for every compact review copy.

Detailed hypotheses, negative results, implementation corrections, citations, and claim limits are in `paper/AI_Grammar_Induction_Experiment_Log.md`. The concise Traditional Chinese study summary is `paper/20260922_RL_Auditability_Study_Summary.md`; the expanded study report is `paper/20260922_RL_Grammar_Fusion_Research_Report.md`. The evidence-aligned manuscript draft is `manuscript/rl_rule_fusion_draft.md`.

## Dataset availability

This release is split across two tracks because the full training trace tree cannot be hosted in Git: it is roughly 4.7 GB and contains individual files above GitHub's 100 MB per-file limit.

**Track 1 — this repository.** Source code (`auditability/`), the compact hashed result package (`results/`), the paper sources and figures (`paper/`, `manuscript/`, `output/`), the reviewer bundles (`review_bundle/`, `review_bundle_v2/`), and the reproduction guide. Approximately 6.7 MB, 194 files. This is sufficient to inspect every reported number: each headline result appears in a JSON file under `results/`, and `results/RESULTS_MANIFEST.json` gives the source path, byte count, and SHA-256 of every compact review copy.

**Track 2 — archived dataset (DOI to be inserted on acceptance).**

> Full training traces and model checkpoints for the eight-seed, two-environment experiment.
> Zenodo DOI: `10.5281/zenodo.XXXXXXX` — **placeholder, to be replaced with the registered DOI.**

The archive contains the complete `runs/` tree: per-step trajectory traces, optimizer-update ledgers, actor and training checkpoints, shared calibration metadata, induced rule banks, FQE critic checkpoints, and the legacy single-seed, grammar-merge, and private-target runs. Unpacking it into the repository root restores the exact layout the commands above expect.

Verify the downloaded archive against the checksum published alongside it on Zenodo before extraction. The archive is an unmodified copy of the local run tree; the same files back every number in `results/`.

If you only need to inspect results rather than replay traces, Track 1 is self-contained and the archive is not required.

## Research materials and references

The local `references/` directory contains public code repositories inspected during method design. They are research references, not dependencies of the core experiment and are excluded from this Git repository. Their upstream names, licenses, and revisions should be checked before any redistribution. The paper draft and log cite primary research and official documentation separately.

## Limitations and next work

- The direct held-out action agreement for seeds 11 and 29 is near chance, so the shared symbolic core is not a reliable proxy for neural-policy behavioral consensus.
- The Acrobot random-arbitration advantage is unexpected and requires follow-up action-frequency, rule-source, and state-occupancy analysis before publication.
- FQE-GPI is currently unsupported as a credible baseline; Q action rankings, held-out value direction, and actor action agreement fail quality checks.
- The Acrobot task still has many capped -500 episodes, and CartPole saturates at 500 for some agents.
- The current claim is about recorded trace integrity/replay and transparent, provenance-linked rule composition under explicit conditions. General causal explanations and “complete auditability” of arbitrary RL training remain unproven.
- The return-weighted rule-bank intervention and broader behavioral tests across more seeds/tasks remain future work.
