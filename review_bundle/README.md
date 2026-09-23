# AIM Auditability Workspace

## Workspace boundary

- The workspace root is `C:\AI\RL\`.
- Do not create, edit, move, or delete project files outside `C:\AI\RL\`.
- Treat `C:\AI\AIM` as a read-only source/reference location. Keep all implementation changes in this workspace.
- `C:\AI\RL\paper\` is reserved for plain-text research logs and notes. Do not put source code, scripts, environments, datasets, or generated model artifacts there.

## Directory layout

```text
C:\AI\RL\
|-- README.md                         # Workspace and file-placement rules
|-- aim_auditability\                 # Executable research project and local AIM source snapshot
|   |-- vqvae_agents_AIM.py            # Working copy of the inspected AIM implementation
|   |-- aim_dictionary_json.py         # Working copy of AIM dictionary implementation
|   |-- ai_grammar_induction.py        # Grammar-induction prototype entry point
|   |-- requirements.txt               # Isolated project dependencies
|   |-- auditability\                  # Audit runners, grammar experiment, fusion, MDL, and analysis code
|   |-- manuscript\                    # English paper draft and revisions
|   |-- references\                    # Shallow clones of public research code for inspection
|   `-- runs\                          # Datasets, checkpoints, traces, and other run outputs
`-- paper\                            # Plain-text notes/logs only
    |-- AIM_Auditability_Complete_Report.md
    `-- AIM_Auditability_Research_Log.md
```

Generated run outputs, datasets, checkpoints, traces, and plots belong under `aim_auditability\runs\`. Public reference repositories belong under `aim_auditability\references\`. Do not put code or generated artifacts in `paper\`.

## Working rules

1. Read this README before creating or changing files.
2. Use `C:\AI\RL\` as the command working directory. Keep every created or edited path inside it.
3. Read the original AIM source from `C:\AI\AIM` only when needed; copy any source required for experimentation into `aim_auditability\` first, and record its upstream revision and hash.
4. Keep research logs and plain-text paper notes in `paper\`. Distinguish proposed analyses from completed experiments, and record run paths, settings, results, and limitations.
5. After code changes, run the relevant checks and record their results in the research log. Do not claim a hypothesis is supported by a smoke test or a single collapsed run.
6. If unsure where an artifact belongs, place it nowhere until its destination is confirmed against this layout.

## Current experiment status

- The controlled AIM private-target experiment has six 100,000-decision seeds with matched-support token interventions. It supports symbol-mediated decision causes in that designed task, not complete auditability of arbitrary RL training.
- `ai_grammar_induction.py` is a legacy prototype wrapper. Its executable path runs the controlled CartPole comparison in `auditability\grammar_audit_experiment.py`; the older classes are retained for inspection and must not be treated as validated replay evidence.
- The grammar-induction study distinguishes lossless grammar coding of observed traces, held-out state/action-rule coverage, effects of grammar constraints on return, and per-update process provenance. Each run stores a step-level trajectory, a per-episode gradient/weights/optimizer/RNG fingerprint ledger, and a final actor/optimizer/RNG state checkpoint under `aim_auditability\runs\`.
- `auditability\verify_deterministic_repeat.py` performs same-seed cold reruns and compares final weights, full transition traces, update ledgers, and checkpoints, then replays environment transitions. This currently validates exact repeatability for CartPole-v1 on the same CPU/software setup; it does not establish general cross-hardware or arbitrary-environment auditability.
- `auditability\compare_seed_grammars.py` measures exact rule-set overlap and compares both grammars on shared raw states; `auditability\merge_seed_grammars.py` performs provenance-bound rule fusion with explicit conflict arbitration and fallback blind spots, then verifies and replays the merged trace.
- `auditability\analyze_trace_mdl_by_round.py` records a declared two-part MDL score for every trace-grammar round. The score evaluates the lossless pair grammar; it is distinct from state-action policy-rule fidelity.
- Example commands from `C:\AI\RL\aim_auditability\`:
  - `python -m auditability.grammar_audit_experiment --output-root .\runs\grammar_audit_pilot --seeds 11 29 --episodes 300`
  - `python -m auditability.verify_deterministic_repeat --output-root .\runs\repeatability_full --seeds 11 29 --episodes 300`
- See `paper\AI_Grammar_Induction_Experiment_Log.md` for current hypotheses, primary references, experiment settings, failures, results, and claim limits.
- See `paper\20260922_RL_Grammar_Fusion_Research_Report.md` for the completed two-agent fusion study, publication positioning, and evidence boundaries.
- The initial English submission-style draft is `aim_auditability\manuscript\rl_rule_fusion_draft.md`; it is a draft for author review, not a submitted paper.
