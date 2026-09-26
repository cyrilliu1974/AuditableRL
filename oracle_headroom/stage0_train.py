"""Stage 0: retrain 8 CartPole baseline seeds with the repo's own pipeline.

Uses auditability.grammar_audit_experiment.train_one and
shared_calibration_symbolizer verbatim, so the actors / rule banks /
symbolizer are produced by the exact same code path as the paper.

Why the "grammar_audit_only" arm: train_one only induces a grammar for
non-baseline arms (baseline writes an empty production list). The audit-only
arm trains identically to baseline (verified by the paper's
baseline/audit-only equivalence check) AND writes induced_grammar.json.

Output: <this_dir>/runs/CartPole-v1/{shared_calibration.json,grammar_audit_only/seed-<s>_<stamp>/...}
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SNAPSHOT = Path.home() / "workspace" / "RL" / "RL_snapshot"
sys.path.insert(0, str(SNAPSHOT))

from auditability.grammar_audit_experiment import (  # noqa: E402
    StateSymbolizer,
    canonical_json,
    shared_calibration_symbolizer,
    train_one,
)

SEEDS = [11, 29, 43, 71, 101, 149, 211, 307]
ENV = "CartPole-v1"
EPISODES = 300
CALIBRATION_EPISODES = 24
N_BINS = 4
MIN_SUPPORT = 8
MIN_CONFIDENCE = 0.70

TASK_ROOT = HERE / "runs" / ENV
LOG = HERE / "logs" / "stage0_train.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main() -> None:
    TASK_ROOT.mkdir(parents=True, exist_ok=True)
    log(f"building shared calibration symbolizer for {ENV} "
        f"(seeds={SEEDS}, cal_episodes={CALIBRATION_EPISODES}, bins={N_BINS})")
    t0 = time.perf_counter()
    symbolizer = shared_calibration_symbolizer(
        SEEDS, episodes_per_seed=CALIBRATION_EPISODES, n_bins=N_BINS,
        environment_name=ENV,
    )
    log(f"symbolizer built in {time.perf_counter() - t0:.1f}s")
    calib_record = {
        "method": "pooled empirical quantiles from separate random-policy roll-ins; "
                  "frozen for every seed (repo protocol)",
        "environment": ENV,
        "calibration_seeds": SEEDS,
        "calibration_episodes_per_seed": CALIBRATION_EPISODES,
        "state_bins": symbolizer.to_json(),
        "state_bins_canonical_sha256": hashlib.sha256(
            canonical_json(symbolizer.to_json()).encode("utf-8")).hexdigest(),
    }
    (TASK_ROOT / "shared_calibration.json").write_text(
        json.dumps(calib_record, indent=2) + "\n", encoding="utf-8")

    manifest = {"environment": ENV, "seeds": {}, "config": {
        "episodes": EPISODES, "min_support": MIN_SUPPORT,
        "min_confidence": MIN_CONFIDENCE, "arm": "grammar_audit_only",
        "symbolizer_scope": "pooled_across_all_experiment_seeds"}}
    for seed in SEEDS:
        t0 = time.perf_counter()
        log(f"training seed {seed} (grammar_audit_only, {EPISODES} episodes) ...")
        run_dir = train_one(
            seed, "grammar_audit_only", EPISODES, TASK_ROOT, symbolizer,
            min_support=MIN_SUPPORT, min_confidence=MIN_CONFIDENCE,
            environment_name=ENV,
            symbolizer_scope="pooled_across_all_experiment_seeds",
        )
        elapsed = time.perf_counter() - t0
        meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        n_rules = len(json.loads(
            (run_dir / "induced_grammar.json").read_text(encoding="utf-8")).get("productions", []))
        log(f"seed {seed} done in {elapsed:.1f}s -> {run_dir.name}; "
            f"train_mean={meta['episode_return_mean']:.1f} rules={n_rules}")
        manifest["seeds"][str(seed)] = {
            "run_dir": str(run_dir.relative_to(HERE)),
            "training_mean_return": meta["episode_return_mean"],
            "rule_count": n_rules,
            "actor_sha256": meta["actor_sha256"],
        }
    (TASK_ROOT / "stage0_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    log("stage 0 complete")


if __name__ == "__main__":
    main()
