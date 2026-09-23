# -*- coding: utf-8 -*-
"""Generate the rule-constrained-arm figure for merged_main.tex Section 5.3.

Unlike gen_figures.py (which plots composition and arbitration results from
results/shared_multienv_gpi), this script reads the per-run metadata of the rule-constrained
experiment under runs/shared_multienv_gpi and plots the paired per-seed change in held-out rule
coverage and in action agreement on covered states.

Every value is read from the recorded metadata.json of the eight seeds; nothing is recomputed or
estimated. Each arm is evaluated on its own state occupancy, so the pairing is by seed, not by state.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(r"C:\AI\RL")
TASK_ROOT = ROOT / "runs" / "shared_multienv_gpi" / "CartPole-v1"
OUT_DIR = ROOT / "paper" / "docs" / "paper_figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_OUT = ROOT / "results" / "constraint_coverage_agreement_by_seed.json"

SEEDS = [11, 29, 43, 71, 101, 149, 211, 307]
ARMS = ("grammar_audit_only", "grammar_constrained")
COVERAGE_COLOR = "#3182bd"
AGREEMENT_COLOR = "#d62728"


def latest_metadata(arm: str, seed: int) -> dict:
    matches = sorted((TASK_ROOT / arm).glob(f"seed-{seed}_*"))
    completed = [path for path in matches if (path / "metadata.json").exists()]
    if not completed:
        raise FileNotFoundError(f"no completed {arm} run for seed {seed}")
    return json.loads((completed[-1] / "metadata.json").read_text(encoding="utf-8"))


def collect() -> list[dict]:
    rows = []
    for seed in SEEDS:
        row = {"seed": seed}
        for arm in ARMS:
            evaluation = latest_metadata(arm, seed)["evaluation"]
            row[f"{arm}_coverage"] = evaluation["grammar_state_coverage"]
            row[f"{arm}_agreement"] = evaluation["grammar_action_agreement_when_covered"]
        rows.append(row)
    return rows


def main() -> int:
    rows = collect()
    payload = {
        "environment": "CartPole-v1",
        "thresholds": {"min_support": 8, "min_confidence": 0.70},
        "episodes_per_evaluation": 20,
        "reset_seed_formula": "seed * 10000 + 900000 + episode_index",
        "measurement_caveat": (
            "Each arm is evaluated on its own state occupancy, so the per-seed pairing is a "
            "same-seed comparison, not a matched-state one."
        ),
        "per_seed": rows,
        "aggregate": {
            "audit_only_coverage_mean": sum(r["grammar_audit_only_coverage"] for r in rows) / len(rows),
            "constrained_coverage_mean": sum(r["grammar_constrained_coverage"] for r in rows) / len(rows),
            "audit_only_agreement_mean": sum(r["grammar_audit_only_agreement"] for r in rows) / len(rows),
            "constrained_agreement_mean": sum(r["grammar_constrained_agreement"] for r in rows) / len(rows),
            "seeds_with_coverage_increase": sum(
                1 for r in rows if r["grammar_constrained_coverage"] > r["grammar_audit_only_coverage"]
            ),
            "seeds_with_agreement_decrease": sum(
                1 for r in rows if r["grammar_constrained_agreement"] < r["grammar_audit_only_agreement"]
            ),
        },
    }
    DATA_OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.4))
    panels = (
        ("coverage", "Held-out rule coverage", COVERAGE_COLOR),
        ("agreement", "Action agreement on covered states", AGREEMENT_COLOR),
    )
    for ax, (key, title, colour) in zip(axes, panels):
        for row in rows:
            low = row[f"grammar_audit_only_{key}"]
            high = row[f"grammar_constrained_{key}"]
            ax.plot([0, 1], [low, high], color=colour, lw=1.2, alpha=0.75, zorder=1)
            ax.plot([0], [low], marker="o", mfc="white", mec=colour, mew=1.2, ms=4.5, zorder=2)
            ax.plot([1], [high], marker="o", color=colour, ms=4.5, zorder=2)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["audit-only", "constrained"], fontsize=8)
        ax.set_xlim(-0.25, 1.25)
        ax.set_ylim(-0.03, 1.03)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", color="0.9", lw=0.6)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("Fraction", fontsize=8)
    fig.suptitle("CartPole-v1 rule-constrained arm, 8 seeds: coverage rises, agreement falls",
                 fontsize=9)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(OUT_DIR / "fig_constraint_coverage.pdf")
    plt.close()

    print("WROTE", OUT_DIR / "fig_constraint_coverage.pdf")
    print("WROTE", DATA_OUT)
    print(json.dumps(payload["aggregate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
