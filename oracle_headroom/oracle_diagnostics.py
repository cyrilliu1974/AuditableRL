"""Oracle headroom diagnostics: decomposition + pre-registered stop rule.

Reads results/oracle_headroom_cartpole.json, computes:

    T = R_best - R_conf          (total gap)
    H = R_oc   - R_conf          (arbitration headroom)
    G_agree = R_of - R_oc        (agreed-action gap)
    G_cover = R_best - R_of      (coverage/fallback gap; residual)

Stop rule (pre-registered in PROTOCOL.md):
    STOP (banks lack usable signal) if H / T < 1/3
    GO   (arbitration is the bottleneck) if H / T >= 1/3

95% bootstrap CIs (5,000 draws) on means and on H/T. The rule is applied to
point estimates; CIs are reported alongside.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = HERE / "logs" / "oracle_diagnostics.log"
DRAW = 5000
STOP_FRACTION = 1.0 / 3.0


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def bootstrap_ci(values: list[float], seed: int = 20260926,
                 draws: int = DRAW) -> list[float]:
    rng = random.Random(seed)
    n = len(values)
    boots = sorted(statistics.mean(values[rng.randrange(n)] for _ in values)
                   for _ in range(draws))
    return [boots[int(0.025 * draws)], boots[int(0.975 * draws) - 1]]


def main() -> None:
    path = HERE / "results" / "oracle_headroom_cartpole.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("oracle_measurement") == "VOID":
        log(f"oracle measurement VOID: {data.get('reason')}")
        return
    pols = data["policies"]

    def series(name: str) -> list[float]:
        return [float(v) for v in pols[name]["return_by_episode"]]

    s_conf = series("grammar_fusion")
    s_oc = series("oracle_conflict")
    s_of = series("oracle_full")
    s_best = series("best_single")

    # Paired bootstrap: resample episode indices jointly so shared reset seeds
    # keep their pairing (variance reduction, same logic as repo's paired test).
    rng = random.Random(20260926)
    n = len(s_conf)
    ratios = []
    for _ in range(DRAW):
        idx = [rng.randrange(n) for _ in range(n)]
        m = lambda s: statistics.mean(s[i] for i in idx)
        T = m(s_best) - m(s_conf)
        H = m(s_oc) - m(s_conf)
        ratios.append(H / T if T > 1e-9 else float("nan"))
    ratios = sorted(r for r in ratios if r == r)
    ratio_ci = [ratios[int(0.025 * len(ratios))], ratios[int(0.975 * len(ratios)) - 1]]

    R = {k: statistics.mean(series(k)) for k in
         ("grammar_fusion", "oracle_conflict", "oracle_full", "best_single",
          "actor_mean_logits")}
    T = R["best_single"] - R["grammar_fusion"]
    H = R["oracle_conflict"] - R["grammar_fusion"]
    G_agree = R["oracle_full"] - R["oracle_conflict"]
    G_cover = R["best_single"] - R["oracle_full"]
    frac = H / T if T > 1e-9 else float("nan")

    verdict = ("STOP" if frac < STOP_FRACTION else "GO")
    report = {
        "means": {k: v for k, v in R.items()},
        "mean_cis": {k: bootstrap_ci(series(k)) for k in R},
        "decomposition": {
            "total_gap_T": T,
            "arbitration_headroom_H": H,
            "agreed_action_gap": G_agree,
            "coverage_fallback_gap": G_cover,
            "H_over_T": frac,
            "H_over_T_ci95": ratio_ci,
        },
        "stop_rule": {
            "threshold": STOP_FRACTION,
            "verdict": verdict,
            "reading": ("banks lack usable signal; do not proceed to Stage 1-4 on these banks"
                        if verdict == "STOP" else
                        "arbitration is the bottleneck; improving the arbiter is worthwhile"),
        },
        "decision_mix": {k: pols[k]["decision_counts"] for k in pols},
        "mean_oracle_q_spread": {k: pols[k]["mean_oracle_q_spread"] for k in pols
                                 if pols[k]["mean_oracle_q_spread"] is not None},
    }
    out = HERE / "results" / "oracle_diagnostics.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    log("==== ORACLE HEADROOM VERDICT ====")
    for k, v in R.items():
        ci = report["mean_cis"][k]
        log(f"  {k:18s} mean={v:8.2f}  95%CI=[{ci[0]:.2f}, {ci[1]:.2f}]")
    log(f"  T={T:.2f}  H={H:.2f}  agreed_gap={G_agree:.2f}  cover_gap={G_cover:.2f}")
    log(f"  H/T = {frac:.3f}  95%CI=[{ratio_ci[0]:.3f}, {ratio_ci[1]:.3f}]  "
        f"threshold={STOP_FRACTION:.3f}")
    log(f"  VERDICT: {verdict} -- {report['stop_rule']['reading']}")
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
