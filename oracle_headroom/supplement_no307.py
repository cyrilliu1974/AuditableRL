"""SUPPLEMENTARY (post-hoc, not pre-registered): sensitivity of the verdict to the
collapsed seed-307 bank.

Reruns grammar_fusion and oracle_conflict WITHOUT the seed-307 bank
(7 banks, 7-actor ensemble fallback). R_best is unchanged (seed 149).
Computes H/T on the no-307 world and compares with the pre-registered verdict.

This does NOT override the pre-registered STOP verdict; it is reported
alongside as a robustness check because the primary H/T CI marginally
touches the threshold.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SNAPSHOT = Path.home() / "workspace" / "RL" / "RL_snapshot"
sys.path.insert(0, str(SNAPSHOT))
sys.path.insert(0, str(HERE))

from auditability.grammar_audit_experiment import StateSymbolizer  # noqa: E402
from auditability.run_shared_multienv_gpi import (  # noqa: E402
    actor_action,
    grammar_fusion_decision,
    load_actor,
    load_rules,
)
from oracle_q import exact_q  # noqa: E402
from oracle_fusion import load_artifacts, evaluate_policy  # noqa: E402

ENV = "CartPole-v1"
EPISODES = 100
LOG = HERE / "logs" / "supplement_no307.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main() -> None:
    sym, actors, banks, action_dim, best_index, _ = load_artifacts()
    keep = [i for i in range(len(actors)) if i != 7]  # seed 307 is index 7
    actors7 = [actors[i] for i in keep]
    banks7 = [banks[i] for i in keep]
    log(f"excluded seed-307 bank; {len(banks7)} banks remain")
    best_actor = actors[best_index]

    def continuation(obs):
        return actor_action(best_actor, obs)

    q_cache = {}

    def q_cached(state, action):
        key = (np.ascontiguousarray(state, dtype=np.float64).tobytes(), int(action))
        if key not in q_cache:
            q_cache[key] = exact_q(ENV, state, int(action), continuation)
        return q_cache[key]

    def ensemble_action(obs):
        state = torch.as_tensor(obs, dtype=torch.float32)
        with torch.inference_mode():
            return int(torch.stack([a(state) for a in actors7]).mean(0).argmax().item())

    def make_policy(mode):
        def policy(obs):
            condition = sym.encode(obs)
            candidates = [b[condition] for b in banks7 if condition in b]
            if not candidates:
                return ensemble_action(obs), {"covered": False, "conflict": False,
                                              "candidate_count": 0}
            cand_actions = sorted({r["action"] for r in candidates})
            is_conflict = len(cand_actions) > 1
            if mode == "grammar_fusion":
                return grammar_fusion_decision(obs, sym, banks7, actors7)
            if mode == "oracle_conflict":
                if not is_conflict:
                    return grammar_fusion_decision(obs, sym, banks7, actors7)
                qs = {a: q_cached(obs, a) for a in cand_actions}
                best = max(cand_actions, key=lambda a: (qs[a], -a))
                return best, {"covered": True, "conflict": True,
                              "candidate_count": len(candidates),
                              "candidate_actions": cand_actions}
            raise ValueError(mode)
        return policy

    out = {}
    for mode in ["grammar_fusion", "oracle_conflict"]:
        t0 = time.perf_counter()
        res = evaluate_policy(make_policy(mode), EPISODES)
        log(f"no307/{mode}: mean={res['mean_return']:.2f} in {time.perf_counter()-t0:.1f}s")
        out[mode] = res["mean_return"]

    # R_best unchanged from primary run (seed 149, no bank involved)
    primary = json.loads((HERE / "results" / "oracle_headroom_cartpole.json").read_text(encoding="utf-8"))
    r_best = primary["policies"]["best_single"]["mean_return"]
    r_conf, r_oc = out["grammar_fusion"], out["oracle_conflict"]
    T, H = r_best - r_conf, r_oc - r_conf
    log(f"no307: R_conf={r_conf:.2f} R_oc={r_oc:.2f} R_best={r_best:.2f}")
    log(f"no307: H/T = {H/T:.3f} (primary: 0.306, threshold 0.333)")
    (HERE / "results" / "supplement_no307.json").write_text(json.dumps({
        "note": "post-hoc supplementary; does not override pre-registered verdict",
        "excluded": "seed-307 bank (index 7)",
        "R_conf_no307": r_conf, "R_oc_no307": r_oc, "R_best": r_best,
        "H_over_T_no307": H / T if T > 1e-9 else None,
    }, indent=2) + "\n", encoding="utf-8")
    log("done")


if __name__ == "__main__":
    main()
