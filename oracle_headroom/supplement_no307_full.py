"""SUPPLEMENTARY-2 (post-hoc): oracle_full WITHOUT the seed-307 bank.

Completes the interpretation ladder in the no-307 world:
  R_conf (335.98) -> R_oc (322.99) -> R_of (?) -> R_best (500)

Same construction as oracle_fusion.py's oracle_full, but with the
seed-307 bank excluded (7 banks, 7-actor ensemble fallback for blind spots).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SNAPSHOT = Path.home() / "workspace" / "RL" / "RL_snapshot"
sys.path.insert(0, str(SNAPSHOT))
sys.path.insert(0, str(HERE))

from auditability.run_shared_multienv_gpi import actor_action  # noqa: E402
from oracle_q import exact_q  # noqa: E402
from oracle_fusion import load_artifacts, evaluate_policy  # noqa: E402

ENV = "CartPole-v1"
EPISODES = 100
LOG = HERE / "logs" / "supplement_no307_full.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main() -> None:
    sym, actors, banks, action_dim, best_index, _ = load_artifacts()
    keep = [i for i in range(len(actors)) if i != 7]
    actors7 = [actors[i] for i in keep]
    banks7 = [banks[i] for i in keep]
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

    def policy(obs):
        condition = sym.encode(obs)
        candidates = [b[condition] for b in banks7 if condition in b]
        if not candidates:
            return ensemble_action(obs), {"covered": False, "conflict": False,
                                          "candidate_count": 0}
        cand_actions = sorted({r["action"] for r in candidates})
        qs = {a: q_cached(obs, a) for a in range(action_dim)}
        best = max(range(action_dim), key=lambda a: (qs[a], -a))
        return best, {"covered": True,
                      "conflict": len(cand_actions) > 1,
                      "candidate_count": len(candidates),
                      "candidate_actions": cand_actions,
                      "q_spread": max(qs.values()) - min(qs.values())}

    t0 = time.perf_counter()
    log("evaluating no307/oracle_full ...")
    res = evaluate_policy(policy, EPISODES)
    log(f"no307/oracle_full: mean={res['mean_return']:.2f} median={res['median_return']:.2f} "
        f"in {time.perf_counter()-t0:.1f}s; counts={res['decision_counts']}")
    (HERE / "results" / "supplement_no307_full.json").write_text(json.dumps({
        "note": "post-hoc supplementary; does not override pre-registered verdict",
        "excluded": "seed-307 bank (index 7)",
        "mean_return": res["mean_return"],
        "median_return": res["median_return"],
        "decision_counts": res["decision_counts"],
    }, indent=2) + "\n", encoding="utf-8")
    log("done")


if __name__ == "__main__":
    main()
