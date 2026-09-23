"""
Experiment B: Behavioral Agreement Rate
原始狀態層級的動作一致性測試

使用方式（從 aim_auditability 目錄執行）：
  python -m auditability.behavioral_agreement \
    --checkpoint-11 runs/repeatability_full/seed11/checkpoint.pt \
    --checkpoint-29 runs/repeatability_full/seed29/checkpoint.pt \
    --trace-11      runs/grammar_audit_pilot/seed11/trajectory.jsonl \
    --trace-29      runs/grammar_audit_pilot/seed29/trajectory.jsonl \
    --output        runs/behavioral_agreement/result.json \
    --n-states      5000
"""

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np


# ── 1. 輕量 Actor（與 GrammarConstrainedPolicy.actor 結構一致）──────────
class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def argmax_action(self, state: torch.Tensor) -> int:
        with torch.no_grad():
            return int(self.net(state).argmax().item())


# ── 2. Checkpoint 載入（相容 GrammarConstrainedPolicy 存檔格式）──────────
def load_actor(path: str, state_dim: int, action_dim: int,
               hidden_dim: int = 128) -> Actor:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)

    actor = Actor(state_dim, action_dim, hidden_dim)

    # 接受三種常見存檔格式
    if isinstance(ckpt, dict):
        if "actor" in ckpt:                        # {"actor": state_dict}
            actor.load_state_dict(ckpt["actor"])
        elif "actor_state_dict" in ckpt:           # {"actor_state_dict": ...}
            actor.load_state_dict(ckpt["actor_state_dict"])
        elif all(k.startswith("net.") or k.startswith("actor.")
                 for k in ckpt):                   # 直接就是 state_dict
            # 去掉可能的 "actor." 前綴
            cleaned = {k.replace("actor.", "", 1): v for k, v in ckpt.items()}
            actor.load_state_dict(cleaned)
        else:
            actor.load_state_dict(ckpt)            # 直接 state_dict
    else:
        raise ValueError(f"無法識別的 checkpoint 格式：{path}")

    actor.eval()
    return actor


# ── 3. 從 JSONL 軌跡讀取原始狀態 ─────────────────────────────────────────
def load_states_from_jsonl(path: str, n: int, rng: random.Random) -> list:
    """
    讀取 JSONL 軌跡，提取 state 欄位。
    相容兩種欄位名：'state' 或 'obs'（CartPole 常見兩種命名）。
    """
    rows = Path(path).read_text(encoding="utf-8").splitlines()
    rows = [r for r in rows if r.strip()]
    rng.shuffle(rows)

    states = []
    for row in rows:
        rec = json.loads(row)
        s = rec.get("state") or rec.get("obs") or rec.get("observation")
        if s is not None:
            states.append(s)
        if len(states) >= n:
            break

    if not states:
        raise ValueError(f"找不到 state/obs/observation 欄位：{path}")
    return states


# ── 4. 核心：計算動作一致率 ───────────────────────────────────────────────
def compute_agreement(actor_11: Actor, actor_29: Actor,
                      states: list) -> dict:
    agree = 0
    details = []          # 記錄衝突的前 20 筆（供 debug）

    for s in states:
        t = torch.tensor(s, dtype=torch.float32)
        a11 = actor_11.argmax_action(t)
        a29 = actor_29.argmax_action(t)
        matched = (a11 == a29)
        agree += int(matched)
        if not matched and len(details) < 20:
            details.append({"state": s, "action_11": a11, "action_29": a29})

    n = len(states)
    rate = agree / n
    return {
        "n_states":         n,
        "n_agree":          agree,
        "n_disagree":       n - agree,
        "agreement_rate":   round(rate, 6),
        "disagreement_rate": round(1 - rate, 6),
        "interpretation":   _interpret(rate),
        "conflict_samples": details,
    }


def _interpret(rate: float) -> str:
    if rate >= 0.90:
        return "STRONG: 行為層級共同核心存在，可作為論文主張的直接證據"
    if rate >= 0.85:
        return "MODERATE-STRONG: 達論文可引用門檻，建議補充狀態分層分析"
    if rate >= 0.70:
        return "MODERATE: 共同核心存在但有噪音，融合仍需仲裁機制"
    return "WEAK: 共同核心比語法重疊暗示的小，需誠實記錄於 Limitations"


# ── 5. 分層分析（按 CartPole 狀態維度分組）────────────────────────────────
def stratified_agreement(actor_11: Actor, actor_29: Actor,
                         states: list) -> dict:
    """
    把狀態依 pole_angle（index 2）分成 3 個區間，
    分別計算一致率——看共識是否集中在某些狀態區域。
    """
    bins = {"near_zero": [], "mid": [], "large": []}

    for s in states:
        angle = abs(s[2])          # CartPole: index 2 = pole angle
        if angle < 0.05:
            bins["near_zero"].append(s)
        elif angle < 0.15:
            bins["mid"].append(s)
        else:
            bins["large"].append(s)

    result = {}
    for region, bucket in bins.items():
        if not bucket:
            result[region] = {"n": 0, "agreement_rate": None}
            continue
        r = compute_agreement(actor_11, actor_29, bucket)
        result[region] = {
            "n": r["n_states"],
            "agreement_rate": r["agreement_rate"],
        }
    return result


# ── 6. 主流程 ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Experiment B: Behavioral Agreement")
    parser.add_argument("--checkpoint-11",  required=True)
    parser.add_argument("--checkpoint-29",  required=True)
    parser.add_argument("--trace-11",       required=True)
    parser.add_argument("--trace-29",       required=True)
    parser.add_argument("--output",         required=True)
    parser.add_argument("--n-states",       type=int, default=5000)
    parser.add_argument("--state-dim",      type=int, default=4)
    parser.add_argument("--action-dim",     type=int, default=2)
    parser.add_argument("--hidden-dim",     type=int, default=128)
    parser.add_argument("--seed",           type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    # 載入兩個 actor
    print(f"載入 seed 11 checkpoint：{args.checkpoint_11}")
    actor_11 = load_actor(args.checkpoint_11,
                          args.state_dim, args.action_dim, args.hidden_dim)

    print(f"載入 seed 29 checkpoint：{args.checkpoint_29}")
    actor_29 = load_actor(args.checkpoint_29,
                          args.state_dim, args.action_dim, args.hidden_dim)

    # 從兩份軌跡合併取樣（各取一半）
    half = args.n_states // 2
    print(f"從軌跡讀取狀態（各 {half} 筆）…")
    states_11 = load_states_from_jsonl(args.trace_11, half, rng)
    states_29 = load_states_from_jsonl(args.trace_29, half, rng)
    all_states = states_11 + states_29
    rng.shuffle(all_states)
    print(f"實際取得狀態數：{len(all_states)}")

    # 整體一致率
    print("計算整體動作一致率…")
    overall = compute_agreement(actor_11, actor_29, all_states)

    # 分層分析
    print("計算分層一致率（依 pole angle）…")
    stratified = stratified_agreement(actor_11, actor_29, all_states)

    # 組合結果
    result = {
        "experiment":        "B_behavioral_agreement",
        "n_states_requested": args.n_states,
        "seed_11_checkpoint": args.checkpoint_11,
        "seed_29_checkpoint": args.checkpoint_29,
        "overall":           overall,
        "stratified_by_pole_angle": stratified,
        "claim_boundary": (
            "argmax policy 下的動作一致性；"
            "不代表隨機取樣策略或訓練過程的一致性"
        ),
    }

    # 寫出結果
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 終端摘要
    print("\n════════════════════════════════")
    print(f"整體一致率：{overall['agreement_rate']:.2%}")
    print(f"一致筆數  ：{overall['n_agree']} / {overall['n_states']}")
    print(f"解讀      ：{overall['interpretation']}")
    print("分層結果：")
    for region, v in stratified.items():
        if v["agreement_rate"] is not None:
            print(f"  {region:12s}：{v['agreement_rate']:.2%}  (n={v['n']})")
    print(f"結果已寫入：{args.output}")
    print("════════════════════════════════")


if __name__ == "__main__":
    main()
