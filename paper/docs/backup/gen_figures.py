# -*- coding: utf-8 -*-
"""Generate the three figures referenced by merged_main.tex from the real
results JSON in C:/AI/RL/results/shared_multienv_gpi. Honest, data-derived plots."""
import json, os, statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = r"C:\AI\RL\results\shared_multienv_gpi"
OUT = r"C:\AI\RL\paper\docs\paper_figs"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- Fig 1: CartPole composition
cp = json.load(open(os.path.join(BASE, "CartPole-v1", "shared_multiseed_gpi_summary.json")))
ce = cp["composition_evaluation"]["policies"]
order = ["best_single", "grammar_fusion", "gpi_mc_q", "actor_mean_logits"]
labels = {"best_single": "Best single actor", "grammar_fusion": "Rule fusion",
          "gpi_mc_q": "Fitted-Q GPI", "actor_mean_logits": "Actor-logit ensemble"}
means = [ce[k]["mean_return"] for k in order]
errs = [ce[k].get("sample_std_return", 0.0) for k in order]
fig, ax = plt.subplots(figsize=(6.4, 3.6))
bars = ax.bar([labels[k] for k in order], means, yerr=errs, capsize=4,
              color=["#9ecae1", "#3182bd", "#fdae6b", "#74c476"], edgecolor="black")
ax.axhline(500.0, ls="--", color="gray", lw=1)
ax.text(3.4, 505, "env. ceiling 500", color="gray", fontsize=8, ha="right")
ax.set_ylabel("Mean held-out return (100 eps)")
ax.set_title("CartPole-v1: composition return by policy")
for b, m in zip(bars, means):
    ax.text(b.get_x() + b.get_width()/2, m + 8, f"{m:.1f}", ha="center", fontsize=8)
ax.set_ylim(-40, 560)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_cartpole_composition.pdf"))
plt.close()

# ---------------------------------------------------------------- Fig 2: Acrobot ablations
ab = json.load(open(os.path.join(BASE, "Acrobot-v1", "grammar_fusion_diagnostics.json")))
abl = ab["ablations"]
name_map = {"confidence_first": "Confidence-ranked (current)", "confidence_only": "Confidence-only",
            "mean_reward_only": "Mean-reward-only", "random_candidate": "Random-candidate",
            "support_first": "Support-first", "defer_conflicts_to_ensemble": "Defer-conflict-to-ensemble",
            "unanimous_only": "Unanimous-only", "source_seed_11": "Source seed 11",
            "source_seed_29": "Source seed 29", "best_single_fallback": "Best-single fallback"}
items = [(name_map.get(k, k), v["mean_return"]) for k, v in abl.items()]
items.sort(key=lambda x: x[1], reverse=True)
names = [n for n, _ in items]; rets = [r for _, r in items]
colors = ["#3182bd" if n == "Confidence-ranked (current)" else
          ("#d62728" if n == "Random-candidate" else "#9ecae1") for n in names]
fig, ax = plt.subplots(figsize=(6.8, 4.0))
ax.barh(names, rets, color=colors, edgecolor="black")
ax.axvline(-500.0, ls="--", color="gray", lw=1)
ax.text(-500.0, len(names)-0.3, "floor -500", color="gray", fontsize=8, ha="right")
ax.set_xlabel("Mean fused return (Acrobot-v1)")
ax.set_title("Acrobot-v1: fused return by arbiter\n(random-candidate beats confidence-ranked)")
for i, r in enumerate(rets):
    ax.text(r + 6, i, f"{r:.1f}", va="center", fontsize=8)
ax.set_xlim(-560, -120)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_acrobot_ablations.pdf"))
plt.close()

# ---------------------------------------------------------------- Fig 3: agreement vs overlap
ov = json.load(open(os.path.join(BASE, "CartPole-v1", "shared_symbolizer_rule_overlap.json")))
pairs = ov["all_pairwise_seed_overlaps"]
jac = [p["state_action_jaccard"] for p in pairs]
ag = json.load(open(os.path.join(BASE, "CartPole-v1", "behavioral_agreement_experiment_B.json")))
res = ag["results"]
agree = res["pooled_state_agreement"]
cba = res.get("episode_cluster_balanced_agreement")
ci = res.get("episode_cluster_balanced_bootstrap_ci95")
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.scatter(jac, [agree] * len(jac), s=28, color="#3182bd", zorder=3,
           label=f"behavioral agreement (measured 11v29) = {agree:.3f}")
# highlight the 11v29 point
for p in pairs:
    if p["left_seed"] == 11 and p["right_seed"] == 29:
        ax.scatter([p["state_action_jaccard"]], [agree], s=70, facecolors="none",
                   edgecolors="black", linewidths=1.6, zorder=4)
if cba is not None and ci is not None:
    ax.axhspan(ci[0], ci[1], color="#3182bd", alpha=0.12, zorder=1)
    ax.axhline(cba, ls=":", color="#3182bd", lw=1)
ax.axhline(0.5, ls="--", color="gray", lw=1)
ax.text(max(jac), 0.5, " chance 0.5", color="gray", fontsize=8, va="bottom", ha="right")
ax.set_xlabel("Rule-set Jaccard overlap (28 seed pairs)")
ax.set_ylabel("Behavioral agreement (fresh states)")
ax.set_title("CartPole-v1: rule overlap vs behavioral agreement\n(overlap varies widely; agreement near chance)")
ax.set_ylim(0.45, 0.57)
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_agreement.pdf"))
plt.close()

print("WROTE figures to", OUT)
print("CartPole policy means:", dict(zip(order, [round(m, 2) for m in means])))
print("Acrobot ablations (sorted):", {n: round(r, 2) for n, r in items})
print("Jaccard min/mean/max:", round(min(jac), 3), round(statistics.mean(jac), 3), round(max(jac), 3))
print("Agreement:", agree, "cluster-balanced:", cba, "CI:", ci)
