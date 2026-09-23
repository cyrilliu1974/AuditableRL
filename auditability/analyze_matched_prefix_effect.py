"""Estimate token-1 effects while matching token-0 support across target classes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from vqvae_agents_AIM import AgentA, AgentB, VQVAE


def analyze(run_dir: Path) -> dict:
    checkpoint = torch.load(run_dir / "policy_checkpoint.pt", map_location="cpu", weights_only=True)
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    config = checkpoint["config"]
    K, D = config["K"], config["D"]
    if config["seq_len"] != 2:
        raise ValueError("matched-prefix analysis currently requires two-token messages")

    vqvae = VQVAE(K=K, D=D)
    vqvae.load_state_dict(checkpoint["vqvae_state_dict"])
    agent_a = AgentA(vqvae, aim_seq_len=2, K=K)
    agent_b = AgentB(vqvae, aim_seq_len=2, K=K)
    agent_a.load_state_dict(checkpoint["agent_a_state_dict"])
    agent_b.load_state_dict(checkpoint["agent_b_state_dict"])
    agent_a.eval()
    agent_b.eval()

    labels = torch.arange(10)
    blank_images = torch.zeros((10, 1, 28, 28))
    empty_messages = torch.zeros((10, 2), dtype=torch.long)
    with torch.inference_mode():
        sender_logits, _ = agent_a(
            blank_images, labels, mode="policy", opponent_aim_sequence=empty_messages
        )
        sender_probabilities = sender_logits.softmax(dim=-1)
        joint_by_parity = {}
        first_by_parity = {}
        for parity in (0, 1):
            class_probabilities = sender_probabilities[labels.remainder(2) == parity]
            # Average each label's product distribution, preserving correlations induced by label.
            joint = torch.einsum("n i,n j->i j", class_probabilities[:, 0, :], class_probabilities[:, 1, :])
            joint = joint / class_probabilities.shape[0]
            joint_by_parity[parity] = joint
            first_by_parity[parity] = joint.sum(dim=1)

        messages = torch.cartesian_prod(torch.arange(K), torch.arange(K))
        neutral_labels = torch.zeros(K * K, dtype=torch.long)
        neutral_images = torch.zeros((K * K, 1, 28, 28))
        receiver_logits = agent_b(messages, neutral_labels, neutral_images, mode="policy")
        receiver_p_c = receiver_logits[:, 0, :].softmax(-1)[:, : K // 2].sum(-1).reshape(K, K)

        position_results = {}
        for target_position in (0, 1):
            matched_position = 1 - target_position
            if matched_position == 0:
                joint = {parity: joint_by_parity[parity] for parity in (0, 1)}
                other_marginal = first_by_parity
                receiver_grid = receiver_p_c
            else:
                joint = {parity: joint_by_parity[parity].T for parity in (0, 1)}
                other_marginal = {parity: first_by_parity[parity] for parity in (0, 1)}
                other_marginal = {
                    parity: joint[parity].sum(dim=1) for parity in (0, 1)
                }
                receiver_grid = receiver_p_c.T

            matched_distribution = torch.minimum(other_marginal[0], other_marginal[1])
            matched_mass = float(matched_distribution.sum())
            if matched_mass <= 0:
                continue
            matched_distribution = matched_distribution / matched_distribution.sum()
            expected_p_c = {}
            for parity in (0, 1):
                conditional_target = joint[parity] / other_marginal[parity].clamp_min(1e-30).unsqueeze(1)
                expected_p_c[parity] = float(
                    (matched_distribution.unsqueeze(1) * conditional_target * receiver_grid).sum()
                )
            position_results[str(target_position)] = {
                "varied_token_position": target_position,
                "matched_other_token_position": matched_position,
                "shared_other_token_probability_mass": matched_mass,
                "shared_other_token_effective_symbols": int((matched_distribution > 0).sum()),
                "P_B_C_under_even_token_given_matched_other": expected_p_c[0],
                "P_B_C_under_odd_token_given_matched_other": expected_p_c[1],
                "delta_P_B_C_even_minus_odd": expected_p_c[0] - expected_p_c[1],
            }

        # Select the position with the strongest measured target information, not a fixed slot.
        evaluation = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
        mi = evaluation["per_position_message_parity_mutual_information"]
        active_position = max(
            mi,
            key=lambda position: mi[position]["observed_mutual_information_bits"],
        )
        active_result = position_results[active_position]
        result = {
            "run_dir": str(run_dir),
            "seed": metadata["seed"],
            "design": "for each message position, compare exact sender-policy conditionals for that token across target parity while matching the other token's distribution by min-mass overlap; receiver context is fixed",
            "active_position_selected_by_highest_observed_mutual_information": int(active_position),
            "active_position_matched_support_causal_contrast": active_result,
            "all_position_matched_support_causal_contrasts": position_results,
            "per_position_parity_mutual_information_bits": mi,
            "interpretation_limit": "This is a causal contrast under matched sender-policy prefix distributions in the designed parity game; it does not establish natural-language explanation or general RL auditability.",
        }
    out = run_dir / "matched_prefix_effect.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run_dir), indent=2))


if __name__ == "__main__":
    main()
