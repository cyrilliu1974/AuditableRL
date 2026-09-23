"""Analyze token-level support and fixed-context interventions for a trained run."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.distributions import Categorical


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from vqvae_agents_AIM import AgentA, AgentB, VQVAE


def sequence_key(message):
    return tuple(int(value) for value in message)


def main(run_dir: Path, repetitions_override: int | None = None):
    checkpoint = torch.load(run_dir / "policy_checkpoint.pt", map_location="cpu", weights_only=True)
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    config = checkpoint["config"]
    K, D, seq_len = config["K"], config["D"], config["seq_len"]
    repetitions = repetitions_override or metadata["config"]["eval_repetitions_per_label"]

    vqvae = VQVAE(K=K, D=D)
    vqvae.load_state_dict(checkpoint["vqvae_state_dict"])
    vqvae.eval()
    agent_a = AgentA(vqvae, aim_seq_len=seq_len, K=K)
    agent_b = AgentB(vqvae, aim_seq_len=seq_len, K=K)
    agent_a.load_state_dict(checkpoint["agent_a_state_dict"])
    agent_b.load_state_dict(checkpoint["agent_b_state_dict"])
    agent_a.eval()
    agent_b.eval()

    torch.manual_seed(int(metadata["seed"]) + 100003)
    labels = torch.arange(10).repeat_interleave(repetitions)
    target_parity = labels.remainder(2).tolist()
    n = labels.numel()
    images = torch.zeros((n, 1, 28, 28))
    neutral_labels = torch.zeros(n, dtype=torch.long)
    empty = torch.zeros((n, seq_len), dtype=torch.long)
    with torch.inference_mode():
        logits_a, _ = agent_a(images, labels, mode="policy", opponent_aim_sequence=empty)
        messages = Categorical(logits=logits_a).sample()
        logits_b = agent_b(messages, neutral_labels, images, mode="policy")
        p_c_observed = logits_b[:, 0, :].softmax(-1)[:, : K // 2].sum(-1)

    message_rows = messages.tolist()
    support = {0: Counter(), 1: Counter()}
    token_counts = {
        0: [Counter() for _ in range(seq_len)],
        1: [Counter() for _ in range(seq_len)],
    }
    p_c_by_parity = {0: [], 1: []}
    for message, parity, probability in zip(message_rows, target_parity, p_c_observed.tolist()):
        key = sequence_key(message)
        support[parity][key] += 1
        p_c_by_parity[parity].append(probability)
        for position, symbol in enumerate(message):
            token_counts[parity][position][int(symbol)] += 1

    modes = {
        parity: max(counts, key=counts.get)
        for parity, counts in support.items()
    }
    candidates = {
        "even_mode": modes[0],
        "odd_mode": modes[1],
    }
    for position in range(seq_len):
        e_to_o = list(modes[0])
        e_to_o[position] = modes[1][position]
        candidates[f"even_mode_replace_position_{position}_with_odd_token"] = tuple(e_to_o)
        o_to_e = list(modes[1])
        o_to_e[position] = modes[0][position]
        candidates[f"odd_mode_replace_position_{position}_with_even_token"] = tuple(o_to_e)

    shared_first_pair = None
    if seq_len == 2:
        sender_probabilities = logits_a.softmax(dim=-1)
        parity_tensor = torch.tensor(target_parity, dtype=torch.long)
        even_mask = parity_tensor == 0
        odd_mask = ~even_mask
        even_joint = torch.zeros((K, K))
        odd_joint = torch.zeros((K, K))
        for first_symbol in range(K):
            sequence_probs = (
                sender_probabilities[:, 0, first_symbol].unsqueeze(1)
                * sender_probabilities[:, 1, :]
            )
            even_joint[first_symbol] = sequence_probs[even_mask].mean(dim=0)
            odd_joint[first_symbol] = sequence_probs[odd_mask].mean(dim=0)

        best_common_pair = None
        for first_symbol in range(K):
            even_second = int(even_joint[first_symbol].argmax())
            odd_second = int(odd_joint[first_symbol].argmax())
            even_probability = float(even_joint[first_symbol, even_second])
            odd_probability = float(odd_joint[first_symbol, odd_second])
            score = min(even_probability, odd_probability)
            if best_common_pair is None or score > best_common_pair["min_policy_probability"]:
                best_common_pair = {
                    "first_symbol": first_symbol,
                    "even_message": (first_symbol, even_second),
                    "odd_message": (first_symbol, odd_second),
                    "even_policy_probability": even_probability,
                    "odd_policy_probability": odd_probability,
                    "min_policy_probability": score,
                }
        shared_first_pair = best_common_pair
        candidates["common_support_even_message"] = best_common_pair["even_message"]
        candidates["common_support_odd_message"] = best_common_pair["odd_message"]

    unique_candidates = sorted(set(candidates.values()))
    sender_log_probs = logits_a.log_softmax(dim=-1)
    sender_probability_by_candidate = {}
    for candidate in unique_candidates:
        candidate_tensor = torch.tensor(candidate, dtype=torch.long).view(1, seq_len, 1)
        candidate_tensor = candidate_tensor.expand(n, -1, -1)
        log_probability = sender_log_probs.gather(2, candidate_tensor).squeeze(-1).sum(dim=1)
        probability = log_probability.exp()
        parity_probabilities = {}
        for parity, name in ((0, "even"), (1, "odd")):
            parity_mask = torch.tensor([value == parity for value in target_parity])
            parity_probabilities[name] = float(probability[parity_mask].mean())
        sender_probability_by_candidate[candidate] = parity_probabilities

    with torch.inference_mode():
        candidate_tensor = torch.tensor(unique_candidates, dtype=torch.long)
        candidate_images = torch.zeros((len(unique_candidates), 1, 28, 28))
        candidate_labels = torch.zeros(len(unique_candidates), dtype=torch.long)
        candidate_logits = agent_b(candidate_tensor, candidate_labels, candidate_images, mode="policy")
        candidate_probabilities = (
            candidate_logits[:, 0, :].softmax(-1)[:, : K // 2].sum(-1).tolist()
        )
    p_c_by_candidate = dict(zip(unique_candidates, candidate_probabilities))

    intervention_results = {}
    for name, key in candidates.items():
        intervention_results[name] = {
            "message": list(key),
            "P_B_C_at_fixed_neutral_observation": p_c_by_candidate[key],
            "observed_count_under_even_sender_policy": support[0][key],
            "observed_count_under_odd_sender_policy": support[1][key],
            "mean_sender_probability_under_target_parity": sender_probability_by_candidate[key],
        }

    token_tv = {}
    for position in range(seq_len):
        even_total = sum(token_counts[0][position].values())
        odd_total = sum(token_counts[1][position].values())
        tv = 0.5 * sum(
            abs(
                token_counts[0][position][symbol] / even_total
                - token_counts[1][position][symbol] / odd_total
            )
            for symbol in range(K)
        )
        token_tv[str(position)] = {
            "total_variation_between_even_and_odd_token_marginals": tv,
            "even_token_support": sorted(token_counts[0][position]),
            "odd_token_support": sorted(token_counts[1][position]),
            "shared_token_support": sorted(
                set(token_counts[0][position]) & set(token_counts[1][position])
            ),
        }

    result = {
        "run_dir": str(run_dir),
        "reproduced_eval_seed": int(metadata["seed"]) + 100003,
        "eval_examples": n,
        "sender_message_mode_by_target_parity": {
            "even": {"message": list(modes[0]), "count": support[0][modes[0]]},
            "odd": {"message": list(modes[1]), "count": support[1][modes[1]]},
        },
        "mean_P_B_C_under_sampled_sender_messages": {
            "even": sum(p_c_by_parity[0]) / len(p_c_by_parity[0]),
            "odd": sum(p_c_by_parity[1]) / len(p_c_by_parity[1]),
        },
        "token_position_marginal_support": token_tv,
        "fixed_context_mode_message_interventions": intervention_results,
        "shared_first_token_common_support_pair": (
            {
                **shared_first_pair,
                "messages_differ_only_at_position_1": (
                    shared_first_pair["even_message"][1]
                    != shared_first_pair["odd_message"][1]
                ),
                "even_message_receiver_probability_and_support": intervention_results[
                    "common_support_even_message"
                ],
                "odd_message_receiver_probability_and_support": intervention_results[
                    "common_support_odd_message"
                ],
            }
            if shared_first_pair is not None
            else None
        ),
        "interpretation_limit": (
            "A one-token hybrid is a supported intervention only if its complete sequence was observed "
            "under that target policy; otherwise label it out-of-support sensitivity."
        ),
    }
    output_path = run_dir / "position_intervention_analysis.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--repetitions", type=int, default=None)
    parsed = parser.parse_args()
    main(parsed.run_dir, parsed.repetitions)
