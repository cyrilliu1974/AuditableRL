"""Train and audit an AIM-mediated private-target coordination game."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.distributions import Categorical


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vqvae_agents_AIM import AgentA, AgentB, VQVAE


ZERO_HASH = "0" * 64


def canonical_json(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def chain_record(record: dict, previous_hash: str) -> dict:
    chained = dict(record)
    chained["previous_hash"] = previous_hash
    chained["record_hash"] = hashlib.sha256(
        (previous_hash + canonical_json(chained)).encode("utf-8")
    ).hexdigest()
    return chained


def audit_trace(trace_path: Path, K: int) -> dict:
    previous_hash = ZERO_HASH
    rows = 0
    valid = True
    with trace_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            row = json.loads(line)
            claimed_hash = row.pop("record_hash", None)
            if row.get("previous_hash") != previous_hash:
                valid = False
            calculated_hash = hashlib.sha256(
                (previous_hash + canonical_json(row)).encode("utf-8")
            ).hexdigest()
            if claimed_hash != calculated_hash:
                valid = False
            message_a = row["A_aim"]
            message_b = row["B_aim"]
            target_action = "C" if row["private_label"] % 2 == 0 else "D"
            decoded_action = "C" if message_b[0] < K // 2 else "D"
            expected_reward = 1 if decoded_action == target_action else -1
            if (
                row["target_action"] != target_action
                or row["B_action"] != decoded_action
                or row["reward"] != expected_reward
                or len(message_a) != len(message_b)
            ):
                valid = False
            previous_hash = claimed_hash or ""
            rows = line_number
    return {"valid": valid, "record_count": rows, "final_record_hash": previous_hash}


def action_c_probability(agent_b, messages, neutral_labels, blank_images, K):
    logits = agent_b(messages, neutral_labels, blank_images, mode="policy")
    first_token_probs = logits[:, 0, :].softmax(dim=-1)
    return first_token_probs[:, : K // 2].sum(dim=-1)


def bootstrap_mean_difference_ci(values_even, values_odd, seed, reps=2000):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    n_even = values_even.numel()
    n_odd = values_odd.numel()
    estimates = torch.empty(reps)
    for index in range(reps):
        even_ix = torch.randint(n_even, (n_even,), generator=generator)
        odd_ix = torch.randint(n_odd, (n_odd,), generator=generator)
        estimates[index] = values_even[even_ix].mean() - values_odd[odd_ix].mean()
    lower, upper = torch.quantile(estimates, torch.tensor([0.025, 0.975]))
    return [float(lower), float(upper)]


def empirical_mi_bits(labels_parity, messages):
    def key(message):
        return tuple(message) if isinstance(message, (list, tuple)) else int(message)

    joint = Counter((int(parity), key(message)) for parity, message in zip(labels_parity, messages))
    label_counts = Counter(int(parity) for parity in labels_parity)
    message_counts = Counter(key(message) for message in messages)
    total = len(messages)
    mutual_information = 0.0
    for (parity, message), count in joint.items():
        p_joint = count / total
        p_label = label_counts[parity] / total
        p_message = message_counts[message] / total
        mutual_information += p_joint * math.log2(p_joint / (p_label * p_message))
    return mutual_information


def permutation_mi_test(labels_parity, symbols, observed_mi, seed, permutations=499):
    rng = random.Random(seed)
    null_mi = []
    labels = list(labels_parity)
    symbols = list(symbols)
    for _ in range(permutations):
        shuffled_labels = labels.copy()
        rng.shuffle(shuffled_labels)
        null_mi.append(empirical_mi_bits(shuffled_labels, symbols))
    greater_equal = sum(value >= observed_mi for value in null_mi)
    return {
        "permutations": permutations,
        "p_value_one_sided": (greater_equal + 1) / (permutations + 1),
        "null_mean_bits": sum(null_mi) / permutations,
        "null_95pct_upper_bits": sorted(null_mi)[int(0.95 * (permutations - 1))],
    }


def evaluate(agent_a, agent_b, K, seq_len, repetitions, seed):
    torch.manual_seed(seed)
    agent_a.eval()
    agent_b.eval()
    labels = torch.arange(10).repeat_interleave(repetitions)
    n = labels.numel()
    blank_images = torch.zeros((n, 1, 28, 28))
    neutral_labels = torch.zeros(n, dtype=torch.long)
    empty_opponent = torch.zeros((n, seq_len), dtype=torch.long)
    with torch.inference_mode():
        logits_a, _ = agent_a(
            blank_images, labels, mode="policy", opponent_aim_sequence=empty_opponent
        )
        messages_argmax = logits_a.argmax(dim=-1)
        messages_sampled = Categorical(logits=logits_a).sample()
        logits_b_argmax = agent_b(
            messages_argmax, neutral_labels, blank_images, mode="policy"
        )
        p_c_argmax = logits_b_argmax[:, 0, :].softmax(-1)[:, : K // 2].sum(-1)
        actions_argmax = torch.where(
            logits_b_argmax[:, 0, :].argmax(-1) < K // 2, 0, 1
        )
        target_parity = labels.remainder(2)
        accuracy_argmax = float((actions_argmax == target_parity).float().mean())

        p_c_sampled = action_c_probability(
            agent_b, messages_sampled, neutral_labels, blank_images, K
        )
        even_mask = target_parity == 0
        odd_mask = ~even_mask
        p_c_even = p_c_sampled[even_mask]
        p_c_odd = p_c_sampled[odd_mask]
        causal_delta = float(p_c_even.mean() - p_c_odd.mean())
        expected_accuracy = float(
            torch.cat([p_c_even, 1.0 - p_c_odd]).mean()
        )

        baseline_message = torch.zeros((n, seq_len), dtype=torch.long)
        p_c_no_message = action_c_probability(
            agent_b, baseline_message, neutral_labels, blank_images, K
        )
        no_message_accuracy = float(
            torch.where(p_c_no_message >= 0.5, 0, 1).eq(target_parity).float().mean()
        )
        permutation = torch.randperm(n)
        shuffled_messages = messages_sampled[permutation]
        p_c_shuffled = action_c_probability(
            agent_b, shuffled_messages, neutral_labels, blank_images, K
        )
        shuffled_expected_accuracy = float(
            torch.cat(
                [p_c_shuffled[even_mask], 1.0 - p_c_shuffled[odd_mask]]
            ).mean()
        )

    labels_parity = target_parity.tolist()
    sampled_list = messages_sampled.tolist()
    even_messages = {tuple(m) for m, parity in zip(sampled_list, labels_parity) if parity == 0}
    odd_messages = {tuple(m) for m, parity in zip(sampled_list, labels_parity) if parity == 1}
    p_c_by_message = {}
    all_messages = sorted(even_messages | odd_messages)
    with torch.inference_mode():
        for start in range(0, len(all_messages), 512):
            batch_messages = torch.tensor(all_messages[start : start + 512], dtype=torch.long)
            batch_labels = torch.zeros(batch_messages.shape[0], dtype=torch.long)
            batch_images = torch.zeros((batch_messages.shape[0], 1, 28, 28))
            probs = action_c_probability(agent_b, batch_messages, batch_labels, batch_images, K)
            p_c_by_message.update(
                {tuple(message): float(prob) for message, prob in zip(batch_messages.tolist(), probs)}
            )

    parity_counts = {"even": Counter(), "odd": Counter()}
    for message, parity in zip(sampled_list, labels_parity):
        parity_counts["even" if parity == 0 else "odd"][tuple(message)] += 1
    message_support = set(parity_counts["even"]) | set(parity_counts["odd"])
    overlap = set(parity_counts["even"]) & set(parity_counts["odd"])
    overlap_mass_even = sum(parity_counts["even"][m] for m in overlap) / sum(parity_counts["even"].values())
    overlap_mass_odd = sum(parity_counts["odd"][m] for m in overlap) / sum(parity_counts["odd"].values())
    message_counts_json = {
        parity: {" ".join(map(str, message)): count for message, count in counts.items()}
        for parity, counts in parity_counts.items()
    }

    position_mi = {}
    for position in range(seq_len):
        symbols = [int(message[position]) for message in sampled_list]
        observed_mi = empirical_mi_bits(labels_parity, symbols)
        position_mi[str(position)] = {
            "observed_mutual_information_bits": observed_mi,
            "permutation_test": permutation_mi_test(
                labels_parity, symbols, observed_mi, seed=seed + 1543 + position
            ),
        }

    return {
        "eval_examples": n,
        "policy_mode_for_argmax_accuracy": "argmax sender and receiver",
        "argmax_receiver_action_accuracy": accuracy_argmax,
        "sampled_sender_expected_receiver_accuracy": expected_accuracy,
        "shuffled_message_expected_receiver_accuracy": shuffled_expected_accuracy,
        "no_message_receiver_accuracy": no_message_accuracy,
        "per_position_message_parity_mutual_information": position_mi,
        "message_support": {
            "unique_even": len(parity_counts["even"]),
            "unique_odd": len(parity_counts["odd"]),
            "overlap_unique": len(overlap),
            "even_probability_mass_on_overlap": overlap_mass_even,
            "odd_probability_mass_on_overlap": overlap_mass_odd,
            "sequence_counts_by_parity": message_counts_json,
        },
        "causal_intervention": {
            "receiver_observation": "fixed neutral label 0 and all-zero image",
            "intervention": "do(A_message) using messages sampled from the trained sender policy conditional on target parity",
            "mean_P_B_C_under_even_messages": float(p_c_even.mean()),
            "mean_P_B_C_under_odd_messages": float(p_c_odd.mean()),
            "delta_P_C_even_minus_odd": causal_delta,
            "delta_bootstrap_95pct_ci": bootstrap_mean_difference_ci(
                p_c_even, p_c_odd, seed=seed + 7919
            ),
            "P_C_by_supported_message": {
                " ".join(map(str, message)): p_c_by_message[message]
                for message in all_messages
            },
        },
    }


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)

    source_checkpoint = args.vqvae_checkpoint
    if source_checkpoint is None:
        source_checkpoint = PROJECT_ROOT / "runs" / "seed-0_20260922T013053Z" / "policy_checkpoint.pt"
    source_checkpoint = Path(source_checkpoint)
    if not source_checkpoint.is_absolute():
        source_checkpoint = PROJECT_ROOT / source_checkpoint
    base_state = torch.load(source_checkpoint, map_location="cpu", weights_only=True)
    base_config = base_state["config"]
    K = args.K or base_config["K"]
    D = args.D or base_config["D"]
    seq_len = args.seq_len
    if K != base_config["K"] or D != base_config["D"]:
        raise ValueError("For this first comparison, K and D must match the fixed pretrained VQ-VAE.")

    vqvae = VQVAE(K=K, D=D)
    vqvae.load_state_dict(base_state["vqvae_state_dict"])
    vqvae.eval()
    for parameter in vqvae.parameters():
        parameter.requires_grad_(False)
    agent_a = AgentA(vqvae, aim_seq_len=seq_len, K=K)
    agent_b = AgentB(vqvae, aim_seq_len=seq_len, K=K)
    optimizer_a = torch.optim.Adam(agent_a.parameters(), lr=args.learning_rate)
    optimizer_b = torch.optim.Adam(agent_b.parameters(), lr=args.learning_rate)

    run_name = f"seed-{args.seed}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    trace_path = output_dir / "symbol_training_trace.jsonl"
    neutral_label_cache = None
    baseline = 0.0
    previous_hash = ZERO_HASH
    reward_sum = 0.0
    correct_sum = 0
    recent_batch_accuracy = deque(maxlen=100)
    episode = 0
    optimizer_updates = 0

    with trace_path.open("w", encoding="utf-8", newline="\n") as trace:
        for batch_start in range(0, args.episodes, args.batch_size):
            batch_n = min(args.batch_size, args.episodes - batch_start)
            parity = torch.arange(batch_n).remainder(2)
            class_offset = torch.randint(0, 5, (batch_n,))
            private_labels = (2 * class_offset + parity).long()
            order = torch.randperm(batch_n)
            private_labels = private_labels[order]
            target_parity = private_labels.remainder(2)
            target_actions = torch.where(target_parity == 0, 0, 1)
            blank_images = torch.zeros((batch_n, 1, 28, 28))
            neutral_labels = torch.zeros(batch_n, dtype=torch.long)
            empty_opponent = torch.zeros((batch_n, seq_len), dtype=torch.long)

            logits_a, _ = agent_a(
                blank_images,
                private_labels,
                mode="policy",
                opponent_aim_sequence=empty_opponent,
            )
            dist_a = Categorical(logits=logits_a)
            message_a = dist_a.sample()
            log_prob_a = dist_a.log_prob(message_a).sum(dim=-1)
            entropy_a = dist_a.entropy().sum(dim=-1)

            logits_b = agent_b(message_a, neutral_labels, blank_images, mode="policy")
            dist_b = Categorical(logits=logits_b)
            message_b = dist_b.sample()
            log_prob_b = dist_b.log_prob(message_b).sum(dim=-1)
            entropy_b = dist_b.entropy().sum(dim=-1)

            action_b = torch.where(message_b[:, 0] < K // 2, 0, 1)
            rewards = torch.where(action_b == target_actions, 1.0, -1.0)
            advantage = rewards - baseline
            loss_a = -(advantage.detach() * log_prob_a).mean() - args.entropy_coef * entropy_a.mean()
            loss_b = -(advantage.detach() * log_prob_b).mean() - args.entropy_coef * entropy_b.mean()

            optimizer_a.zero_grad(set_to_none=True)
            loss_a.backward()
            torch.nn.utils.clip_grad_norm_(agent_a.parameters(), 1.0)
            optimizer_a.step()
            optimizer_b.zero_grad(set_to_none=True)
            loss_b.backward()
            torch.nn.utils.clip_grad_norm_(agent_b.parameters(), 1.0)
            optimizer_b.step()
            optimizer_updates += 1

            batch_reward_mean = float(rewards.mean())
            baseline = args.baseline_momentum * baseline + (1.0 - args.baseline_momentum) * batch_reward_mean
            reward_sum += float(rewards.sum())
            correct_sum += int((rewards > 0).sum())
            recent_batch_accuracy.append(float((rewards > 0).float().mean()))

            labels_list = private_labels.tolist()
            messages_a_list = message_a.tolist()
            messages_b_list = message_b.tolist()
            actions_b_list = action_b.tolist()
            rewards_list = rewards.to(torch.int64).tolist()
            for local_index in range(batch_n):
                record = {
                    "episode": episode,
                    "private_label": labels_list[local_index],
                    "target_action": "C" if labels_list[local_index] % 2 == 0 else "D",
                    "A_aim": messages_a_list[local_index],
                    "B_aim": messages_b_list[local_index],
                    "B_action": "C" if actions_b_list[local_index] == 0 else "D",
                    "reward": rewards_list[local_index],
                }
                chained = chain_record(record, previous_hash)
                trace.write(canonical_json(chained) + "\n")
                previous_hash = chained["record_hash"]
                episode += 1

            if (batch_start // args.batch_size + 1) % max(1, args.log_every_batches) == 0:
                print(
                    f"episode={episode}/{args.episodes} "
                    f"rolling_sampled_accuracy={sum(recent_batch_accuracy) / len(recent_batch_accuracy):.4f} "
                    f"mean_reward={reward_sum / episode:.4f}"
                )

    validation = audit_trace(trace_path, K)
    if not validation["valid"] or validation["record_count"] != args.episodes:
        raise RuntimeError(f"Symbol-only trace replay failed: {validation}")
    tamper_probe = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[0])
    tamper_probe["reward"] = -tamper_probe["reward"]
    tamper_probe["record_hash"] = chain_record(
        {key: value for key, value in tamper_probe.items() if key not in {"previous_hash", "record_hash"}},
        tamper_probe["previous_hash"],
    )["record_hash"]
    if tamper_probe["record_hash"] == json.loads(trace_path.read_text(encoding="utf-8").splitlines()[0])["record_hash"]:
        raise RuntimeError("Tamper probe unexpectedly retained the original chain hash")

    eval_result = evaluate(agent_a, agent_b, K, seq_len, args.eval_repetitions, args.seed + 100003)
    eval_result["symbol_only_trace_audit"] = validation
    eval_result["tamper_probe_detected"] = True

    checkpoint_out = output_dir / "policy_checkpoint.pt"
    torch.save(
        {
            "config": {
                "seed": args.seed,
                "episodes": args.episodes,
                "K": K,
                "D": D,
                "seq_len": seq_len,
                "learning_rate": args.learning_rate,
                "entropy_coef": args.entropy_coef,
                "task": "A observes private MNIST-style label; B receives neutral label and blank image; B must act on label parity",
                "vqvae_checkpoint": str(source_checkpoint),
            },
            "vqvae_state_dict": vqvae.state_dict(),
            "agent_a_state_dict": agent_a.state_dict(),
            "agent_b_state_dict": agent_b.state_dict(),
        },
        checkpoint_out,
    )
    (output_dir / "evaluation.json").write_text(
        json.dumps(eval_result, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "episodes": episode,
                "optimizer_updates": optimizer_updates,
                "sampled_accuracy_over_training_episodes": correct_sum / episode if episode else None,
                "recent_rolling_sampled_accuracy": (
                    sum(recent_batch_accuracy) / len(recent_batch_accuracy)
                    if recent_batch_accuracy
                    else None
                ),
                "mean_reward_over_training_episodes": reward_sum / episode if episode else None,
                "checkpoint_file": checkpoint_out.name,
                "trace_file": trace_path.name,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    run_metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "config": {
            "episodes": args.episodes,
            "optimizer_updates": optimizer_updates,
            "batch_size": args.batch_size,
            "K": K,
            "D": D,
            "seq_len": seq_len,
            "learning_rate": args.learning_rate,
            "entropy_coef": args.entropy_coef,
            "baseline_momentum": args.baseline_momentum,
            "eval_repetitions_per_label": args.eval_repetitions,
        },
        "task_contract": {
            "sender_observation": "private label in [0,9] and a blank image",
            "receiver_observation": "constant label 0 and a blank image",
            "target": "C for even private label; D for odd private label",
            "reward": "+1 when receiver action matches target, otherwise -1",
            "receiver_side_channel": "none; image and label are constant across target classes",
        },
        "source_checkpoint": str(source_checkpoint),
        "training_trace": trace_path.name,
        "trace_record_count": validation["record_count"],
        "trace_final_hash": validation["final_record_hash"],
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"run_dir": str(output_dir), "evaluation": eval_result}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--D", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--baseline-momentum", type=float, default=0.9)
    parser.add_argument("--eval-repetitions", type=int, default=128)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--log-every-batches", type=int, default=20)
    parser.add_argument(
        "--vqvae-checkpoint",
        type=Path,
        default=None,
        help="Defaults to the prior MNIST-pretrained AIM checkpoint under runs/.",
    )
    parser.add_argument(
        "--output-root", type=Path, default=PROJECT_ROOT / "runs" / "private_target"
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
