"""Train one isolated AIM audit run and evaluate deterministic policies on MNIST test data."""

import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aim_dictionary_json import AIMDictionary
from vqvae_agents_AIM import (
    AgentA,
    AgentB,
    VQVAE,
    multi_agent_game,
    payoff,
    train_vqvae,
    interpret_aim_as_action,
)


def git_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def source_fingerprints():
    tracked_source = [
        REPO_ROOT / "vqvae_agents_AIM.py",
        Path(__file__).resolve(),
    ]
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        )
        dirty_paths = [line[3:] for line in status.splitlines() if line]
    except (OSError, subprocess.CalledProcessError):
        dirty_paths = None
    hashes = {}
    for source_path in tracked_source:
        try:
            hashes[str(source_path.relative_to(REPO_ROOT))] = hashlib.sha256(
                source_path.read_bytes()
            ).hexdigest()
        except OSError:
            hashes[str(source_path)] = None
    return {"dirty_paths": dirty_paths, "source_sha256": hashes}


def evaluate_argmax(checkpoint, output_dir, batch_size=128):
    """Evaluate a saved policy on every MNIST test example and emit a JSONL trace."""
    args = checkpoint["config"]
    vqvae = VQVAE(K=args["K"], D=args["D"])
    vqvae.load_state_dict(checkpoint["vqvae_state_dict"])
    agent_a = AgentA(vqvae, args["aim_seq_len"], args["K"])
    agent_b = AgentB(vqvae, args["aim_seq_len"], args["K"])
    agent_a.load_state_dict(checkpoint["agent_a_state_dict"])
    agent_b.load_state_dict(checkpoint["agent_b_state_dict"])
    vqvae.eval()
    agent_a.eval()
    agent_b.eval()

    dataset = torchvision.datasets.MNIST(
        root=str(REPO_ROOT / "data"), train=False, download=True,
        transform=transforms.ToTensor(),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    trace_path = output_dir / "argmax_test_trace.jsonl"
    action_counts = {"A": {"C": 0, "D": 0}, "B": {"C": 0, "D": 0}}
    reward_totals = {"A": 0, "B": 0}
    reward_pairs = {}
    record_count = 0

    with trace_path.open("w", encoding="utf-8", newline="\n") as trace_file:
        with torch.inference_mode():
            for batch_start, (images, labels) in enumerate(loader):
                batch_n = int(labels.shape[0])
                zeros = torch.zeros((batch_n, args["aim_seq_len"]), dtype=torch.long)
                a_logits, _ = agent_a(
                    images, labels, mode="policy", opponent_aim_sequence=zeros
                )
                a_symbols = a_logits.argmax(dim=-1)
                b_logits = agent_b(a_symbols, labels, images, mode="policy")
                b_symbols = b_logits.argmax(dim=-1)

                for offset in range(batch_n):
                    sample_index = batch_start * batch_size + offset
                    a_sequence = a_symbols[offset]
                    b_sequence = b_symbols[offset]
                    a_action = interpret_aim_as_action(a_sequence, args["K"])
                    b_action = interpret_aim_as_action(b_sequence, args["K"])
                    label = int(labels[offset].item())
                    reward_a, reward_b = payoff(a_action, b_action, label, sample_index + 1)
                    action_counts["A"][a_action] += 1
                    action_counts["B"][b_action] += 1
                    reward_totals["A"] += reward_a
                    reward_totals["B"] += reward_b
                    pair = f"{a_action}/{b_action}"
                    reward_pairs[pair] = reward_pairs.get(pair, 0) + 1
                    image_sha256 = hashlib.sha256(
                        bytes(dataset.data[sample_index].reshape(-1).tolist())
                    ).hexdigest()
                    row = {
                        "sample_index": sample_index,
                        "image_sha256": image_sha256,
                        "mnist_label": label,
                        "A_aim": a_sequence.tolist(),
                        "B_aim": b_sequence.tolist(),
                        "A_action": a_action,
                        "B_action": b_action,
                        "reward_A": reward_a,
                        "reward_B": reward_b,
                    }
                    trace_file.write(json.dumps(row, separators=(",", ":")) + "\n")
                    record_count += 1

    summary = {
        "evaluation": "deterministic_argmax",
        "dataset": "MNIST test split (train=False)",
        "sample_count": record_count,
        "action_decoder": "first emitted symbol < K // 2 => C, otherwise D",
        "action_counts": action_counts,
        "mean_reward": {
            name: value / record_count if record_count else None
            for name, value in reward_totals.items()
        },
        "action_pair_counts": reward_pairs,
        "trace_file": trace_path.name,
        "interpretation_note": (
            "The C/D decoder is hand-specified. Replay accuracy verifies the logged decoder/reward contract; "
            "it does not establish learned symbol semantics or causal message influence."
        ),
    }
    (output_dir / "argmax_test_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an isolated AIM audit baseline and evaluate on untouched MNIST test data."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=10000)
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--D", type=int, default=64)
    parser.add_argument("--aim_seq_len", type=int, default=2)
    parser.add_argument(
        "--reflection_strategy", choices=["none", "aim_context_value", "predictive_bias"],
        default="predictive_bias",
    )
    parser.add_argument("--reflection_coeff", type=float, default=0.05)
    parser.add_argument("--entropy_coeff", type=float, default=0.01)
    parser.add_argument("--run_root", type=Path, default=Path("runs/auditability"))
    return parser.parse_args()


def main():
    args = parse_args()
    run_root = args.run_root if args.run_root.is_absolute() else REPO_ROOT / args.run_root
    run_root.mkdir(parents=True, exist_ok=True)
    run_name = f"seed-{args.seed}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    output_dir = run_root / run_name
    output_dir.mkdir(exist_ok=False)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = {
        "seed": args.seed,
        "epochs": args.epochs,
        "rounds": args.rounds,
        "K": args.K,
        "D": args.D,
        "aim_seq_len": args.aim_seq_len,
        "reflection_strategy": args.reflection_strategy,
        "reflection_coeff": args.reflection_coeff,
        "entropy_coeff": args.entropy_coeff,
        "rl_training_split": "MNIST train (train=True)",
        "evaluation_split": "MNIST test (train=False)",
        "gamma_rl_note": "The existing core accepts gamma_rl but does not use it in the update.",
    }
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        **source_fingerprints(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "torchvision": torchvision.__version__,
        "config": config,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    aim_dictionary = AIMDictionary(str(output_dir / "aim_dictionary.json"))
    data_root = str(REPO_ROOT / "data")
    vqvae = train_vqvae(args.epochs, args.K, args.D, data_root=data_root)
    rewards_a, rewards_b, agent_a, agent_b = multi_agent_game(
        vqvae,
        aim_dictionary,
        rounds=args.rounds,
        aim_seq_len=args.aim_seq_len,
        K_val=args.K,
        reflection_strategy=args.reflection_strategy,
        reflection_coeff=args.reflection_coeff,
        entropy_coeff=args.entropy_coeff,
        rl_train_on_mnist_train=True,
        return_agents=True,
        data_root=data_root,
    )
    aim_dictionary.save()

    checkpoint_path = output_dir / "policy_checkpoint.pt"
    torch.save(
        {
            "config": config,
            "vqvae_state_dict": vqvae.state_dict(),
            "agent_a_state_dict": agent_a.state_dict(),
            "agent_b_state_dict": agent_b.state_dict(),
        },
        checkpoint_path,
    )
    (output_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "rounds": len(rewards_a),
                "mean_reward_A": sum(rewards_a) / len(rewards_a) if rewards_a else None,
                "mean_reward_B": sum(rewards_b) / len(rewards_b) if rewards_b else None,
                "checkpoint_file": checkpoint_path.name,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = evaluate_argmax(
        torch.load(checkpoint_path, map_location="cpu", weights_only=True), output_dir
    )
    print(json.dumps({"run_dir": str(output_dir), "evaluation": summary}, indent=2))


if __name__ == "__main__":
    main()
