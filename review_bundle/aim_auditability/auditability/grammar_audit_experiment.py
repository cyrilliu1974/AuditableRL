"""Controlled experiment: induced state-action grammars, constraints, and exact trace replay."""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import io
import json
import platform
import random
import re
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


ROOT = Path(__file__).resolve().parents[1]
ZERO_HASH = "0" * 64
LEXER = re.compile(r'"(?:\\.|[^"\\])*"|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null|[{}\[\]:,]|\n')


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def actor_state_sha256(actor: nn.Module) -> str:
    buffer = io.BytesIO()
    torch.save(actor.state_dict(), buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def serialized_sha256(value: object) -> str:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def gradient_sha256(actor: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in actor.named_parameters():
        digest.update(name.encode("utf-8"))
        if parameter.grad is None:
            digest.update(b"<none>")
        else:
            digest.update(parameter.grad.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def rng_state_hashes() -> dict[str, str]:
    numpy_state = np.random.get_state()
    numpy_payload = {
        "generator": numpy_state[0],
        "keys": numpy_state[1].tolist(),
        "position": int(numpy_state[2]),
        "has_gauss": int(numpy_state[3]),
        "cached_gaussian": float(numpy_state[4]),
    }
    python_payload = json.dumps(random.getstate(), separators=(",", ":"))
    return {
        "python_rng_sha256": hashlib.sha256(python_payload.encode("utf-8")).hexdigest(),
        "numpy_rng_sha256": hashlib.sha256(canonical_json(numpy_payload).encode("utf-8")).hexdigest(),
        "torch_rng_sha256": hashlib.sha256(torch.get_rng_state().cpu().numpy().tobytes()).hexdigest(),
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


class StateSymbolizer:
    """Quantile bins fitted on a separate calibration roll-in and then frozen."""

    def __init__(self, n_bins: int = 4):
        self.n_bins = n_bins
        self.edges: torch.Tensor | None = None

    def fit(self, states: list[np.ndarray]) -> None:
        values = torch.as_tensor(np.asarray(states), dtype=torch.float64)
        quantiles = torch.linspace(0, 1, self.n_bins + 1, dtype=torch.float64)[1:-1]
        self.edges = torch.stack([torch.quantile(values[:, d], quantiles) for d in range(values.shape[1])])

    def encode(self, state: np.ndarray | list[float]) -> tuple[int, ...]:
        if self.edges is None:
            raise RuntimeError("state symbolizer must be calibrated before use")
        values = torch.as_tensor(state, dtype=torch.float64)
        return tuple(int(torch.bucketize(values[d], self.edges[d], right=True)) for d in range(values.numel()))

    def to_json(self) -> dict:
        if self.edges is None:
            raise RuntimeError("state symbolizer not fitted")
        return {"method": "per-dimension empirical quantiles", "n_bins": self.n_bins, "edges": self.edges.tolist()}


@dataclass
class RuleStats:
    count: int = 0
    rewards_sum: float = 0.0
    source_steps: list[int] | None = None


class StateActionGrammar:
    """A finite CFG whose productions are induced from observed state/action rules."""

    def __init__(self, min_support: int = 8, min_confidence: float = 0.90):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.counts: dict[tuple[int, ...], collections.Counter[int]] = collections.defaultdict(collections.Counter)
        self.rewards: dict[tuple[tuple[int, ...], int], float] = collections.defaultdict(float)
        self.source_steps: dict[tuple[tuple[int, ...], int], list[int]] = collections.defaultdict(list)
        self.generation = 0

    def observe(self, condition: tuple[int, ...], action: int, reward: float, step_id: int) -> None:
        self.counts[condition][int(action)] += 1
        key = (condition, int(action))
        self.rewards[key] += float(reward)
        self.source_steps[key].append(int(step_id))

    def induce(self) -> dict[tuple[int, ...], dict]:
        self.generation += 1
        return self._eligible_rules()

    def _eligible_rules(self) -> dict[tuple[int, ...], dict]:
        rules: dict[tuple[int, ...], dict] = {}
        for condition, action_counts in self.counts.items():
            total = sum(action_counts.values())
            action, count = min(action_counts.items(), key=lambda item: (-item[1], item[0]))
            confidence = count / total if total else 0.0
            if count >= self.min_support and confidence >= self.min_confidence:
                rules[condition] = {
                    "action": action,
                    "support": count,
                    "condition_count": total,
                    "confidence": confidence,
                    "mean_reward": self.rewards[(condition, action)] / count,
                    "source_step_ids": list(self.source_steps[(condition, action)]),
                }
        return rules

    def formal_grammar(self) -> dict:
        rules = self._eligible_rules()
        productions = []
        for condition, rule in sorted(rules.items()):
            state_symbol = "STATE_" + "_".join(map(str, condition))
            action_symbol = f"ACTION_{rule['action']}"
            productions.append({
                "lhs": "S",
                "rhs": [state_symbol, action_symbol],
                "support": rule["support"],
                "condition_count": rule["condition_count"],
                "confidence": rule["confidence"],
                "mean_reward": rule["mean_reward"],
                "source_step_ids": rule["source_step_ids"],
            })
        return {
            "kind": "induced finite state-action grammar",
            "start_symbol": "S",
            "terminals": sorted({symbol for p in productions for symbol in p["rhs"]}),
            "non_terminals": ["S"],
            "generation": self.generation,
            "min_support": self.min_support,
            "min_confidence": self.min_confidence,
            "productions": productions,
        }


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = 2, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, action_dim))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


def calibration_symbolizer(
    seed: int,
    episodes: int = 24,
    n_bins: int = 4,
    environment_name: str = "CartPole-v1",
) -> StateSymbolizer:
    """Fit state boundaries on random-policy data excluded from scored RL runs."""
    env = gym.make(environment_name)
    env.action_space.seed(seed + 100_000)
    states: list[np.ndarray] = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + 100_000 + episode)
        states.append(np.asarray(observation, dtype=np.float64))
        done = False
        while not done:
            action = int(env.action_space.sample())
            next_observation, _, terminated, truncated, _ = env.step(action)
            states.append(np.asarray(next_observation, dtype=np.float64))
            done = bool(terminated or truncated)
    env.close()
    symbolizer = StateSymbolizer(n_bins=n_bins)
    symbolizer.fit(states)
    return symbolizer


def shared_calibration_symbolizer(
    seeds: list[int], episodes_per_seed: int = 24, n_bins: int = 4,
    environment_name: str = "CartPole-v1",
) -> StateSymbolizer:
    """Fit one frozen quantizer from independent random roll-ins pooled across seeds."""
    states: list[np.ndarray] = []
    for seed in seeds:
        env = gym.make(environment_name)
        env.action_space.seed(seed + 100_000)
        try:
            for episode in range(episodes_per_seed):
                observation, _ = env.reset(seed=seed * 10_000 + 100_000 + episode)
                states.append(np.asarray(observation, dtype=np.float64))
                terminated = truncated = False
                while not (terminated or truncated):
                    action = int(env.action_space.sample())
                    observation, _, terminated, truncated, _ = env.step(action)
                    states.append(np.asarray(observation, dtype=np.float64))
        finally:
            env.close()
    symbolizer = StateSymbolizer(n_bins=n_bins)
    symbolizer.fit(states)
    return symbolizer


def chain_record(record: dict, previous_hash: str) -> dict:
    row = dict(record)
    row["previous_hash"] = previous_hash
    row["record_hash"] = hashlib.sha256((previous_hash + canonical_json(row)).encode("utf-8")).hexdigest()
    return row


def verify_trace(path: Path) -> dict:
    previous = ZERO_HASH
    count = 0
    with path.open("r", encoding="utf-8") as stream:
        for count, line in enumerate(stream, 1):
            row = json.loads(line)
            claimed = row.pop("record_hash")
            if row.get("previous_hash") != previous:
                raise ValueError(f"trace line {count}: previous-hash mismatch")
            actual = hashlib.sha256((previous + canonical_json(row)).encode("utf-8")).hexdigest()
            if actual != claimed:
                raise ValueError(f"trace line {count}: record-hash mismatch")
            previous = claimed
    return {"valid": True, "records": count, "final_hash": previous}


def verify_update_trace(path: Path) -> dict:
    """Verify the episode-level optimizer-update ledger independently of trajectory rows."""
    previous = ZERO_HASH
    count = 0
    with path.open("r", encoding="utf-8") as stream:
        for count, line in enumerate(stream, 1):
            row = json.loads(line)
            claimed = row.pop("record_hash")
            if row.get("previous_hash") != previous:
                raise ValueError(f"update ledger line {count}: previous-hash mismatch")
            actual = hashlib.sha256((previous + canonical_json(row)).encode("utf-8")).hexdigest()
            if actual != claimed:
                raise ValueError(f"update ledger line {count}: record-hash mismatch")
            previous = claimed
    return {"valid": True, "records": count, "final_hash": previous}


def training_update_core_sha256(path: Path) -> str:
    """Hash only optimizer-relevant values; exclude ledger chain and legacy sidecar generation tags."""
    digest = hashlib.sha256()
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            core = {
                key: value for key, value in row.items()
                if key not in {"previous_hash", "record_hash", "grammar_generation"}
            }
            digest.update(canonical_json(core).encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()


def replay_environment_trace(path: Path, environment_name: str = "CartPole-v1") -> dict:
    """Replay logged actions in the recorded Gymnasium task and compare transitions."""
    env = gym.make(environment_name)
    current_episode = None
    expected_step = 0
    transitions = 0
    total_reward = 0.0
    ended = True
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                row = json.loads(line)
                episode = int(row["episode"])
                step = int(row["step"])
                if episode != current_episode:
                    if current_episode is not None and not ended:
                        raise ValueError(f"line {line_number}: new episode began before terminal/truncated")
                    if step != 0 or row["reset_seed"] is None:
                        raise ValueError(f"line {line_number}: episode start missing reset seed")
                    observation, _ = env.reset(seed=int(row["reset_seed"]))
                    current_episode = episode
                    expected_step = 0
                    ended = False
                if step != expected_step:
                    raise ValueError(f"line {line_number}: expected step {expected_step}, found {step}")
                if np.asarray(observation, dtype=np.float32).tolist() != row["state"]:
                    raise ValueError(f"line {line_number}: state does not match replayed environment")
                observation, reward, terminated, truncated, _ = env.step(int(row["action"]))
                next_state = np.asarray(observation, dtype=np.float32).tolist()
                if next_state != row["next_state"]:
                    raise ValueError(f"line {line_number}: next state does not match replay")
                if float(reward) != float(row["reward"]):
                    raise ValueError(f"line {line_number}: reward mismatch")
                if bool(terminated) != bool(row["terminated"]) or bool(truncated) != bool(row["truncated"]):
                    raise ValueError(f"line {line_number}: termination/truncation mismatch")
                transitions += 1
                total_reward += float(reward)
                expected_step += 1
                ended = bool(terminated or truncated)
    finally:
        env.close()
    return {
        "valid": True,
        "transition_count": transitions,
        "episode_count": 0 if current_episode is None else current_episode + 1,
        "reconstructed_reward_sum": total_reward,
        "replay_method": "Gymnasium reset seed + recorded action, exact float32 state/reward/terminated/truncated comparison",
    }


def tokenize_losslessly(text: str) -> list[str]:
    tokens = LEXER.findall(text)
    if "".join(tokens) != text:
        raise ValueError("trace tokenizer failed exact lexical round-trip")
    return tokens


def gamma_code_bits(value: int) -> int:
    """Bit length of Elias gamma coding for a positive integer."""
    if value < 1:
        raise ValueError("Elias gamma codes positive integers only")
    return 2 * (value.bit_length() - 1) + 1


def measure_pair_grammar_mdl(encoded: dict) -> dict:
    """Compute a declared two-part code length for PairGrammarCodec output.

    The terminal dictionary stores each UTF-8 token once with a gamma-coded byte
    length. Rule RHS references and the start sequence use fixed-width symbol IDs.
    Fixed schema/software metadata is excluded; provenance cost is reported apart.
    """
    vocabulary = encoded["terminal_vocabulary"]
    productions = encoded["production_rhs_terminal_or_nonterminal_ids"]
    start_sequence = encoded["start_sequence_ids"]
    vocabulary_bits = gamma_code_bits(len(vocabulary) + 1)
    for token in vocabulary:
        size = len(token.encode("utf-8"))
        vocabulary_bits += gamma_code_bits(size + 1) + 8 * size
    symbol_count = len(vocabulary) + len(productions)
    symbol_width = max(1, (max(1, symbol_count) - 1).bit_length())
    production_bits = gamma_code_bits(len(productions) + 1) + 2 * len(productions) * symbol_width
    grammar_bits = vocabulary_bits + production_bits
    trace_bits = gamma_code_bits(len(start_sequence) + 1) + len(start_sequence) * symbol_width
    provenance_bytes = len(canonical_json(encoded["production_source_token_offsets"]).encode("utf-8"))
    raw_bits = encoded["input_bytes"] * 8
    return {
        "code_length_schema": "mdl-pair-grammar-v1; Elias-gamma counts/UTF-8 lengths, literal UTF-8 terminal dictionary, fixed-width symbol references",
        "symbol_id_width_bits": symbol_width,
        "grammar_description_bits": vocabulary_bits + production_bits,
        "encoded_trace_bits_given_grammar": trace_bits,
        "mdl_total_bits": grammar_bits + trace_bits,
        "uncompressed_trace_bits": raw_bits,
        "mdl_to_uncompressed_ratio": (grammar_bits + trace_bits) / max(1, raw_bits),
        "provenance_metadata_bytes_excluded_from_mdl": provenance_bytes,
        "roundtrip_exact": bool(encoded["roundtrip_exact"]),
        "source_trace_sha256": encoded["input_sha256"],
    }


class PairGrammarCodec:
    """Deterministic frequency-pair grammar compressor with exact lossless decoding."""

    def __init__(self, max_rules: int = 96):
        self.max_rules = max_rules
        self.productions: list[tuple[int, int]] = []
        self.provenance: list[dict] = []

    def compress(self, tokens: list[str]) -> dict:
        vocabulary: list[str] = []
        token_ids: dict[str, int] = {}
        sequence: list[int] = []
        for token in tokens:
            token_id = token_ids.get(token)
            if token_id is None:
                token_id = len(vocabulary)
                token_ids[token] = token_id
                vocabulary.append(token)
            sequence.append(token_id)
        spans = [(index, index + 1) for index in range(len(tokens))]
        for rule_no in range(self.max_rules):
            counts = collections.Counter(zip(sequence, sequence[1:]))
            candidates = [(pair, count) for pair, count in counts.items() if count >= 2]
            if not candidates:
                break
            pair, count = min(candidates, key=lambda item: (-item[1], item[0]))
            lhs = len(vocabulary) + rule_no
            new_sequence: list[str] = []
            new_spans: list[tuple[int, int]] = []
            occurrence_starts: list[int] = []
            index = 0
            while index < len(sequence):
                if index + 1 < len(sequence) and (sequence[index], sequence[index + 1]) == pair:
                    start = spans[index][0]
                    end = spans[index + 1][1]
                    occurrence_starts.append(start)
                    new_sequence.append(lhs)
                    new_spans.append((start, end))
                    index += 2
                else:
                    new_sequence.append(sequence[index])
                    new_spans.append(spans[index])
                    index += 1
            if len(new_sequence) == len(sequence):
                break
            starts_json = canonical_json(occurrence_starts)
            self.productions.append(pair)
            self.provenance.append({
                "occurrence_count": len(occurrence_starts),
                "occurrence_start_offsets_sha256": hashlib.sha256(starts_json.encode("utf-8")).hexdigest(),
                "first_occurrence_start_offsets": occurrence_starts[:8],
                "last_occurrence_start_offsets": occurrence_starts[-8:],
            })
            sequence, spans = new_sequence, new_spans

        decoded_index = 0
        decoded_hash = hashlib.sha256()
        decoded_size = 0
        def expand(symbol: int) -> None:
            nonlocal decoded_index, decoded_size
            if symbol < len(vocabulary):
                actual = vocabulary[symbol]
                if decoded_index >= len(tokens) or tokens[decoded_index] != actual:
                    raise AssertionError("induced pair grammar failed lossless round-trip")
                encoded_symbol = actual.encode("utf-8")
                decoded_hash.update(encoded_symbol)
                decoded_size += len(encoded_symbol)
                decoded_index += 1
            else:
                production = self.productions[symbol - len(vocabulary)]
                expand(production[0])
                expand(production[1])
        for symbol in sequence:
            expand(symbol)
        input_digest = hashlib.sha256()
        input_size = 0
        for token in tokens:
            encoded_token = token.encode("utf-8")
            input_digest.update(encoded_token)
            input_size += len(encoded_token)
        if decoded_index != len(tokens) or decoded_size != input_size or decoded_hash.hexdigest() != input_digest.hexdigest():
            raise AssertionError("induced pair grammar failed lossless round-trip")
        return {
            "schema": "lossless-pair-substitution-cfg-v1",
            "kind": "pair-substitution grammar; nonterminals and repeated pairs induced from the observed trace",
            "max_rules": self.max_rules,
            "token_count": len(tokens),
            "compressed_symbol_count": len(sequence),
            "production_count": len(self.productions),
            "token_symbol_ratio": len(sequence) / max(1, len(tokens)),
            "roundtrip_exact": True,
            "input_sha256": input_digest.hexdigest(),
            "input_bytes": input_size,
            "decoded_sha256": decoded_hash.hexdigest(),
            "terminal_vocabulary": vocabulary,
            "production_rhs_terminal_or_nonterminal_ids": [list(rhs) for rhs in self.productions],
            "production_source_token_offsets": self.provenance,
            "start_sequence_ids": sequence,
        }


def grammar_predict(rules: dict[tuple[int, ...], dict], condition: tuple[int, ...]) -> int | None:
    rule = rules.get(condition)
    return None if rule is None else int(rule["action"])


def train_one(
    seed: int,
    arm: str,
    episodes: int,
    output_root: Path,
    symbolizer: StateSymbolizer,
    update_interval: int = 50,
    min_support: int = 8,
    min_confidence: float = 0.90,
    environment_name: str = "CartPole-v1",
    symbolizer_scope: str = "seed_specific",
) -> Path:
    experiment_code_hash = sha256_file(Path(__file__).resolve())
    prototype_source_hash = sha256_file(ROOT / "ai_grammar_induction.py")
    seed_everything(seed)
    env = gym.make(environment_name)
    state_dim = int(np.prod(env.observation_space.shape))
    if not isinstance(env.action_space, gym.spaces.Discrete):
        env.close()
        raise TypeError("this experiment runner requires a discrete action space")
    action_dim = int(env.action_space.n)
    actor = Actor(state_dim=state_dim, action_dim=action_dim)
    optimizer = torch.optim.Adam(actor.parameters(), lr=3e-3)
    grammar = StateActionGrammar(min_support=min_support, min_confidence=min_confidence)
    active_rules: dict[tuple[int, ...], dict] = {}

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / f"{arm}" / f"seed-{seed}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    trace_path = run_dir / "trajectory.jsonl"
    update_path = run_dir / "training_updates.jsonl"
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    previous_hash = ZERO_HASH
    previous_update_hash = ZERO_HASH
    step_global = 0

    with trace_path.open("w", encoding="utf-8", newline="\n") as trace, update_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as update_ledger:
        for episode in range(episodes):
            observation, reset_info = env.reset(seed=seed * 10_000 + episode)
            observation = np.asarray(observation, dtype=np.float32)
            log_probs: list[torch.Tensor] = []
            rewards: list[float] = []
            total_reward = 0.0
            episode_step = 0
            terminated = truncated = False

            while not (terminated or truncated):
                state_symbol = symbolizer.encode(observation)
                state_tensor = torch.as_tensor(observation, dtype=torch.float32)
                logits = actor(state_tensor)
                mask = [True] * action_dim
                applied_rule = None
                if arm == "grammar_constrained":
                    applied_rule = active_rules.get(state_symbol)
                    if applied_rule is not None:
                        mask = [False, False]
                        mask[int(applied_rule["action"])] = True
                masked_logits = logits.masked_fill(~torch.tensor(mask, dtype=torch.bool), float("-inf"))
                dist = Categorical(logits=masked_logits)
                action_tensor = dist.sample()
                action = int(action_tensor.item())
                action_probs = dist.probs.detach().tolist()
                log_probs.append(dist.log_prob(action_tensor))

                next_observation, reward, terminated, truncated, info = env.step(action)
                next_observation = np.asarray(next_observation, dtype=np.float32)
                rewards.append(float(reward))
                total_reward += float(reward)
                if arm != "baseline":
                    grammar.observe(state_symbol, action, float(reward), step_global)

                event = {
                    "episode": episode,
                    "step": episode_step,
                    "global_step": step_global,
                    "state": observation.astype(float).tolist(),
                    "state_symbol": list(state_symbol),
                    "action": action,
                    "action_probabilities_after_constraint": action_probs,
                    "reward": float(reward),
                    "next_state": next_observation.astype(float).tolist(),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    # Sidecar grammar state is intentionally excluded so baseline and audit-only
                    # runs can be compared by identical trajectory hashes.
                    "grammar_generation": 0,
                    "constraint_rule": applied_rule,
                    "reset_seed": seed * 10_000 + episode if episode_step == 0 else None,
                }
                chained = chain_record(event, previous_hash)
                trace.write(canonical_json(chained) + "\n")
                previous_hash = chained["record_hash"]
                observation = next_observation
                step_global += 1
                episode_step += 1

            returns = []
            running_return = 0.0
            for reward in reversed(rewards):
                running_return = float(reward) + 0.99 * running_return
                returns.append(running_return)
            returns.reverse()
            returns_tensor = torch.tensor(returns, dtype=torch.float32)
            if returns_tensor.numel() > 1 and float(returns_tensor.std(unbiased=False)) > 1e-8:
                returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std(unbiased=False) + 1e-8)
            loss = -(torch.stack(log_probs) * returns_tensor).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            gradient_hash = gradient_sha256(actor)
            optimizer.step()

            update = {
                "episode": episode,
                "global_step": step_global,
                "episode_length": episode_step,
                "episode_return": total_reward,
                "loss": float(loss.detach()),
                "gradient_sha256_after_clipping": gradient_hash,
                "actor_state_sha256_after_update": actor_state_sha256(actor),
                "optimizer_state_sha256_after_update": serialized_sha256(optimizer.state_dict()),
                "rng_state_sha256_after_update": rng_state_hashes(),
            }
            chained_update = chain_record(update, previous_update_hash)
            update_ledger.write(canonical_json(chained_update) + "\n")
            previous_update_hash = chained_update["record_hash"]

            episode_returns.append(total_reward)
            episode_lengths.append(episode_step)
            if arm != "baseline" and (episode + 1) % update_interval == 0:
                active_rules = grammar.induce()

    env.close()
    grammar_json = grammar.formal_grammar() if arm != "baseline" else {
        "kind": "none", "productions": [], "generation": 0
    }
    grammar_path = run_dir / "induced_grammar.json"
    grammar_path.write_text(json.dumps(grammar_json, indent=2) + "\n", encoding="utf-8")
    checkpoint_path = run_dir / "actor.pt"
    torch.save({"state_dict": actor.state_dict(), "seed": seed, "arm": arm}, checkpoint_path)

    full_checkpoint_path = run_dir / "training_checkpoint.pt"
    torch.save({
        "actor_state_dict": actor.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "completed_episodes": episodes,
        "seed": seed,
        "arm": arm,
    }, full_checkpoint_path)

    evaluation = evaluate_policy(
        actor, symbolizer, active_rules, arm, seed, n_episodes=20,
        environment_name=environment_name,
    )
    trace_check = verify_trace(trace_path)
    update_check = verify_update_trace(update_path)
    config = {
        "environment": environment_name,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "episodes": episodes,
        "optimizer": "Adam",
        "learning_rate": 0.003,
        "return_discount": 0.99,
        "hidden_dim": 64,
        "state_bins": symbolizer.to_json(),
        "symbolizer_scope": symbolizer_scope,
        "grammar_update_interval": update_interval,
        "grammar_min_support": min_support,
        "grammar_min_confidence": min_confidence,
        "seed": seed,
        "arm": arm,
    }
    metadata = {
        "created_at_utc": stamp,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "gymnasium": gym.__version__,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "config": config,
        "calibration_seed_offset": 100_000,
        "training_episode_reset_seed_formula": "seed * 10000 + episode_index",
        "trace_audit": trace_check,
        "training_update_audit": update_check,
        "trace_sha256": sha256_file(trace_path),
        "training_updates_sha256": sha256_file(update_path),
        "training_update_core_sha256": training_update_core_sha256(update_path),
        "actor_sha256": actor_state_sha256(actor),
        "training_checkpoint_sha256": sha256_file(full_checkpoint_path),
        "grammar_sha256": sha256_file(grammar_path),
        "experiment_code_sha256": experiment_code_hash,
        "prototype_source_sha256": prototype_source_hash,
        "episode_return_mean": statistics.mean(episode_returns),
        "episode_return_last_50_mean": statistics.mean(episode_returns[-50:]),
        "episode_length_mean": statistics.mean(episode_lengths),
        "evaluation": evaluation,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (run_dir / "episode_returns.json").write_text(json.dumps(episode_returns) + "\n", encoding="utf-8")
    return run_dir


def evaluate_policy(
    actor: Actor,
    symbolizer: StateSymbolizer,
    rules: dict[tuple[int, ...], dict],
    arm: str,
    seed: int,
    n_episodes: int = 20,
    environment_name: str = "CartPole-v1",
) -> dict:
    env = gym.make(environment_name)
    returns: list[float] = []
    lengths: list[int] = []
    covered = correct = 0
    total = 0
    actor.eval()
    with torch.inference_mode():
        for episode in range(n_episodes):
            observation, _ = env.reset(seed=seed * 10_000 + 900_000 + episode)
            total_reward = 0.0
            length = 0
            terminated = truncated = False
            while not (terminated or truncated):
                condition = symbolizer.encode(observation)
                logits = actor(torch.as_tensor(observation, dtype=torch.float32))
                neural_action = int(logits.argmax().item())
                grammar_action = grammar_predict(rules, condition)
                total += 1
                if grammar_action is not None:
                    covered += 1
                    correct += int(grammar_action == neural_action)
                action = neural_action
                if arm == "grammar_constrained" and grammar_action is not None:
                    action = grammar_action
                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                length += 1
            returns.append(total_reward)
            lengths.append(length)
    env.close()
    actor.train()
    return {
        "episodes": n_episodes,
        "mean_return": statistics.mean(returns),
        "median_return": statistics.median(returns),
        "mean_length": statistics.mean(lengths),
        "grammar_state_coverage": covered / max(total, 1),
        "grammar_action_agreement_when_covered": correct / max(covered, 1),
        "covered_steps": covered,
        "total_steps": total,
        "return_by_episode": returns,
    }


def replay_induced_trace(trace_path: Path, output_path: Path, max_rules: int = 96) -> dict:
    text = trace_path.read_text(encoding="utf-8")
    tokens = tokenize_losslessly(text)
    codec = PairGrammarCodec(max_rules=max_rules)
    encoded = codec.compress(tokens)
    encoded["source_trace_sha256"] = sha256_file(trace_path)
    raw_bytes = text.encode("utf-8")
    raw_gzip_bytes = gzip.compress(raw_bytes, compresslevel=9, mtime=0)
    encoded["raw_trace_bytes"] = len(raw_bytes)
    encoded["raw_trace_gzip_bytes"] = len(raw_gzip_bytes)
    payload = canonical_json(encoded).encode("utf-8")
    grammar_gzip_bytes = gzip.compress(payload, compresslevel=9, mtime=0)

    # Verify the actual serialized/gzipped archive, not only the in-memory induction.
    loaded = json.loads(gzip.decompress(grammar_gzip_bytes).decode("utf-8"))
    vocabulary = loaded["terminal_vocabulary"]
    productions = loaded["production_rhs_terminal_or_nonterminal_ids"]
    cursor = 0
    def check_expansion(symbol_id: int) -> None:
        nonlocal cursor
        if symbol_id < len(vocabulary):
            if cursor >= len(tokens) or tokens[cursor] != vocabulary[symbol_id]:
                raise AssertionError("serialized grammar archive failed exact decode")
            cursor += 1
        else:
            left, right = productions[symbol_id - len(vocabulary)]
            check_expansion(left)
            check_expansion(right)
    for symbol_id in loaded["start_sequence_ids"]:
        check_expansion(symbol_id)
    if cursor != len(tokens):
        raise AssertionError("serialized grammar archive decoded an incomplete token stream")

    encoded["grammar_json_bytes"] = len(payload)
    encoded["grammar_gzip_bytes"] = len(grammar_gzip_bytes)
    encoded["grammar_gzip_to_raw_trace_bytes_ratio"] = len(grammar_gzip_bytes) / max(1, len(raw_bytes))
    encoded["grammar_gzip_to_raw_gzip_bytes_ratio"] = len(grammar_gzip_bytes) / max(1, len(raw_gzip_bytes))
    output_path.write_bytes(grammar_gzip_bytes)
    manifest = {
        "archive_file": output_path.name,
        "archive_sha256": hashlib.sha256(grammar_gzip_bytes).hexdigest(),
        "source_trace_sha256": encoded["source_trace_sha256"],
        "roundtrip_exact": encoded["roundtrip_exact"],
        "raw_trace_bytes": encoded["raw_trace_bytes"],
        "raw_trace_gzip_bytes": encoded["raw_trace_gzip_bytes"],
        "grammar_json_bytes": encoded["grammar_json_bytes"],
        "grammar_gzip_bytes": len(grammar_gzip_bytes),
        "grammar_gzip_to_raw_trace_bytes_ratio": encoded["grammar_gzip_to_raw_trace_bytes_ratio"],
        "grammar_gzip_to_raw_gzip_bytes_ratio": encoded["grammar_gzip_to_raw_gzip_bytes_ratio"],
    }
    output_path.with_name("lossless_trace_grammar_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return encoded


def summarize_existing_runs(output_root: Path, seeds: list[int], codec_rules: int = 16) -> dict:
    """Resume analysis after training artifacts already exist; encode one unique full trace only."""
    arms = ("baseline", "grammar_audit_only", "grammar_constrained")
    by_seed: dict[int, dict[str, Path]] = {}
    per_seed = []
    trace_audits = []
    paired_equivalence = []

    for seed in seeds:
        by_seed[seed] = {}
        row = {"seed": seed}
        for arm in arms:
            matches = sorted((output_root / arm).glob(f"seed-{seed}_*"))
            if len(matches) != 1:
                raise ValueError(f"expected exactly one completed run for seed={seed}, arm={arm}; found {len(matches)}")
            run_dir = matches[0]
            metadata_path = run_dir / "metadata.json"
            if not metadata_path.exists():
                raise ValueError(f"run is incomplete: {run_dir}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            trace_check = verify_trace(run_dir / "trajectory.jsonl")
            update_path = run_dir / "training_updates.jsonl"
            update_check = verify_update_trace(update_path) if update_path.exists() else None
            env_check = replay_environment_trace(run_dir / "trajectory.jsonl")
            if trace_check["final_hash"] != metadata["trace_audit"]["final_hash"]:
                raise ValueError(f"stored trace audit mismatch: {run_dir}")
            if update_check and update_check["final_hash"] != metadata.get("training_update_audit", {}).get("final_hash"):
                raise ValueError(f"stored optimizer-update audit mismatch: {run_dir}")
            by_seed[seed][arm] = run_dir
            evaluation = metadata["evaluation"]
            grammar = json.loads((run_dir / "induced_grammar.json").read_text(encoding="utf-8"))
            row[arm] = {
                "run_dir": str(run_dir),
                "train_return_mean": metadata["episode_return_mean"],
                "train_return_last_50_mean": metadata["episode_return_last_50_mean"],
                "eval_mean_return": evaluation["mean_return"],
                "grammar_state_coverage": evaluation["grammar_state_coverage"],
                "grammar_action_agreement_when_covered": evaluation["grammar_action_agreement_when_covered"],
                "grammar_production_count": len(grammar.get("productions", [])),
                "actor_sha256": metadata["actor_sha256"],
                "trace_sha256": metadata["trace_sha256"],
                "training_updates_sha256": metadata.get("training_updates_sha256"),
                "training_update_core_sha256": metadata.get(
                    "training_update_core_sha256", training_update_core_sha256(update_path) if update_path.exists() else None
                ),
                "training_checkpoint_sha256": metadata.get("training_checkpoint_sha256"),
            }
            trace_audits.append({
                "seed": seed, "arm": arm, **trace_check,
                "training_update_audit": update_check or {"valid": None, "records": 0, "availability": "not captured in legacy run"},
                "environment_replay": env_check,
            })
        base = json.loads((by_seed[seed]["baseline"] / "metadata.json").read_text(encoding="utf-8"))
        observer = json.loads((by_seed[seed]["grammar_audit_only"] / "metadata.json").read_text(encoding="utf-8"))
        paired_equivalence.append({
            "seed": seed,
            "same_actor_hash": base["actor_sha256"] == observer["actor_sha256"],
            "same_transition_trace_hash": base["trace_sha256"] == observer["trace_sha256"],
            "same_training_update_hash": (
                base.get("training_updates_sha256") == observer.get("training_updates_sha256")
                if base.get("training_updates_sha256") and observer.get("training_updates_sha256") else None
            ),
            "same_optimizer_update_core_hash": (
                row["baseline"]["training_update_core_sha256"] is not None
                and row["baseline"]["training_update_core_sha256"] == row["grammar_audit_only"]["training_update_core_sha256"]
            ),
        })
        row["grammar_constraint_return_delta_vs_baseline"] = (
            row["grammar_constrained"]["eval_mean_return"] - row["baseline"]["eval_mean_return"]
        )
        per_seed.append(row)

    # Baseline and observer-only traces are identical; one complete trace is enough to test lossless coding.
    codec_results = []
    for seed in seeds[:1]:
        run_dir = by_seed[seed]["baseline"]
        codec_path = run_dir / "lossless_trace_grammar.json.gz"
        codec = replay_induced_trace(run_dir / "trajectory.jsonl", codec_path, max_rules=codec_rules)
        codec_results.append({
            "seed": seed,
            "run_dir": str(run_dir),
            **{key: codec[key] for key in (
                "token_count", "compressed_symbol_count", "production_count", "token_symbol_ratio",
                "roundtrip_exact", "input_sha256", "decoded_sha256", "raw_trace_bytes",
                "raw_trace_gzip_bytes", "grammar_json_bytes", "grammar_gzip_bytes",
                "grammar_gzip_to_raw_trace_bytes_ratio", "grammar_gzip_to_raw_gzip_bytes_ratio"
            )},
        })

    return_deltas = [float(row["grammar_constraint_return_delta_vs_baseline"]) for row in per_seed]
    episode_counts = {
        int(json.loads((by_seed[seed]["baseline"] / "metadata.json").read_text(encoding="utf-8"))["config"]["episodes"])
        for seed in seeds
    }
    if len(episode_counts) != 1:
        raise ValueError(f"inconsistent episode counts across seeds: {sorted(episode_counts)}")
    result = {
        "claim_scope": "CartPole-v1, REINFORCE, fixed CPU software environment; evidence is bounded to this implementation/task",
        "experiment": {"seeds": seeds, "episodes_per_seed_arm": episode_counts.pop(), "arms": list(arms)},
        "paired_baseline_vs_audit_only_exact_equivalence": paired_equivalence,
        "all_baseline_audit_only_pairs_exactly_equal": all(
            row["same_actor_hash"] and row["same_transition_trace_hash"]
            and row["same_optimizer_update_core_hash"]
            for row in paired_equivalence
        ),
        "trace_hash_chain_validation": {
            "all_valid": all(row["valid"] for row in trace_audits),
            "all_environment_transitions_replayed": all(row["environment_replay"]["valid"] for row in trace_audits),
            "all_update_ledgers_valid": all(row["training_update_audit"]["valid"] is not False for row in trace_audits),
            "all_runs_have_update_ledgers": all(row["training_update_audit"]["valid"] is not None for row in trace_audits),
            "total_records": sum(row["records"] for row in trace_audits),
            "total_optimizer_updates": sum(row["training_update_audit"]["records"] for row in trace_audits),
            "runs": trace_audits,
        },
        "per_seed_metrics": per_seed,
        "grammar_constraint_return_delta": {
            "per_seed": return_deltas,
            "mean": statistics.mean(return_deltas),
            "minimum": min(return_deltas),
            "maximum": max(return_deltas),
        },
        "lossless_trace_grammar_codec": codec_results,
        "all_encoded_traces_roundtrip_exactly": all(
            item["roundtrip_exact"] and item["input_sha256"] == item["decoded_sha256"] for item in codec_results
        ),
        "interpretation": {
            "lossless_grammar": "Exact round-trip is a proof about a recorded trace only; it is not evidence the grammar explains the policy.",
            "state_action_grammar": "Held-out coverage and agreement measure generalization separately; low coverage must not be hidden by high covered-subset accuracy.",
            "training_process_audit": "Hash-chained trajectory and same-seed instrumentation equivalence are partial audit evidence; they do not prove cross-hardware exact training replay or complete capture of every optimizer/RNG state.",
        },
    }
    summary_path = output_root / "experiment_summary.json"
    summary_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def run_experiment(
    output_root: Path,
    seeds: list[int],
    episodes: int,
    calibration_episodes: int,
    codec_rules: int,
    state_bins: int = 4,
    min_support: int = 8,
    min_confidence: float = 0.90,
) -> dict:
    runs: list[dict] = []
    by_seed: dict[int, dict[str, Path]] = {}
    for seed in seeds:
        symbolizer = calibration_symbolizer(seed, episodes=calibration_episodes, n_bins=state_bins)
        by_seed[seed] = {}
        for arm in ("baseline", "grammar_audit_only", "grammar_constrained"):
            run_dir = train_one(
                seed, arm, episodes, output_root, symbolizer,
                min_support=min_support,
                min_confidence=min_confidence,
            )
            by_seed[seed][arm] = run_dir
            metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
            runs.append({"seed": seed, "arm": arm, "run_dir": str(run_dir), **metadata})

    return summarize_existing_runs(output_root, seeds, codec_rules=codec_rules)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs" / "grammar_audit")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 29])
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--calibration-episodes", type=int, default=24)
    parser.add_argument("--codec-rules", type=int, default=48)
    parser.add_argument("--state-bins", type=int, default=4)
    parser.add_argument("--min-support", type=int, default=8)
    parser.add_argument("--min-confidence", type=float, default=0.90)
    parser.add_argument("--summarize-existing", action="store_true", help="analyze completed run folders without retraining")
    args = parser.parse_args()
    if args.summarize_existing:
        result = summarize_existing_runs(args.output_root, args.seeds, codec_rules=args.codec_rules)
    else:
        result = run_experiment(
            args.output_root, args.seeds, args.episodes, args.calibration_episodes, args.codec_rules,
            state_bins=args.state_bins,
            min_support=args.min_support,
            min_confidence=args.min_confidence,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
