"""Legacy prototype for AIM grammar induction.

The classes below are retained for inspection, but their original CartPole demo
and ReplayAuditor are not evidence of lossless replay or held-out generalization.
Run the validated comparison experiment with ``python ai_grammar_induction.py``;
its implementation is in ``auditability/grammar_audit_experiment.py``.
"""

import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import random

# ============================================================
# 基礎資料結構定義
# ============================================================

@dataclass
class Rule:
    """
    AI 母語的最小語義單元。
    設計原則：規則必須是「有條件的動作傾向」，不是純動作。
    """
    condition: tuple        # 狀態特徵的離散化表示（見 StateAbstractor）
    action_dist: tuple      # 動作傾向（離散化後的 action index 序列）
    frequency: int = 1      # 觀察到此規則的次數
    confidence: float = 0.0 # P(action | condition)，Layer 2 計算

    def __hash__(self):
        return hash((self.condition, self.action_dist))

    def __eq__(self, other):
        return self.condition == other.condition and self.action_dist == other.action_dist


@dataclass
class ProductionRule:
    """
    形式語法的產生式規則：A → B C 或 A → terminal
    """
    lhs: str                    # 左側非終端符號
    rhs: tuple                  # 右側符號序列
    weight: float = 1.0         # 語法歸納後的規則權重
    source_rules: list = field(default_factory=list)  # 來源 Rule 編號（可審計）


@dataclass  
class FormalGrammar:
    """
    歸納出的形式語法容器。
    invariant: productions 中的每條規則都有 source_rules 可追溯到 Layer 1 觀察。
    """
    terminals: set = field(default_factory=set)
    non_terminals: set = field(default_factory=set)
    productions: list = field(default_factory=list)
    start_symbol: str = "S"
    generation: int = 0         # 第幾輪語法精煉

    def is_stable(self, prev_grammar) -> bool:
        """收斂判定：產生式數量不再減少"""
        if prev_grammar is None:
            return False
        return len(self.productions) == len(prev_grammar.productions)


# ============================================================
# LAYER 1：規則觀察期
# ============================================================

class StateAbstractor:
    """
    將連續狀態空間離散化為可觀察的條件符號。
    這是讓「規則」能被記錄的關鍵前處理。
    
    設計選擇：k-means 離散化 + 線上更新
    替代方案：VQ-VAE（若狀態維度高）、手動分箱（若有領域知識）
    """
    def __init__(self, state_dim: int, n_bins: int = 16):
        self.state_dim = state_dim
        self.n_bins = n_bins
        # 初始化為均勻分箱，訓練過程中動態更新
        self.bin_edges = torch.zeros(state_dim, n_bins + 1)
        self._initialized = False
        self._state_buffer = []
        self._buffer_size = 1000

    def update_bins(self, state: torch.Tensor):
        """線上更新分箱邊界（每 buffer_size 步更新一次）"""
        self._state_buffer.append(state.detach().cpu())
        if len(self._state_buffer) >= self._buffer_size:
            states = torch.stack(self._state_buffer)
            for d in range(self.state_dim):
                quantiles = torch.quantile(
                    states[:, d],
                    torch.linspace(0, 1, self.n_bins + 1)
                )
                self.bin_edges[d] = quantiles
            self._state_buffer = []
            self._initialized = True

    def abstract(self, state: torch.Tensor) -> tuple:
        """
        state (shape: [state_dim]) → condition tuple（可哈希，用作 Rule.condition）
        若分箱尚未初始化，回傳 None（Layer 1 跳過此樣本）
        """
        if not self._initialized:
            self.update_bins(state)
            return None
        
        indices = []
        for d in range(self.state_dim):
            idx = torch.bucketize(state[d], self.bin_edges[d])
            idx = idx.clamp(0, self.n_bins - 1)
            indices.append(idx.item())
        return tuple(indices)


class ActionAbstractor:
    """
    將動作（離散或連續）轉為符號 token。
    離散動作：直接用 index。
    連續動作：量化到 n_action_bins 個符號。
    """
    def __init__(self, action_type: str = "discrete",
                 action_dim: int = 1, n_action_bins: int = 8):
        self.action_type = action_type
        self.action_dim = action_dim
        self.n_action_bins = n_action_bins
        # 連續動作的量化邊界
        self.action_bin_edges = torch.linspace(-1, 1, n_action_bins + 1)

    def abstract(self, action) -> tuple:
        if self.action_type == "discrete":
            if isinstance(action, int):
                return (action,)
            return (action.item(),)
        else:
            # 連續動作量化
            if isinstance(action, (int, float)):
                action = torch.tensor([action])
            tokens = []
            for d in range(self.action_dim):
                idx = torch.bucketize(action[d], self.action_bin_edges)
                idx = idx.clamp(0, self.n_action_bins - 1)
                tokens.append(idx.item())
            return tuple(tokens)


class RuleObserver:
    """
    Layer 1 核心：同步觀察 RL 訓練，記錄 (condition → action) 規則。
    
    使用方式：在標準 RL 訓練迴圈的每個 step 後呼叫 observe()。
    不干涉 RL 訓練本身。
    """
    def __init__(self, state_dim: int, action_type: str = "discrete",
                 action_dim: int = 1, n_state_bins: int = 16,
                 n_action_bins: int = 8, window_size: int = 3):
        self.state_abstractor = StateAbstractor(state_dim, n_state_bins)
        self.action_abstractor = ActionAbstractor(action_type, action_dim, n_action_bins)
        self.window_size = window_size  # 多步規則的視窗大小

        # 規則儲存：condition → Counter(action_sequence)
        self.rule_bank: dict[tuple, Counter] = defaultdict(Counter)
        
        # 滑動視窗緩衝
        self._condition_buffer = []
        self._action_buffer = []
        
        self.total_observations = 0

    def observe(self, state: torch.Tensor, action,
                reward: float, done: bool):
        """
        每個 RL step 後呼叫。
        state: 當前狀態（進入動作前）
        action: 執行的動作
        reward: 獲得的回饋（目前僅記錄，Layer 2 加權時使用）
        done: episode 是否結束
        """
        condition = self.state_abstractor.abstract(state)
        if condition is None:
            return  # 分箱尚未初始化
        
        action_token = self.action_abstractor.abstract(action)
        
        self._condition_buffer.append(condition)
        self._action_buffer.append(action_token)
        
        # 維持視窗大小
        if len(self._condition_buffer) > self.window_size:
            self._condition_buffer.pop(0)
            self._action_buffer.pop(0)
        
        # 記錄單步規則（condition → action）
        self.rule_bank[condition][action_token] += 1
        
        # 記錄多步規則（若視窗已滿）
        if len(self._condition_buffer) == self.window_size:
            multi_condition = self._condition_buffer[0]  # 以第一步狀態為條件
            multi_action = tuple(
                t for tokens in self._action_buffer for t in tokens
            )
            self.rule_bank[multi_condition][multi_action] += 1
        
        if done:
            self._condition_buffer.clear()
            self._action_buffer.clear()
        
        self.total_observations += 1

    def get_top_rules(self, min_frequency: int = 5) -> list[Rule]:
        """
        從 rule_bank 提取高頻規則，輸入 Layer 2。
        min_frequency：過濾低頻雜訊用。
        """
        rules = []
        for condition, action_counter in self.rule_bank.items():
            total = sum(action_counter.values())
            for action_seq, count in action_counter.items():
                if count >= min_frequency:
                    confidence = count / total
                    rules.append(Rule(
                        condition=condition,
                        action_dist=action_seq,
                        frequency=count,
                        confidence=confidence
                    ))
        # 依頻率降序排列
        rules.sort(key=lambda r: r.frequency, reverse=True)
        return rules

    def summary(self) -> dict:
        return {
            "total_observations": self.total_observations,
            "unique_conditions": len(self.rule_bank),
            "unique_rules": sum(len(v) for v in self.rule_bank.values()),
        }


# ============================================================
# LAYER 2：語法歸納（Grammar Induction）
# ============================================================

class SequiturInducer:
    """
    基於 SEQUITUR 算法的上下文無關語法歸納。
    
    SEQUITUR 原理：
    1. 掃描符號序列，發現重複的 bigram
    2. 用新的非終端符號替換重複 bigram
    3. 遞迴直到無重複
    
    輸入：Rule 列表（來自 Layer 1）
    輸出：FormalGrammar
    """
    def __init__(self):
        self.symbol_counter = 0
        self.bigram_index: dict = {}    # bigram → 出現次數
        self.rule_index: dict = {}      # bigram → 非終端符號

    def _new_symbol(self) -> str:
        sym = f"NT{self.symbol_counter}"
        self.symbol_counter += 1
        return sym

    def induce(self, rules: list[Rule]) -> FormalGrammar:
        """
        從 Rule 列表歸納 FormalGrammar。
        每條 Rule 的 action_dist 作為一個符號序列輸入。
        """
        grammar = FormalGrammar()
        grammar.start_symbol = "S"
        grammar.non_terminals.add("S")

        # 將所有 rule 的 action_dist 轉為字串序列
        all_sequences = []
        for i, rule in enumerate(rules):
            # 條件符號化
            cond_sym = f"C{'_'.join(map(str, rule.condition))}"
            grammar.terminals.add(cond_sym)
            
            # 動作符號化
            action_syms = [f"A{t}" for t in rule.action_dist]
            for sym in action_syms:
                grammar.terminals.add(sym)
            
            seq = [cond_sym] + action_syms
            all_sequences.append((seq, i, rule.frequency))

        # 合併所有序列（依頻率加權）
        weighted_seq = []
        for seq, rule_idx, freq in all_sequences:
            for _ in range(min(freq, 10)):  # 上限避免爆炸
                weighted_seq.extend(seq)
                weighted_seq.append("SEP")  # 規則分隔符

        # 執行 SEQUITUR 壓縮
        compressed = self._sequitur_compress(weighted_seq, grammar, rules)

        # 建立 S 規則
        grammar.productions.append(ProductionRule(
            lhs="S",
            rhs=tuple(compressed[:50]),  # 取代表性前綴
            weight=1.0,
            source_rules=list(range(len(rules)))
        ))

        grammar.generation += 1
        return grammar

    def _sequitur_compress(self, sequence: list,
                            grammar: FormalGrammar,
                            source_rules: list[Rule]) -> list:
        """核心壓縮迴圈"""
        seq = list(sequence)
        changed = True
        iteration = 0
        max_iterations = 100

        while changed and iteration < max_iterations:
            changed = False
            bigram_counts = Counter()
            
            # 統計 bigram 頻率
            for i in range(len(seq) - 1):
                if seq[i] != "SEP" and seq[i+1] != "SEP":
                    bigram = (seq[i], seq[i+1])
                    bigram_counts[bigram] += 1
            
            # 找最高頻 bigram
            if not bigram_counts:
                break
            top_bigram, top_count = bigram_counts.most_common(1)[0]
            
            if top_count < 2:
                break  # 沒有重複 bigram，語法已穩定
            
            # 建立新的非終端符號
            if top_bigram not in self.rule_index:
                new_sym = self._new_symbol()
                self.rule_index[top_bigram] = new_sym
                grammar.non_terminals.add(new_sym)
                grammar.productions.append(ProductionRule(
                    lhs=new_sym,
                    rhs=top_bigram,
                    weight=top_count,
                    source_rules=[
                        i for i, r in enumerate(source_rules)
                        if any(
                            f"A{t}" in top_bigram or
                            f"C{'_'.join(map(str, r.condition))}" in top_bigram
                            for t in r.action_dist
                        )
                    ]
                ))
            else:
                new_sym = self.rule_index[top_bigram]
            
            # 替換序列中的 bigram
            new_seq = []
            i = 0
            while i < len(seq):
                if (i < len(seq) - 1 and
                    seq[i] == top_bigram[0] and
                    seq[i+1] == top_bigram[1]):
                    new_seq.append(new_sym)
                    i += 2
                    changed = True
                else:
                    new_seq.append(seq[i])
                    i += 1
            seq = new_seq
            iteration += 1

        return seq


class GrammarInducer:
    """
    Layer 2 入口：選擇歸納算法並管理語法版本。
    """
    def __init__(self, algorithm: str = "sequitur"):
        self.algorithm = algorithm
        self.history: list[FormalGrammar] = []
        
        if algorithm == "sequitur":
            self._inducer = SequiturInducer()
        else:
            raise NotImplementedError(f"Algorithm {algorithm} not yet implemented")

    def induce(self, rules: list[Rule]) -> FormalGrammar:
        prev = self.history[-1] if self.history else None
        grammar = self._inducer.induce(rules)
        self.history.append(grammar)
        return grammar

    def is_converged(self) -> bool:
        """連續兩輪產生式數量相同 → 收斂"""
        if len(self.history) < 2:
            return False
        return self.history[-1].is_stable(self.history[-2])

    def convergence_report(self) -> dict:
        return {
            "generations": len(self.history),
            "production_counts": [len(g.productions) for g in self.history],
            "converged": self.is_converged()
        }


# ============================================================
# LAYER 3：語法回饋壓縮（Feedback Loop）
# ============================================================

class GrammarConstrainedPolicy(nn.Module):
    """
    受語法約束的策略網路。
    
    核心設計：在標準 Actor 的 logits 上疊加語法 mask。
    語法允許的動作保持原始 logit；不允許的動作設為 -inf。
    
    這樣 RL 梯度仍然正常流動，只是動作空間被語法剪枝。
    """
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 標準 Actor 網路
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # 語法 mask（初始全 1，Layer 3 更新後會有 0）
        self.register_buffer(
            'grammar_mask',
            torch.ones(action_dim, dtype=torch.bool)
        )
        
        # 當前有效的語法約束（用於審計）
        self.current_grammar: Optional[FormalGrammar] = None
        self.constraint_history: list[dict] = []

    def update_grammar_mask(self, grammar: FormalGrammar,
                            state_abstractor: StateAbstractor,
                            action_abstractor: ActionAbstractor):
        """
        從形式語法更新動作 mask。
        
        邏輯：提取語法中所有合法的 action token，
              對應到原始動作空間的 index。
        """
        allowed_actions = set()
        
        for prod in grammar.productions:
            for sym in prod.rhs:
                if sym.startswith("A"):
                    try:
                        action_idx = int(sym[1:])
                        allowed_actions.add(action_idx)
                    except ValueError:
                        pass
        
        if not allowed_actions:
            # 語法還沒歸納出有效動作，保持全開放
            return
        
        new_mask = torch.zeros(self.action_dim, dtype=torch.bool)
        for idx in allowed_actions:
            if idx < self.action_dim:
                new_mask[idx] = True
        
        # 確保至少有一個動作可選
        if new_mask.sum() == 0:
            new_mask = torch.ones(self.action_dim, dtype=torch.bool)
            return
        
        self.grammar_mask = new_mask
        self.current_grammar = grammar
        
        # 記錄約束歷史（用於審計）
        self.constraint_history.append({
            "grammar_generation": grammar.generation,
            "allowed_action_count": int(new_mask.sum()),
            "total_actions": self.action_dim,
            "pruning_ratio": 1.0 - float(new_mask.sum()) / self.action_dim
        })

    def forward(self, state: torch.Tensor) -> Categorical:
        """
        輸出受語法約束的動作分佈。
        """
        logits = self.actor(state)
        
        # 應用語法 mask
        masked_logits = logits.clone()
        masked_logits[~self.grammar_mask] = float('-inf')
        
        # 若所有動作都被 mask（edge case），回退到原始 logits
        if torch.all(masked_logits == float('-inf')):
            masked_logits = logits
        
        return Categorical(logits=masked_logits)

    def get_audit_trace(self) -> dict:
        """
        返回可審計的決策軌跡摘要。
        """
        return {
            "grammar_generation": (
                self.current_grammar.generation
                if self.current_grammar else None
            ),
            "active_productions": (
                len(self.current_grammar.productions)
                if self.current_grammar else 0
            ),
            "constraint_history": self.constraint_history,
        }


class ReplayAuditor:
    """
    重播驗證器：給定決策軌跡，驗證是否可依語法規則重播。
    
    「可重播」的操作定義：
    每個 (condition, action) 對都能在 grammar 的 productions 中找到對應規則。
    """
    def __init__(self):
        self.audit_log: list[dict] = []

    def verify_trajectory(
        self,
        trajectory: list[tuple],   # [(state_tensor, action), ...]
        grammar: FormalGrammar,
        state_abstractor: StateAbstractor,
        action_abstractor: ActionAbstractor
    ) -> dict:
        """
        驗證一條完整 episode 是否可重播。
        
        回傳：
        - replayable: bool
        - coverage: float（可重播步數 / 總步數）
        - failed_steps: list（無法重播的步驟索引）
        - audit_entries: list（每步的審計條目）
        """
        total_steps = len(trajectory)
        failed_steps = []
        audit_entries = []

        # 建立語法規則的快速查找表
        valid_pairs = set()
        for prod in grammar.productions:
            rhs = prod.rhs
            if len(rhs) >= 2:
                cond_str = rhs[0] if isinstance(rhs[0], str) else str(rhs[0])
                act_str = rhs[1] if isinstance(rhs[1], str) else str(rhs[1])
                valid_pairs.add((cond_str, act_str))

        for step_idx, (state, action) in enumerate(trajectory):
            condition = state_abstractor.abstract(state)
            if condition is None:
                audit_entries.append({
                    "step": step_idx,
                    "status": "SKIP",
                    "reason": "abstractor not initialized"
                })
                continue

            action_token = action_abstractor.abstract(action)
            
            cond_str = f"C{'_'.join(map(str, condition))}"
            act_str = f"A{action_token[0]}"
            
            replayable = (cond_str, act_str) in valid_pairs
            
            entry = {
                "step": step_idx,
                "condition_symbol": cond_str,
                "action_symbol": act_str,
                "status": "PASS" if replayable else "FAIL",
                "matching_production": None
            }
            
            if replayable:
                # 找到對應的產生式（審計用）
                for prod in grammar.productions:
                    if (len(prod.rhs) >= 2 and
                        prod.rhs[0] == cond_str and
                        prod.rhs[1] == act_str):
                        entry["matching_production"] = {
                            "lhs": prod.lhs,
                            "rhs": prod.rhs,
                            "source_rules": prod.source_rules
                        }
                        break
            else:
                failed_steps.append(step_idx)
            
            audit_entries.append(entry)

        coverage = (total_steps - len(failed_steps)) / max(total_steps, 1)
        
        result = {
            "replayable": len(failed_steps) == 0,
            "coverage": coverage,
            "total_steps": total_steps,
            "failed_steps": failed_steps,
            "audit_entries": audit_entries,
            "grammar_generation": grammar.generation
        }
        
        self.audit_log.append(result)
        return result


# ============================================================
# 主訓練迴圈整合
# ============================================================

class AIMotherTongueRL:
    """
    三層架構的完整整合類別。
    
    使用方式：
        env = YourGymEnv()
        system = AIMotherTongueRL(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
        )
        system.train(env, total_episodes=5000)
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_type: str = "discrete",
        hidden_dim: int = 128,
        # Layer 1 超參數
        n_state_bins: int = 16,
        n_action_bins: int = 8,
        rule_window_size: int = 3,
        min_rule_frequency: int = 5,
        # Layer 2 超參數
        grammar_update_interval: int = 500,   # 每幾個 episode 歸納一次語法
        grammar_algorithm: str = "sequitur",
        # Layer 3 超參數
        max_grammar_generations: int = 20,
        lr: float = 3e-4,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Layer 1
        self.observer = RuleObserver(
            state_dim=state_dim,
            action_type=action_type,
            action_dim=1 if action_type == "discrete" else action_dim,
            n_state_bins=n_state_bins,
            n_action_bins=n_action_bins,
            window_size=rule_window_size
        )
        self.min_rule_frequency = min_rule_frequency
        
        # Layer 2
        self.inducer = GrammarInducer(algorithm=grammar_algorithm)
        self.grammar_update_interval = grammar_update_interval
        self.max_grammar_generations = max_grammar_generations
        
        # Layer 3
        self.policy = GrammarConstrainedPolicy(state_dim, action_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.auditor = ReplayAuditor()
        
        self.current_grammar: Optional[FormalGrammar] = None
        self.training_log: list[dict] = []

    def _reinforce_update(self, log_probs: list, rewards: list, gamma: float = 0.99):
        """標準 REINFORCE 更新（可替換為 PPO/SAC）"""
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns)
        if returns.std() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        loss = torch.stack([
            -lp * R for lp, R in zip(log_probs, returns)
        ]).sum()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()
        return loss.item()

    def train(self, env, total_episodes: int = 5000, verbose: bool = True):
        """
        主訓練迴圈。
        env 需符合 gym 介面：reset() → state, step(action) → (next_state, reward, done, info)
        """
        episode_rewards = []
        
        for ep in range(total_episodes):
            state, _ = env.reset() if hasattr(env.reset(), '__iter__') else (env.reset(), {})
            state = torch.FloatTensor(state)
            
            log_probs, rewards, trajectory = [], [], []
            done = False
            ep_reward = 0
            
            # ── Episode 執行 ──
            while not done:
                dist = self.policy(state)
                action = dist.sample()
                log_probs.append(dist.log_prob(action))
                
                next_state, reward, done, *_ = env.step(action.item())
                next_state = torch.FloatTensor(next_state)
                
                # Layer 1：觀察規則
                self.observer.observe(state, action.item(), reward, done)
                
                trajectory.append((state.clone(), action.item()))
                rewards.append(reward)
                ep_reward += reward
                state = next_state
            
            # ── REINFORCE 更新 ──
            loss = self._reinforce_update(log_probs, rewards)
            episode_rewards.append(ep_reward)
            
            # ── Layer 2：定期歸納語法 ──
            if (ep + 1) % self.grammar_update_interval == 0:
                rules = self.observer.get_top_rules(self.min_rule_frequency)
                
                if len(rules) >= 3:  # 規則數量足夠才歸納
                    new_grammar = self.inducer.induce(rules)
                    self.current_grammar = new_grammar
                    
                    # Layer 3：更新語法 mask
                    self.policy.update_grammar_mask(
                        new_grammar,
                        self.observer.state_abstractor,
                        self.observer.action_abstractor
                    )
                    
                    # 驗證最後一條 trajectory
                    audit_result = self.auditor.verify_trajectory(
                        trajectory,
                        new_grammar,
                        self.observer.state_abstractor,
                        self.observer.action_abstractor
                    )
                    
                    log_entry = {
                        "episode": ep + 1,
                        "grammar_generation": new_grammar.generation,
                        "n_productions": len(new_grammar.productions),
                        "replay_coverage": audit_result["coverage"],
                        "fully_replayable": audit_result["replayable"],
                        "converged": self.inducer.is_converged(),
                        "avg_reward_last_100": np.mean(episode_rewards[-100:])
                    }
                    self.training_log.append(log_entry)
                    
                    if verbose:
                        print(
                            f"[Ep {ep+1}] "
                            f"Grammar Gen={new_grammar.generation} | "
                            f"Productions={len(new_grammar.productions)} | "
                            f"Replay Coverage={audit_result['coverage']:.2%} | "
                            f"Converged={self.inducer.is_converged()} | "
                            f"Avg Reward={log_entry['avg_reward_last_100']:.2f}"
                        )
                    
                    # 收斂判定
                    if (self.inducer.is_converged() and
                        new_grammar.generation >= self.max_grammar_generations):
                        if verbose:
                            print(f"[收斂] 語法在第 {new_grammar.generation} 代穩定。")
                        break
        
        return self.training_log

    def audit_policy(self, env, n_episodes: int = 10) -> dict:
        """
        訓練後審計：測試策略的重播覆蓋率。
        """
        if self.current_grammar is None:
            return {"error": "No grammar induced yet"}
        
        results = []
        for _ in range(n_episodes):
            state, _ = env.reset() if hasattr(env.reset(), '__iter__') else (env.reset(), {})
            state = torch.FloatTensor(state)
            trajectory = []
            done = False
            
            with torch.no_grad():
                while not done:
                    dist = self.policy(state)
                    action = dist.sample()
                    next_state, _, done, *_ = env.step(action.item())
                    trajectory.append((state.clone(), action.item()))
                    state = torch.FloatTensor(next_state)
            
            result = self.auditor.verify_trajectory(
                trajectory,
                self.current_grammar,
                self.observer.state_abstractor,
                self.observer.action_abstractor
            )
            results.append(result)
        
        return {
            "n_episodes": n_episodes,
            "mean_coverage": np.mean([r["coverage"] for r in results]),
            "fully_replayable_episodes": sum(r["replayable"] for r in results),
            "grammar_generation": self.current_grammar.generation,
            "n_productions": len(self.current_grammar.productions),
            "policy_audit_trace": self.policy.get_audit_trace()
        }


# ============================================================
# 使用範例（CartPole）
# ============================================================

def example_cartpole(
    seeds: list[int] | None = None,
    episodes: int = 300,
    calibration_episodes: int = 24,
    codec_rules: int = 48,
    output_root=None,
):
    """Run the tested baseline / grammar-audit / constrained-policy comparison."""
    from pathlib import Path
    from auditability.grammar_audit_experiment import run_experiment

    project_root = Path(__file__).resolve().parent
    return run_experiment(
        output_root=Path(output_root) if output_root else project_root / "runs" / "grammar_audit",
        seeds=seeds or [11, 29],
        episodes=episodes,
        calibration_episodes=calibration_episodes,
        codec_rules=codec_rules,
    )


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 29])
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--calibration-episodes", type=int, default=24)
    parser.add_argument("--codec-rules", type=int, default=48)
    parser.add_argument("--output-root", type=Path, default=project_root / "runs" / "grammar_audit")
    parser.add_argument("--summarize-existing", action="store_true", help="analyze completed runs without retraining")
    args = parser.parse_args()
    if args.summarize_existing:
        from auditability.grammar_audit_experiment import summarize_existing_runs
        result = summarize_existing_runs(args.output_root, args.seeds, codec_rules=args.codec_rules)
    else:
        result = example_cartpole(args.seeds, args.episodes, args.calibration_episodes, args.codec_rules, args.output_root)
    print(json.dumps(result, indent=2))
