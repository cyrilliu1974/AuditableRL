我建议把 2.RL 的实验重心收缩到一个清晰问题：

> 离散规则层能否可靠地描述不同 RL 策略之间的行为关系，并在什么条件下支持策略组合？

现在论文最大的不足，不是实验数量少，而是关键结论仍建立在“两种环境、单组 8 个 agent、部分 seed pair、单一状态分布”上。尤其当前结果显示：

- 规则 Jaccard 约 0.56，但新状态行为一致率只有 51.38%；
- CartPole 规则融合为 205.96，低于 actor 平均值 330.86 和 logit ensemble 的 500；
- Acrobot 从 sampled-rule 的 −443.37 改为 greedy-rule 后变成 −141.77。

这些结果其实已经提示：最值得研究的不是“我们的融合算法很强”，而是：

1. 为什么规则重叠不代表行为一致；
2. 什么情况下规则置信度可以用于组合；
3. induction–deployment mismatch 为什么会导致灾难性失败。

下面是我建议的实验优先级。

## 一、必须做的五组实验

| 优先级 | 实验 | 解决的核心问题 |
|---|---|---|
| P0 | 全 seed、全任务的重叠—行为一致性实验 | 中央负结论是否稳健 |
| P0 | 规则忠实度与 coverage–fidelity 曲线 | 规则到底有没有描述神经策略 |
| P0 | induction × arbitration 因子实验 | Acrobot 的逆转究竟由什么造成 |
| P0 | 完整组合基线与独立 agent population | 融合结果能否泛化 |
| P1 | Oracle headroom 与冲突状态诊断 | 问题出在规则库还是仲裁器 |

---

## 1. 全面的“规则重叠是否预测行为一致”实验

这是论文最重要的实验，应当升级为主实验。

### 当前问题

目前最醒目的证据是 CartPole seed 11/29：

- 规则集合重叠较高；
- fresh-state action agreement 约 51.38%；
- 接近由动作频率决定的基准。

但一个 seed pair 不足以支持一般性结论。即使使用全部 28 个 pair，如果所有 pair 来自同一批 8 个 seed，也不能把 28 个 pair 当成独立样本。

### 建议设计

每个任务训练至少 24–40 个独立策略：

- 最低方案：3 组 × 8 agents；
- 推荐方案：5 组 × 8 agents；
- 每组独立训练、独立构建规则库、独立执行融合。

对每组中的所有 agent pair，计算：

- 普通 Jaccard；
- support-weighted Jaccard；
- visitation-weighted Jaccard；
- action-conditional rule overlap；
- fresh-state action agreement；
- Cohen’s \(\kappa\) 或 chance-adjusted agreement；
- 策略 logits 的 Jensen–Shannon divergence。

### 不要只使用一种 fresh-state 分布

至少在四种状态分布下测量：

1. **Mixture occupancy**：所有 source policy 的状态混合；
2. **Random/calibration occupancy**：随机策略产生的状态；
3. **Conflict distribution**：规则库给出不同动作的状态，进行过采样；
4. **Shifted distribution**：扰动初始状态、动力学参数或观测噪声。

这样能够回答：

> “重叠不预测行为一致”是普遍现象，还是只在某一种 occupancy distribution 下成立？

### 统计方法

不要简单对所有 state 或所有 pair 做普通 bootstrap。pair 共享同一个 agent，不独立。

建议：

- 按训练 seed 或独立 agent population 做 cluster bootstrap；
- 或使用 dyadic/QAP permutation；
- 给出 Spearman 相关系数及置信区间；
- 如果要声称“没有有意义的关联”，做 equivalence test，而不只是报告 \(p>0.05\)。

例如预注册：

\[
H_{\text{meaningful}}: |\rho| \geq 0.20
\]

只有相关系数置信区间落在 \([-0.20,0.20]\) 内，才能比较有力地支持“重叠没有实际预测价值”。

---

## 2. 规则忠实度与 selective prediction 实验

这是目前稿件里最缺的一块。

论文构建了 condition→action rules，但没有充分回答：

> 在规则覆盖的状态上，这些规则究竟多大程度上重现了原神经策略？

“规则 coverage 高”不等于“规则预测正确”。必须把二者同时报告。

### 三种规则库构建协议

建议比较：

1. **Sampled-action bank**  
   用训练时实际采样动作作为标签；

2. **Greedy/argmax bank**  
   对轨迹中的状态重新查询 actor，以 argmax 动作为标签；

3. **Balanced-query bank**  
   从共同状态池、冲突区域和低覆盖区域采样状态，再查询 actor 的 argmax 动作。

第三种可以判断：问题究竟是动作标签不一致，还是训练轨迹覆盖不足。

### 主要指标

在独立 held-out state corpus 上报告：

- Rule coverage；
- Covered-state fidelity：
  $\[P(\hat a_{\text{rule}}(s)=\arg\max_a\pi(a|s)\mid s\text{ covered})\]$
- Overall fidelity，把 fallback 也计算进去；
- Macro-F1，避免动作不平衡造成虚高；
- selective risk–coverage curve；
- AURC；
- rule confidence 的 ECE/Brier score；
- 与多数动作基线的差值。

尤其需要画出：

> admission threshold \(\tau\) 增加时，coverage 如何下降、fidelity 是否真的提高。

如果阈值从 0.7 提到 0.9 后 coverage 崩溃，但 fidelity 并没有显著提高，这会成为很有价值的负结果。

---

## 3. 完整的 induction × arbitration 因子实验

当前 Acrobot 的结果很重要，但还无法确定机制：

- sampled bank + confidence：−443.37；
- sampled bank + random：−187.02；
- greedy bank + confidence：−141.77；
- greedy bank + random：−234.26。

这说明存在 induction–deployment mismatch，但还没有分解：

- greedy bank 是否整体更好；
- confidence 是否只在 greedy bank 中才有意义；
- 改善来自 coverage、冲突结构，还是规则置信度校准。

### 建议至少做 3×5 因子设计

规则诱导方式：

- sampled action；
- greedy action；
- balanced greedy query。

仲裁方式：

- confidence-first；
- support-first；
- mean-reward-first；
- random candidate；
- value/advantage-first。

fallback 需要固定，否则无法判断变化来自仲裁还是 fallback。

### 每个 cell 报告

- episodic return；
- success rate；
- 10% CVaR 或最低分位表现；
- coverage；
- conflict rate；
- fallback rate；
- candidate count；
- arbiter 选择动作与 source actor argmax 的一致率；
- 选择动作的估计 advantage。

这个实验最好同时在所有环境运行，而不是只在 Acrobot 补一次。

如果交互效应显著，即：

\[
\text{greedy bank} \times \text{confidence arbitration}
\]

明显优于其他组合，那么论文就能提出一个更有价值的原则：

> 经验规则置信度只有在规则诱导标签与部署决策协议一致时，才具有仲裁意义。

这比“我们在 Acrobot 上修复了一个 bug”更有论文价值。

---

## 4. 更严格的组合性能实验

当前 CartPole 的“+42.31”容易被审稿人质疑，因为 comparator 是按训练表现事后选择的，而且同一批 actor 中有四个达到 500。

### 必须预先固定基线

至少包括：

- 随机选一个 source actor；
- source actors 的平均表现；
- 用独立 validation set 选择的最佳 actor；
- action majority vote；
- probability/logit ensemble；
- random-candidate rule arbiter；
- confidence rule fusion；
- greedy-bank confidence fusion；
- oracle candidate selector。

不要再把“训练表现最佳 actor”作为主要 comparator，除非它是在分析前明确规定的部署选择规则。

### 数据拆分

建议严格拆成：

- calibration：构建 symbolizer；
- induction：建立规则库；
- validation：选择 threshold、symbolizer 和 comparator；
- test：只做一次最终评估。

所有超参数，包括：

- bin 数量；
- confidence threshold；
- support threshold；
- arbitration priority；
- fallback；

都不能根据 test return 决定。

### 统计单位

当前 100 个 common-reset episodes 可以比较固定策略之间的差异，但不能支持“对新的训练 seed 也成立”。

因此需要多个独立 agent populations，例如：

- 每个任务 5 组；
- 每组 8 个独立 actor；
- 每组形成一个 fused policy；
- 每个 fused policy 在相同的 200 个 evaluation seeds 上评估。

推断时先在 population 层面计算差异，再进行 hierarchical bootstrap。否则 episode 数再多，也只是重复评估同一组 agent。

### 建议报告的指标

除了平均 return，还应包括：

- median return；
- success rate；
- 10% CVaR；
- regret to best validation-selected actor；
- regret to logit ensemble；
- 冲突状态上的条件 return；
- 性能—可审计性 Pareto frontier。

如果 rule fusion 仍然输给 logit ensemble，也不一定意味着论文失败。可以诚实定位为：

> Rule fusion 提供了可追溯、可离散检查的决策层，但需要支付明确的性能代价。

---

## 5. Oracle headroom：区分“规则库问题”和“仲裁器问题”

这是我最推荐增加的机制诊断实验。

当前融合表现差时，无法判断究竟是：

1. 规则库没有包含正确动作；
2. 正确动作在 candidate set 中，但 confidence arbiter 选错；
3. 状态符号过粗，不同真实状态被压到同一个 symbol；
4. fallback 太弱。

### Oracle candidate selector

对于发生规则冲突的状态，使用 Monte Carlo rollout 或可靠的 \(Q(s,a)\) 估计，计算候选动作的真实后续回报，然后选择最优候选动作。

得到：

\[
F_{\text{oracle}}(s)
=
\arg\max_{a\in A_{\text{rules}}(s)}
Q(s,a)
\]

比较：

- confidence fusion；
- random fusion；
- oracle fusion；
- unrestricted optimal/ensemble policy。

### 解释方式

| 结果 | 结论 |
|---|---|
| Oracle fusion 也很差 | 规则库或 symbolizer 没有提供足够信息 |
| Oracle 很强，confidence 很差 | 主要是仲裁器和 confidence 失效 |
| Greedy bank 的 oracle 明显更强 | sampled-label induction 损坏了候选集合 |
| Candidate oracle 强，但仍低于 ensemble | 离散规则表示存在性能上限 |

还可以计算：

\[
\operatorname{corr}\big(
\text{rule confidence},
Q(s,a)-Q(s,a_{\text{alternative}})
\big)
\]

如果相关性接近零，就能直接证明：

> 动作出现频率意义上的 confidence，并不等于控制意义上的 action quality。

这会是非常有价值的机制结果。

---

## 二、强烈建议增加的实验

### 6. Symbolizer 消融

当前固定 quantile symbolizer 是整个框架的关键瓶颈，但目前只使用了一种设置。

建议比较：

- Quantile bins：\(B=2,4,8\)；
- uniform-width bins；
- global k-means：\(K=64,256,1024\)；
- 学习型但冻结的 VQ symbolizer；
- random-policy calibration；
- mixed-policy calibration；
- source-policy occupancy calibration。

报告：

- occupied vocabulary；
- held-out coverage；
- fidelity；
- rule overlap；
- behavioral agreement；
- conflict rate；
- fusion return；
-规则数及审计成本。

Acrobot 不适合盲目增加每维 bin，因为 \(8^6=262{,}144\) 个符号会导致严重稀疏。这里更适合增加 global clustering 或 product quantization。

最终应给出一条 Pareto 曲线：

> 符号复杂度／规则数量 vs. 策略忠实度与组合性能。

---

### 7. 环境扩展

至少应从 2 个任务扩展到 4 个，推荐最终做到 5–6 个。

最低可行组合：

- CartPole：低冲突、明显 ceiling；
- Acrobot：高冲突、稀疏覆盖；
- MountainCar：稀疏奖励、状态空间低维；
- LunarLander：更高维、四动作。

更强的版本再增加：

- 一个 MiniGrid 任务：部分可观测、语义状态；
- 一个 MinAtar 或离散视觉任务：测试 symbolizer 扩展性。

环境选择不要只追求数量，而应覆盖不同 regime：

| Regime | 需要的任务性质 |
|---|---|
| 低冲突 | 多个策略行为相近 |
| 高冲突 | 相同 symbol 下动作分歧大 |
| 稀疏奖励 | rule frequency 与 action value 容易脱钩 |
| 分布偏移 | calibration 与 deployment occupancy 不一致 |
| 高维状态 | 检验符号空间爆炸 |

---

### 8. Audit 机制的威胁模型实验

单纯报告 hash 和 replay 全部通过，实验信息量很低，因为这是实现正确时应当发生的结果。

更有价值的是主动注入不同类型的篡改：

| 篡改方式 | Hash | Replay | 理论上应否检测 |
|---|---:|---:|---:|
| 修改 action，不重新计算 hash | 失败 | 可能失败 | 是 |
| 修改 reward，不重新计算 hash | 失败 | 失败 | 是 |
| 重排 transition | 失败 | 失败 | 是 |
| 修改并重新计算整个 hash chain | 通过 | 失败 | 是 |
| 构造完全可 replay 的虚假轨迹并重新 hash | 通过 | 通过 | 否 |
| 改变环境版本或浮点平台 | 可能通过 | 可能失败 | 应明确测试 |

最后一种“检测不到”的情况非常重要，它能清楚界定：

> 该 ledger 证明记录未被事后修改并且可以重放，但不能证明记录最初由所声明的训练过程产生。

这比“100% hash validity”更有审稿价值。

另外建议补：

- 不同 CPU/操作系统上的 replay；
- 不同 Gymnasium 版本；
- 随机环境中是否完整记录 RNG state；
- 浮点 exact equality 与 tolerance-based equality 的差异。

---

### 9. 修复或降级 FQE/GPI 实验

当前 FQE-GPI 在两个任务都 collapse，因此它不能作为可信的竞争基线。

有两个选择。

**选择 A：认真修复**

- cross-fitting；
- 增加行为覆盖；
- 对每个 source policy 单独验证 critic；
- 报告 Bellman residual；
- 与 Monte Carlo return 比较；
- 报告 \(Q\) 排序相关性；
- 检查 action gap；
- 只有 critic 通过有效性检查后才运行 GPI。

**选择 B：从主要基线中移除**

把它放到附录，定位为：

> 一次失败的诊断性尝试，不能用于证明 rule fusion 优于 value-based composition。

我更倾向于选择 B，除非作者愿意投入大量工作把 value baseline 做扎实。

---

## 三、可选但能显著增强“auditability”主张的实验

### 10. 人类审计效用实验

如果论文想声称规则层对审计有实际价值，可以让具有 RL 背景的参与者完成诊断任务。

实验条件：

- 只看训练曲线和日志；
- 看神经策略 rollout；
- 看规则库和 provenance；
- 看规则库、provenance 和 replay 工具。

任务例如：

- 判断两个策略是否真正行为一致；
- 找到一次失败决策来自哪个 source bank；
- 识别 sampled/argmax protocol mismatch；
- 判断某个融合失败是 coverage 还是 conflict 导致。

指标：

- 判断准确率；
- 完成时间；
- 错误类型；
- 主观信心。

这个实验可以把“auditability”从工程属性提升到用户效用，但不是最低限度必须做。

---

## 四、我建议的最小可发表实验包

如果计算和时间有限，我建议至少完成下面六项：

1. 扩展到 4 个环境；
2. 每个环境至少 3 组独立的 8-agent populations；
3. 所有 pair、四种状态分布下的 overlap–agreement 分析；
4. sampled/greedy/balanced 三种规则诱导的 coverage–fidelity 曲线；
5. induction × arbitration 因子实验；
6. oracle candidate selector，用来分解 representation 与 arbitration 问题。

统计上必须做到：

- population/训练 seed 是主要推断单位；
- evaluation episode 不是唯一统计单位；
- 预先固定 comparator；
- 使用 validation/test 分离；
- 中央负结论使用 equivalence test 或有意义的效应区间。

## 五、论文叙事也应该相应调整

我不建议把目标定成：

> “证明 rule fusion 优于标准策略集成。”

因为现有数据明显不支持：CartPole 上 logit ensemble 是 500，而规则融合只有 205.96。

更好的主线是：

> 我们提出一个可验证的符号行为层，并系统展示三个容易被混淆但并不等价的关系：规则覆盖不代表忠实度，规则重叠不代表行为一致，审计可追溯性不代表组合性能。进一步地，规则诱导与部署协议的一致性决定了经验置信度是否可用于仲裁。

如果上述实验做扎实，即便规则融合最终仍然输给 logit ensemble，这篇论文依然可以成为一篇有价值的“诊断框架＋边界条件＋负结果”论文。相反，如果只是继续增加几个 fusion return 表格，论文的核心问题仍然不会解决。原稿可在这里打开：:chatgpt-content-reference{index="0"}[2.RL(1).pdf](sandbox:/workspace/scratch/8f466bb31712/upload/2.RL(1).pdf)。
