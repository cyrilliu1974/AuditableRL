# Oracle Headroom 分支研究報告 (v1.0)

**日期**：2026-09-26
**分支代號**：`oracle_headroom`
**所屬計畫**：AIM Auditability and Rule-Level RL Fusion
**發表包（本報告所有連結的基準）**：https://github.com/cyrilliu1974/AuditableRL
**實驗狀態**：預註冊實驗已完成；主要判定為 **STOP（borderline）**；Post-hoc 穩健性檢查已完成
**報告定位**：這是一份「決策紀錄（decision record）」型報告。它不只報告數字，而是說明**這批數字如何改變了研究路線的選擇**。

---

## 摘要

原論文的實驗結果，直覺上會把人推向一個結論：**該投入資源打造一個「超強裁判（arbiter）」**——用 GNN 學狀態表徵、用 RL 學價值函數，來裁決規則庫之間的衝突。這個方向在當時看起來非常自然，因為 Acrobot 上「隨機仲裁」竟然大幅勝過「信心仲裁」。

Oracle Headroom 分支用一次**預註冊、低成本、精確**的診斷實驗，把這個直覺**證偽**了：

- 即使把裁判換成「完美裁判」（在衝突狀態直接查真實 Q 值選最優），也只能補回 **31%** 的融合—最佳差距（`H/T = 0.306 < 1/3` → STOP）。
- 進一步拿掉一個訓練時就塌陷的 bank（seed 307）後，**完美裁判反而輸給現行的信心仲裁**（322.99 vs 335.98，`H/T = −0.079`）；而 7 個健康 bank 在 **43,510 個覆蓋步裡只衝突了 5 步**。

結論：**仲裁問題是空集合。** 剩下的缺口全部在表示層——規則沒覆蓋到（覆蓋率）＋規則本身不夠好（品質）。

因此，下一步的工作**不是**設計更強的仲裁器，而是**提升網格（symbolizer 離散化）與協議（規則誘導／准入）**。這個轉向讓後續研究從「高成本、高不確定性的表徵學習工程」，降維成「低維度超參數搜尋 ＋ 資料協定設計」——**工作反而變簡單了**。

---

## 0. 這份報告的資料來源與可追溯性

本報告所有數字都可在發表包中直接查到。連結基準為：

```
https://github.com/cyrilliu1974/AuditableRL/blob/main/
```

| 類別 | 路徑 | 用途 |
| --- | --- | --- |
| 原論文結果摘要 | [`README.md`](https://github.com/cyrilliu1974/AuditableRL/blob/main/README.md) | 第 1 節所有原論文數字 |
| 原論文草稿 | [`manuscript/rl_rule_fusion_draft.md`](https://github.com/cyrilliu1974/AuditableRL/blob/main/manuscript/rl_rule_fusion_draft.md) | 主張與限制 |
| 原論文精簡結果 | [`results/shared_multienv_gpi/`](https://github.com/cyrilliu1974/AuditableRL/tree/main/results/shared_multienv_gpi) | Jaccard、行為一致率、仲裁消融、FQE |
| 預註冊協議 | [`oracle_headroom/PROTOCOL.md`](https://github.com/cyrilliu1974/AuditableRL/blob/main/oracle_headroom/PROTOCOL.md) | 實驗開始**之前**寫定的判定規則 |
| 技術判定報告 | [`oracle_headroom/REPORT.md`](https://github.com/cyrilliu1974/AuditableRL/blob/main/oracle_headroom/REPORT.md) | 主要結論的技術版本 |
| 實驗日誌 | [`oracle_headroom/EXPERIMENT_LOG.md`](https://github.com/cyrilliu1974/AuditableRL/blob/main/oracle_headroom/EXPERIMENT_LOG.md) | 時序紀錄、異常處置 |
| 主要結果（機器可讀） | [`oracle_headroom/results/oracle_headroom_cartpole.json`](https://github.com/cyrilliu1974/AuditableRL/blob/main/oracle_headroom/results/oracle_headroom_cartpole.json) | 五策略逐局分數、決策計數 |
| 分解與判定 | [`oracle_headroom/results/oracle_diagnostics.json`](https://github.com/cyrilliu1974/AuditableRL/blob/main/oracle_headroom/results/oracle_diagnostics.json) | 差距分解、bootstrap CI、STOP 判定 |
| 穩健性檢查（仲裁） | [`oracle_headroom/results/supplement_no307.json`](https://github.com/cyrilliu1974/AuditableRL/blob/main/oracle_headroom/results/supplement_no307.json) | 排除 seed 307 後的融合分數 |
| 穩健性檢查（覆蓋） | [`oracle_headroom/results/supplement_no307_full.json`](https://github.com/cyrilliu1974/AuditableRL/blob/main/oracle_headroom/results/supplement_no307_full.json) | 排除 seed 307 後的衝突步數 |
| 訓練清單 | [`oracle_headroom/runs/CartPole-v1/stage0_manifest.json`](https://github.com/cyrilliu1974/AuditableRL/blob/main/oracle_headroom/runs/CartPole-v1/stage0_manifest.json) | 8 個 seed 的訓練分數、規則數、actor 雜湊 |
| 後續實驗規劃 | [`oracle_headroom/future_experiments.md`](https://github.com/cyrilliu1974/AuditableRL/blob/main/oracle_headroom/future_experiments.md) | 第 6 節所指的網格／協議工作項 |

---

## 1. 背景：原論文的結果，與它看似指向的下一步

### 1.1 原論文的核心數字

原研究訓練 8 個 REINFORCE agent（seeds 11/29/43/71/101/149/211/307），在 CartPole-v1 與 Acrobot-v1 上抽取狀態—動作規則，並嘗試以規則層合成策略。

| 結果 | CartPole-v1 | Acrobot-v1 |
| --- | ---: | ---: |
| 8 seed 平均成對規則 Jaccard | 0.5572 | 0.0956 |
| 最佳單一 actor 平均報酬 | 163.65 | −500.00 |
| Actor-logit ensemble 平均報酬 | 500.00 | −500.00 |
| **規則融合（grammar fusion）平均報酬** | **205.96** | **−443.37** |
| 規則融合 vs 最佳單一（95% bootstrap CI） | +42.31 [23.98, 61.93] | +56.63 [39.32, 75.15] |

來源：[`README.md`](https://github.com/cyrilliu1974/AuditableRL/blob/main/README.md)

### 1.2 兩個讓「裁判」變成頭號嫌疑犯的證據

**證據一：Acrobot 的仲裁消融（最刺眼的一筆）。**

| 仲裁方式 | Acrobot 平均報酬 |
| --- | ---: |
| 規則融合（信心優先） | −443.37 |
| 僅信心 | −444.49 |
| 平均報酬優先 / 支撐度優先 | −500 |
| 衝突時交給 actor ensemble | −500 |
| **隨機選一個候選規則** | **−187.02** |
| 五條獨立隨機選擇流 | −206.17 ～ −194.10（平均 −198.82） |
| 對照：無條件隨機動作 | −498.71 |

來源：[`README.md`](https://github.com/cyrilliu1974/AuditableRL/blob/main/README.md)、[`results/shared_multienv_gpi/Acrobot-v1/grammar_fusion_diagnostics.json`](https://github.com/cyrilliu1974/AuditableRL/blob/main/results/shared_multienv_gpi/Acrobot-v1/grammar_fusion_diagnostics.json)

「**隨機**仲裁竟然比**精心設計的信心**仲裁好 256 分」——這在任何工程直覺下都只指向一個結論：**仲裁器的排序邏輯壞了**。

**證據二：CartPole 的融合輸給 ensemble。** 規則融合 205.96，logit ensemble 500.00。中間的 294 分看起來也像是「合成規則的決策品質不足」。

### 1.3 直覺結論：投入超強 GNN/RL 裁判

於是最自然的推論是：

> 既然衝突時的仲裁是瓶頸，那就用更強的表徵與價值估計來當裁判——GNN 學狀態圖表徵、RL 學 Q 函數，取代手寫的「信心優先」啟發式。

這個方向在當時**幾乎沒有被質疑**，因為它同時解釋了 Acrobot 的隨機優勢與 CartPole 的融合劣勢。

### 1.4 但這個結論有一個未被檢驗的前提

它默默假設了：**衝突狀態的候選集合裡，真的存在「好動作」，只是裁判挑不出來。**

這個前提從未被檢驗。而它其實有兩種截然不同的失效模式（見下圖的邏輯）：

| 病因 | 現象 | 藥方 |
| --- | --- | --- |
| **A：裁判爛** | 候選動作裡有好的，裁判選錯 | 做更好的裁判（GNN/RL） |
| **B：秘笈本身爛** | 候選動作**全都是壞的**，或根本沒覆蓋到 | 換更好的規則／網格；做裁判是白工 |

**Oracle Headroom 分支的唯一任務，就是把 A 與 B 分開。**

---

## 2. 實驗設計

### 2.1 問題形式化

在 CartPole-v1 上，以 8 個重訓 baseline actor（同 seeds、同設定：300 episodes、共用 4-bin 校準符號化器、`min_support=8`、`min_confidence=0.70`），定義五個策略並在**同一批 100 局**（共用 reset seeds `20_000_000 + episode_index`）上評估：

| 符號 | 策略 | 定義 |
| --- | --- | --- |
| `R_conf` | confidence fusion | 現行的信心優先規則融合 |
| `R_oc` | **oracle_conflict** | **只在衝突狀態**開天眼，從候選動作中選真實 Q 最大者；其餘與信心融合相同 |
| `R_of` | **oracle_full** | **在每一個被覆蓋的狀態**都開天眼，從**完整動作空間**選真實 Q 最大者；盲區仍用 ensemble 備援 |
| `R_best` | best single actor | 依**訓練**報酬選出的最佳單一 actor（seed 149） |
| `R_ens` | actor mean-logit ensemble | 8-actor logit 平均（僅作參考） |

**分解恆等式**：

```
R_best − R_conf = (R_best − R_of) + (R_of − R_oc) + (R_oc − R_conf)
                = 覆蓋/備援缺口   + 集體同意但選錯   + 仲裁缺口
```

來源：[`PROTOCOL.md`](https://github.com/cyrilliu1974/AuditableRL/blob/main/oracle_headroom/PROTOCOL.md) §1

### 2.2 為什麼這裡的 oracle 是「精確」而非「估計」

這是本實驗能便宜又乾淨的關鍵。CartPole-v1 的動力學在給定 `(state, action)` 下是**確定性**的，且延續策略（最佳 actor 的 argmax）也是確定性的。因此：

> **從 `(s, a)` 做「一次」rollout 得到的總報酬，就是精確的 `Q(s, a)`——不需要 Monte Carlo 平均，沒有估計噪聲。**

前置條件在主要實驗**之前**已驗證：

| 前置檢查 | 結果 | 來源 |
| --- | --- | --- |
| set-state roundtrip | 20/20 通過 | [`oracle_headroom_cartpole.json`](https://github.com/cyrilliu1974/AuditableRL/blob/main/oracle_headroom/results/oracle_headroom_cartpole.json) → `continuation.premises` |
| rollout determinism | 20/20 通過 | 同上 |
| 延續策略健全性閘門（> 400） | 500.00 → **PASS** | 同上 → `continuation.sanity_gate_mean` |

> 這個閘門是預註冊的。它存在的原因是一個前車之鑑：先前的 A2 實驗在 Acrobot 上用了 ensemble 作為延續策略，結果該策略永不終止，所有 MC 值並列 −500，整個 oracle 量測失效（VOID）。本實驗要求延續策略必須先通過健全性檢查。

來源：[`EXPERIMENT_LOG.md`](https://github.com/cyrilliu1974/AuditableRL/blob/main/oracle_headroom/EXPERIMENT_LOG.md)（2026-09-26 ~03:05 條目）、[`results/mc_degeneracy_probe_8actor.json`](https://github.com/cyrilliu1974/AuditableRL/blob/main/results/mc_degeneracy_probe_8actor.json)

### 2.3 預註冊的判定規則

協議在任何訓練或量測**之前**寫定：

> 令 `H = R_oc − R_conf`（仲裁可回收空間），`T = R_best − R_conf`（總缺口）。
>
> - **STOP**（規則庫缺乏可用訊號，不得在此批 bank 上建 Stage 1–4）：`H/T < 1/3`
> - **GO**（仲裁是瓶頸，值得改進仲裁器）：`H/T ≥ 1/3`

不確定性以 95% bootstrap CI（5,000 次抽樣）報告；判定套用於點估計，CI 並列呈現。

來源：[`PROTOCOL.md`](https://github.com/cyrilliu1974/AuditableRL/blob/main/oracle_headroom/PROTOCOL.md) §3

**為什麼要預註冊？** 因為這個實驗的結論會決定一大筆研究投資的方向。若在看到結果後才決定「多少算有意義」，就等於自己給自己發通行證。協議先寫死，結論才有約束力。

---

## 3. 主要結果

### 3.1 五種策略（100 個共用 reset 局）

| 策略 | 平均 | 中位數 | 標準差 | 95% CI |
| --- | ---: | ---: | ---: | --- |
| best_single（seed 149） | **500.00** | 500.00 | 0.00 | [500.0, 500.0] |
| actor_mean_logits（8） | 293.83 | 285.00 | 37.04 | [286.9, 301.2] |
| **oracle_full**（每步開天眼） | **364.60** | 500.00 | 147.59 | [336.6, 393.2] |
| **oracle_conflict**（完美裁判） | **194.59** | 178.50 | 52.97 | [184.8, 205.4] |
| **grammar_fusion**（現況） | **59.68** | 12.00 | 74.78 | [45.7, 74.7] |

來源：[`results/oracle_headroom_cartpole.json`](https://github.com/cyrilliu1974/AuditableRL/blob/main/oracle_headroom/results/oracle_headroom_cartpole.json) → `policies.*.mean_return`、`results/oracle_diagnostics.json` → `means` / `mean_cis`

> **注意 `R_conf = 59.68` 遠低於原論文的 205.96。** 這不是重現失敗，而是本批次重訓的 bank 集合中有一個塌陷的 seed 307（見第 4 節）。這個落差本身就是後續整個機制的入口。

### 3.2 差距分解

```
R_best − R_conf = (R_best − R_of) + (R_of − R_oc) + (R_oc − R_conf)
     440.32     =      135.40     +     170.01     +     134.91
                =   覆蓋/備援缺口  +  集體同意但選錯  +    仲裁缺口
                =       31%       +       39%       +      31%
```

| 缺口 | 分數 | 佔比 | 可否靠更好的裁判修復 |
| --- | ---: | ---: | --- |
| 仲裁缺口（裁判選錯） | 134.91 | 31% | **可以** |
| 集體同意但選錯（表示層） | 170.01 | 39% | **不行**——所有 bank 都同意同一個壞動作 |
| 覆蓋／備援缺口（盲區） | 135.40 | 31% | **不行**——沒有規則可用 |

來源：[`results/oracle_diagnostics.json`](https://github.com/cyrilliu1974/AuditableRL/blob/main/oracle_headroom/results/oracle_diagnostics.json) → `decomposition`

**讀法**：一個完美裁判只能補回 31% 的缺口；其餘 69% 是表示層問題。

### 3.3 判定

```
H = 134.91,  T = 440.32
H / T = 0.30639
95% CI = [0.2739, 0.3373]
閾值 = 1/3 = 0.33333
```

> **VERDICT：STOP**（點估計低於閾值）。
> CI 上界 0.337 僅**勉強**觸及閾值——因此這是一個 **borderline STOP，而非慘敗**。

來源：[`results/oracle_diagnostics.json`](https://github.com/cyrilliu1974/AuditableRL/blob/main/oracle_headroom/results/oracle_diagnostics.json) → `stop_rule`、`decomposition.H_over_T_ci95`

### 3.4 決策組成統計（每個策略在 100 局中走過的步數）

| 策略 | 總步數 | 被覆蓋 | 衝突 | 盲區 |
| --- | ---: | ---: | ---: | ---: |
| best_single | 50,000 | 50,000 | 0 | 0 |
| actor_mean_logits | 29,383 | 29,383 | 0 | 0 |
| grammar_fusion | 5,968 | 5,264 | 800 | 704 |
| oracle_conflict | 19,459 | 17,791 | 1,458 | 1,668 |
| oracle_full | 36,460 | 34,487 | 1,656 | 1,973 |

來源：[`results/oracle_diagnostics.json`](https://github.com/cyrilliu1974/AuditableRL/blob/main/oracle_headroom/results/oracle_diagnostics.json) → `decision_mix`

這裡已經埋下了第 4 節的伏筆：`oracle_full` 的 oracle 呼叫數是 34,487 次，而 `oracle_conflict` 只有 1,458 次——**完美裁判實際上只被叫上場 1,458 次**，但即使如此也只補回 31%。

---

## 4. Post-hoc 穩健性檢查：seed 307 毒化效應

### 4.1 為什麼要做

因為判定是 borderline（CI 上界擦到閾值），單一結論不足以支撐一個重大的路線決定。於是做了一個**明確標記為 exploratory** 的補充檢查：把訓練時就塌陷的 seed 307 bank 拿掉，用剩下 7 個 bank 重跑。

**seed 307 是什麼狀態？** 它在 300 場訓練中報酬卡在 9.74（其他 seed 為 151～345），3.0 秒就跑完（每局約 10 步）。其 metadata 顯示了關鍵的一組數字：

| 欄位 | 值 | 意義 |
| --- | ---: | --- |
| `episode_return_mean` | 9.74 | 訓練表現完全塌陷 |
| `evaluation.grammar_state_coverage` | **0.96875** | 規則**幾乎覆蓋所有狀態** |
| `evaluation.grammar_action_agreement_when_covered` | **1.0** | 規則與 actor **100% 一致** |
| `rule_count` | 33 | 規則數最少 |

來源：[`runs/.../seed-307_20260925T185803Z/metadata.json`](https://github.com/cyrilliu1974/AuditableRL/blob/main/oracle_headroom/runs/CartPole-v1/grammar_audit_only/seed-307_20260925T185803Z/metadata.json)、[`runs/CartPole-v1/stage0_manifest.json`](https://github.com/cyrilliu1974/AuditableRL/blob/main/oracle_headroom/runs/CartPole-v1/stage0_manifest.json)

> 一組「高覆蓋率、高一致性」的規則——看起來是最健康的 bank，實際上是最毒的。因為一個飽和的確定性策略，會對**壞動作**產出**高信心**的規則。

處置說明：307 被**保留**在主要實驗的 bank 集合中（事後替換 seed 會違反預註冊協議），只在標記為 exploratory 的補充分析中排除。

### 4.2 結果：結論整個翻轉

| 策略 | 含 307（主要） | 拿掉 307 後 |
| --- | ---: | ---: |
| grammar_fusion（信心） | 59.68 | **335.98** |
| oracle_conflict（完美裁判） | 194.59 | **322.99** |
| oracle_full（每步開天眼） | 364.60 | **450.87** |
| best_single | 500.00 | 500.00 |

來源：[`results/supplement_no307.json`](https://github.com/cyrilliu1974/AuditableRL/blob/main/oracle_headroom/results/supplement_no307.json)、[`results/supplement_no307_full.json`](https://github.com/cyrilliu1974/AuditableRL/blob/main/oracle_headroom/results/supplement_no307_full.json)

三個關鍵事實：

1. **「oracle 有幫助」的效應，完全由這一個塌陷的 bank 驅動。** 移除它，信心融合從 59.68 跳到 335.98。
2. **完美裁判打不贏信心仲裁。** `H/T = −0.079`——仲裁缺口是 **−12.99**，也就是噪聲。
3. **仲裁問題是空集合。** 7 個健康 bank 在 **43,510 個覆蓋步中只有 5 步衝突**。它們幾乎完全一致，沒有任何東西留給裁判裁決。

拿掉 307 後的階梯：

```
R_conf = 335.98  →  R_oc = 322.99  →  R_of = 450.87  →  R_best = 500
          仲裁缺口 −12.99（噪聲）
                     集體同意但選錯 127.88（剩餘 164 分的 78%）
                                  覆蓋缺口 49.13（30%）
```

### 4.3 機制：頻率 ≠ 品質

```
塌陷的確定性策略
   → 每次都輸出同一個動作
   → 規則數量少、但出現頻率極高
   → 信心值（= 頻率）極高
   → 「信心優先」的仲裁邏輯主動選中它們
   → 整個融合決策被單一壞 bank 帶歪
```

這不是「規則庫完全沒有訊號」，而是**整個 bank 集合的穩健性失敗（robustness failure）**。

### 4.4 三點精煉發現（明確標記為 exploratory）

1. **毒化範圍超出衝突狀態。** 完美裁判只能把分數從 60 救回 195——因為 307 的規則在**非衝突**步驟也佔主導，仲裁器根本沒有介入的機會。
2. **對健康 bank 而言，仲裁不是瓶頸。** 信心優先已經追平完美裁判（`H ≤ 0`）。A2 時期對仲裁器的擔憂，在這個 regime 下正式解除。
3. **剩餘缺口（336 → 500）純粹在表示層。** 78% 是「集體同意但次優的動作」，30% 是盲區。**一致 ≠ 正確。**

---

## 5. 統一讀法

把兩個世界並排看，問題的結構就完全清楚了：

| 問題 | 含 307（主要） | 拿掉 307（健康 bank） | 結論 |
| --- | ---: | ---: | --- |
| 仲裁缺口 | 134.91（31%） | −12.99（≈0） | **不是瓶頸** |
| 集體同意但選錯 | 170.01（39%） | 127.88（78%） | **主瓶頸** |
| 覆蓋／盲區 | 135.40（31%） | 49.13（30%） | **次瓶頸** |
| 衝突步數 | 800 / 5,264 覆蓋步 | **5 / 43,510 覆蓋步** | 仲裁是空集合 |

來源：[`results/oracle_diagnostics.json`](https://github.com/cyrilliu1974/AuditableRL/blob/main/oracle_headroom/results/oracle_diagnostics.json)、[`results/supplement_no307_full.json`](https://github.com/cyrilliu1974/AuditableRL/blob/main/oracle_headroom/results/supplement_no307_full.json)

Oracle 乾淨地把兩個長期被混為一談的問題分開：

- **「banks 會不會吵架？」**（仲裁問題）→ 幾乎不會；而且信心優先已經等於完美裁判。
- **「banks 一致同意的動作好不好？」**（表示問題）→ 不好，而且**這才是分數損失的所在**。

---

## 6. 對研究路線的影響（本報告的核心）

### 6.1 被證偽的方向：GNN/RL 超強裁判

原論文結果給出的直覺是「升級成超強裁判」。Oracle Headroom 對這個方向給出的是**否證**，而且是雙重的：

| 檢驗 | 結果 | 對 GNN/RL 裁判的意涵 |
| --- | --- | --- |
| 完美裁判能補回多少？（主要世界） | 31%（`H/T = 0.306 < 1/3`） | 上限就在那裡，做多好都只值 31% |
| 完美裁判 vs 信心仲裁（健康 bank） | **322.99 < 335.98**（`H/T = −0.079`） | 連「免費的完美」都贏不了現行做法 |
| 裁判有多少事可做？ | 43,510 覆蓋步中 5 次衝突 | **仲裁的可用空間幾乎為零** |

> **投資一個超強的 GNN/RL 裁判，等於在解一個幾乎不存在的問題。** 而且它是一筆高成本、高不確定性的投資：要設計圖表徵、訓練價值函數、處理非平穩性，而且失敗時難以歸因（是表徵錯、還是價值估計錯、還是資料不足？）。

補充一點方法論上的風險：GNN/RL 裁判的「改進」很容易被單一壞 bank 掩蓋。在含 307 的世界裡，任何仲裁器看起來都「有改善空間」（因為完美裁判能從 60 拉到 195），但那個改善空間是**毒化造成的假象**，不是仲裁能力不足。若沒有先做 Oracle Headroom 這個分解，很可能會花大量資源去追一個不存在的瓶頸。

### 6.2 被指向的方向：提升網格與協議

剩下的缺口是**兩個已被精確量化的表示層缺口**：

| 缺口 | 佔比（主要 / 健康） | 對應工作項 | 具體內容 |
| --- | --- | --- | --- |
| **覆蓋／盲區** | 31% / 30% | **提升網格** | symbolizer 離散化：`B=2/4/8` quantile、uniform-width、global k-means `K=64/256/1024`、學習型凍結 VQ；報告 occupied vocabulary、held-out coverage、fidelity、conflict rate、fusion return，最終給出「符號複雜度 vs 忠實度／效能」的 Pareto 曲線 |
| **集體同意但選錯** | 39% / 78% | **提升協議** | ① 規則誘導協議：sampled / greedy / balanced-query bank 三種標籤方式的比較；② 准入協議：`min_support` / `min_confidence` 的 coverage–fidelity 曲線，找出 `τ` 的甜蜜點；③ **bank 入場健康檢查**：以 source actor 的 eval mean 為門檻，塌陷的 bank 不得進入集合；④ 仲裁協議：固定 fallback 的 3×5 因子設計 |

來源：[`oracle_headroom/future_experiments.md`](https://github.com/cyrilliu1974/AuditableRL/blob/main/oracle_headroom/future_experiments.md) §2、§3、§6

**其中「bank 入場健康檢查」是 Oracle Headroom 直接新增的一條工作項。** 在本實驗中，排除 307 是 post-hoc 的（看到融合結果之後才做的）；下一步要做的是把它**預註冊**成集合建構時的前置條件——在還沒看到融合表現之前，就依 source actor 的評測表現決定 bank 是否准入。

### 6.3 為什麼工作反而變簡單了

這是本分支最重要的實務結論。

| | 原本的直覺路線（GNN/RL 裁判） | Oracle Headroom 修正後的路線（網格＋協議） |
| --- | --- | --- |
| 問題定位 | 假想的瓶頸（仲裁） | **已量化的兩個缺口**（覆蓋 31%、品質 39～78%） |
| 技術性質 | 表徵學習工程：GNN 架構、RL 訓練、非平穩性 | 低維度超參數搜尋 ＋ 資料協定設計 |
| 實驗成本 | 高（每輪訓練一個新仲裁器） | 低（可直接在既有 pipeline 上做網格掃描） |
| 可預註冊性 | 低（架構與訓練有太多自由度） | 高（`B`、`K`、`τ`、`min_support` 都是明確的離散選擇） |
| 失敗歸因 | 困難（表徵／價值／資料三者糾纏） | 容易（覆蓋率與 fidelity 是可直接量測的中介指標） |
| 風險 | 投入大量資源後可能仍無法歸因 | 每次實驗便宜、結論明確、可逐步收斂 |

換句話說：**Oracle Headroom 把問題從「設計一個更強的仲裁器」降維成「調網格與協議」。** 前者是開放式的研究工程，後者是收斂的實驗設計。

### 6.4 對論文敘事的影響

本分支為論文新增了第四個「不代表」，而且**附帶了機制，不只是數字**：

| 既有的三個「不代表」 | 本分支新增 |
| --- | --- |
| 規則覆蓋不代表忠實度 | **候選集合不代表存在可選的好動作** |
| 規則重疊不代表行為一致 | 機制：政策塌陷下，**頻率 ≠ 品質** |
| 審計可追溯性不代表組合性能 | 後果：**一致 ≠ 正確**；單一劣質 bank 即可毒化整個集合 |

預註冊的 STOP 判定**維持不變**（`H/T = 0.306 < 1/3`，主要世界）。補充分析並沒有推翻它，而是**重新定義了 STOP 的含義**：

> STOP 不是「完全沒有訊號」，而是「**在沒有 bank 准入機制與規則品質工程的前提下，這批 bank 不是穩健的組合基底**」。

這對論文是正面的：它把一個阻斷性的負結果，轉換成一條明確、可執行、成本可控的後續路徑。

---

## 7. 重現指引

環境（見 [`runs/.../seed-307.../metadata.json`](https://github.com/cyrilliu1974/AuditableRL/blob/main/oracle_headroom/runs/CartPole-v1/grammar_audit_only/seed-307_20260925T185803Z/metadata.json)）：

```
Python 3.12.3, torch 2.14.0+cpu, numpy 2.5.3, gymnasium 1.3.0
deterministic_algorithms = true
```

執行順序：

| 步驟 | 腳本 | 產出 |
| --- | --- | --- |
| 1 | [`stage0_train.py`](https://github.com/cyrilliu1974/AuditableRL/blob/main/oracle_headroom/stage0_train.py) | 8 個 seed 的 actor、規則庫、符號化器 → `runs/CartPole-v1/` |
| 2 | [`oracle_q.py`](https://github.com/cyrilliu1974/AuditableRL/blob/main/oracle_headroom/oracle_q.py) | 精確 rollout Q ＋ 前置條件驗證（set-state / determinism） |
| 3 | [`oracle_fusion.py`](https://github.com/cyrilliu1974/AuditableRL/blob/main/oracle_headroom/oracle_fusion.py) | 五策略於共用 reset seeds 評估 → `results/oracle_headroom_cartpole.json` |
| 4 | [`oracle_diagnostics.py`](https://github.com/cyrilliu1974/AuditableRL/blob/main/oracle_headroom/oracle_diagnostics.py) | 差距分解、bootstrap CI、STOP 判定 → `results/oracle_diagnostics.json` |
| 5 | [`supplement_no307.py`](https://github.com/cyrilliu1974/AuditableRL/blob/main/oracle_headroom/supplement_no307.py) | 排除 307 的仲裁比較 → `results/supplement_no307.json` |
| 6 | [`supplement_no307_full.py`](https://github.com/cyrilliu1974/AuditableRL/blob/main/oracle_headroom/supplement_no307_full.py) | 排除 307 的完整覆蓋分解 → `results/supplement_no307_full.json` |

**關鍵重現參數**：

- 評估局數 100；reset seed 公式 `20_000_000 + episode_index`（所有策略共用）
- 訓練 reset seed 公式 `seed * 10000 + episode_index`；校準 offset `100000`
- 符號化器：per-dimension empirical quantiles，`n_bins = 4`，跨所有實驗 seed 池化
- 規則准入：`min_support = 8`，`min_confidence = 0.70`
- 延續策略：最佳訓練報酬 actor（seed 149）的 argmax，固定不變
- 仲裁 CI：5,000 次 bootstrap

**注意**：本批次 actor 是重訓的（原論文 checkpoint 不在快照中）。所有比較都在重訓集合**內部**進行，因此內部效度不依賴原始權重的複製。seed 307 在此環境（torch 2.14.0 / gymnasium 1.3.0）塌陷，歸因於套件版本造成的 RNG 漂移——這是誠實的重現變異，不是程式錯誤（證據：其餘 7 個 seed 用同一份程式正常學習）。

---

## 8. 工程備註（可重現性紀錄）

| 項目 | 處置 |
| --- | --- |
| `stage0_train.py` | 修正 auditability import 路徑（`sys.path`）1 處 → 跑通 |
| `oracle_fusion.py` | 修正 `StateSymbolizer` 初始化簽名 1 處 → 跑通 |
| `oracle_q.py`、`oracle_diagnostics.py`、`supplement_no307.py` | 一次跑通，無修正 |
| `supplement_no307_full.py` | **未改程式**。首次背景執行於約 9 分鐘時被執行環境無聲回收（無 traceback、無 OOM、stderr 為空）；原封不動重跑一次，1,478.6 秒完成 |

「無聲回收」的判斷依據：同一份程式形狀的主要 20 分鐘執行正常完成；記憶體檢查時仍有 5.5 GB 可用；無 `dmesg` OOM 紀錄；中止時點與一次 context compaction 事件重合。

### 打包進發表包時修復的換行問題

將本分支納入發表包時，發現並修復了一個會**破壞雜湊驗證**的問題：

- **現象**：repo 設定為 `* text=auto` 且 `core.autocrlf=true`，檢出時會把 LF 改寫為 CRLF。本分支所有產物都是 bare LF，因此新 clone 出來的 `trajectory.jsonl` 會與 `metadata.json` 記錄的 `trace_sha256` 對不上。
- **實測**：以 `git checkout-index` 模擬檢出，`trajectory.jsonl` 的 SHA-256 由 `109a5b55…` 變成 `4ecd9dc3…`——驗證確實會失敗。
- **修復**：在 `.gitattributes` 加入 `oracle_headroom/** -text`，將整棵樹設為位元組精確。本分支是凍結的證據包而非持續編輯的原始碼，因此不需要換行正規化。
- **驗證**：再次模擬檢出後，**88 / 88 個檔案位元組完全一致**，`metadata.json` 的 32 項檔案層級雜湊全部通過。

完整時序紀錄：[`EXPERIMENT_LOG.md`](https://github.com/cyrilliu1974/AuditableRL/blob/main/oracle_headroom/EXPERIMENT_LOG.md)

---

## 9. 結論

1. **預註冊判定成立**：`H/T = 0.306 < 1/3` → STOP（borderline，CI 上界 0.337）。
2. **完美裁判只補回 31% 的缺口**；69% 在表示層。
3. **拿掉一個塌陷 bank 後，完美裁判反而輸給信心仲裁**（`H/T = −0.079`）；7 個健康 bank 在 43,510 覆蓋步中僅 5 次衝突——**仲裁問題是空集合**。
4. **真正的機制是「頻率 ≠ 品質」**：塌陷的確定性策略產出高信心爛規則，而信心優先的仲裁會主動選中它們。這是 bank 集合的穩健性失敗。
5. **路線修正**：放棄「超強 GNN/RL 裁判」，轉向**提升網格（symbolizer 離散化）與協議（規則誘導／准入／bank 健康檢查）**。
6. **淨效果**：一次便宜的診斷實驗，擋掉了一條昂貴且難以歸因的錯誤投資，並把後續工作從開放式研究工程降維成收斂的實驗設計。**工作反而變簡單了。**

---

## 附錄 A：完整數字對照表

| 指標 | 含 307（主要，預註冊） | 拿掉 307（exploratory） |
| --- | ---: | ---: |
| `R_best`（best_single） | 500.00 | 500.00 |
| `R_conf`（信心融合） | 59.68 | 335.98 |
| `R_oc`（完美裁判） | 194.59 | 322.99 |
| `R_of`（完美動作） | 364.60 | 450.87 |
| `R_ens`（logit ensemble） | 293.83 | — |
| 總缺口 `T` | 440.32 | 164.02 |
| 仲裁缺口 `H` | 134.91 | −12.99 |
| `H/T` | **0.3064** | **−0.0792** |
| `H/T` 95% CI | [0.2739, 0.3373] | — |
| 集體同意但選錯 | 170.01（39%） | 127.88（78%） |
| 覆蓋／備援缺口 | 135.40（31%） | 49.13（30%） |
| 衝突步數 | 800 | **5** |
| 覆蓋步數 | 5,264 | 43,510 |
| 盲區步數 | 704 | 1,577 |
| 判定 | **STOP** | 仲裁無效 |

來源：[`results/oracle_diagnostics.json`](https://github.com/cyrilliu1974/AuditableRL/blob/main/oracle_headroom/results/oracle_diagnostics.json)、[`results/supplement_no307.json`](https://github.com/cyrilliu1974/AuditableRL/blob/main/oracle_headroom/results/supplement_no307.json)、[`results/supplement_no307_full.json`](https://github.com/cyrilliu1974/AuditableRL/blob/main/oracle_headroom/results/supplement_no307_full.json)

## 附錄 B：8 個 seed 的訓練結果

| seed | 訓練平均報酬 | 規則數 | `actor_sha256`（前 16 碼） |
| --- | ---: | ---: | --- |
| 11 | 151.74 | 100 | `e3f6d277f229a33c` |
| 29 | 199.67 | 113 | `ed0d9bb75e17f7db` |
| 43 | 267.46 | 131 | `41aec624ad269174` |
| 71 | 179.08 | 98 | `330093982cfcf8ef` |
| 101 | 231.19 | 123 | `e3b07f09a3144a86` |
| 149 | 345.52 | 146 | `a849595440d35c4c` |
| 211 | 192.72 | 107 | `0802457a2612b47f` |
| **307** | **9.74** | **33** | `17ea6a98d727a900` |

來源：[`runs/CartPole-v1/stage0_manifest.json`](https://github.com/cyrilliu1974/AuditableRL/blob/main/oracle_headroom/runs/CartPole-v1/stage0_manifest.json)

### 判讀注意：`actor_sha256` 不是 `actor.pt` 的檔案摘要

這一項容易誤判，特別說明。`actor_sha256` 是**模型 state dict 的序列化摘要**，來自：

```python
def actor_state_sha256(actor: nn.Module) -> str:
    buffer = io.BytesIO()
    torch.save(actor.state_dict(), buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()
```

（`auditability/grammar_audit_experiment.py`）

而磁碟上的 `actor.pt` 是一個 PyTorch zip 容器（`torch.save` 的完整物件）。兩者是**不同的序列化**，摘要天生不同。因此：

| 欄位 | 是否為對應檔案的位元組摘要 | 可否用 `sha256sum` 直接驗證 |
| --- | --- | --- |
| `trace_sha256` → `trajectory.jsonl` | 是 | 可以 |
| `training_updates_sha256` → `training_updates.jsonl` | 是 | 可以 |
| `training_checkpoint_sha256` → `training_checkpoint.pt` | 是 | 可以 |
| `grammar_sha256` → `induced_grammar.json` | 是 | 可以 |
| **`actor_sha256` → `actor.pt`** | **否（是 state dict 摘要）** | **不可以** |

若用 `sha256sum actor.pt` 去比對 `metadata.json` 的 `actor_sha256`，**一定不會相等**。這是欄位語意問題，不是資料損毀。8 個 seed 的 `actor.pt` 檔案本身在儲存與搬運過程中皆保持位元組不變（已驗證）。

**建議後續修正**：將該欄位更名為 `actor_state_sha256`，或另外補上 `actor_file_sha256` 以消歧義。

---

*本報告由 Oracle Headroom 分支的預註冊實驗、技術判定報告與實驗日誌整理而成。所有數字的權威來源為 `results/` 下的 JSON 檔；本報告不引入任何未在該處記錄的數字。*
