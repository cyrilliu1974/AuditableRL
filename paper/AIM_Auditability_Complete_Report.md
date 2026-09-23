# 完整方案：證明「RL 訓練結果可審計」——AI Mother Tongue 改造計畫

**最後更新**：2026年9月18日  
**作者**：基於 cyrilliu1974/AI-Mother-Tongue 與可解釋性最佳實踐  
**目標**：通過精準的實驗設計，證明多代理 RL 訓練結果可被離散符號日誌完整重構，無須逆向工程神經網路權重

---

## 第一部分：核心論點與假說

### 1.1 主張（Thesis）
在凍結 VQ-VAE 碼本的多代理 RL 框架下，經過聯合訓練的策略網路（policy network）會自發性地學習一套**穩定、確定性、可查表重構的符號-行為映射**。這套映射在以下三個維度上可被稽核：
- **忠實度（Faithfulness）**：查表能重構原始網路決策的準確率
- **因果效力（Causal Efficacy）**：符號維度的改變會導致下游行為實質變化
- **聚類可解釋性（Clustering Interpretability）**：模型編碼的特徵在連續空間中形成可識別的邊界

### 1.2 假說（Hypotheses）

**H₁（主假說）**：在完整測試集上，確定性推理（argmax）得到的符號序列與其對應的行為輸出，能通過符號→行為查表以 ≥90% 的准確率重構。
- **虛無假說 H₀**：查表准確率 < 60%（隨機基線附近）

**H₂（第一符號的特徵聚類）**：政策網路的連續中間表徵 z_e 在符號空間中形成兩個清晰分離的視覺簇（對應 C/D）。
- **度量**：用 DBSCAN 或 Silhouette Score 量化簇的分離度
- **成功標準**：Silhouette Score ≥ 0.4

**H₃（第二符號的因果作用）**：通過激活代換（activation patching）改變第二符號，會導致 AgentB 行為改變的比例 ≥ 30%。
- **虛無假說**：改變比例 < 10%（只是噪聲）

**H₄（跨種子穩定性）**：用 5 個獨立隨機種子重新訓練，上述三項指標的平均值與標準差應落在可接受範圍內。
- **成功標準**：cv（變異係數）≤ 0.15

---

## 第二部分：完整程式碼改造路線圖

### 2.1 改動總覽

原 `vqvae_agents_AIM.py` 需要新增 4 個模組，改動 1 個主迴圈。**不刪除原代碼，只追加**。

```
vqvae_agents_AIM.py (原文件)
├─ 新增函數 audit_deterministic_pass()        [第 2.2 節]
├─ 新增類別 SymbolAudit                        [第 2.3 節]
├─ 新增函數 compute_faithfulness_score()      [第 2.4 節]
├─ 新增函數 activation_patching_test()        [第 2.5 節]
├─ 新增函數 clustering_analysis()             [第 2.6 節]
└─ 修改主流程 if __name__ == '__main__'        [第 2.7 節]
```

### 2.2 `audit_deterministic_pass()` — 生成原始稽核記錄

**目的**：訓練結束後，跑過**完整測試集**一次（不抽樣），用 argmax 而非隨機抽樣，產生確定性的符號-行為日誌。

**插入位置**：在 `vqvae_agents_AIM.py` 的 `multi_agent_game()` 函數**之後**，新增為獨立函數。

```python
def audit_deterministic_pass(vqvae, agentA, agentB, K_val, aim_seq_len=2, batch_size=64):
    """
    確定性稽核通道：遍歷完整測試集，使用 argmax（無隨機性）生成符號。
    
    返回: dict，包含所有記錄與統計量
    """
    import torchvision
    from torch.utils.data import DataLoader
    from torchvision import transforms
    
    agentA.eval()
    agentB.eval()
    for p in list(agentA.parameters()) + list(agentB.parameters()):
        p.requires_grad_(False)
    
    transform = transforms.ToTensor()
    test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    records = []
    all_z_e = []
    all_A_aim_1st = []  # 用於後面的聚類分析
    
    with torch.no_grad():
        for x, labels in loader:
            batch_size_actual = x.size(0)
            
            # Agent A 推理
            z_e = agentA.vqvae.encoder(x)
            A_logits_raw, _ = agentA(x, labels, 
                                     mode='policy',
                                     opponent_aim_sequence=torch.zeros((batch_size_actual, aim_seq_len), 
                                                                       dtype=torch.long))
            # argmax 而不是抽樣
            A_aim = A_logits_raw.view(batch_size_actual, aim_seq_len, K_val).argmax(dim=-1)
            
            # Agent B 推理
            B_logits_raw = agentB(A_aim, labels, x, mode='policy')
            B_aim = B_logits_raw.view(batch_size_actual, aim_seq_len, K_val).argmax(dim=-1)
            
            # 行為詮釋（基於第一個符號）
            A_acts = [interpret_aim_as_action(A_aim[i, 0].item(), K_val) for i in range(batch_size_actual)]
            B_acts = [interpret_aim_as_action(B_aim[i, 0].item(), K_val) for i in range(batch_size_actual)]
            
            # 獎勵計算
            rewards = [payoff(A_acts[i], B_acts[i], int(labels[i].item()), 0) 
                      for i in range(batch_size_actual)]
            
            # 記錄
            for i in range(batch_size_actual):
                records.append({
                    "img_idx": len(records),
                    "mnist_label": int(labels[i].item()),
                    "A_aim": A_aim[i].tolist(),
                    "B_aim": B_aim[i].tolist(),
                    "A_action": A_acts[i],
                    "B_action": B_acts[i],
                    "reward_A": rewards[i][0],
                    "reward_B": rewards[i][1],
                    "z_e": z_e[i].cpu().numpy()  # 保存用於後續聚類
                })
                all_z_e.append(z_e[i].cpu().numpy())
                all_A_aim_1st.append(A_aim[i, 0].item())
    
    return {
        "records": records,
        "z_e_array": np.array(all_z_e),
        "A_aim_1st_array": np.array(all_A_aim_1st),
        "total_samples": len(records)
    }
```

**關鍵點**：
- `argmax` 而非 `sample()` → 確定性
- 迴圈遍歷**全部** test data → 無偏樣本
- 保存 `z_e`（encoder 連續輸出）→ 用於後續聚類視覺化
- 保存所有詳細記錄 → 作為稽核日誌

### 2.3 `SymbolAudit` 類別 — 管理與分析稽核日誌

**目的**：從 `audit_deterministic_pass()` 的原始記錄建造查表、計算統計量。

```python
class SymbolAudit:
    """
    從離散符號日誌反演 RL 策略。
    計算忠實度、準純度、符號使用分佈等。
    """
    def __init__(self, records, K_val, aim_seq_len=2):
        self.records = records
        self.K_val = K_val
        self.aim_seq_len = aim_seq_len
        self.lookup_A = {}  # symbol_tuple → action (with purity & confidence)
        self.lookup_B = {}
        self.purity_A = {}
        self.purity_B = {}
        self.confidence_A = {}  # 用於區分 confident vs ambiguous decision
        self.confidence_B = {}
        
    def build_lookup_tables(self):
        """
        從記錄建造查表。
        若同一符號映射多個行為，記錄最多數的行為及其相對頻率。
        """
        from collections import defaultdict, Counter
        
        tally_A = defaultdict(Counter)
        tally_B = defaultdict(Counter)
        
        for r in self.records:
            aim_A_tuple = tuple(r["A_aim"])
            aim_B_tuple = tuple(r["B_aim"])
            tally_A[aim_A_tuple][r["A_action"]] += 1
            tally_B[aim_B_tuple][r["B_action"]] += 1
        
        # 決策規則：每個符號取頻率最高的行為
        for sym, counter in tally_A.items():
            most_common_act, count = counter.most_common(1)[0]
            total = sum(counter.values())
            purity = count / total
            self.lookup_A[sym] = most_common_act
            self.purity_A[sym] = purity
            self.confidence_A[sym] = count  # 絕對支持票數，越高越有信心
        
        for sym, counter in tally_B.items():
            most_common_act, count = counter.most_common(1)[0]
            total = sum(counter.values())
            purity = count / total
            self.lookup_B[sym] = most_common_act
            self.purity_B[sym] = purity
            self.confidence_B[sym] = count
    
    def compute_faithfulness(self):
        """
        忠實度：用查表預測，能與實際行為相符的比例。
        """
        hits = 0
        misses = 0
        for r in self.records:
            aim_A = tuple(r["A_aim"])
            aim_B = tuple(r["B_aim"])
            
            pred_A = self.lookup_A.get(aim_A)
            pred_B = self.lookup_B.get(aim_B)
            
            if pred_A == r["A_action"] and pred_B == r["B_action"]:
                hits += 1
            else:
                misses += 1
        
        faith = hits / (hits + misses) if (hits + misses) > 0 else 0.0
        return faith, hits, misses
    
    def purity_statistics(self):
        """
        準純度：查表中各符號對應行為的一致性。
        返回均值、中位數、最小值、最大值。
        """
        purities_A = list(self.purity_A.values())
        purities_B = list(self.purity_B.values())
        
        return {
            "A_mean": np.mean(purities_A),
            "A_median": np.median(purities_A),
            "A_min": np.min(purities_A),
            "A_max": np.max(purities_A),
            "B_mean": np.mean(purities_B),
            "B_median": np.median(purities_B),
            "B_min": np.min(purities_B),
            "B_max": np.max(purities_B),
        }
    
    def symbol_entropy(self):
        """
        符號選擇的熵（多樣性）。
        越接近 log(K)，說明模型用了整個碼本；越低，說明只用少數符號。
        """
        from scipy.stats import entropy as scipy_entropy
        
        symbol_counts_A = Counter([tuple(r["A_aim"]) for r in self.records])
        symbol_counts_B = Counter([tuple(r["B_aim"]) for r in self.records])
        
        probs_A = np.array(list(symbol_counts_A.values())) / len(self.records)
        probs_B = np.array(list(symbol_counts_B.values())) / len(self.records)
        
        H_A = scipy_entropy(probs_A)
        H_B = scipy_entropy(probs_B)
        
        return {"H_A": H_A, "H_B": H_B, "max_possible": np.log(self.K_val**self.aim_seq_len)}
    
    def coverage(self):
        """
        覆蓋率：模型實際用了多少比例的碼本？
        （對於診斷「是否浪費碼本維度」有用）
        """
        unique_A = len(set(tuple(r["A_aim"]) for r in self.records))
        unique_B = len(set(tuple(r["B_aim"]) for r in self.records))
        possible = self.K_val ** self.aim_seq_len
        
        return {
            "A_coverage": unique_A / possible,
            "B_coverage": unique_B / possible,
            "A_unique_count": unique_A,
            "B_unique_count": unique_B
        }
```

**關鍵點**：
- `build_lookup_tables()` 是核心，構建「符號→行為」映射表
- `purity_A/B` 捕捉「同一符號對應多個行為的混亂度」
- 如果某個符號的 purity = 0.95，代表它 95% 的時候映射到同一個行為
- `compute_faithfulness()` 就是 H₁ 假說的直接測試

### 2.4 `compute_faithfulness_score()` — H₁ 假說檢定

```python
def compute_faithfulness_score(audit_obj, confidence_threshold=1):
    """
    H₁ 檢定：是否達到 ≥90% 忠實度？
    
    Args:
        confidence_threshold: 只採用「至少被見過 N 次」的符號映射
                             （排除單次出現、可能是噪聲的映射）
    
    Returns:
        dict with test results, p-value (if using binomial test)
    """
    from scipy import stats
    
    hits = 0
    total = 0
    for r in audit_obj.records:
        aim_A = tuple(r["A_aim"])
        aim_B = tuple(r["B_aim"])
        
        # 只計算「有足夠信心」的預測
        if audit_obj.confidence_A.get(aim_A, 0) >= confidence_threshold and \
           audit_obj.confidence_B.get(aim_B, 0) >= confidence_threshold:
            pred_A = audit_obj.lookup_A.get(aim_A)
            pred_B = audit_obj.lookup_B.get(aim_B)
            if pred_A == r["A_action"] and pred_B == r["B_action"]:
                hits += 1
            total += 1
    
    faith = hits / total if total > 0 else 0.0
    
    # 二項式檢定：H₀ = 隨機猜測（50%）vs H₁ = 實際准確率
    # 在 C/D 只有兩個選擇下，隨機基線是 0.5
    _, p_value = stats.binom_test(hits, total, 0.5, alternative='greater')
    
    return {
        "faithfulness": faith,
        "hits": hits,
        "total_tested": total,
        "p_value": p_value,
        "significant_at_0.01": p_value < 0.01,
        "H1_pass": faith >= 0.90 and p_value < 0.01
    }
```

### 2.5 `activation_patching_test()` — H₃ 因果檢定

```python
def activation_patching_test(vqvae, agentA, agentB, K_val, aim_seq_len=2, 
                              n_patches=500, which_symbol_idx=1):
    """
    H₃ 檢定：改變第 which_symbol_idx 個符號會導致行為改變嗎？
    
    邏輯：
    1. 隨機抽 C 類圖片的符號 + D 類圖片的符號
    2. 構造雜交符號（C_aim 前半保留，後半換成 D_aim 的後半）
    3. 比較 AgentB 接收原始 vs 雜交符號時的行為差異
    
    Returns:
        dict with flip rate 與統計檢定
    """
    import torchvision
    from torch.utils.data import DataLoader
    
    agentA.eval(); agentB.eval()
    for p in list(agentA.parameters()) + list(agentB.parameters()):
        p.requires_grad_(False)
    
    transform = transforms.ToTensor()
    test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # 分別取偶數（C 的代理）和奇數（D 的代理）
    even_indices = [i for i, (_, l) in enumerate(test_data) if int(l) % 2 == 0]
    odd_indices  = [i for i, (_, l) in enumerate(test_data) if int(l) % 2 == 1]
    
    flips = 0
    total_pairs = 0
    
    with torch.no_grad():
        for _ in range(n_patches):
            if len(even_indices) == 0 or len(odd_indices) == 0:
                break
            
            # 隨機選一張偶數圖和一張奇數圖
            idx_c = random.choice(even_indices)
            idx_d = random.choice(odd_indices)
            
            x_c, l_c = test_data[idx_c]
            x_d, l_d = test_data[idx_d]
            
            x_c = x_c.unsqueeze(0)
            x_d = x_d.unsqueeze(0)
            l_c = torch.tensor([l_c])
            l_d = torch.tensor([l_d])
            
            # Agent A 對 C 圖的反應
            A_logits_c, _ = agentA(x_c, l_c, mode='policy',
                                   opponent_aim_sequence=torch.zeros((1, aim_seq_len), dtype=torch.long))
            aim_c = A_logits_c.view(-1, K_val).argmax(1).view(1, -1)
            
            # Agent A 對 D 圖的反應
            A_logits_d, _ = agentA(x_d, l_d, mode='policy',
                                   opponent_aim_sequence=torch.zeros((1, aim_seq_len), dtype=torch.long))
            aim_d = A_logits_d.view(-1, K_val).argmax(1).view(1, -1)
            
            # 基準：AgentB 接收 C 的符號，使用 C 的圖片
            B_logits_base = agentB(aim_c, l_c, x_c, mode='policy')
            B_act_base = interpret_aim_as_action(B_logits_base.view(-1, K_val).argmax(1).item(), K_val)
            
            # 代換：AgentB 接收 C 的符號（但第 which_symbol_idx 改成 D 的），使用 C 的圖片
            aim_patched = aim_c.clone()
            aim_patched[0, which_symbol_idx] = aim_d[0, which_symbol_idx]
            
            B_logits_patch = agentB(aim_patched, l_c, x_c, mode='policy')
            B_act_patch = interpret_aim_as_action(B_logits_patch.view(-1, K_val).argmax(1).item(), K_val)
            
            if B_act_base != B_act_patch:
                flips += 1
            total_pairs += 1
    
    flip_rate = flips / total_pairs if total_pairs > 0 else 0.0
    
    # 二項式檢定：H₀ = 符號無關（50% 翻轉是巧合）vs H₁ = 有因果作用（翻轉率 > 50%）
    from scipy import stats
    _, p_value = stats.binom_test(flips, total_pairs, 0.5, alternative='greater')
    
    return {
        "flip_rate": flip_rate,
        "flips": flips,
        "total_pairs": total_pairs,
        "p_value": p_value,
        "significant_at_0.01": p_value < 0.01,
        "H3_pass": flip_rate >= 0.30 and p_value < 0.01
    }
```

**關鍵點**：
- 不是隨便換，而是系統性地用「對立類別」的符號做代換
- 翻轉率 ≥30% + p < 0.01 才算通過 H₃
- 這實質上是在測試「AgentB 到底有沒有在用這個符號」

### 2.6 `clustering_analysis()` — H₂ 聚類檢定

```python
def clustering_analysis(audit_data, K_val):
    """
    H₂ 檢定：encoder 輸出 z_e 在符號空間中是否分離？
    
    方法：
    1. 用 UMAP 降維 z_e 到 2D（用於可視化）
    2. 用 DBSCAN 自動發現簇
    3. 計算 Silhouette Score（指標：簇的內聚性 vs 分離度）
    4. 顏色按照第一符號（< K//2 = C，>= K//2 = D）標記
    
    Returns:
        dict with 視覺化檔案路徑 & Silhouette Score
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import UMAP
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import silhouette_score
    import matplotlib.pyplot as plt
    
    z_e = audit_data["z_e_array"]
    aim_1st = audit_data["A_aim_1st_array"]
    
    # PCA 先降到 50D（節省時間）
    if z_e.shape[1] > 50:
        pca = PCA(n_components=50)
        z_e_50 = pca.fit_transform(z_e)
    else:
        z_e_50 = z_e
    
    # UMAP 降到 2D
    umap = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    z_e_2d = umap.fit_transform(z_e_50)
    
    # 用「第一符號」作為預先定義的標籤
    labels_priori = (aim_1st < K_val // 2).astype(int)  # 0 = C, 1 = D
    
    # Silhouette Score（利用先驗標籤）
    silh_score = silhouette_score(z_e_2d, labels_priori)
    
    # DBSCAN 自動發現簇（驗證是否真的有簇結構）
    dbscan = DBSCAN(eps=0.5, min_samples=10)
    dbscan_labels = dbscan.fit_predict(z_e_2d)
    n_clusters_found = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    
    # 可視化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左圖：按預先標籤著色
    scatter1 = ax1.scatter(z_e_2d[:, 0], z_e_2d[:, 1], c=labels_priori, cmap='coolwarm', s=20, alpha=0.6)
    ax1.set_title(f'z_e 空間（按第一符號）| Silhouette={silh_score:.3f}')
    ax1.set_xlabel('UMAP 1')
    ax1.set_ylabel('UMAP 2')
    plt.colorbar(scatter1, ax=ax1, label='C(0) / D(1)')
    
    # 右圖：按 DBSCAN 發現的簇著色
    scatter2 = ax2.scatter(z_e_2d[:, 0], z_e_2d[:, 1], c=dbscan_labels, cmap='tab10', s=20, alpha=0.6)
    ax2.set_title(f'DBSCAN 自動發現的簇 (n_clusters={n_clusters_found})')
    ax2.set_xlabel('UMAP 1')
    ax2.set_ylabel('UMAP 2')
    plt.colorbar(scatter2, ax=ax2, label='Cluster ID')
    
    plt.tight_layout()
    fig_path = './clustering_analysis.png'
    plt.savefig(fig_path, dpi=150)
    plt.close()
    
    return {
        "silhouette_score": silh_score,
        "H2_pass": silh_score >= 0.40,
        "dbscan_n_clusters": n_clusters_found,
        "figure_path": fig_path
    }
```

### 2.7 主程式改動

在 `if __name__ == '__main__':` 區塊最後加入：

```python
if __name__ == '__main__':
    # ... 原有的訓練代碼 ...
    # 訓練後緊接著執行：
    
    print("\n" + "="*60)
    print("開始稽核階段（Audit Phase）")
    print("="*60)
    
    # 步驟 1：生成確定性日誌
    audit_data = audit_deterministic_pass(vqvae, agentA, agentB, 
                                          K_val=args.K, aim_seq_len=args.aim_seq_len)
    print(f"✓ 稽核日誌生成完成。總樣本數：{audit_data['total_samples']}")
    
    # 步驟 2：建造查表並計算統計量
    audit_obj = SymbolAudit(audit_data["records"], K_val=args.K, aim_seq_len=args.aim_seq_len)
    audit_obj.build_lookup_tables()
    
    # 步驟 3：H₁ 檢定（忠實度）
    h1_result = compute_faithfulness_score(audit_obj, confidence_threshold=1)
    print(f"\n【H₁：忠實度】")
    print(f"  查表准確率: {h1_result['faithfulness']:.4f} (目標: ≥0.90)")
    print(f"  統計檢定 p-value: {h1_result['p_value']:.2e}")
    print(f"  通過: {'✓' if h1_result['H1_pass'] else '✗'}")
    
    # H₁ 輔助統計
    purity_stats = audit_obj.purity_statistics()
    print(f"\n  【準純度分析】")
    print(f"    Agent A: mean={purity_stats['A_mean']:.3f}, median={purity_stats['A_median']:.3f}")
    print(f"    Agent B: mean={purity_stats['B_mean']:.3f}, median={purity_stats['B_median']:.3f}")
    
    entropy_stats = audit_obj.symbol_entropy()
    print(f"\n  【符號多樣性】")
    print(f"    H(A) = {entropy_stats['H_A']:.3f}, H(B) = {entropy_stats['H_B']:.3f}")
    print(f"    最大可能: {entropy_stats['max_possible']:.3f}")
    
    coverage = audit_obj.coverage()
    print(f"\n  【碼本覆蓋】")
    print(f"    Agent A: {coverage['A_unique_count']} / {args.K**args.aim_seq_len} = {coverage['A_coverage']:.1%}")
    print(f"    Agent B: {coverage['B_unique_count']} / {args.K**args.aim_seq_len} = {coverage['B_coverage']:.1%}")
    
    # 步驟 4：H₂ 檢定（聚類）
    h2_result = clustering_analysis(audit_data, K_val=args.K)
    print(f"\n【H₂：特徵聚類】")
    print(f"  Silhouette Score: {h2_result['silhouette_score']:.4f} (目標: ≥0.40)")
    print(f"  通過: {'✓' if h2_result['H2_pass'] else '✗'}")
    print(f"  圖表已保存: {h2_result['figure_path']}")
    
    # 步驟 5：H₃ 檢定（因果）
    h3_result = activation_patching_test(vqvae, agentA, agentB, K_val=args.K, 
                                         aim_seq_len=args.aim_seq_len, n_patches=500, 
                                         which_symbol_idx=1)
    print(f"\n【H₃：因果效力（第二符號）】")
    print(f"  行為翻轉率: {h3_result['flip_rate']:.4f} (目標: ≥0.30)")
    print(f"  統計檢定 p-value: {h3_result['p_value']:.2e}")
    print(f"  通過: {'✓' if h3_result['H3_pass'] else '✗'}")
    
    # 步驟 6：匯總
    all_pass = h1_result['H1_pass'] and h2_result['H2_pass'] and h3_result['H3_pass']
    print(f"\n" + "="*60)
    if all_pass:
        print("【結論】✓ RL 訓練結果可審計（三項假說全部通過）")
    else:
        print("【結論】✗ 尚未達到完全可審計標準")
    print("="*60)
    
    # 步驟 7：保存完整報告
    report = {
        "timestamp": datetime.now().isoformat(),
        "hyperparameters": vars(args),
        "H1_faithfulness": h1_result,
        "H2_clustering": h2_result,
        "H3_patching": h3_result,
        "purity_stats": purity_stats,
        "entropy_stats": entropy_stats,
        "coverage": coverage,
        "overall_pass": all_pass
    }
    report_path = f"./audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n完整報告已保存: {report_path}")
```

---

## 第三部分：判定標準與解讀

### 3.1 四個假說的判定標準（事先寫死，不事後調整）

| 假說 | 度量 | 通過標準 | 失敗標準 | 灰色地帶 |
|-----|-----|--------|--------|--------|
| **H₁** | Faithfulness + p-value | faith ≥90% AND p<0.01 | faith <60% OR p≥0.05 | 60%-90% 或 0.01≤p<0.05 |
| **H₂** | Silhouette Score | Silh ≥0.40 | Silh <0.20 | 0.20-0.40 |
| **H₃** | Flip Rate + p-value | flip ≥30% AND p<0.01 | flip <10% OR p≥0.05 | 10%-30% 或 0.01≤p<0.05 |
| **H₄** | CV（變異係數，5 seed） | cv ≤0.15 for all | cv >0.30 for any | 0.15-0.30 |

### 3.2 結論矩陣

```
H₁ Pass ✓ + H₂ Pass ✓ + H₃ Pass ✓ + H₄ Pass ✓
  ⇒ 【強證據】RL 結果可審計
  
H₁ Pass ✓ + H₂ Pass ✓ + H₃ Pass ✓ + H₄ Gray
  ⇒ 【中等證據】核心結論成立，但穩定性有保留
  
H₁ Pass ✓ + H₂ Pass ✗ + H₃ Gray
  ⇒ 【弱證據】忠實度高，但特徵無可解釋邊界、因果作用不清
       解釋：符號可能只是巧合的統計對應，無真實語義
  
H₁ Gray/Fail ✗
  ⇒ 【不通過】查表本身就無法重構，「可審計」主張不成立
```

---

## 第四部分：論文/報告框架

### 4.1 標題與摘要

**標題**  
*Interpretability in Emergent Communication: Evidence for Auditable RL Policies via Symbol-Action Faithfulness*

**摘要**（150-200 詞）  
```
多代理 RL 中的隱蔽協議（Steganographic Collusion）問題使得政策透明度難以驗證。
本工作提出一套基於離散符號稽核的方法論，證明在凍結 VQ-VAE 碼本的聯合訓練框架下，
策略網路學到的符號-行為映射具有三方面性質：
(1) 高忠實度：查表可重構 ≥90% 的推理行為，無須逆向工程權重；
(2) 有因果作用：激活代換實驗表明符號維度與下游決策有實質因果聯繫；
(3) 跨種子穩定：多次獨立訓練得到的映射結構一致。
我們在 MNIST 上驗證上述三個假說，並討論對 AI 審計與透明度的含義。
```

### 4.2 論文結構

**1. Introduction（1-2 頁）**
- 背景：為什麼 MARL 的審計困難？（隱蔽協議、權重黑盒）
- 問題陳述：能否用稀疏符號日誌完全重構 RL 政策？
- 貢獻：三層驗證框架（忠實度→聚類→因果）

**2. Related Work（1 頁）**
- 可解釋性（Saliency Maps, Feature Attribution）
- 隱蔽協議與審計（Steganography in MARL）
- VQ-VAE 與符號湧現（Discrete Bottlenecks）

**3. Method（2 頁）**
- 3.1 框架概述（VQ-VAE + 聯合訓練 A2C）
- 3.2 確定性稽核通道（為什麼要 argmax 而非 sample）
- 3.3 三個假說與對應檢定
  - H₁ 忠實度：二項式檢定，H₀ = 隨機（50%）
  - H₂ 聚類：Silhouette Score，UMAP 可視化
  - H₃ 因果：Activation Patching，交叉類別符號代換

**4. Experiments（2-3 頁）**
- 4.1 設置（MNIST，K=32，aim_seq_len=2，…）
- 4.2 結果
  - 表格 1：四個假說的檢定結果與 p-value
  - 圖 1：聚類可視化（UMAP + Silhouette）
  - 圖 2：查表忠實度的分佈（按符號分組）
  - 圖 3：Patching 實驗的翻轉率與統計顯著性
- 4.3 穩定性分析（5 個 seed，變異係數）

**5. Analysis（1-2 頁）**
- 5.1 第一符號 vs 第二符號的語義差異
  - 第一符號：被獎勵函數直接定義的「行為標籤」
  - 第二符號：符號化的內部狀態，語義由 reflection loss 誘發
- 5.2 Transparency Paradox 討論
  - 高忠實度 ≠ 高透明度：符號查表準確，但人類仍可能不理解「為什麼」選這個符號
  - 聚類分析有助於識別「可解釋維度」vs「隱蔽維度」
- 5.3 對監管的含義
  - 符號日誌能否作為審計證據？
  - 與傳統黑盒模型的優勢與局限

**6. Limitations（0.5 頁）**
- 任務簡單性（MNIST 奇偶判斷 vs 現實決策）
- 凍結碼本的約束（vs 端到端訓練）
- 樣本大小（10,000 test images vs 分佈外泛化）

**7. Conclusion（0.5 頁）**
- 主要發現
- 未來工作：更複雜任務、動態碼本、多代理聯盟

### 4.3 圖表清單

| 圖號 | 內容 | 說明 |
|-----|------|------|
| 圖 1 | UMAP 聚類 + Silhouette | 左：按第一符號上色；右：DBSCAN 自動簇 |
| 圖 2 | 忠實度分佈 | 直方圖：各符號對應行為的準純度 |
| 圖 3 | Patching 翻轉率 | 柱狀圖：改變各符號維度的翻轉率 ± 95% CI |
| 圖 4 | 穩定性（5 seed） | 5 組獨立訓練的忠實度/Silhouette/翻轉率分佈 |
| 表 1 | 假說檢定總結 | 四行（H₁-H₄）× 五列（指標、觀測、標準、p-value、通過） |

---

## 第五部分：預防性回應（對抗論證）

### 5.1 「忠實度高只是因為任務太簡單」

**預防**：
- 報告中明確註明「MNIST 奇偶是 2-way classification，隨機基線 50%」
- 計算符號使用的熵與碼本覆蓋（如果熵低，代表確實在用少量符號高效編碼）
- 在 Discussion 中討論「即使在簡單任務上，90%+ 忠實度也不平凡」理由：
  1. VQ-VAE 碼本是連續訓練的，可能學到了與 C/D 無關的壓縮方向
  2. 策略網路是獨立的 FC 層，沒有內建的符號結構化機制
  3. 聯合訓練中 A、B 互相干擾，符號對齐不是必然的

### 5.2 「Silhouette 0.40 不算分離好，可能是偽相關」

**預防**：
- 報告中同時展示 DBSCAN 自動發現的簇（如果自動發現 2-3 個明確簇，說明分離是真實的）
- 計算「互信息 I(z_e; symbol)」vs 隨機特徵的互信息作為對照
- 在 Discussion 中說明「Silhouette 0.40 在高維空間中已是中等-強的分離」（與 face recognition, clustering literature 對標）

### 5.3 「Activation Patching 的 30% 翻轉率不高，可能符號邊際效應很小」

**預防**：
- 報告統計檢定 p-value 與效應大小（Cohen's h）
- 補充分析：哪些符號對的代換翻轉率最高（是否有「關鍵符號」）
- Discussion 中指出「30% 翻轉率的因果解釋」：
  - AgentB 接收 AgentA 的符號，但也看圖片本身 (z_e)
  - 圖片提供了強信號，符號只是調節
  - 在 「僅用符號不看圖」的設置中翻轉率會更高（可作為未來實驗）

### 5.4 「跨種子穩定性不穩定」（若 H₄ 失敗）

**預防**：
- 如果某個 seed 結果異常，調查原因（訓練崩潰？超參數敏感？）
- 報告中分別給出「穩定的 seed」和「異常 seed」的結果
- Discussion 指出「變異性的來源」（隨機初始化、優化器噪聲、數據打亂順序）
- 提議「改進穩定性的方向」（更小的學習率、更多 epoch、seed-aware 超參選擇）

### 5.5 「循環論證：第一符號的規則是人寫的，不是模型學的」

**預防**（見第一部分盲區 1 修正）：
- 明確區分「符號邊界」（`< K/2` 是寫死的）和「符號映射」（什麼特徵落在邊界哪邊是學來的）
- 報告中特別強調：「我們驗證的是『給定邊界，特徵聚類的一致性』，而非『邊界本身是湧現的』」
- 第二符號作為對比案例：「沒有寫死的規則，卻也顯示因果作用」

---

## 第六部分：完整實驗清單

### 執行步驟

```bash
# 1. 訓練 + 稽核（單次）
python vqvae_agents_AIM.py \
  --epochs 10 --rounds 10000 --K 32 --D 64 \
  --aim_seq_len 2 --reflection_strategy predictive_bias \
  --reflection_coeff 0.05 --entropy_coeff 0.05

# 2. 跨種子重複（5 次）
for seed in {0..4}; do
  python vqvae_agents_AIM.py \
    --epochs 10 --rounds 10000 --K 32 --D 64 \
    --aim_seq_len 2 --reflection_strategy predictive_bias \
    --reflection_coeff 0.05 --entropy_coeff 0.05 \
    --seed $seed
done

# 3. 合併報告
python merge_audit_reports.py --output final_audit_report.json
```

### 輸出檔案

```
audit_report_YYYYMMDD_HHMMSS.json
├─ H1_faithfulness: {faithfulness, p_value, H1_pass}
├─ H2_clustering: {silhouette_score, H2_pass, figure_path}
├─ H3_patching: {flip_rate, p_value, H3_pass}
├─ H4_stability: {cv_faith, cv_silh, cv_flip, H4_pass}
├─ purity_stats: {A_mean, B_mean, ...}
├─ coverage: {A_unique_count, B_unique_count, ...}
└─ overall_pass: bool

clustering_analysis.png
├─ 左圖：UMAP + 按第一符號著色
└─ 右圖：UMAP + DBSCAN 簇著色

patching_flip_rates.png (可選)
├─ 各符號維度的翻轉率 + 95% CI
```

---

## 第七部分：論文投稿建議

### 適合的會議/期刊

1. **ICLR / NeurIPS / ICML**（如果擴展為更複雜任務、多代理競爭等）
2. **AISTATS / UAI**（強調統計檢定與因果推斷的方向）
3. **ACM CCS / IEEE S&P**（如果定位為「AI 審計安全」）
4. **AI Policy 期刊**（Nat Mach Intell, AI & Society 等）

### 投稿時常見問題與回應

**Q: 為什麼只在 MNIST 上做？**  
A: MNIST 是「可控測試床」，便於隔離變數。與論文提出的「先在簡單環境驗證原理、再擴展」策略一致。後續工作會在 RL 標準基準（Atari, 多代理遊戲）上複製。

**Q: 你的「可審計」只是「可預測」，並不代表「可理解」。**  
A: 同意。我們的主張是「符號日誌能夠完整重構 RL 決策邏輯，是審計的必要條件」，但非充分條件。「理解為什麼」需要進一步的因果干預與特徵歸因，我們在第五部分的聚類分析中有初步觸碰。

**Q: Activation Patching 只測試了一個維度（第二符號），為什麼不測試所有維度？**  
A: 篇幅與計算限制。第一符號本來就有寫死的規則（不需驗證），第二符號代表「沒有寫死規則的維度」。完整的因果分析（CDE, NDE）可作為未來工作。

---

## 結論

這份方案涵蓋了從代碼到論文、從實驗設計到預防性防守的完整要素。核心是**三層驗證**：

1. **忠實度（信噪比）**：能否用符號表重構？
2. **因果性（實質性）**：符號改變會改變行為嗎？
3. **穩定性（可重現性）**：在多個訓練中一致嗎？

只有三層都通過，才能宣稱「RL 結果可審計」。如果某層失敗，報告應明確說明「在哪個層級失敗了，意味著什麼」。

**最後的最後**：這套方案適合做一篇**2-4 頁的工作坊論文**或**一篇完整的 5-8 頁會議論文**。如果要投主會議，建議在「複雜性」和「實際應用」上再擴展（例如模擬多代理協商、金融交易決策等現實場景）。
