# 研究計畫：VJO -- Safety Radius for LLM Jailbreak Evaluation

**目標會議**: NeurIPS 2025
**論文標題**: *The Safety Radius: A Certified Metric for LLM Jailbreak Vulnerability*

---

## 1. 一句話摘要

> 我們定義 safety radius sigma(p)，一個可計算的 continuous metric，量化 LLM 在特定 prompt 下的越獄脆弱程度。類比 randomized smoothing 的 certified radius，sigma 把 LLM 安全評估從 binary 的 "pass/fail" 提升為 quantitative 的 "how safe"。

---

## 2. 核心定位 (Pitch)

### 核心問題

現有的 jailbreak 評估只給你一個 binary 答案: ASR = X%。但這告訴你的是「多少 prompt 被攻破」，而不是「每個 prompt 有多安全」。你無法區分「差一點就被攻破」和「完全不可能被攻破」的 prompt。

### Safety Radius: 一個數字說清安全程度

**sigma(M, p, D) = 所有 jailbreak response 中機率最高的那個的機率。**

- sigma 大 (接近 1): prompt 很脆弱，高機率 jailbreak 存在
- sigma 小 (接近 0): prompt 很安全，所有 jailbreak 都是極低機率事件
- sigma = 0: 不存在任何 jailbreak (certified safe)

這就像 randomized smoothing 的 certified radius:

| Randomized Smoothing | VJO |
|---|---|
| 給定圖片 x，certified radius = 最小對抗擾動距離 | 給定 prompt p，safety radius = 最高 jailbreak 機率 |
| radius 大 = 更安全 | sigma 小 = 更安全 |
| 一個 continuous metric 取代 binary robust/not-robust | 一個 continuous metric 取代 binary safe/unsafe |
| 用 Gaussian noise sampling 計算 | 用 probability-ordered search + binary search 計算 |

### 論文的三個 headline results

1. **Safety radius sigma(p)**: 新 metric，連續、可比較、有形式化定義 (Lean4 machine-checked)。讓 "Llama2 的中位 sigma 是 10^{-8}, Vicuna 是 10^{-1}" 這樣的陳述成為可能。
2. **VJO 能計算 sigma 的上下界**: SAT witness 給下界，Certified UNSAT 給上界。Binary search on epsilon 以 O(log(1/delta)) 收斂。
3. **sigma 驅動的新實驗**: safety radius distribution (violin plot per model), sigma vs decoding strategy, sigma-based deployment threshold. 全是之前不可能畫出的圖。

---

## 3. 完成進度

### DONE: 形式化層 (Lean4)

- [x] **F1: 核心定義** (`Defs.lean`)
  - Model, DecodingStrategy, responseProb, nTokenLikelihood, jailbreakThreshold, jailbreakOracleProblem
  - 所有定義通過型別檢查，無 sorry
- [x] **F2: 可靠性定理** (`Soundness.lean`)
  - `boa_soundness`: Sat witness 滿足雙條件
  - `likelihood_pruning_sound`: 前綴機率 < threshold 則延伸亦然
  - `responseProb_append_singleton/le`: responseProb 在延伸下單調不增
  - `priority_queue_well_founded`: 搜尋空間有限
  - `boa_sat_implies_oracle`: Sat witness 解決 oracle problem
- [x] **F3: 覆蓋率界** (`Coverage.lean`)
  - `totalAboveThreshold_finite`: above-threshold 集合有限
  - `coverage_lower_bound`: 給定 budget 的探索下界
  - `unsat_confidence`: 遺漏比例 <= delta
- [x] **F4: Block List 分析** (`BlockList.lean`)
  - `blocklist_completeness_gap`: block list 造成完備性缺口
  - `unblocked_responses_count`: 未封鎖回應數 = (|V|-|BL|)^n
  - `blocklist_missed_fraction`: 遺漏比例 = (1-p)^n
- [x] **F5: 最優搜尋** (`OptimalSearch.lean`) -- **核心貢獻**
  - `complete_pruning_sound`: 以 threshold(n_max) 剪枝保可靠
  - `complete_search_finds_all`: 所有 above-threshold jailbreak 的前綴不被剪枝（完備性）
  - `first_find_optimal`: 第一個找到的 jailbreak 機率最高（最優性）
  - `exploredSet_finite`, `explored_all_potentially_useful`: 搜尋集最小
  - `random_sample_exponential_decay`: BOA Stage 2 的效率問題量化
- [x] **F6: 次模覆蓋** (`Submodular.lean`)
  - `coverage_submodular`: jailbreak coverage 是 monotone submodular
  - `greedy_gap_decay`: greedy 逐步收斂
  - `one_minus_inv_pow_le_inv_e`: (1-1/k)^k <= 1/e
  - `greedy_achieves_one_minus_inv_e`: greedy 達 (1-1/e) 最優覆蓋
  - `vjo_coverage_guarantee`: VJO 覆蓋率保證
- [x] **F7: Safety Radius** (`SafetyRadius.lean`)
  - `safetyRadius`: 定義 (sSup of jailbreak response probs)
  - `jailbreakOracleProblem_monotone`: oracle problem 對 epsilon 單調
  - `unsat_implies_safety_bound`, `unsat_bounds_safetyRadius`: UNSAT 認證
  - `reachability_monotone`, `jailbreak_set_monotone`: decoding strategy 包含關係
  - `binary_search_sandwich`, `binary_search_halves`, `binary_search_converges`: binary search 正確性
  - `safetyRadius_le_one`, `safetyRadius_nonneg`, `safetyRadius_ge_of_jailbreak`, `safetyRadius_eq_zero_of_no_jailbreak`: 基本性質

### DONE: 系統層 (Python)

- [x] **核心搜尋** (`VJO/search.py`): probability-ordered complete search, priority queue
- [x] **vLLM 後端** (`VJO/backend.py`): TokenOracle + SubstringJudge
- [x] **模型配置** (`VJO/config.py`): 5 模型 (Vicuna, Llama2, Llama3, Llama3.1, Qwen3)，匹配 BOA Table 1
- [x] **實驗框架** (`VJO/experiment.py`): Modal GPU runner, VJO vs BOA 比較
- [x] **實驗腳本** (infrastructure ready, 尚未跑出結果)
  - `experiments/run.py`: Vicuna 快速比較
  - `experiments/completeness.py`: Qwen 上 certified UNSAT demo
  - `experiments/benchmark.py`: LLM judge benchmark
  - `experiments/full_vocab.py`: full-vocabulary certified search

---

## 4. TODO (按優先順序)

### P0: 實驗 (blocking for submission)

- [ ] **Judge 升級**: 整合 Llama-Guard-3 (或 HarmBench classifier) 作為 judge，取代 substring matching
- [ ] **完整 prompt 集**: 90 prompts (JBB 30 + AdvBench 30 + AgentHarm 30)，與 BOA 對齊
- [ ] **BOA 完整 baseline**: 實作 BOA Stage 3 (priority search)，不能只比 Stage 2 (random sampling)
- [ ] **主實驗**: 5 models x 90 prompts x 1500s budget, VJO vs BOA (Stage 2+3)
  - 指標: ASR, certified UNSAT rate, nodes expanded, time to first SAT, witness log-prob
- [ ] **Certification curve**: x = epsilon (1, 1e-1, ..., 1e-4), y = certified UNSAT fraction, 每個 model 一條線
- [ ] **Witness quality**: 比較 VJO witness vs BOA witness 的 log-prob (VJO 應嚴格 >= BOA)
- [ ] **Budget efficiency**: coverage vs nodes expanded, 展示 VJO 不浪費算力
- [ ] **Full-vocab certification**: 在 n_max=20,30 上跑 `full_vocab.py`, 報告可行性

### P1: 新理論 (強化貢獻)

- [x] **Safety radius** (`SafetyRadius.lean`): 定義 safetyRadius = sSup of jailbreak response probs
  - [x] Lean 形式化 (`safetyRadius`)
  - [x] Monotonicity: UNSAT at epsilon_0 implies UNSAT at epsilon_1 >= epsilon_0 (`jailbreakOracleProblem_monotone`)
  - [x] UNSAT certification: UNSAT bounds all jailbreak probs (`unsat_implies_safety_bound`, `unsat_bounds_safetyRadius`)
  - [x] Bounded: 0 <= safetyRadius <= 1 (`safetyRadius_nonneg`, `safetyRadius_le_one`)
  - [x] Jailbreak witness lower bounds safetyRadius (`safetyRadius_ge_of_jailbreak`)
  - [x] No jailbreak implies safetyRadius = 0 (`safetyRadius_eq_zero_of_no_jailbreak`)
  - [x] Binary search correctness: sandwich, halving, convergence (`binary_search_sandwich`, `binary_search_halves`, `binary_search_converges`)
  - [x] Decoding strategy reachability monotonicity (`reachability_monotone`, `jailbreak_set_monotone`)
- [ ] **Telescopic certification theorem**: (D1, epsilon1) 的 certificate 推廣到 (D2, epsilon2), 當 D2 更 restrictive 且 epsilon2 >= epsilon1。Lean 形式化。
- [ ] **實作 binary search**: Python 端實作 binary search on epsilon 來計算 safety radius

### P2: Lean 清理 (supplementary quality)

- [ ] 減少 axiom 數量: 將 `nTokenLikelihood` 從 axiom 改為 constructive definition (discrete finite-sum)
  - `nTokenLikelihood_nonneg`, `nTokenLikelihood_le_one`, `nTokenLikelihood_antitone` 也應改為 lemma
  - `Judge` 保留 axiom (天生為 oracle)
  - 目前 axiom 數: 5 (Judge x1, nTokenLikelihood x4), 目標: 2 (Judge x1, nTokenLikelihood x1 or 0)
- [ ] 修復 warnings: deprecated `push_neg` -> `push Not` (SafetyRadius.lean:231,259), unused variables
- [ ] 補齊所有 public theorem 的 docstring

### P3: 論文撰寫

- [ ] **Section 1 (Intro)**: 現有評估只有 ASR (binary)。我們提出 safety radius sigma (continuous)。類比 randomized smoothing。
- [ ] **Section 2 (Safety Radius)**: 定義 sigma(M,p,D)。性質: 0 <= sigma <= 1, monotone in epsilon, monotone in decoding restrictiveness。形式化在 Lean4。
- [ ] **Section 3 (Computing sigma)**: VJO 演算法。Complete pruning 保證不漏。First-find optimality 保證最高機率 witness。Binary search on epsilon 收斂到 sigma。(1-1/e) budget efficiency。
- [ ] **Section 4 (Formal Guarantees)**: Lean4 定理清單 + 證明 sketch。Axiom 說明。Block list completeness gap 作為 honest negative result。
- [ ] **Section 5 (Experiments)**:
  - Table 1: sigma distribution per model (median, 95th pct, % below threshold)
  - Figure 1: sigma violin plot per model
  - Figure 2: sigma vs decoding strategy (top-k/top-p sweep)
  - Figure 3: binary search convergence (uncertainty interval vs iterations)
  - Figure 4: VJO vs BOA comparison (ASR + certified UNSAT rate + time)
  - Table 2: full-vocab certification feasibility
- [ ] **Section 6 (Discussion)**: limitations, PAC connection, adaptive budget, frontier models
- [ ] **Supplementary**: 完整 Lean source, (1-1/e) bound 證明, 所有實驗細節, prompt lists

---

## 5. Reviewer 預判與應對

### 預期正面

1. **New metric**: safety radius sigma(p) 是全新的 continuous metric，比 ASR 更 informative。Reviewer 看到 violin plot 和 sigma distribution 會覺得有新東西。
2. **Clean analogy**: randomized smoothing 的類比直覺且準確。Reviewer 秒懂 positioning。
3. **Formal rigor**: Lean4 machine-checked proofs，不是 paper proofs。48 個定理，零 sorry。
4. **Actionable**: sigma-based deployment threshold ("上線前要求 99% prompt 的 sigma < 10^{-6}") 是產業界能直接用的。

### 預期攻擊點與應對

| 攻擊點 | 應對 |
|--------|------|
| sigma 依賴 Judge J，Judge 是 axiom | 與 certified radius 依賴 base classifier 相同。Judge 是可替換的模組，sigma 的形式性質不依賴 J 的具體實作。 |
| sigma 計算代價高 (binary search * complete search) | 展示實際 runtime。Randomized smoothing 也需要數千次 forward pass。Binary search 只需 ~20 輪。 |
| top-k caveat 削弱 certificate | full-vocab search 消除此限制。同時展示兩種模式 (practical top-k + certified full-vocab)。 |
| BOA 比較不公平 (只比 Stage 2) | P0 中實作完整 BOA Stage 2+3 baseline。 |
| 只測 7B/8B 模型 | 明確列為 limitation。sigma 是 model-agnostic metric，框架可擴展。 |
| 演算法太 straightforward | 貢獻是 metric + framework，不是演算法。Randomized smoothing 的演算法也很簡單 (加 noise + 投票)，但 certified radius 這個概念改變了整個領域。 |
| Submodularity 連結不緊密 | (1-1/e) bound 回答 "有限算力下 sigma 的估計品質"。是 budget-constrained 版本的保證。 |

---

## 6. 資源需求

| 資源 | 規格 | 用途 |
|------|------|------|
| GPU | 1-2x A100-80GB (Modal) | 主 benchmark + full-vocab search |
| 預估 GPU 時數 | ~200h (5 models x 90 prompts x ~25min) | 主實驗 |
| Lean4 + Mathlib | 鎖定版本 | 證明編譯 |

---

## 7. 參考資源

- BOA 論文: arXiv:2506.17299
- Randomized smoothing: Cohen, Rosenfeld, Kolter (ICML 2019)
- JailBreakBench: https://github.com/JailbreakBench/jailbreakbench
- vLLM: https://github.com/vllm-project/vllm
- Lean4: https://github.com/leanprover/lean4
- Mathlib4: https://github.com/leanprover-community/mathlib4
