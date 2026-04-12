**DJ Oracle: Dual-Judge Safety Certification for Generative Models**

**問題定義**
- Diffusion safety oracle：給定 diffusion model、text prompt、safety classifier、threshold τ，輸出兩類認證
- T₁ 認證（統計）：$\Pr_{z \sim N(0,I)}[f(z) > \tau]$ 的上界
- T₂ 認證（對抗）：$\Pr_{z_0 \sim P}[\text{attack from } z_0 \text{ succeeds}]$ 的下界
- 不追求全域 Sat/Unsat（在 m ≈ 20 的有效維度下 Lipschitz 搜索需 $10^{40}$ queries，不可行）
- 參照密碼學哲學：不證「絕對安全」，證「特定 threat model 下的可量化保證」

**T₁：統計認證**
- Method：Monte Carlo 抽 B 個 $z \sim N(0,I)$，計算 unsafe 比例 $\hat{p}$
- Bound：Clopper-Pearson 95% 上界 $U_{\text{CP}}(k, B)$
- 解讀：「在自然使用下，unsafe 率以 95% 信心 ≤ $U_{\text{CP}}$」
- B=1000 時，$k=0$ 給 $U_{\text{CP}} \approx 0.0037$；$k=100$ 給 $U_{\text{CP}} \approx 0.12$

**T₂：對抗認證**
- Attack：VAE-backprop PGD，loss = CLIP similarity to `["a nude person", "a naked body", "bare skin"]`
- 梯度流過 UNet + VAE + CLIP；PGD 更新：$z \leftarrow z - \eta \cdot \text{sign}(\nabla_z \mathcal{L})$
- 投影回 typical set：$z \leftarrow z / \|z\| \cdot \sqrt{d}$
- 半徑限制：$\|z - z_0\| \le r$
- 成功判定：任一步 NudeNet score > τ

**z₀ 與 r 的選擇**
- z₀ 分層採樣：按 T₁ score 分 4 層 [0.0, 0.05, 0.1, 0.3]，每層 i.i.d. 抽 K 個
- r grid：{2, 5, 10, 20}
- 報告 per-stratum Clopper-Pearson 下界 $L_j(r)$
- 最 punchy 陳述：以 $P = N(0,I) \mid f(z_0) \le 0.1$，給 $\Pr[\text{attack 成功}] \ge L$

**理論貢獻**
- T₁ bound：標準 CP exact binomial，tight for Bernoulli
- T₂ bound：attack success rate 的 CP 下界（對偶）
- Randomized smoothing local cert：對特定 z₀，certified radius $r^*$ 使攻擊失敗機率 ≥ $1 - \alpha$
- 不追求 Lipschitz global bound（已證明 m = 20 不可行）

**主要實驗**
- Hero table：SD v1.5 / ESD / UCE / SalUn 在同一 prompt 下的 $(T_1, T_2)$ 對比
  - Preliminary：SD v1.5 T₁=0.897, T₂=3/3；ESD T₁=0.110, T₂=3/3
- T₂ attack success vs r curves：比較各防禦的 certified radius
- I2P subset（50 prompts）的 $(T_1, T_2)$ batch 評估
- T₂ ablation：CLIP PGD vs random-direction（證明 attack 實作影響結論）

**核心 Finding 模板**
> 「模型 M 在 prompt p 下：T₁ upper 0.14（自然使用下看似安全），但 T₂ lower 0.83（給定 ESD 自己認為安全的起點，攻擊仍 83% 成功）。Concept erasure 是 statistical cover，不是 adversarial barrier。」

**應用場景**
- Concept erasure 方法的 rigorous audit
- Safety benchmark 標準化：取代 ad hoc ASR，用 $(T_1, T_2)$ 雙指標
- 防禦方法設計指標：同時最小化兩個 bound
