**DJ Oracle：Diffusion Model 的可證明最佳安全認證**

**問題定義**
- 提出 diffusion safety oracle 問題：給定 diffusion model、text prompt、safety classifier、likelihood 門檻，判定是否存在 initial latent noise 能生成 unsafe 圖片
- 對標 LLM Jailbreak Oracle（Boa），填補 diffusion model 端的空白
- Oracle 回傳兩種結果：Sat（找到漏洞，附帶具體 witness）或 Unsat（在指定範圍內證明安全，附帶數學誤差上界）
- 與現有 adversarial attack 的關鍵差異：攻擊只回答「我找到了一個」，oracle 回答「是否存在」

**技術挑戰**
- Diffusion model 的 latent space 維度極高（SD v1.5 為 16384 維），暴力搜索不可行
- 與 LLM oracle 的本質不同：LLM 是離散 token space 的 tree search，diffusion 是連續空間的全域優化
- 一般情況下等價於高維非凸優化，屬於 intractable 問題

**核心方法**
- 利用已有頂會論文（LOCO Edit / NeurIPS 2024、Asyrp / ICLR 2023）驗證的低維語意結構，將搜索從全空間降至 m 維子空間
- 透過 gradient PCA 找出影響 safety score 的關鍵方向，每個 prompt 獨立計算
- 在子空間內套用 Piyavskii-Shubert 演算法進行 Lipschitz 全域優化：每評估一個點就縮小未知區域，逐步逼近最佳解

**理論貢獻**
- 證明在 Lipschitz 假設 + 低維子空間假設下，B 次評估後的近似誤差為 O(L·r·B^{−1/m} + δ)
- 證明此 B^{−1/m} 的收斂速率是 information-theoretically optimal，任何演算法在相同條件下都不可能更快
- 分析問題的 computational hardness，說明全空間 oracle 的 intractability
- Unsat 的保證不只是「沒找到」，而是有數學上界的安全認證

**Threat Class 分層**
- T₁（高似然威脅）：只搜索正常使用中可能被抽到的 noise 區域
- T₂（Prompt 對齊威脅）：只搜索與 prompt 語意相關的子空間
- T₃（單一概念威脅）：針對特定 safety category（如裸露）做認證，維度最低，bound 最 tight

**實驗設計**
- 在未防禦模型（SD v1.5、SDXL）上驗證低維假設是否成立
- 在 concept erasure 模型（ESD、SalUn）上執行 oracle，認證防禦是否真正有效
- 與 pure gradient ascent、random search 對比，展示 provably optimal 搜索的優勢
- 量測實際 Lipschitz constant L 與投影誤差 δ，驗證理論 bound 的 tightness
- 支援使用者自定義 safety classifier（NudeNet、NSFW detector 等），oracle 與 classifier 解耦

**應用場景**
- 模型部署前的安全認證：量化特定 prompt 下的 unsafe generation 風險
- 防禦機制的有效性驗證：concept erasure 做完後用 oracle 檢查是否還有漏網之魚
- 不同模型版本的安全性比較：用統一框架橫向對比
- Safety benchmark 的標準化：取代現有 ad hoc 的攻擊成功率指標