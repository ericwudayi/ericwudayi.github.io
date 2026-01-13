---
layout: post
title: "Flow Matching 與相關生成模型資源整理"
date: 2026-01-13
categories: [research, generative-models]
---

以下整理 **conditional flow matching**、各種 **flow matching 類方法**，以及與之相近的 **mean/probability flow**、**diffusion/score-based** 模型資源。內容除了「連結 + 小結」，也補上核心數學形式與更深入的比較重點，方便閱讀 references 後建立整體心智模型。

## Flow Matching 與 Conditional Flow Matching

- **Flow Matching for Generative Modeling** (Lipman et al.)
  - 連結：https://arxiv.org/abs/2210.02747
  - 小結：提出以監督學習直接回歸連續流場（ODE 速度場）$v_\theta(x,t)$ 的方式來學生成模型；避免顯式求解 log-likelihood 或反向積分的複雜度，並可與 OT/bridge 類方法銜接。
  - 核心數學：令 $x_t$ 為由某個「耦合路徑」(coupling path) 連接 $x_0 \sim p_0$ 與 $x_1 \sim p_1$ 的中間狀態，flow matching 的訓練目標可寫為
    $$\nabla_\theta \mathbb{E}_{t \sim \mathcal{U}(0,1),\, x_t} \left\| v_\theta(x_t, t) - \dot{x}_t \right\|^2,$$
    其中 $\dot{x}_t$ 為路徑的真實速度場。推論時解 ODE：$\dot{x}_t = v_\theta(x_t,t)$。

- **Conditional Flow Matching** (Tong et al.)
  - 連結：https://arxiv.org/abs/2302.00482
  - 小結：在 flow matching 中引入條件分佈與對應的流場配對，提供更通用的條件式生成框架，並建立與 OT/bridge、diffusion 的關係。
  - 核心數學：針對條件變數 $c$，目標是回歸 $v_\theta(x_t,t,c)$。若路徑以條件耦合 $(x_0,x_1)\sim \pi(\cdot\mid c)$ 產生，則最小化
    $$\mathbb{E}_{c} \mathbb{E}_{t,x_t\mid c} \left\| v_\theta(x_t,t,c) - \dot{x}_t \right\|^2.$$

## Flow Matching 變體與近似方法

- **Rectified Flow** (Liu et al.)
  - 連結：https://arxiv.org/abs/2209.03003
  - 小結：用更簡化的線性插值與 ODE 回歸方式建立生成流；實作上更輕量，且可視為 flow matching 的特例/變體。
  - 核心數學：常用路徑設為線性插值 $x_t = (1-t)x_0 + t x_1$，則 $\dot{x}_t = x_1 - x_0$ 為常數，訓練回歸定向向量場。

- **Continuous Normalizing Flows (CNF) / Neural ODE**
  - 連結：https://arxiv.org/abs/1806.07366
  - 小結：以連續時間 ODE 建模可逆流，搭配 Neural ODE 設計，可視為 flow matching 的基礎工具之一。
  - 核心數學：定義 $\dot{x}_t = f_\theta(x_t,t)$，對應密度變化滿足
    $$\frac{d}{dt}\log p_t(x_t) = -\nabla \cdot f_\theta(x_t,t),$$
    是 flow matching 與 likelihood-based 連續流的橋樑。

- **FFJORD**
  - 連結：https://arxiv.org/abs/1810.01367
  - 小結：以可擴展的 CNF 訓練策略與 Hutchinson trace estimator 降低計算成本，常被視為 flow-based/continuous-flow 系列的實作基礎。
  - 核心數學：以隨機估計 $\nabla \cdot f_\theta$，用 Hutchinson trace estimator 近似 $\mathrm{tr}\left(\frac{\partial f}{\partial x}\right)$，降低 Jacobian 計算成本。

- **Schrödinger Bridge / Stochastic Control**
  - 連結：https://arxiv.org/abs/2106.01357
  - 小結：以隨機最優控制形式描述從噪聲到資料分佈的轉移；與 flow matching / diffusion 在連續時間建模上有緊密關係。

## Mean / Probability Flow 與 Diffusion / Score-based

- **Score-Based Generative Modeling through SDEs**
  - 連結：https://arxiv.org/abs/2011.13456
  - 小結：以隨機微分方程建模擴散；提供 **probability flow ODE**（可視為「平均流/mean flow」的確定性版本）來連接 diffusion 與 ODE 流建模。
  - 核心數學：對前向 SDE $dx = f(x,t)\,dt + g(t)\,dW_t$，對應 probability flow ODE 為
    $$dx = \left[f(x,t) - \frac{1}{2} g(t)^2 \nabla_x \log p_t(x)\right] dt,$$
    其中 $\nabla_x \log p_t(x)$ 由 score network 估計。

- **Denoising Diffusion Probabilistic Models (DDPM)**
  - 連結：https://arxiv.org/abs/2006.11239
  - 小結：經典 diffusion 模型，透過逐步加噪與反向去噪來生成樣本；與 flow matching 在連續/離散時間與損失設計上可互相對照。
  - 核心數學：離散前向擾動 $q(x_t\mid x_{t-1})$ 定義為高斯加噪；反向模型學習 $\epsilon_\theta(x_t,t)$ 或 $x_0$，以 ELBO/簡化損失訓練。

- **DDIM (Deterministic Diffusion Implicit Models)**
  - 連結：https://arxiv.org/abs/2010.02502
  - 小結：提供更快的 deterministic 取樣軌跡，可視為 diffusion 與 ODE 流之間的橋接方法。
  - 核心數學：在固定噪聲 schedule 下令隨機性項為 0，得到 deterministic sampling trajectory，可視為與 probability flow ODE 類似的離散化。

## 小提示：如何建立「Flow Matching × Diffusion」心智地圖

1. **Flow Matching**：用監督式回歸學「速度場」。
2. **Diffusion / Score-based**：用噪聲擾動學「score/梯度」，透過 SDE/ODE 生成。
3. **Probability/Mean Flow ODE**：把 stochastic 擴散轉成 deterministic ODE，便於與 flow matching 互相理解與轉換。

## 深度比較：假設、目標、可解釋性與數學差異

| 面向 | Flow Matching | Conditional Flow Matching | Rectified Flow | Diffusion / Score-based | Probability Flow ODE |
| --- | --- | --- | --- | --- | --- |
| 目標 | 回歸速度場 $v_\theta$ | 回歸條件速度場 $v_\theta(\cdot,c)$ | 回歸簡化速度場 | 回歸 score $\nabla \log p_t(x)$ 或噪聲 | 以 score 定義 ODE 速度場 |
| 路徑假設 | 自由選擇 coupling path | 條件 coupling path | 線性插值路徑 | 由固定噪聲 schedule | 由 SDE 對應 ODE |
| 損失形式 | $\|v_\theta - \dot{x}_t\|^2$ | $\|v_\theta(\cdot,c) - \dot{x}_t\|^2$ | $\|v_\theta - (x_1-x_0)\|^2$ | $\|\epsilon_\theta - \epsilon\|^2$ 或 score-matching | score 對應 ODE drift |
| 取樣方式 | ODE 解 | ODE 解（條件） | ODE 解 | SDE/ODE 解 | ODE 解 |
| 直覺 | 直接學生成動態 | 直接學條件生成動態 | 用最簡單動態近似 | 噪聲擾動再去噪 | 把隨機過程投影到平均流 |

### 讀 references 時可對照的關鍵問題

1. **路徑選擇**：是否明確定義 coupling path？是否依賴 OT/bridge 或簡單插值？  
2. **損失設計**：是否等價於某種 score-matching 或 likelihood 變分界？  
3. **取樣成本**：ODE/SDE 的解算步數如何？可否 deterministic？  
4. **條件化機制**：條件輸入是作為 network conditioning 還是改變路徑分佈？  
5. **理論保證**：收斂、可逆性、或 path consistency 是否被證明？  

## 驗證方式：如何確認數學是否「講得通」

- **連續性/邊界一致性**：檢查 $t=0$、$t=1$ 的端點是否對應到 $p_0$ 與 $p_1$，且 flow ODE 是否在整段時間保持可解。  
- **路徑速度的一致性**：若採用線性插值路徑，應滿足 $\dot{x}_t = x_1 - x_0$；若採用 OT/bridge，則速度場應與對應的最優解/控制目標相符。  
- **loss 與估計量對齊**：flow matching 的目標是回歸速度場，score-based 的目標是回歸 score；兩者的 target 是否是對應的「真實」物理量。  
- **取樣等價性**：檢查 ODE/SDE 的數值解是否在相同 seed/schedule 下對應理論上的推導（如 probability flow ODE 與 SDE 的等價性）。  

## 數學洞見（Mathematical Insight）

1. **Flow matching 是「直接學動力學」，diffusion 是「間接學密度梯度」**：  
   flow matching 以 $v_\theta(x,t)$ 直接回歸速度場；而 diffusion 以 score $\nabla_x \log p_t(x)$ 推導反向 dynamics。兩者都導向 ODE 取樣，但路徑假設與學習訊號不同。  
2. **Probability flow ODE 是兩派的橋樑**：  
   score-based 模型的 probability flow ODE 把 stochastic 擴散轉成 deterministic ODE，使其與 flow matching 的 ODE 形式直接對齊。  
3. **路徑假設決定可學性與偏差**：  
   線性插值（Rectified Flow）通常帶來更穩定的目標，但可能偏離最優 transport；OT/bridge 的路徑更理論嚴謹，但實作與估計成本較高。  
4. **loss 的選擇反映估計策略**：  
   flow matching 的 $\|v_\theta - \dot{x}_t\|^2$ 是對「速度場」的 supervised regression；score-based 的 $\|\epsilon_\theta - \epsilon\|^2$ 是對「噪聲/score」的回歸，等價於某種 score matching。  

## 可改進方向（Future Work / Ideas）

1. **更好的路徑設計**：探索介於線性插值與 OT/bridge 之間的「可學習路徑」，兼顧穩定性與理論一致性。  
2. **更低成本的 ODE/SDE 解法**：研究自適應步長、distillation 或 closed-form 近似，降低取樣成本。  
3. **一致性/可逆性理論**：加強對 flow matching 的可逆性、收斂與誤差累積的理論界限。  
4. **條件化與多模態擴展**：更系統性地將 condition 融合到 coupling 或 path 設計中，避免僅是「把條件餵進網路」的表層方式。  
5. **評估指標標準化**：建立跨方法一致的比較基準（例如速度場誤差、軌跡偏移、采樣時間），讓方法比較更公平透明。  

> 若你想要我根據特定方向（例如：影像生成、語音、圖結構、或 OT/bridge 方向）再補充更專門的論文與實作 repo，可以告訴我想關注的應用與研究背景。
