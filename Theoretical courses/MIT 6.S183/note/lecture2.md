# MIT 6.S183 2026 Lecture 2：Perspectives on Diffusion

## 1. 本讲主线

Lecture 2 讲 diffusion 的数学视角：同一个 diffusion model 可以从 ODE、SDE、DDIM、DDPM、score、flow 等角度理解。

核心关系：

$$
\text{Diffusion}
\rightarrow
\text{Deterministic Sampling / ODE / DDIM}
\rightarrow
\text{Stochastic Sampling / SDE / DDPM}
$$

本讲重点不是重新训练模型，而是解释：模型到底学了什么，以及采样公式从哪里来。

---

## 2. Diffusion Process 的定义

Diffusion process 可以是离散时间，也可以是连续时间。

离散时间：

$$
t \in \{0,1,\ldots,T\}
$$

连续时间：

$$
t \in [0,1]
$$

课程后面主要用连续时间，但写法仍然用下标，例如 $x_t,\sigma_t$。

基本加噪边缘分布为：

$$
x_t \sim x_0 + \sigma_t \epsilon,
\quad
\epsilon \sim \mathcal{N}(0,I)
$$

注意：这只定义了每个时间点的边缘分布 $p_t(x)$，并没有规定不同时间 $x_s,x_t$ 之间如何连接。

---

## 3. Stochastic Interpolant：训练时的采样方式

最直接的方式是把分布等式写成具体等式：

$$
x_t = x_0 + \sigma_t \epsilon
$$

它可以直接生成任意噪声等级下的训练样本 $x_t$。

但它不能用于真正生成，因为它需要已知 $x_0$，而生成时我们正是要从噪声恢复出 $x_0$。

作用：

- 训练时用于构造带噪样本；
- 不适合作为采样算法。

---

## 4. Deterministic Sampling：ODE 视角

如果想从噪声确定性地走回数据，可以学习一个速度场：

$$
\frac{dx_t}{dt} = v(x_t,t)
$$

只要这个速度场能让样本分布满足 diffusion 的边缘分布变化，就可以从 $x_1$ 积分回 $x_0$。

直觉：模型学习每个位置、每个时间点应该往哪里走。

---

## 5. Transport Equation

ODE 下，密度 $p_t(x)$ 的变化由 transport equation 描述：

$$
\partial_t p_t(x)
=
-\nabla \cdot \left(v(x,t)p_t(x)\right)
$$

其中 $\nabla \cdot$ 表示散度，描述某个点附近的概率质量是流入还是流出。

这条方程回答：如果样本按速度场 $v$ 运动，整体概率分布会如何变化。

---

## 6. Probability Flow ODE 与 DDIM

在 diffusion 中，模型常预测噪声：

$$
\epsilon_\theta(x_t,t) \approx \mathbb{E}[\epsilon|x_t]
$$

连续时间下，速度场可以写成：

$$
v(x_t,t)
=
\dot{\sigma}_t \epsilon_\theta(x_t,t)
$$

于是采样就是解 ODE：

$$
\frac{dx_t}{dt}
=
\dot{\sigma}_t \epsilon_\theta(x_t,t)
$$

DDIM 可以看作这个 ODE 的离散化版本。

直觉：

- Probability Flow ODE：连续确定性采样；
- DDIM：离散确定性采样；
- 初始噪声固定后，生成路径也固定。

---

## 7. Stochastic Sampling：SDE 视角

另一种采样方式是在反向去噪过程中也加入随机噪声。

SDE 形式：

$$
dx_t
=
f(x_t,t)dt
+
\rho_t dB_t
$$

其中：

- $f(x_t,t)$ 是 drift，控制平均运动方向；
- $\rho_t$ 是 diffusion coefficient，控制噪声强度；
- $dB_t$ 是 Brownian noise，表示随机扰动。

离散理解：

$$
x_{t+\Delta t}
=
x_t
+
f(x_t,t)\Delta t
+
\rho_t \sqrt{\Delta t}\epsilon
$$

其中 $\epsilon\sim\mathcal{N}(0,I)$。

---

## 8. 为什么 SDE 更麻烦？

ODE 可以简单反向积分，但 SDE 不能直接反向，因为随机噪声无法“减回去”。

原因是噪声项随 $\sqrt{\Delta t}$ 缩放：

$$
\rho_t \sqrt{\Delta t}\epsilon
$$

这使得 SDE 的反向过程必须重新定义，而不是简单把 $dt$ 改成负数。

直觉：ODE 是确定轨迹，SDE 是随机游走；随机游走没有简单的逐点反函数。

---

## 9. Fokker-Planck Equation

SDE 下，密度演化由 Fokker-Planck equation 描述。

它是 transport equation 的随机版本：

$$
\partial_t p_t(x)
=
-\nabla\cdot(f(x,t)p_t(x))
+
\rho_t^2 \Delta p_t(x)
$$

直觉：

- 第一项：drift 造成的概率流动；
- 第二项：随机噪声造成的扩散。

当 $\rho_t=0$ 时，就退化为 ODE 的 transport equation。

---

## 10. Score Function

Fokker-Planck 中会出现一个重要对象：score。

$$
\nabla_x \log p_t(x)
$$

它表示当前点向更高概率密度区域移动的方向。

对 diffusion 来说，score 告诉我们：当前带噪样本应该往哪里走，才能更像真实数据。

---

## 11. Tweedie's Formula：噪声预测与 Score 的关系

模型预测噪声：

$$
\epsilon_\theta(x_t,t)
\approx
\mathbb{E}[\epsilon|x_t]
$$

score 与噪声预测之间有关系：

$$
\epsilon_\theta(x_t,t)
=
-\sigma_t \nabla_x \log p_t(x_t)
$$

也就是说，预测噪声和预测 score 本质等价。

直觉：

- 噪声预测：告诉你该去掉什么；
- score 预测：告诉你该往哪里走；
- 二者只是不同参数化。

---

## 12. 等价参数化

Diffusion model 可以预测很多等价目标：

### 12.1 Epsilon Prediction

预测加入的噪声：

$$
\epsilon_\theta(x_t,t)
$$

### 12.2 x0 Prediction

预测原始干净样本：

$$
\hat{x}_0(x_t,t)
$$

### 12.3 Score Prediction

预测概率密度上升方向：

$$
\nabla_x \log p_t(x_t)
$$

### 12.4 Flow Prediction

预测 ODE 速度场：

$$
v_\theta(x_t,t)
$$

这些参数化可以相互转换，本质上都在学习同一个反向去噪信息。

---

## 13. Reverse SDE 与 DDPM

通过 Fokker-Planck，可以构造一个 reverse SDE，使它和 probability flow ODE 有相同的边缘分布演化。

连续形式是 reverse SDE，离散化后得到 DDPM sampler。

关系：

$$
\text{Reverse SDE}
\rightarrow
\text{DDPM}
$$

DDPM 是随机采样：

- 每一步既沿模型预测方向去噪；
- 又加入一定随机噪声；
- 同一个初始噪声附近可能走出不同路径。

相比 DDIM，DDPM 通常更慢，但随机性可能提高鲁棒性和生成质量。

---

## 14. DDIM vs DDPM

| 采样器 | 连续视角 | 是否随机 | 特点 |
|---|---|---|---|
| DDIM | Probability Flow ODE | 否 | 速度快，路径确定 |
| DDPM | Reverse SDE | 是 | 速度慢，但随机性更强 |

核心区别：

$$
\text{DDIM} = \text{ODE 离散化}
$$

$$
\text{DDPM} = \text{SDE 离散化}
$$

两者使用同一个训练好的噪声预测网络 $\epsilon_\theta$。

---

## 15. 本讲总结

本讲核心逻辑：

$$
\text{Diffusion 的边缘分布}
\rightarrow
\text{如何连接这些分布}
\rightarrow
\text{ODE / SDE 两种采样视角}
\rightarrow
\text{DDIM / DDPM 两种离散采样器}
$$

关键点：

- diffusion process 只规定每个时间点的噪声分布，不唯一规定采样路径；
- stochastic interpolant 用于训练构造 $x_t$，但不能直接生成；
- ODE 视角得到 deterministic sampler，对应 DDIM；
- SDE 视角得到 stochastic sampler，对应 DDPM；
- transport equation 描述 ODE 下的密度演化；
- Fokker-Planck equation 描述 SDE 下的密度演化；
- $\epsilon$ prediction、$x_0$ prediction、score prediction、flow prediction 本质等价；
- diffusion 训练通常只需要一个 $\epsilon_\theta$，不同采样算法只是使用它的方式不同。