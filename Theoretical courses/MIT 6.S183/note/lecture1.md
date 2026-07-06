# MIT 6.S183 2026 Lecture 1：Introduction to Diffusion Models

## 1. 本讲定位

这门课是 diffusion models 的实践入门，Lecture 1 主要回答三个问题：

- 为什么需要生成模型？
- VAE 和 GAN 为什么不够理想？
- Diffusion 的基本训练和采样流程是什么？

本讲只做非严格、直觉化介绍，真正的数学解释会放到 Lecture 2。

---

## 2. 从监督学习到生成模型

传统监督学习学习的是输入到输出的一个预测值：

$$
\hat y = f_\theta(x)
$$

它通常只建模给定输入后的平均输出，因此会丢失不确定性、多模态等信息。

生成模型希望学习完整条件分布：

$$
p_\theta(y|x)
$$

这样不仅能给出一个预测，还能从分布中采样，得到不同可能结果，并估计方差、模式数量等信息。

直觉：监督学习给“一个答案”，生成模型给“答案的分布”。

---

## 3. 生成模型的用途

生成模型的核心能力是从一个分布中采样，也可以把一个分布的样本变成另一个分布的样本。

典型应用：

- 图像生成：给定属性或文本生成图像；
- sketch-to-image：把草图转成真实图像；
- image editing：重采样图像某一部分；
- image restoration：从损坏图像恢复自然图像。

回归式模型容易产生模糊平均结果，而生成模型能产生更清晰、更具体的样本。

---

## 4. VAE：从最大似然到 ELBO

### 4.1 动机

最自然的生成建模目标是最大化数据概率：

$$
\max_\theta \log p_\theta(x)
$$

VAE 引入隐变量 $z$，把数据生成过程写成：

$$
p_\theta(x)=\int p_\theta(x|z)p(z)\,dz
$$

其中 decoder 学习 $p_\theta(x|z)$，负责从隐变量生成数据。

### 4.2 ELBO

直接优化上面的积分很难，所以引入 encoder：

$$
q_\phi(z|x)
$$

通过 Jensen 不等式得到证据下界 ELBO：

$$
\log p_\theta(x)
\ge
\mathbb{E}_{q_\phi(z|x)}
[\log p_\theta(x|z)]
-
D_{\mathrm{KL}}(q_\phi(z|x)\,\Vert\,p(z))
$$

两项含义：

- 重构项：$x \rightarrow z \rightarrow x$ 后尽量还原原样本；
- KL 项：让 encoder 输出的 $z$ 接近先验分布 $p(z)$。

### 4.3 VAE 的问题

VAE 的第一个问题是生成模糊。因为 $x$ 被压缩到低维 $z$，相似样本会映射到相近 latent，decoder 会对这些样本做平均，导致高频细节丢失。

第二个问题是 posterior collapse：

$$
q_\phi(z|x) \approx p(z)
$$

此时 decoder 可能直接忽略 $z$，自己建模整个数据分布；decoder 越强，这个问题反而越严重。

---

## 5. GAN：从似然下界转向分布匹配

### 5.1 动机

从 VAE 学到的教训是：不要只优化 likelihood bound，而是直接在分布层面优化生成分布。

GAN 的目标是让生成分布 $p_\theta$ 接近真实分布 $p^*$。

### 5.2 Total Variation 视角

课程用 total variation distance 解释 GAN 的思想：

$$
d_{\mathrm{TV}}(p_\theta,p^*)
=
\sup_{\lVert f\rVert_\infty \le 1}
\left(
\mathbb{E}_{p^*}[f(x)]
-
\mathbb{E}_{p_\theta}[f(x)]
\right)
$$

这里的 $f$ 可以理解为 discriminator：它试图区分样本来自真实分布还是生成分布。

### 5.3 GAN 结构

GAN 包含两个部分：

$$
z \sim p(z), \quad G_\theta(z) \rightarrow x
$$

$$
D_\phi(x) \rightarrow \text{real or fake}
$$

生成器 $G$ 把噪声映射到数据空间，判别器 $D$ 判断样本是真实还是生成。

### 5.4 GAN 的问题

GAN 在数学上直接优化分布距离，但训练很难：

- generator 的梯度来自 discriminator；
- 两者是对抗关系，训练不稳定；
- generator 和 discriminator 的容量需要平衡；
- 训练过程像“黑魔法”，很难从 loss 判断是否正常。

GAN 仍然可以很强，但因为训练困难，后来 diffusion 更受欢迎。

---

## 6. Diffusion：固定去噪路径

### 6.1 动机

VAE 有模糊和 posterior collapse，GAN 有 min-max 训练不稳定。Diffusion 的核心改动是：

> 不学习任意的 noise-to-data 映射，而是固定一个具体的去噪过程，然后学习这个过程。

它不需要 ELBO 下界，也不需要 discriminator 和 min-max 对抗训练。

---

## 7. 正向加噪过程

Diffusion 受到物理中 Brownian motion 的启发：给数据不断加入随机噪声，直到变成纯噪声。

正向过程：

$$
x_0 \rightarrow x_\sigma
$$

一种简单写法是：

$$
x_\sigma = x_0 + \sigma \epsilon,
\quad
\epsilon \sim \mathcal{N}(0,I)
$$

其中 $\sigma$ 表示噪声强度。$\sigma$ 越大，数据越接近纯噪声。

---

## 8. 训练目标：预测噪声

模型输入带噪数据 $x_\sigma$ 和噪声等级 $\sigma$，输出被加入的噪声：

$$
\epsilon_\theta(x_\sigma,\sigma) \approx \epsilon
$$

训练损失是简单的平方误差：

$$
\mathcal{L}
=
\mathbb{E}_{x_0,\sigma,\epsilon}
\left[
\lVert
\epsilon_\theta(x_\sigma,\sigma)-\epsilon
\rVert^2
\right]
$$

如果模型能预测噪声，就能近似恢复干净样本：

$$
\hat x_0 = x_\sigma - \sigma \epsilon_\theta(x_\sigma,\sigma)
$$

直觉：模型不是直接生成图像，而是学习“当前样本里哪些部分是噪声”。

---

## 9. Noise Schedule

训练和采样时需要选择一系列噪声等级：

$$
\sigma_{\min} \rightarrow \sigma_{\max}
$$

课程中提到常见范围大致是：

$$
\sigma_{\min} \approx 0.01,
\quad
\sigma_{\max} \approx 100
$$

噪声等级数量可以从几十到上千不等，常见做法是在 log space 中均匀采样：

$$
\log \sigma \sim \text{Uniform}(\log \sigma_{\min}, \log \sigma_{\max})
$$

这样小噪声和大噪声区间都能得到足够覆盖。

---

## 10. 模型结构

### 10.1 Toy Example：2D Spiral

课程用二维 spiral dataset 解释 diffusion。训练时：

1. 采样干净点 $x_0$；
2. 采样噪声等级 $\sigma$；
3. 构造 $x_\sigma = x_0 + \sigma\epsilon$；
4. 把 $x_\sigma$ 和 $\sigma$ 的 embedding 输入 MLP；
5. 预测噪声 $\epsilon$。

这里的 $\sigma$ 会通过 sin/cos embedding 变成特征向量，再与 $x_\sigma$ 拼接输入模型。

### 10.2 Image Model

图像 diffusion 常用两类结构：

- U-Net：通过下采样提取全局信息，再通过上采样恢复空间细节；
- Diffusion Transformer：把图像切成 patch token，再用 Transformer 建模。

课程中强调：diffusion 对架构选择相对宽容，只要架构合理，通常不会像 GAN 那样训练崩掉。

---

## 11. Denoiser 学到的东西

从几何直觉看，denoiser 学到的是“如何回到数据流形”。

小噪声时，模型更像是在预测当前点到最近数据点的方向；大噪声时，模型看到的是更模糊的整体分布信息。

因此 diffusion 可以理解为学习一个方向场：

$$
x_\sigma \rightarrow x_0
$$

也就是从带噪样本指向真实数据流形的方向。

---

## 12. 采样过程

训练好 $\epsilon_\theta$ 后，从纯噪声开始：

$$
x_T \sim \text{noise}
$$

然后逐步反向去噪：

$$
x_T \rightarrow x_{T-1} \rightarrow \cdots \rightarrow x_0
$$

所有采样算法都使用同一个噪声预测网络 $\epsilon_\theta$，区别只在于如何更新 $x_t$。

### DDIM

DDIM 是确定性采样器。给定初始噪声后，它沿着模型预测的方向确定性地流向数据。

可以理解为一种 ODE-like 的反向过程。

### DDPM

DDPM 是随机采样器。它在反向过程中还会加入额外噪声，因此同一个中间状态可能生成不同结果。

可以理解为一种 stochastic reverse process。

---

## 13. 本讲总结

本讲核心逻辑：

$$
\text{监督学习只预测均值}
\rightarrow
\text{生成模型学习完整分布}
\rightarrow
\text{VAE / GAN 各有问题}
\rightarrow
\text{Diffusion 用固定去噪过程稳定生成}
$$

关键点：

- 生成模型学习的是完整分布，而不是单个输出；
- VAE 从最大似然出发，用 ELBO 训练，但容易模糊和 posterior collapse；
- GAN 直接做分布匹配，但对抗训练不稳定；
- Diffusion 固定正向加噪过程，只学习反向去噪；
- 训练 diffusion 的核心就是预测噪声：

$$
\epsilon_\theta(x_\sigma,\sigma) \approx \epsilon
$$

- 采样时从纯噪声出发，利用同一个噪声预测网络逐步走回数据分布。