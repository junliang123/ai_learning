# CS231n 2025 Lecture 13 —— Generative Models 1

## 一、生成模型概述

### 监督学习

监督学习通常学习条件分布 $p(y|x)$，即给定输入 $x$ 预测标签 $y$，典型任务包括分类、检测、分割、图像描述等。

---

### 无监督学习

无监督学习只有数据 $x$，没有人工标签，目标是从数据本身发现结构，例如聚类、降维、密度估计和表征学习。

---

### 判别模型（Discriminative Model）

判别模型学习 $p(y|x)$，即给定输入预测标签。它只让不同标签竞争概率质量，不建模输入 $x$ 本身是否合理。

---

### 生成模型（Generative Model）

生成模型学习 $p(x)$ 或 $p(x|y)$，即建模数据本身的概率分布。它需要判断哪些样本像真实数据，哪些样本不像真实数据。

---

### 无条件生成模型

无条件生成模型学习 $p(x)$，可以采样生成数据，也可用于异常检测，但无法控制生成内容。

---

### 条件生成模型

条件生成模型学习 $p(x|y)$，即给定条件生成数据，例如 Text-to-Image、Text-to-Video、LLM 等。实际应用中最常用的是条件生成模型。

---

### 为什么需要生成模型？

生成模型适合处理“一个输入对应多个合理输出”的任务。例如同一句 prompt 可以对应无数张合理图片，因此模型应学习整个输出分布 $p(x|y)$，而不是一个确定映射。

---

## 二、生成模型分类

### Explicit Density

显式密度模型可以计算或近似计算 $p(x)$，代表方法包括 Autoregressive Model 和 VAE。

---

### Implicit Density

隐式密度模型不能直接给出 $p(x)$，但可以从模型中采样，代表方法包括 GAN 和 Diffusion。

---

### 本讲内容位置

本讲主要讲 Autoregressive Model 和 VAE；GAN 与 Diffusion 放到下一讲。

---

## 三、Maximum Likelihood Estimation

生成模型的核心目标是最大化训练数据在模型中的概率。给定数据集 $\{x^{(1)},\dots,x^{(N)}\}$，优化：

$$
\theta^*
=
\arg\max_\theta
\prod_{i=1}^N p_\theta(x^{(i)})
$$

取 log 后得到：

$$
\theta^*
=
\arg\max_\theta
\sum_{i=1}^N
\log p_\theta(x^{(i)})
$$

直觉：调整模型分布，让真实样本获得更高概率。

---

## 四、Autoregressive Model

自回归模型把样本拆成序列 $x=(x_1,x_2,\dots,x_T)$，利用概率链式法则：

$$
p(x)
=
\prod_{t=1}^{T}
p(x_t|x_1,\dots,x_{t-1})
$$

即每一步根据前面内容预测下一个元素，RNN 和 Transformer 语言模型都属于这种思路。

---

### 文本自回归

文本天然是一维离散序列，因此很适合自回归建模：

$$
p(w_1,\dots,w_T)
=
\prod_{t=1}^{T}
p(w_t|w_{<t})
$$

这就是语言模型预测 next token 的概率形式。

---

### 图像自回归

图像也可以展平成像素序列，但高分辨率图像序列极长，例如 $1024\times1024\times3$ 有三百多万个数，因此直接逐像素自回归生成非常昂贵。

---

## 五、AutoEncoder

AutoEncoder 用 Encoder 把输入压缩为 latent code，再用 Decoder 重建输入：

$$
z=f_\phi(x),
\qquad
\hat{x}=g_\theta(z)
$$

训练目标是让重建图接近原图：

$$
\mathcal L_{rec}
=
\|x-\hat{x}\|_2^2
$$

中间的 $z$ 是 bottleneck，模型不能简单复制输入，必须学习压缩后的有效表示。

---

### 普通 AutoEncoder 的问题

普通 AE 可以重建图像，但 latent space 没有被约束成已知分布。训练后我们不知道该从哪里采样一个合理的 $z$，因此它不适合直接生成新图像。

---

## 六、VAE 核心思想

VAE = AutoEncoder + 概率 latent space。

VAE 假设图片 $x$ 背后有隐变量 $z$，并规定先验分布：

$$
p(z)=\mathcal N(0,I)
$$

生成过程为：

$$
z\sim p(z),
\qquad
x\sim p_\theta(x|z)
$$

生成图片时只需要采样 $z\sim\mathcal N(0,I)$，再输入 Decoder 得到图像。

---

## 七、VAE 的训练困境：只有 $x$，没有 $z$

VAE 的生成目标是训练出一个 Decoder：

$$
z \rightarrow p_\theta(x|z)
$$

生成图片时，只需要从先验采样 $z\sim\mathcal N(0,I)$，再输入 Decoder 得到图像。因此真正需要学的是 $p_\theta(x|z)$。

但训练集中只有真实图片 $x$，没有对应的 latent variable $z$。也就是说，我们不知道某张真实图片 $x$ 应该由哪个 $z$ 生成。

所以训练时需要一个反向过程：给定当前图片 $x$，找到可能生成它的 $z$，也就是希望得到后验分布：

$$
p_\theta(z|x)
$$

---

## 八、真实后验 $p_\theta(z|x)$ 的困境

根据贝叶斯公式：

$$
p_\theta(z|x)
=
\frac{
p_\theta(x|z)p(z)
}{
p_\theta(x)
}
$$

其中分子 $p_\theta(x|z)p(z)$ 可以计算：$p(z)$ 是标准高斯先验，$p_\theta(x|z)$ 由 Decoder 给出。

真正困难的是分母：

$$
p_\theta(x)
=
\int p_\theta(x|z)p(z)\,dz
$$

这个积分需要遍历所有可能的 $z$，而 $p_\theta(x|z)$ 又由神经网络表示，因此通常无法解析计算。

所以 $p_\theta(x)$ 算不了，进一步导致真实后验 $p_\theta(z|x)$ 也算不了。

---

## 九、引入 Encoder：用 $q_\phi(z|x)$ 近似真实后验

既然真实后验 $p_\theta(z|x)$ 无法计算，VAE 引入一个 Encoder 网络来近似它：

$$
q_\phi(z|x)
\approx
p_\theta(z|x)
$$

Encoder 输入图片 $x$，输出一个高斯分布的参数：

$$
q_\phi(z|x)
=
\mathcal N
\left(
\mu_\phi(x),
\operatorname{diag}(\sigma_\phi^2(x))
\right)
$$

也就是说，Encoder 输出 $\mu_\phi(x)$ 和 $\sigma_\phi(x)$，表示当前图片可能对应的 latent 分布。

为了从 $q_\phi(z|x)$ 中采样且能反向传播，使用 reparameterization trick：

$$
\epsilon\sim\mathcal N(0,I)
$$

$$
z
=
\mu_\phi(x)
+
\sigma_\phi(x)\odot\epsilon
$$

然后将 $z$ 输入 Decoder：

$$
p_\theta(x|z)
=
\mathcal N
\left(
\mu_\theta(z),
\sigma^2 I
\right)
$$

实际训练中通常直接把 Decoder 输出的均值 $\mu_\theta(z)$ 当作重建图像 $\hat{x}$。如果固定方差 $\sigma^2$，最大化高斯 likelihood 等价于最小化 MSE：

$$
-\log p_\theta(x|z)
\propto
\|x-\mu_\theta(z)\|_2^2
$$

因此训练流程可以概括为：

$$
x
\rightarrow
q_\phi(z|x)
\rightarrow
z
\rightarrow
p_\theta(x|z)
\rightarrow
\hat{x}
$$

但如果只使用重建损失 $\|x-\hat{x}\|^2$，模型会退化成普通 AutoEncoder，latent space 仍然不可采样。因此还需要后面的 ELBO 推导，引出 KL 项来约束 $q_\phi(z|x)$ 接近先验 $p(z)=\mathcal N(0,I)$。

---

## 十、VAE 的训练目标：最大化 $\log p_\theta(x)$

VAE 的目标是最大化数据似然：$\log p_\theta(x)$

也就是让真实图片 $x$ 在模型分布中概率尽可能高。由于 $p_\theta(x)$ 中的积分不可计算，所以不能直接优化 $\log p_\theta(x)$，需要推导它的下界 ELBO。

---

## 十一、ELBO 推导

目标是最大化：

$$
\log p_\theta(x)
$$

因为 $\log p_\theta(x)$ 与 $z$ 无关，所以可以写成对 $q_\phi(z|x)$ 的期望：

$$
\log p_\theta(x)
=
\mathbb E_{q_\phi(z|x)}
[\log p_\theta(x)]
$$

由贝叶斯公式：

$$
p_\theta(x)
=
\frac{
p_\theta(x|z)p(z)
}{
p_\theta(z|x)
}
$$

代入并引入 $q_\phi(z|x)$：

$$
\log p_\theta(x)
=
\mathbb E_{q_\phi(z|x)}
\left[
\log
\frac{
p_\theta(x|z)p(z)q_\phi(z|x)
}{
p_\theta(z|x)q_\phi(z|x)
}
\right]
$$

拆开：

$$
\log p_\theta(x)
=
\mathbb E_{q_\phi}
[\log p_\theta(x|z)]
-
D_{KL}
(q_\phi(z|x)\|p(z))
+
D_{KL}
(q_\phi(z|x)\|p_\theta(z|x))
$$

由于 KL 散度非负：

$$
D_{KL}
(q_\phi(z|x)\|p_\theta(z|x))
\ge 0
$$

所以得到下界：

$$
\log p_\theta(x)
\ge
\mathbb E_{q_\phi}
[\log p_\theta(x|z)]
-
D_{KL}
(q_\phi(z|x)\|p(z))
$$

右边称为 ELBO：

$$
\boxed{
\mathcal L_{ELBO}
=
\mathbb E_{q_\phi(z|x)}
[\log p_\theta(x|z)]
-
D_{KL}
(q_\phi(z|x)\|p(z))
}
$$

VAE 训练就是最大化 ELBO。

---

## 十二、由 ELBO 得到 VAE Loss

最大化 ELBO：

$$
\max_{\theta,\phi}
\left[
\mathbb E_{q_\phi(z|x)}
[\log p_\theta(x|z)]
-
D_{KL}
(q_\phi(z|x)\|p(z))
\right]
$$

等价于最小化负 ELBO：

$$
\mathcal L_{VAE}
=
-
\mathbb E_{q_\phi(z|x)}
[\log p_\theta(x|z)]
+
D_{KL}
(q_\phi(z|x)\|p(z))
$$

因此 VAE Loss 可以写成：

$$
\boxed{
\mathcal L_{VAE}
=
\mathcal L_{rec}
+
\mathcal L_{KL}
}
$$

其中：

$$
\mathcal L_{rec}
=
-
\mathbb E_{q_\phi(z|x)}
[\log p_\theta(x|z)]
$$

$$
\mathcal L_{KL}
=
D_{KL}
(q_\phi(z|x)\|p(z))
$$

重建项让 $z$ 保留当前图片信息，KL 项让 latent 分布接近标准高斯。

---

## 十三、Reconstruction Loss

如果 Decoder 假设固定方差高斯：

$$
p_\theta(x|z)
=
\mathcal N
\left(
\mu_\theta(z),
\sigma^2I
\right)
$$

则负 log likelihood 等价于 MSE：

$$
\mathcal L_{rec}
\propto
\|x-\hat{x}\|_2^2
$$

其中 $\hat{x}=\mu_\theta(z)$。

---

## 十四、KL Loss

KL 项约束 Encoder 输出的分布接近先验：

$$
q_\phi(z|x)
\approx
p(z)=\mathcal N(0,I)
$$

若：

$$
q_\phi(z|x)
=
\mathcal N
(\mu,\operatorname{diag}(\sigma^2))
$$

则：

$$
\mathcal L_{KL}
=
\frac{1}{2}
\sum_j
\left(
\mu_j^2+\sigma_j^2-\log\sigma_j^2-1
\right)
$$

它鼓励 $\mu\rightarrow 0$，$\sigma\rightarrow 1$，从而让生成时可以直接从 $\mathcal N(0,I)$ 采样。

---

## 十五、VAE 训练流程

1. 输入真实图片 $x$。
2. Encoder 输出 $\mu_\phi(x)$ 和 $\sigma_\phi(x)$。
3. 采样 $\epsilon\sim\mathcal N(0,I)$，计算 $z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon$。
4. Decoder 输出 $\hat{x}=\mu_\theta(z)$。
5. 计算重建损失 $\mathcal L_{rec}\approx\|x-\hat{x}\|^2$。
6. 计算 KL 损失 $\mathcal L_{KL}=D_{KL}(q_\phi(z|x)\|p(z))$。
7. 最小化 $\mathcal L_{VAE}=\mathcal L_{rec}+\mathcal L_{KL}$，同时更新 Encoder 和 Decoder。

---

## 十六、VAE 生成流程

训练完成后不再需要 Encoder，直接采样：

$$
z\sim\mathcal N(0,I)
$$

然后输入 Decoder：

$$
\hat{x}=\mu_\theta(z)
$$

由于训练时 KL 项已经让 $q_\phi(z|x)$ 接近 $\mathcal N(0,I)$，所以 Decoder 在生成时能处理从标准高斯采样得到的 $z$。

---

## 十七、两个 Loss 的矛盾

重建损失希望 $z$ 尽量保留每张图的独特信息，因此倾向于让 $\sigma\rightarrow 0$，每张图对应一个几乎确定的 latent code。

KL 损失希望 $q_\phi(z|x)$ 接近标准高斯，因此倾向于让 $\mu\rightarrow 0$、$\sigma\rightarrow 1$。

VAE 的训练本质就是在“重建质量”和“latent space 可采样性”之间折中。

---

## 十八、VAE 的特点

VAE 的优点是有明确概率解释，能学习连续平滑的 latent space，并且可以通过采样生成新数据。

VAE 的缺点是重建项通常类似 MSE，容易导致生成结果偏模糊，视觉质量一般不如 GAN 和 Diffusion。

---

## 十九、本讲总结

Lecture 13 主线：生成模型用于建模数据分布，尤其适合输出存在多种可能的任务。判别模型学习 $p(y|x)$，生成模型学习 $p(x)$ 或 $p(x|y)$。Autoregressive Model 通过链式法则逐步建模序列，VAE 则用概率 latent space 改造 AutoEncoder。

VAE 的核心逻辑：

1. 假设 $z\sim\mathcal N(0,I)$，并由 $p_\theta(x|z)$ 生成图片。
2. 训练集中只有 $x$，没有 $z$，所以需要 Encoder $q_\phi(z|x)$ 近似真实后验。
3. 最大化 $\log p_\theta(x)$ 不可直接计算，因此最大化 ELBO。
4. ELBO 最终得到“重建项 + KL 项”的损失函数。

核心公式：

$$
\boxed{
\log p_\theta(x)
\ge
\mathbb E_{q_\phi(z|x)}
[\log p_\theta(x|z)]
-
D_{KL}
(q_\phi(z|x)\|p(z))
}
$$

最终损失：

$$
\boxed{
\mathcal L_{VAE}
=
\mathcal L_{rec}
+
\mathcal L_{KL}
}
$$

一句话理解：

> VAE = AutoEncoder + Gaussian latent space + Maximum Likelihood / ELBO。