# CS231N Lecture 14：Generative Models 2 复习提纲

## 1. 生成模型的定位

生成模型的目标不是只预测标签，而是学习数据分布本身，核心问题是：如何从一个简单分布中采样，生成看起来像真实数据的新样本。

判别模型建模标签条件分布：

$$
p(y|x)
$$

生成模型建模数据分布或条件数据分布：

$$
p(x) \quad \text{或} \quad p(x|y)
$$

其中 $y$ 可以是类别、文本 prompt、输入图像等控制信号；实际应用中更重要的是条件生成模型，因为它能控制生成内容。

## 2. 显式密度模型与隐式密度模型

显式密度模型直接或近似给出 $p_\theta(x)$，因此训练目标通常来自最大似然。

常见例子：

- Autoregressive Model：精确建模 $p(x)$；
- VAE：近似建模 $p(x)$，最大化 ELBO；
- Normalizing Flow：可精确计算密度，但架构受限。

隐式密度模型不直接输出 $p_\theta(x)$，但可以从模型中采样：

$$
z \sim p(z), \quad x = G(z)
$$

GAN 和 Diffusion 都属于这一类：重点不是算出某个样本的概率，而是生成高质量样本。

## 3. 回顾：Autoregressive 与 VAE

### 3.1 Autoregressive Model

自回归模型把数据拆成序列，逐步预测下一个元素。

$$
p(x)=\prod_{i=1}^{n}p(x_i|x_{<i})
$$

对图像来说，可以把像素或子像素看成 0～255 的离散 token，再用 RNN/Transformer 建模。优点是似然明确，缺点是采样通常较慢。

### 3.2 VAE

VAE 假设数据 $x$ 由潜变量 $z$ 生成，并通过编码器学习 $q_\phi(z|x)$，通过解码器学习 $p_\theta(x|z)$。

核心目标是最大化 ELBO：

$$
\log p_\theta(x) \geq 
\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]
-
D_{KL}(q_\phi(z|x)\|p(z))
$$

第一项要求重建好，第二项要求潜空间接近先验 $p(z)$。VAE 的好处是有 $x \to z$ 的编码关系，但生成结果常偏模糊。

## 4. GAN：用判别器逼近分布匹配

### 4.1 为什么提出 GAN

VAE 和自回归模型都要显式写出或近似 $p(x)$，但真实数据分布 $p_{\text{data}}$ 极其复杂。GAN 的想法是：不直接计算密度，只训练一个生成器 $G$ 把简单噪声 $z$ 变成样本 $x$。

$$
z \sim p(z), \quad x = G(z), \quad x \sim p_G
$$

目标是让生成分布 $p_G$ 尽量接近真实分布 $p_{\text{data}}$。

### 4.2 架构

GAN 有两个网络：

- 生成器 $G$：输入噪声 $z$，输出假样本 $G(z)$；
- 判别器 $D$：输入样本 $x$，输出它是真实样本的概率 $D(x)$。

直觉：判别器越会分辨真假，生成器为了骗过它，就必须生成更接近真实分布的样本。

### 4.3 训练目标从哪里来

判别器希望真实样本判为 1，生成样本判为 0：

$$
\max_D 
\mathbb{E}_{x\sim p_{\text{data}}}[\log D(x)]
+
\mathbb{E}_{z\sim p(z)}[\log(1-D(G(z)))]
$$

生成器希望假样本被判为真，因此要让 $D(G(z))$ 越大越好。GAN 的原始 minimax 目标为：

$$
\min_G \max_D V(D,G)
=
\mathbb{E}_{x\sim p_{\text{data}}}[\log D(x)]
+
\mathbb{E}_{z\sim p(z)}[\log(1-D(G(z)))]
$$

其中判别器最大化 $V(D,G)$，生成器最小化 $V(D,G)$。

### 4.4 训练流程

每轮训练交替更新：

1. 固定 $G$，更新 $D$，让 $D(x)$ 接近 1，$D(G(z))$ 接近 0；
2. 固定 $D$，更新 $G$，让 $D(G(z))$ 接近 1；
3. 生成器的梯度来自判别器，反向传播路径是 $D(G(z)) \to G(z) \to G$。

注意：GAN 的 $V(D,G)$ 不是普通意义上的 loss，数值大小不能稳定反映生成质量，因为它同时依赖 $G$ 和 $D$ 的强弱。

### 4.5 理论解释：为什么最优时能匹配分布

固定生成器 $G$ 时，可以写出最优判别器：

$$
D_G^*(x)=
\frac{p_{\text{data}}(x)}
{p_{\text{data}}(x)+p_G(x)}
$$

含义：如果真实概率比生成概率大，则 $D_G^*(x)>0.5$；如果两者相等，则 $D_G^*(x)=0.5$。

把 $D_G^*$ 代回 GAN 目标，可以得到：

$$
V(D_G^*,G)
=
-\log 4
+
2D_{JS}(p_{\text{data}}\|p_G)
$$

因此外层最小化等价于最小化 JS 散度；当且仅当

$$
p_G = p_{\text{data}}
$$

时达到最优。

但这个结论依赖理想条件：无限模型容量、真实分布可表达、优化能收敛。实际训练中这些都不保证。

### 4.6 生成器损失的实用修正

原始生成器目标是最小化：

$$
\mathcal{L}_G
=
\mathbb{E}_{z\sim p(z)}[\log(1-D(G(z)))]
$$

训练初期，生成器很差，判别器很容易判断假样本，所以 $D(G(z))\approx 0$。此时 $\log(1-D(G(z)))$ 的梯度很小，生成器学不动。

实践中常改为 non-saturating loss：

$$
\mathcal{L}_G
=
-\mathbb{E}_{z\sim p(z)}[\log D(G(z))]
$$

它和原始目标方向相近，但在训练初期能给生成器更强梯度。

### 4.7 推理流程

推理时丢弃判别器，只保留生成器：

$$
z \sim p(z), \quad x = G(z)
$$

优点是生成速度快，一次前向传播即可得到样本。

### 4.8 GAN 的优缺点

GAN 的优点：公式简单，推理快，调得好时图像清晰锐利，潜空间插值常比较平滑。

GAN 的缺点：训练不稳定，没有可靠 loss 曲线，容易 mode collapse，可能出现 NaN/Inf，且没有显式的 $x \to z$ 编码器。

## 5. Diffusion Model：从噪声逐步去噪生成数据

### 5.1 为什么扩散模型取代 GAN 成为主流

GAN 的核心困难是对抗训练不稳定，而且无法通过 loss 曲线判断训练是否变好。扩散模型换了一种思路：把生成问题变成监督式去噪问题。

GAN 是一步映射：

$$
z \to x
$$

扩散模型是多步反向去噪：

$$
x_T \to x_{T-1} \to \cdots \to x_0
$$

其中 $x_T$ 是纯噪声，$x_0$ 是干净数据。

### 5.2 基本直觉

扩散模型先构造从数据到噪声的路径，再训练模型学会反向走回来。

噪声 $z$ 通常与数据 $x$ 形状相同，例如图像是 $H\times W\times 3$，噪声也必须是 $H\times W\times 3$。

噪声程度用 $t\in[0,1]$ 表示：

- $t=0$：干净数据；
- $t=1$：纯噪声；
- 中间 $t$：数据与噪声的混合。

训练时，模型输入带噪样本和噪声等级，学习去掉一部分噪声；推理时，从纯噪声开始，多次调用模型逐步生成数据。

## 6. Rectified Flow：用向量场学习从噪声到数据的路径

### 6.1 为什么引入 Rectified Flow

扩散模型有很多数学形式，符号复杂。Rectified Flow 是一种直观版本：直接让网络预测“从数据点到噪声点的方向向量”，再在推理时沿反方向积分回数据分布。

核心思想：学习一个向量场，而不是直接学习 $p(x)$。

### 6.2 训练数据构造

每次训练采样：

$$
x \sim p_{\text{data}}, \quad z \sim \mathcal{N}(0,I), \quad t\sim U(0,1)
$$

其中 $x$ 是真实样本，$z$ 是同形状高斯噪声。

构造带噪样本：

$$
x_t = (1-t)x + tz
$$

构造真实速度向量：

$$
v_{\text{gt}} = z - x
$$

这里 $v_{\text{gt}}$ 指向“从真实数据 $x$ 到噪声 $z$”的方向。

### 6.3 模型目标

模型 $f_\theta$ 输入 $x_t$ 和 $t$，预测速度向量：

$$
\hat v = f_\theta(x_t,t)
$$

损失函数是简单的 MSE：

$$
\mathcal{L}(\theta)
=
\mathbb{E}_{x,z,t}
\left[
\|f_\theta(x_t,t)-(z-x)\|_2^2
\right]
$$

这个损失从哪里来：训练时我们知道配对的真实样本 $x$ 和噪声 $z$，所以可以直接监督模型预测二者之间的方向向量。

### 6.4 训练流程

训练过程可以概括为：

1. 从数据集中取真实图像 $x$；
2. 采样同形状高斯噪声 $z$；
3. 采样噪声等级 $t$；
4. 线性插值得到 $x_t=(1-t)x+tz$；
5. 网络预测 $\hat v=f_\theta(x_t,t)$；
6. 用 $\|\hat v-(z-x)\|^2$ 更新模型。

相比 GAN，Rectified Flow 有普通监督学习式的 loss，loss 下降通常意味着模型确实学得更好。

### 6.5 推理流程

推理从纯噪声开始：

$$
x_1 \sim \mathcal{N}(0,I)
$$

然后从 $t=1$ 逐步走向 $t=0$。由于训练目标 $v=z-x$ 指向噪声方向，推理时要沿反方向更新：

$$
x_{t-\Delta t}
=
x_t - \Delta t \cdot f_\theta(x_t,t)
$$

重复 30～50 步左右，可以从纯噪声生成干净样本。

### 6.6 直觉总结

训练时学的是“这个带噪点应该朝哪个方向回到数据分布”。推理时模型不断给出方向，采样过程就是沿着这个向量场从噪声分布走回数据分布。

## 7. 条件生成与 Classifier-Free Guidance

### 7.1 条件生成

条件生成模型学习的是：

$$
p(x|y)
$$

其中 $y$ 可以是类别、文本 prompt、草图、参考图像等。Rectified Flow 中只需要把 $y$ 作为额外输入：

$$
\hat v = f_\theta(x_t,t,y)
$$

这样模型预测的方向就不是回到整个数据分布，而是回到条件分布 $p(x|y)$。

### 7.2 CFG 的动机

普通条件扩散模型有时不够听 prompt。Classifier-Free Guidance 的目标是给模型一个旋钮，控制它多大程度上服从条件 $y$。

### 7.3 CFG 的训练方式

训练时随机丢弃条件信息：

$$
y \to \varnothing
$$

例如 50% 概率把文本条件换成 null token。这样同一个模型会同时学到：

无条件速度：

$$
v_{\varnothing}=f_\theta(x_t,t,\varnothing)
$$

有条件速度：

$$
v_y=f_\theta(x_t,t,y)
$$

直觉上，$v_{\varnothing}$ 指向整体数据分布，$v_y$ 指向满足条件 $y$ 的子分布。

### 7.4 CFG 的推理公式

推理时同时计算两次模型输出：一次带条件，一次不带条件。

$$
v_y = f_\theta(x_t,t,y)
$$

$$
v_{\varnothing} = f_\theta(x_t,t,\varnothing)
$$

然后做线性组合：

$$
v_{\mathrm{CFG}} = (1+w)v_y - wv_{\varnothing}
$$

当 $w=0$ 时：

$$
v_{\mathrm{CFG}} = v_y
$$

即普通条件生成；$w$ 越大，越强调条件信号。

代价：每一步采样要跑两次模型，因此推理成本大约翻倍。

## 8. Noise Schedule：为什么不总是均匀采样 $t$

最简单的 Rectified Flow 使用：

$$
t \sim U(0,1)
$$

但直觉上，$t$ 接近 0 或 1 时任务反而容易：接近干净数据或纯噪声时，模型要预测的方向更简单；中间噪声等级最难，因为一个 $x_t$ 可能对应多个潜在的 $(x,z)$ 配对。

因此实际训练中常用非均匀噪声分布，例如 logit-normal schedule，让训练更关注中间区域。

高分辨率图像还可能需要 shifted noise schedule，因为不同分辨率下像素相关性不同，破坏图像信息所需的噪声强度也不同。

## 9. Diffusion 的一般形式

不同扩散模型的区别，常体现在如何构造 $x_t$ 和让模型预测什么。

一般可以写成：

$$
x_t = a(t)x + b(t)z
$$

预测目标可以写成：

$$
y_{\text{gt}} = c(t)x + d(t)z
$$

模型损失通常是：

$$
\mathcal{L}(\theta)
=
\mathbb{E}
\left[
\|f_\theta(x_t,t)-y_{\text{gt}}\|_2^2
\right]
$$

不同选择对应不同形式：

- 预测 clean data：模型直接预测 $x$；
- 预测 noise：模型预测加入的噪声 $z$ 或 $\epsilon$；
- 预测 velocity：模型预测 $z-x$；
- VP / VE diffusion：通过不同的 $a(t),b(t)$ 控制方差变化方式。

## 10. Diffusion 的三种数学视角

### 10.1 Latent Variable Model 视角

干净数据 $x_0$ 对应一串不可见的带噪变量：

$$
x_1,x_2,\dots,x_T
$$

这类似 VAE：有观测变量，也有潜变量。训练目标可以通过最大化似然下界推出来。

### 10.2 Score Matching 视角

Score function 是：

$$
\nabla_x \log p(x)
$$

它表示在数据空间中，哪个方向概率密度更高。扩散模型可以理解为学习不同噪声等级下的 score field：

$$
\nabla_x \log p_t(x)
$$

推理时沿着这些方向逐步走向高概率数据区域。

### 10.3 SDE / ODE 视角

扩散采样也可以看成求解微分方程：模型学习一个连续时间的动力系统，把噪声分布运输到数据分布。

Rectified Flow 中的简单更新：

$$
x_{t-\Delta t}=x_t-\Delta t f_\theta(x_t,t)
$$

可以看作一种 Euler 积分近似。更复杂的采样器本质上是在使用更好的数值积分方法。

## 11. Latent Diffusion Model：为什么在潜空间里扩散

### 11.1 动机

直接在像素空间做扩散很贵，尤其是高分辨率图像。Latent Diffusion 的核心思想是：先把图像压缩到低维潜空间，再在潜空间中做扩散。

原图像空间：

$$
x \in \mathbb{R}^{H\times W\times 3}
$$

潜空间：

$$
h = E(x)
$$

常见做法是空间下采样，例如 $8\times$ 下采样，并把通道数增加。

### 11.2 架构

LDM 通常包含三部分：

1. 编码器 $E$：把图像压缩成 latent；
2. 解码器 $D$：把 latent 还原成图像；
3. 扩散模型：在 latent 空间中学习去噪或速度场。

训练扩散模型时，编码器通常冻结：

$$
h=E(x)
$$

然后在 $h$ 上加噪，训练模型去噪。

### 11.3 训练流程

第一阶段：训练 autoencoder / VAE。

$$
x \xrightarrow{E} h \xrightarrow{D} \hat x
$$

它需要保证重建质量足够好，否则后续扩散模型生成的 latent 再好，解码出来也会模糊。

第二阶段：冻结编码器，在 latent 空间训练扩散模型。

$$
h_t=(1-t)h+tz
$$

$$
\mathcal{L}
=
\|f_\theta(h_t,t)-(z-h)\|_2^2
$$

第三阶段：推理时先在 latent 空间采样，再用解码器还原图像。

$$
z \to h_0 \to D(h_0)=x
$$

### 11.4 为什么 LDM 里会同时出现 VAE、GAN、Diffusion

VAE 提供可压缩、可采样的潜空间；GAN 或判别器帮助 autoencoder 的重建结果更清晰；Diffusion 负责在 latent 空间中生成高质量样本。

所以现代生成模型不是单纯的 VAE、GAN 或 Diffusion，而是三者组合：

$$
\text{VAE/GAN Autoencoder} + \text{Latent Diffusion}
$$

## 12. Diffusion Transformer，简称 DiT

### 12.1 为什么用 Transformer

扩散模型早期常用 U-Net，但现代大模型更倾向于 Transformer，因为 Transformer 易于扩展到大规模数据和大参数量。

DiT 的输入通常包括：

- 带噪 latent tokens；
- 时间步 $t$；
- 条件信号 $y$，如文本 embedding。

### 12.2 条件信息如何注入

时间步 $t$ 常通过 scale-shift 或 AdaLN 一类机制注入，即根据 $t$ 生成缩放和平移参数，调制中间激活。

文本条件常通过 cross-attention 或 joint attention 注入：

$$
\text{image latent tokens} \leftrightarrow \text{text tokens}
$$

这样模型可以根据文本内容调整去噪方向。

## 13. Text-to-Image 与 Text-to-Video Pipeline

### 13.1 Text-to-Image

文本生成图像的典型流程：

$$
\text{prompt}
\to
\text{text encoder}
\to
\text{text embedding}
$$

然后扩散模型在 latent 空间中迭代去噪：

$$
h_T \to h_{T-1}\to \cdots \to h_0
$$

最后通过 VAE decoder 得到图像：

$$
x = D(h_0)
$$

整体结构：

$$
\text{Text Encoder} + \text{Latent Diffusion/DiT} + \text{VAE Decoder}
$$

### 13.2 Text-to-Video

视频生成与图像生成类似，但 latent 多了时间维度：

$$
h \in \mathbb{R}^{T\times H\times W\times C}
$$

因此 token 数量大幅增加，Transformer 序列长度会非常长。图像模型可能处理上千 token，视频模型可能处理数万 token，这也是视频生成训练和推理昂贵的主要原因。

## 14. Diffusion 的推理慢与 Distillation

扩散模型的问题是采样慢，因为推理要多次调用大模型：

$$
x_T \to x_{T-1}\to \cdots \to x_0
$$

原始模型可能需要 30、50、100 步。Distillation 的目标是把多步采样压缩成更少步骤，甚至一步生成。

核心权衡：

$$
\text{更少采样步数} \quad \Longleftrightarrow \quad \text{可能损失生成质量}
$$

所以 distillation 的重点是尽量减少推理步数，同时保持样本质量。

## 15. Autoregressive Models 的回归：离散 latent 生成

现代生成模型中，自回归模型也会重新出现。做法是先用离散 VAE 或 tokenizer 把图像压缩成离散 latent tokens，再在 token 序列上训练自回归模型：

$$
x \to z_1,z_2,\dots,z_n
$$

$$
p(z)=\prod_i p(z_i|z_{<i})
$$

这说明现代生成 pipeline 经常不是单一范式，而是把 VAE、GAN、Diffusion、Autoregressive Model 组合使用。

## 本讲总结

1. 生成模型的核心问题是学习从简单分布到真实数据分布的采样方法，而不是只做分类。
2. GAN 放弃显式密度，通过生成器和判别器的对抗，让 $p_G$ 逼近 $p_{\text{data}}$；理论上优雅，但训练不稳定。
3. Diffusion 把生成问题改写成逐步去噪问题，因此有稳定的监督式损失，训练更可控。
4. Rectified Flow 用最直观的方式表达扩散：采样 $x,z,t$，构造 $x_t$，训练模型预测速度 $z-x$，推理时沿反方向从噪声走回数据。
5. CFG 通过同时学习有条件和无条件方向，在推理时放大条件信号，是现代文本生成图像模型的重要技巧。
6. LDM 解决了像素空间扩散太贵的问题：先用 VAE/GAN autoencoder 压缩图像，再在 latent 空间扩散。
7. DiT 表示扩散模型正在从 U-Net 架构走向 Transformer 架构；文本、图像、视频都可以统一成 token 序列建模。
8. 现代生成模型不是 VAE、GAN、Diffusion 三选一，而是经常把它们组合成完整 pipeline。