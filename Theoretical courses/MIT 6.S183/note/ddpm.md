# DDPM 最终总结

DDPM 可以一句话概括：

**人为规定一个从图像到噪声的加噪过程，然后训练模型预测噪声；生成时从纯噪声开始，反过来一步步去噪成图像。**

---

## 1. 前向过程：人为定义加噪规则

DDPM 先人为设定一组噪声强度参数：

$$
\beta_1,\beta_2,\ldots,\beta_T
$$

其中 $\beta_t$ 表示第 $t$ 步加入多少噪声。

然后定义：

$$
\alpha_t = 1-\beta_t
$$

$$
\bar{\alpha}_t = \prod_{s=1}^{t}\alpha_s
$$

含义是：

- $\beta_t$：第 $t$ 步加多少噪声；
- $\alpha_t$：第 $t$ 步保留多少信号；
- $\bar{\alpha}_t$：从 $0$ 到 $t$ 累计还保留多少原图信息。

前向过程是：

$$
x_0 \rightarrow x_1 \rightarrow x_2 \rightarrow \cdots \rightarrow x_T
$$

其中 $x_0$ 是真实图像，$x_T$ 接近纯高斯噪声。

---

## 2. 训练时：直接构造任意时刻的 $x_t$

理论上可以一步步加噪：

$$
x_0 \rightarrow x_1 \rightarrow \cdots \rightarrow x_t
$$

但实际训练时通常直接采样一个时间步 $t$，然后一步构造 $x_t$：

$$
x_t
=
\sqrt{\bar{\alpha}_t}x_0
+
\sqrt{1-\bar{\alpha}_t}\epsilon
$$

其中：

$$
\epsilon \sim \mathcal{N}(0,I)
$$

这个公式表示：

$$
x_t
=
\text{原图残留部分}
+
\text{总噪声部分}
$$

也就是说，$x_t$ 不是简单的 $x_0+\epsilon$，而是原图和噪声都带有权重。

---

## 3. 训练目标：预测总噪声

训练时，模型输入带噪图像 $x_t$ 和时间步 $t$，输出预测噪声：

$$
\epsilon_\theta(x_t,t)
$$

训练目标是让预测噪声接近真实加入的噪声 $\epsilon$：

$$
L
=
\left\|
\epsilon
-
\epsilon_\theta(x_t,t)
\right\|^2
$$

所以 DDPM 训练的不是：

$$
x_t \rightarrow x_{t-1}
$$

而是：

$$
(x_t,t) \rightarrow \epsilon
$$

也就是模型学习：

**给定任意噪声程度的图像 $x_t$，预测它相对于原图 $x_0$ 的总噪声 $\epsilon$。**

这可以看成一种自监督学习：标签 $\epsilon$ 不是人工标注的，而是训练时自己采样出来的。

---

## 4. 生成时：从纯噪声开始反向去噪

生成时没有真实图像 $x_0$，所以先采样纯噪声：

$$
x_T \sim \mathcal{N}(0,I)
$$

然后反向迭代：

$$
x_T \rightarrow x_{T-1} \rightarrow x_{T-2} \rightarrow \cdots \rightarrow x_0
$$

每一步都做：

1. 输入当前图像 $x_t$ 和时间步 $t$；
2. 模型预测总噪声 $\epsilon_\theta(x_t,t)$；
3. 根据预测噪声计算更干净的 $x_{t-1}$。

---

## 5. 为什么不能直接减噪声？

因为模型预测的是从 $x_0$ 到 $x_t$ 的总噪声，不是 $x_t \rightarrow x_{t-1}$ 这一小步的局部噪声。

所以生成时不能简单写成：

$$
x_{t-1}
=
x_t
-
\epsilon_\theta(x_t,t)
$$

正确做法是先利用预测噪声估计原图：

$$
\hat{x}_0
=
\frac{
x_t-\sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t,t)
}{
\sqrt{\bar{\alpha}_t}
}
$$

然后再根据 $\hat{x}_0$ 和 $x_t$ 反推出 $x_{t-1}$。

等价地，也可以直接写成 DDPM 的一步反向采样公式：

$$
x_{t-1}
=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t
-
\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}
\epsilon_\theta(x_t,t)
\right)
+
\sigma_t z
$$

其中：

$$
z\sim \mathcal{N}(0,I)
$$

这里的 $\sigma_t z$ 表示采样时额外加入的一点随机性。最后一步通常不再额外加随机噪声。

---

## 6. 最终理解版

DDPM 的完整逻辑可以记成三步：

**第一，前向加噪是人为定义的。**

$$
x_0 \rightarrow x_1 \rightarrow \cdots \rightarrow x_T
$$

通过预设的 $\beta_t$、$\alpha_t$、$\bar{\alpha}_t$，我们知道每个时刻 $x_t$ 中原图和噪声的比例。

**第二，训练模型预测总噪声。**

$$
x_t
=
\sqrt{\bar{\alpha}_t}x_0
+
\sqrt{1-\bar{\alpha}_t}\epsilon
$$

模型学习：

$$
\epsilon_\theta(x_t,t)\approx \epsilon
$$

**第三，生成时把总噪声预测转换成一步步去噪。**

$$
x_T \rightarrow x_{T-1} \rightarrow \cdots \rightarrow x_0
$$

每一步都预测当前 $x_t$ 中的总噪声，再根据 DDPM 的反向公式得到 $x_{t-1}$。

---

## 7. 最简记忆

DDPM 不是直接学“如何画图”，而是学：

**这张带噪图里有哪些噪声。**

训练时：

$$
(x_t,t) \rightarrow \epsilon
$$

生成时：

$$
x_T \rightarrow x_{T-1} \rightarrow \cdots \rightarrow x_0
$$

核心一句话：

**人为规定加噪过程，训练模型预测噪声，生成时从纯噪声开始反复预测噪声并按公式去噪。**