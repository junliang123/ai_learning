# CS231n 2026 Lecture 5 —— Image Classification with CNNs

## 一、为什么图像需要 CNN

传统方法先人工提取 Color Histogram、HOG、SIFT 等特征，再训练分类器。全连接网络则把图像直接展平，但会破坏二维空间结构，并产生大量参数。

CNN 使用针对图像的两个核心假设：

- **局部连接**：相邻像素之间通常更相关；
- **权重共享**：同一种局部模式可能出现在图像任意位置。

因此 CNN 能保留空间结构，并通过端到端反向传播自动学习特征。

---

## 二、卷积层

输入通常写成 $C_{in}\times H\times W$。一个卷积核大小为 $C_{in}\times K_H\times K_W$，它覆盖输入的全部通道，在空间上滑动并做点积，产生一张 activation map。使用 $C_{out}$ 个卷积核，就得到 $C_{out}$ 个输出通道。

卷积层参数：

$$
W\in\mathbb R^{C_{out}\times C_{in}\times K_H\times K_W},
\qquad b\in\mathbb R^{C_{out}}
$$

可学习参数量为：

$$
C_{out}(C_{in}K_HK_W+1)
$$

它与输入的空间尺寸无关，这是卷积比全连接层高效的重要原因。

---

## 三、Padding、Stride 与输出尺寸

对输入宽度 $W$、卷积核大小 $K$、padding $P$、stride $S$：

$$
W'=\frac{W-K+2P}{S}+1
$$

高度同理。结果必须为整数。

- **Padding**：在边界补值，避免特征图过快缩小；奇数卷积核常用 $P=(K-1)/2$ 实现 same padding。
- **Stride**：控制滑动步长；$S>1$ 会下采样空间尺寸。
- 常见配置：$3\times3$、stride 1、padding 1。

---

## 四、感受野与层次特征

单个卷积输出只依赖输入中的局部区域，这个区域叫 **receptive field**。stride 为 1 且每层核大小均为 $K$ 时，堆叠 $L$ 层后的理论感受野为：

$$
1+L(K-1)
$$

浅层卷积通常学习边缘、方向和对立颜色；更深层把局部特征组合成纹理、部件乃至完整物体。网络通过逐层组合形成层次化表示。

---

## 五、池化与下采样

池化在每个通道内独立聚合局部区域，常见为 Max Pooling 和 Average Pooling。它没有可学习参数。

经典配置是 $2\times2$ Max Pool、stride 2，使高和宽减半。池化可降低计算量、扩大后续层的有效感受野，并降低模型对微小位置变化的敏感度。现代架构也常用 stride convolution 代替池化完成下采样。

---

## 六、平移等变性

卷积核在所有位置共享，因此输入发生平移时，特征图通常会发生对应平移：

$$
\operatorname{Conv}(T(x))\approx T(\operatorname{Conv}(x))
$$

这叫 **translation equivariance（平移等变）**，不是严格的平移不变。池化、下采样和最终的全局聚合才会逐步增强对小幅平移的不敏感性。

---

## 七、本讲总结

1. CNN 保留图像二维结构，利用局部连接和权重共享高效学习特征。
2. 一个卷积核产生一个输出通道，输出形状由 kernel、padding 和 stride 决定。
3. 深层卷积逐步扩大感受野，并从低级边缘组合出高级语义。
4. 池化或步幅卷积负责下采样。
5. CNN 的核心归纳偏置是局部性与平移等变性。
