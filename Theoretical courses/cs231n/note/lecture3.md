# CS231n 2026 Lecture 3 —— Regularization and Optimization

## 一、完整训练目标

训练分类器既要拟合数据，也要限制模型复杂度：

$$
L(W)=\frac{1}{N}\sum_{i=1}^{N}L_i(W)+\lambda R(W)
$$

- **Data Loss**：让预测匹配训练标签。
- **Regularization**：抑制对训练噪声的过度拟合，提升测试集泛化。
- $\lambda$：正则化强度，是通过验证集选择的超参数。

常见正则项：

$$
R_{L2}(W)=\sum_j W_j^2,\qquad
R_{L1}(W)=\sum_j |W_j|
$$

L2 倾向于让信息分散在多个较小权重中；L1 更容易产生稀疏权重。Elastic Net 同时使用 L1 与 L2。

---

## 二、梯度与梯度检查

梯度给出损失上升最快的方向，因此参数应沿负梯度更新：

$$
W\leftarrow W-\eta\nabla_W L
$$

其中 $\eta$ 是学习率。

- **数值梯度**：通过微小扰动近似导数，容易实现，但慢且只是近似值。
- **解析梯度**：由微积分和反向传播得到，快速且精确，但实现可能出错。

实践中使用解析梯度训练，并用数值梯度做 **gradient check**。

---

## 三、SGD 与 Mini-batch

完整梯度需要遍历整个训练集，数据大时成本过高。SGD 用一个 mini-batch 近似完整梯度：

$$
g_t\approx \frac{1}{B}\sum_{i\in\mathcal B_t}\nabla_W L_i
$$

mini-batch 提升计算效率，但也引入梯度噪声。普通 SGD 的典型困难包括：

- 在不同方向曲率差异很大时来回震荡；
- 在鞍点或平坦区域进展缓慢；
- mini-batch 梯度噪声使更新方向不稳定。

---

## 四、常用优化器

### 1. SGD + Momentum

Momentum 累积历史更新方向，像带惯性的速度：

$$
v_t=\rho v_{t-1}+g_t,\qquad W_t=W_{t-1}-\eta v_t
$$

它能减少陡峭方向上的震荡，并加速稳定方向上的移动。常见 $\rho$ 为 0.9 或 0.99。

### 2. RMSProp

RMSProp 维护梯度平方的移动平均，为每个参数自动缩放学习率：陡峭方向更新变小，平坦方向相对加快。

### 3. Adam 与 AdamW

Adam 同时结合：

- 梯度一阶矩：类似 Momentum；
- 梯度二阶矩：类似 RMSProp；
- 初始时刻的 bias correction。

AdamW 将 weight decay 与 Adam 的矩估计解耦，通常比直接把 L2 项混入 Adam 梯度更符合预期。AdamW 是现代深度学习中很常见的默认选择；SGD + Momentum 经过充分调参后也可能泛化得更好。

---

## 五、学习率策略

学习率通常比优化器名称更重要：

- 太大：损失震荡甚至发散；
- 太小：训练极慢，可能停在平坦区域；
- 常用衰减：Step、Cosine、Linear、Inverse-Square-Root；
- **Warmup**：训练初期从很小的学习率逐步升高，避免大模型或大 batch 初期不稳定。

典型训练策略是前期使用较大学习率快速探索，后期逐步减小学习率进行精细收敛。

---

## 六、本讲总结

1. 正则化在训练误差与泛化能力之间做权衡。
2. 训练使用解析梯度，数值梯度主要用于检查实现。
3. SGD 用 mini-batch 提升效率，但带来噪声和优化困难。
4. Momentum 改善更新方向，RMSProp 自适应缩放，Adam 结合二者，AdamW 正确处理 weight decay。
5. 优化器必须与合适的学习率、衰减策略和验证集调参一起使用。
