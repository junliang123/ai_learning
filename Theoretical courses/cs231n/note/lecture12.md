# CS231n 2025 Lecture 12 —— Self-Supervised Learning

## 一、为什么需要 Self-Supervised Learning？

### 1. 核心问题
监督学习依赖大量人工标注，而现实中无标签数据远多于有标签数据，因此希望利用无标签数据学习通用特征。

### 2. 核心思想
设计 **Pretext Task（预训练任务）**，利用数据本身生成监督信号，训练 Encoder 学习高质量 Representation，再迁移到下游任务。

### 3. 整体流程
Pretext Task → Encoder → Representation → Downstream Task（分类、检测、分割等）

---

# 二、Pretext Task（经典自监督任务）

## 1. Rotation Prediction（旋转预测）
**任务：** 预测图片旋转角度（0°/90°/180°/270°）。

**思想：** 能判断图片方向，说明模型理解了物体结构（Visual Common Sense）。

**特点：**
- 四分类任务
- 预训练效果明显优于随机初始化
- 简单数据集最终仍可能接近监督学习

---

## 2. Jigsaw Puzzle（拼图）
**任务：** 打乱图片 Patch，预测正确排列。

两种方式：
- Patch 相对位置预测（8分类）
- 整体排列预测（论文简化为64分类）

**思想：** 学习图像整体空间结构。

---

## 3. Image Inpainting（图像补全）
**任务：** 遮挡部分图片，恢复缺失区域。

**结构：**
Encoder → Decoder → Reconstruction

**Loss：**
仅计算 Mask 区域。

**特点：**
- 属于 AutoEncoder 思想
- 重建结果容易模糊，因此加入 GAN 提升真实性

---

## 4. Image Colorization（图像上色）
**任务：** 给灰度图预测颜色。

LAB 色彩空间：
- L：亮度
- AB：颜色

Split-Brain AutoEncoder：
- L → AB
- AB → L

**扩展：**
视频上色过程中，模型还能学习像素对应关系，可用于 Tracking。

---

# 三、Masked AutoEncoder（MAE）

## 核心思想
随机 Mask 大量 Patch（通常75%），仅利用剩余 Patch 重建完整图片。

## 结构
Encoder（ViT） → Decoder → Reconstruction

## Loss
仅计算 Mask Patch 的 MSE。

## 优势
- Mask 比例高，任务更困难，特征更好
- 同一图片可随机 Mask，多次利用
- Encoder 仅处理少量 Patch，训练效率高

## 下游训练
- **Linear Probing**：冻结 Encoder，仅训练分类器，用于评价 Representation。
- **Fine-tuning**：继续训练 Encoder，适用于实际任务。

---

# 四、Contrastive Learning（对比学习）

## 核心思想
- Positive Pair：同一图片不同增强
- Negative Pair：不同图片

目标：
- Positive 更近
- Negative 更远

---

## InfoNCE Loss
本质类似 Softmax Cross Entropy，通过最大化 Positive 相似度、最小化 Negative 相似度学习特征。

特点：
- Negative 越多效果越好
- 大 Batch 更容易训练

---

# 五、SimCLR

流程：
Image → Data Augmentation → Encoder → Projection Head → InfoNCE Loss

特点：
- 每张图片生成两个 View
- Batch 内其它样本均作为 Negative
- 需要较大的 Batch Size

---

# 六、MoCo

提出原因：
解决 SimCLR 对大 Batch 的依赖。

核心改进：
- Queue：保存历史 Negative Sample
- Momentum Encoder：稳定更新特征

优点：
小 Batch 下仍能获得大量 Negative Sample。

---

# 七、本课总结

**核心目标：**
利用无标签数据，通过 Pretext Task 学习通用视觉表示（Representation），提升下游任务性能。

**重点掌握：**
- Self-Supervised Learning 思想
- Pretext Task
- Representation
- Rotation / Jigsaw / Inpainting / Colorization
- MAE
- Linear Probing vs Fine-tuning
- Contrastive Learning
- InfoNCE
- SimCLR
- MoCo