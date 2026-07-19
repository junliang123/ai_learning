# CS231n 2026 Lecture 8 —— Attention and Transformers

## 一、从 Seq2Seq 的信息瓶颈到注意力

传统 RNN 编码器把整个输入序列压缩进一个固定向量，长序列中的细节很容易丢失。注意力机制让解码器在每个生成步骤重新查看所有编码器状态，并根据当前需求加权汇总信息。

其核心是三组向量：

- **Query**：当前想查找什么；
- **Key**：每个位置可以用什么特征被匹配；
- **Value**：匹配后真正取回的内容。

缩放点积注意力写作：

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

$QK^\top$ 计算所有查询与键的相似度，softmax 把它变成权重，再对 $V$ 加权求和。除以 $\sqrt{d_k}$ 可以避免维度较大时点积过大、softmax 过早饱和。

---

## 二、自注意力、交叉注意力与掩码

**交叉注意力** 的 $Q$ 来自一个序列，$K,V$ 来自另一个序列，例如文本解码器读取图像特征。

**自注意力** 的 $Q,K,V$ 都来自同一输入 $X$，但通过不同线性层投影：

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$$

这样每个 token 都能根据内容直接聚合其他 token 的信息，任意两个位置之间只需一次交互。

自回归语言模型不能偷看未来 token，因此使用 **causal mask**：把未来位置的注意力 logit 设为 $-\infty$，使 softmax 后的权重为 0。视觉编码器通常不需要因果掩码，可以进行双向全局交互。

---

## 三、位置编码

纯自注意力只关心输入向量的集合，对位置重新排列会得到相应重排的输出，本身不知道先后与空间位置。因此必须加入位置信息，例如：

- 固定的正弦、余弦位置编码；
- 可学习位置 embedding；
- 相对位置编码或 RoPE。

文本需要表达 token 顺序，图像则需要同时表达水平和垂直位置。没有位置编码时，模型无法区分相同 token 以不同顺序出现的序列。

---

## 四、多头注意力

单个注意力可能只学习一种相似关系。多头注意力把通道分成多个子空间，各自独立计算注意力，再拼接并投影：

$$
\operatorname{MHA}(X)
=\operatorname{Concat}(head_1,\ldots,head_H)W_O
$$

不同头可以关注局部邻近、长距离依赖、语义对应或空间关系。多个头并不保证自动产生清晰可解释的分工，但显著提高了表达能力。

对长度为 $N$ 的序列，标准注意力要形成 $N\times N$ 的权重关系，时间和显存通常为 $O(N^2)$。FlashAttention 等实现通过分块计算减少中间矩阵的显存读写，但不会改变标准全局注意力的二次计算量。

---

## 五、Transformer Block

Transformer 将注意力作为基本计算单元，并重复堆叠结构相同的 block。一个典型 block 包括：

1. LayerNorm；
2. 多头自注意力；
3. 残差连接；
4. LayerNorm；
5. 对每个 token 独立应用两层 MLP；
6. 再次残差连接。

MLP 通常先把通道维度扩张，再压回原维度：

$$
\operatorname{MLP}(x)=W_2\,\phi(W_1x+b_1)+b_2
$$

注意力负责 token 之间的信息交换，MLP 负责每个 token 内部的通道变换；残差连接和归一化帮助深层网络稳定训练。原始 Transformer 常用 post-norm，现代大模型中 pre-norm 更常见。

---

## 六、Transformer 语言模型

语言模型先将 token 映射为 embedding，叠加位置信息，再经过多层带因果掩码的 Transformer。最后把每个位置的特征投影到词表大小，并以交叉熵训练下一 token 预测。

与 RNN 相比，训练时所有序列位置可以并行计算，且远距离 token 能直接交互；代价是标准注意力的显存和计算随序列长度平方增长。生成仍然是逐 token 的，因为下一步输入依赖刚生成的结果。

---

## 七、Vision Transformer

Vision Transformer（ViT）把图像切成固定大小的 patch，将每个 patch 展平并线性投影为 token：

$$
\text{image}\rightarrow\text{patches}\rightarrow\text{token embeddings}
$$

随后加入二维位置信息，使用不带因果掩码的 Transformer 编码。最终可以汇聚所有 patch 特征，或使用专门的分类 token 完成预测。Patch embedding 也可以看作卷积核大小和步幅都等于 patch 大小的卷积。

ViT 减少了 CNN 固有的局部连接与平移先验，依靠数据学习空间关系；在大规模预训练下表现突出，但标准全局注意力在高分辨率图像上仍然昂贵。

---

## 八、本讲总结

1. 注意力根据 Query 与 Key 的匹配，从 Value 中动态提取信息。
2. 自注意力让同一序列中的所有位置直接交互，因果掩码阻止语言模型读取未来。
3. 位置编码弥补注意力自身不感知顺序的问题，多头机制并行学习不同关系。
4. Transformer block 由注意力、MLP、残差连接和归一化组成。
5. Transformer 易于并行并能建模全局关系，但标准注意力对序列长度具有二次复杂度。
6. ViT 将图像切成 patch token，使同一套架构可以用于视觉任务。
