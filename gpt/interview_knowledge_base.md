# Mini GPT 项目面试知识库

> 用途：这是给面试辅助软件上传的项目知识库。内容围绕“我如何讲这个 mini GPT 项目、面试官可能怎么追问、我该如何回答”组织，而不是泛泛介绍 GPT。

## 1. 30 秒项目介绍

我做了一个从零实现 mini GPT 的项目，目标不是直接调用现成大模型接口，而是理解 GPT 的核心训练闭环和 decoder-only Transformer 的内部结构。

我的实现路线是先从最简单的 Bigram 模型开始，跑通数据读取、字符级 tokenizer、batch 采样、loss、反向传播、优化器和生成；然后逐步替换模型结构，加入 token embedding、position embedding、causal self-attention、multi-head attention、feedforward、残差连接、LayerNorm、TransformerBlock、参数初始化和验证集评估。

这个项目让我比较清楚地理解了 GPT 的核心流程：

```text
文本 -> tokenizer -> token id -> batch -> GPT forward -> logits -> cross entropy -> backward -> AdamW -> autoregressive generation
```

## 2. 2 分钟项目详解

这个项目复现的是一个小型字符级 GPT。训练数据是 tiny Shakespeare 文本，模型目标是根据前面的字符预测下一个字符。

我没有一开始就写完整 GPT，而是按“先跑通闭环，再逐步替换结构”的方式实现：

1. 先读取文本，构建字符级 tokenizer，把每个字符映射成整数 id。
2. 把整篇文本编码成一维 token 序列，并切分成训练集和验证集。
3. 写 batch 采样函数。每个样本里，`x` 是一段 token，`y` 是整体右移一位的目标序列。
4. 先写 Bigram 模型，验证训练闭环能跑通。
5. 再加入 token embedding 和 position embedding。
6. 实现 causal self-attention，用 mask 防止模型看到未来 token。
7. 实现 multi-head attention，让模型从多个子空间读取上下文。
8. 实现 feedforward 和 TransformerBlock。
9. 堆叠多个 block，形成 GPTLanguageModel。
10. 使用 AdamW 训练，并用 validation loss 评估。
11. 最后用自回归方式生成文本。

我现在的本地版本主要用于 CPU 小规模调试；更大的参数组会放到租的 GPU VM 上训练。

## 3. 面试中我想突出什么

这个项目最值得强调的不是“模型效果有多强”，而是我真的理解了 GPT 的核心组件：

- tokenizer 如何把文本变成模型能处理的 token id。
- 为什么训练目标是 next-token prediction。
- batch 里的 `x` 和 `y` 为什么要错一位。
- logits 和 targets 的 shape 如何对应 cross entropy。
- causal mask 为什么能防止偷看未来。
- multi-head attention 为什么要把多个 head 的输出拼接。
- TransformerBlock 为什么需要残差连接和 LayerNorm。
- 生成时为什么只取最后一个时间步的 logits。
- 生成时为什么要裁剪到最后 `block_size` 个 token。
- `model.train()`、`model.eval()` 和 `torch.no_grad()` 的区别。
- CPU 调试参数和 GPU 训练参数如何取舍。

## 4. 如果面试官问“你这个项目是不是照着视频抄的？”

可以回答：

我参考了 Karpathy 的 GPT from scratch 思路，但我不是直接一口气抄完整代码。我是按模块逐步复现的：先 Bigram 跑通训练闭环，再逐步加入 embedding、attention、multi-head、feedforward、block 和完整 GPT。每一步我都会检查 tensor shape、loss 是否下降、生成是否能跑，以及常见 bug，比如 ModuleList 不能直接调用、position embedding 越界、forgot backward、device mismatch 等。

如果被追问：

我认为这个项目的价值在于，我能解释每个模块为什么存在、输入输出 shape 是什么、错误时怎么 debug，而不是只知道最终代码长什么样。

## 5. 如果面试官问“你怎么使用 AI 辅助？”

可以回答：

这个岗位允许 AI 辅助，所以我在项目里把 AI 当作代码 review 和学习辅助工具。我会先自己实现每个模块，然后让 AI 帮我检查 shape、训练逻辑和常见错误。比如我自己写 self-attention、multi-head、feedforward 和 block，再让 AI 检查是否有维度拼接错误、是否漏了 residual connection、是否忘记 `loss.backward()`。

我不会让 AI 直接替我生成完整项目，因为我的目标是理解每个模块。AI 更像一个 pair reviewer，帮我暴露盲点，但最终代码和概念我都要能自己解释。

## 6. 当前项目架构

当前 mini GPT 的结构是：

```text
输入 idx: (B, T)
  -> token embedding: (B, T, n_embd)
  -> position embedding: (T, n_embd)
  -> 相加得到 x: (B, T, n_embd)
  -> 多层 TransformerBlock
  -> final LayerNorm
  -> lm_head
  -> logits: (B, T, vocab_size)
```

训练时：

```text
logits reshape 为 (B*T, vocab_size)
targets reshape 为 (B*T)
用 cross entropy 计算 loss
```

生成时：

```text
context -> 裁剪最后 block_size 个 token -> forward -> 取 logits[:, -1, :] -> softmax -> multinomial 采样 -> 拼接新 token
```

## 7. 关键 Shape 速查

```text
B = batch size
T = sequence length / block_size
C = n_embd
V = vocab_size
H = head_size
```

输入输出：

```text
idx:      (B, T)
targets:  (B, T)
logits:   (B, T, V)
loss 用:  logits -> (B*T, V), targets -> (B*T)
```

embedding：

```text
token embedding:    (B, T, C)
position embedding: (T, C)
x:                  (B, T, C)
```

self-attention：

```text
q, k, v:       (B, T, H)
attention wei: (B, T, T)
head output:   (B, T, H)
```

multi-head：

```text
每个 head: (B, T, H)
拼接后:    (B, T, n_head * H)
投影后:    (B, T, C)
```

## 8. Bigram 为什么重要

面试回答：

Bigram 模型本身很弱，它只根据当前 token 预测下一个 token。但我先实现它是为了验证整个训练闭环：数据读取、tokenizer、batch 采样、loss、backward、optimizer 和 generate。如果 Bigram 都跑不通，直接写 Transformer 会很难定位 bug。

Bigram 的本质：

```text
当前 token id -> embedding table -> 下一个 token 的 logits
```

它不真正利用长上下文，所以效果有限。

## 9. 字符级 tokenizer vs token/subword tokenizer

当前项目使用字符级 tokenizer，这是为了学习和调试简单。

字符级 tokenizer：

- 每个字符是一个 token。
- 词表小。
- 实现简单。
- 序列长，训练慢。
- 不适合生产级 GPT。

token/subword tokenizer：

- 一个 token 可以是单词、子词或 byte-level BPE 片段。
- 词表大。
- 序列短，效率高。
- 现代 GPT 常用。

面试回答：

我这个项目用字符级 tokenizer 是为了从零理解语言模型训练流程。但真实 GPT 通常不会用字符级 tokenizer，而是使用 subword 或 byte-level BPE tokenizer。原因是字符级序列太长，会浪费上下文窗口并增加 attention 的二次方计算成本。

## 10. BPE 和 byte-level tokenizer 怎么说

如果面试官问 BPE：

BPE 的核心思想是从小单位开始，比如字符或字节，反复统计最常见的相邻 pair，并把高频 pair 合并成新的 token。这样可以让常见词或词片段变成较短 token 序列，同时保留处理罕见词的能力。

如果问 byte-level BPE：

byte-level BPE 以字节作为基础单位，因此理论上可以覆盖任意 Unicode 文本，不容易遇到 unknown token。现代 GPT 类模型常用这类 tokenizer，因为它兼顾通用性和压缩效率。

## 11. Causal Self-Attention 如何解释

Self-attention 让每个位置可以从前面的 token 读取信息。

Q、K、V 的直觉：

```text
Query: 我这个位置想找什么？
Key:   每个位置提供什么匹配标签？
Value: 真正被读取的信息内容。
```

计算过程：

```text
q @ k.T -> attention score
除以 sqrt(head_size) -> 稳定 softmax
causal mask -> 禁止看未来
softmax -> 注意力权重
权重 @ v -> 聚合上下文
```

为什么需要 mask：

训练语言模型时，第 t 个位置只能根据 `0...t` 的 token 预测下一个 token。如果它能看到未来 token，就相当于作弊。

## 12. Multi-Head Attention 如何解释

可以这样说：

单头 attention 是从一个表示子空间看上下文关系；multi-head attention 是多个 head 并行看同一个序列，每个 head 可以学习不同关系，比如局部字符组合、换行结构、标点模式或更远距离依赖。

流程：

```text
x -> head1 -> (B, T, head_size)
x -> head2 -> (B, T, head_size)
...
concat -> (B, T, n_embd)
linear projection -> (B, T, n_embd)
```

关键条件：

```text
n_embd 必须能被 n_head 整除
head_size = n_embd // n_head
```

## 13. FeedForward 如何解释

FeedForward 是每个位置独立应用的 MLP。

Attention 负责 token 之间通信；FeedForward 负责对每个 token 的 hidden vector 做非线性加工。

结构：

```text
Linear(n_embd, 4*n_embd)
ReLU
Linear(4*n_embd, n_embd)
Dropout
```

它不会混合不同时间位置的信息，因为 `nn.Linear` 只作用在最后一维。

## 14. TransformerBlock 如何解释

当前实现是 GPT 风格的 pre-norm block：

```text
x = x + MultiHeadAttention(LayerNorm(x))
x = x + FeedForward(LayerNorm(x))
```

两个关键点：

1. 残差连接：保留原始信息，让每个子层学习补充量。
2. LayerNorm：稳定每个 token 的 hidden vector 分布。

为什么是 pre-norm：

先 LayerNorm 再进入 attention/MLP，训练深层 Transformer 更稳定。

## 15. 训练循环如何解释

一轮训练：

```text
xb, yb = sample batch
logits, loss = model(xb, yb)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

重点：

- forward 时 PyTorch 自动建立计算图。
- `loss.backward()` 根据 loss 反向传播，得到每个参数的梯度。
- PyTorch 默认梯度累加，所以每轮要 `zero_grad()`。
- `optimizer.step()` 根据梯度更新参数。

如果问 AdamW：

AdamW 是 Transformer 常用优化器，使用自适应学习率和动量信息，并把 weight decay 解耦，通常比朴素 SGD 更稳定。

## 16. eval、train、no_grad 的区别

```text
model.train()
```

切换到训练模式，dropout 会启用。

```text
model.eval()
```

切换到评估模式，dropout 会关闭。

```text
torch.no_grad()
```

关闭计算图构建，节省内存和时间。

面试回答：

`eval()` 不等于不计算梯度，它只改变模型中某些层的行为；真正关闭梯度记录要用 `torch.no_grad()`。

## 17. 常见 Bug 与我如何 Debug

1. 忘记 `loss.backward()`

表现：optimizer.step() 后 loss 不下降。

2. `ModuleList` 直接调用

错误：`ModuleList is missing forward`。

解决：用 for 循环执行每个 block，或改用 `nn.Sequential`。

3. position embedding 越界

表现：生成时序列越来越长，位置 id 超过 block_size。

解决：generate 时只把最后 `block_size` 个 token 送进模型。

4. `torch.cat` 拼错维度

Multi-head attention 应该沿最后一维拼接，即 `dim=-1`。

5. device mismatch

模型、batch、context 必须都在同一个 device。

6. cross entropy shape 错误

训练时要把 logits 从 `(B, T, V)` reshape 成 `(B*T, V)`，targets 从 `(B, T)` reshape 成 `(B*T)`。

## 18. 如果面试官问“为什么生成质量不好？”

可以回答：

首先，这是一个很小的字符级模型，数据量、模型规模和训练步数都很有限。其次，字符级建模比 subword 建模更难，因为序列更长，模型要先学会拼字母再学语言结构。再次，我本地主要是 CPU 调试参数，重点是验证结构正确，不是追求生成质量。真正提升效果需要更大模型、更长训练、更合理 tokenizer 和更多数据。

## 19. 如果面试官问“这个项目和真实 GPT 差距在哪里？”

差距包括：

- 真实 GPT 使用 subword/byte-level tokenizer，不是简单字符级 tokenizer。
- 真实模型参数量大得多。
- 真实训练数据规模大得多。
- 真实训练用分布式 GPU/TPU。
- 会有学习率调度、混合精度、checkpoint、梯度裁剪等工程。
- 真实产品模型还会有指令微调、偏好优化或安全对齐。
- 生产推理会用 KV cache，而当前 generate 每步都重新跑上下文。

## 20. 如果问 KV Cache

当前项目没有实现 KV cache。

可以这样回答：

我现在的 generate 每生成一个 token，都会把最近 `block_size` 个 token 重新送进模型计算，所以会重复计算历史上下文。KV cache 的思想是在自回归生成时缓存历史 token 的 key 和 value，新 token 只需要计算自己的 q/k/v，并和缓存的 k/v 做 attention，从而显著加速推理。这是生产级 GPT 推理的重要优化。

## 21. 如果问学习率、参数量和 CPU/GPU 调参

本地 CPU 调试，我会用较小参数：

```python
batch_size = 8
block_size = 32
max_iters = 1000
eval_iters = 10
n_embd = 48
n_head = 4
n_layer = 2
```

GPU VM 训练可以放大：

```python
batch_size = 32
block_size = 128
max_iters = 2000
eval_iters = 20
n_embd = 180
n_head = 6
n_layer = 4
```

注意：

```text
attention 计算大致随 block_size^2 增长
n_embd 和 n_layer 增大都会明显增加计算量
```

## 22. 面试高频快答

### 问：为什么 logits 只取最后一个时间步生成？

答：自回归生成时，每一步只需要预测当前上下文之后的下一个 token。模型虽然对每个位置都输出 logits，但最后一个位置才对应“下一个 token”的预测。

### 问：为什么 validation loss 有意义？

答：validation 数据没有参与参数更新，可以评估泛化能力。如果 train loss 很低但 val loss 高，说明可能过拟合。

### 问：为什么 dropout 在 eval 时要关闭？

答：训练时 dropout 是正则化，评估和生成时需要稳定输出，所以 eval 模式会关闭 dropout。

### 问：为什么要初始化权重？

答：合理初始化可以让训练更稳定。GPT 常用 Linear 和 Embedding 权重按均值 0、标准差 0.02 的正态分布初始化。

### 问：为什么 feedforward 中间扩展到 4 倍？

答：这是 Transformer 常见设计。中间维度更大可以增强每个 token 位置的非线性表达能力，然后再投影回 `n_embd`。

## 23. 面试结尾总结

如果要总结这个项目，我会说：

这个项目让我从底层理解了 GPT 的关键机制。我不是直接调用 API，而是实现了从 tokenizer、batch、Bigram baseline 到完整 decoder-only Transformer 的训练与生成流程。虽然模型规模很小，但它包含了 GPT 的核心结构：token/position embedding、causal multi-head self-attention、feedforward、residual connection、LayerNorm、AdamW 训练和自回归生成。

