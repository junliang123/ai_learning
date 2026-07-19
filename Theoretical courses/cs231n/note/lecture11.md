# CS231n 2026 Lecture 11 —— Large-Scale Distributed Training

## 一、大规模训练的三个瓶颈

单个 GPU 拥有大量并行计算核心和高带宽显存，矩阵乘法还可由 Tensor Cores 以低精度高效执行。但训练大模型不只受峰值算力限制，还要考虑：

- **显存容量**：参数、梯度、优化器状态和中间激活都要占空间；
- **显存与网络带宽**：数据搬运或跨 GPU 通信可能比计算更慢；
- **并行利用率**：依赖关系、同步等待和负载不均会让设备空闲。

集群内部通信速度并不均匀：同一服务器内的 GPU 通常连接更快，跨服务器或跨机架更慢。因此并行策略需要让通信密集的操作尽量发生在高速互联的小范围设备之间。

---

## 二、通信集合操作

分布式训练依靠少数 collective primitives 交换张量：

- **All-Reduce**：聚合所有 GPU 的张量，并把结果发回每个 GPU；数据并行常用它同步梯度。
- **Reduce-Scatter**：先聚合，再让每个 GPU 只保留结果的一部分；FSDP 反向传播常用。
- **All-Gather**：每个 GPU 持有一个分片，通信后每个 GPU 都得到完整张量；FSDP 前向传播常用。
- **All-to-All**：把各 GPU 的张量重新切分并交换，常用于改变分片维度或 MoE token 路由。

通信量相同时，实际耗时还取决于拓扑、带宽、延迟以及能否与计算重叠。

---

## 三、数据并行与参数分片

### Distributed Data Parallel（DDP）

DDP 在每个 GPU 上复制完整模型，把大 batch 沿样本维度切分。每张卡独立完成前向和反向，再对梯度求平均：

$$
\nabla_W L=\frac{1}{M}\sum_{i=1}^{M}\nabla_W L_i
$$

它概念简单，计算也容易扩展，但每个 GPU 都保存完整参数、梯度和优化器状态，所以模型仍必须能装入单卡显存。GPU 数增加还会扩大有效 batch，需要相应调整学习率和训练配置。

### Fully Sharded Data Parallel（FSDP）

FSDP（与 ZeRO 思路相关）把参数、梯度和优化器状态分片到不同 GPU。计算某一层前，使用 All-Gather 临时收集该层完整参数；反向后使用 Reduce-Scatter 聚合并重新分片梯度，然后释放不再需要的完整参数。

这样可把模型状态的单卡显存降到大约原来的 $1/M$，但代价是更频繁的通信。HSDP 则把 GPU 分组：组内执行 FSDP，组间复制模型并做数据并行，在显存节省和跨节点通信之间折中。

---

## 四、激活检查点

即使参数能够装下，中间激活也可能耗尽显存。普通反向传播保存每一层前向激活，$N$ 层网络大致需要 $O(N)$ 激活显存。

**Activation Checkpointing** 只保存少量边界状态，反向传播需要某段内部激活时，再从最近的 checkpoint 重算前向。它用额外计算换显存：保存的 checkpoint 越少，显存越省，但重算越多。

极端地完全不保存中间结果会反复从头重算，计算代价可能达到 $O(N^2)$。实际系统通常把网络划分为若干段，在显存与重算时间之间选择合适粒度。

---

## 五、沿不同维度切分模型

Transformer 中间激活可概括为：

$$
(\text{Layers},\ \text{Batch},\ \text{Sequence},\ \text{Channel})
$$

几种并行策略分别切分不同维度：

| 策略 | 切分维度 | 核心用途 | 主要代价 |
| --- | --- | --- | --- |
| DP / FSDP | Batch / 模型状态 | 扩大吞吐、分摊模型状态 | 梯度或参数通信 |
| CP | Sequence | 处理单卡放不下的长序列 | 注意力跨卡通信 |
| PP | Layers | 把不同层放在不同 GPU | 流水线气泡与负载均衡 |
| TP | Channel | 跨 GPU 拆分单层矩阵乘法 | 频繁、低延迟的层内通信 |

### Context Parallelism（CP）

CP 把一条长序列分到多张 GPU。逐 token 的归一化和 MLP 较容易并行，但全局注意力仍需要访问其他分片的 $K,V$。Ulysses 通过 All-to-All 把序列分片重排为注意力头分片；Ring Attention 则让 $K,V$ 分块在设备间环形流动，适合更长序列。

### Pipeline Parallelism（PP）

PP 把连续层分配给不同 GPU，激活在阶段边界传递。若一次只处理一个 batch，大量设备会等待前后阶段，形成 pipeline bubble。把 batch 切成多个 microbatch 并交错执行，可以填充流水线、提高利用率，但仍要平衡各阶段计算量。

### Tensor Parallelism（TP）

TP 把单层权重矩阵按行或列拆分，使多个 GPU 共同完成一次矩阵乘法。相邻线性层可采用互补的分片方向，减少中间结果的收集次数，最后通过 All-Reduce 合并结果。TP 通信频繁，通常优先放在高速互联的 GPU 组内。

MoE 模型还可使用 **Expert Parallelism（EP）**，把不同专家放在不同设备，并用 All-to-All 把 token 路由到相应专家。

---

## 六、混合并行与 MFU

超大模型通常同时使用 DP、CP、PP、TP，形成多维 GPU 网格。选择方式取决于模型能否装入显存、序列长度、全局 batch、网络拓扑以及通信能否与计算重叠，不存在对所有集群都最优的固定配置。

Model FLOPs Utilization（MFU）衡量实际训练中，有多少理论峰值算力用于模型本身的有效矩阵计算：

$$
t_{theory}=\frac{\text{model FLOPs per step}}{\text{device peak FLOPs}},
\qquad
\operatorname{MFU}=\frac{t_{theory}}{t_{actual}}
$$

$t_{actual}$ 包含前向、反向、优化器、通信和等待等完整 step 时间。MFU 低可能来自小矩阵、显存带宽限制、通信、数据加载、重计算或设备空闲；优化的目标不是简单增加 GPU 数，而是提高整体有效吞吐。

---

## 七、本讲总结

1. 大规模训练同时受算力、显存、带宽和通信拓扑限制。
2. DDP 复制模型并切分 batch；FSDP 进一步分片参数、梯度和优化器状态。
3. 激活检查点通过反向时重算前向结果来节省显存。
4. DP、CP、PP、TP 分别沿 batch、sequence、layers、channel 维度切分工作。
5. 超大模型通常组合多种并行方式，并让高频通信匹配高速互联拓扑。
6. MFU 用于衡量有效模型计算占理论峰值的比例，是调整分布式训练方案的重要指标。
