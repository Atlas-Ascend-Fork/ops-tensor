# Kernel ConfidenceHeadFused Mix

> [代码位置](../../../../include/blaze/gemm/kernel/kernel_confidence_head_fused_mix.h)

## 功能说明

`GemmUniversal` 的 `KernelConfidenceHeadMix` 特化实现 GEMV 与置信度后处理融合。hidden 和可选 markov 特征沿 K 逻辑拼接，与一维 weight 投影，加上可选 bias[N] 后执行温度缩放、截断和 Sigmoid，输出 FP32。

```text
z = hidden @ weight[:K1] + markov_emb @ weight[K1:] + bias
u = -z / T_eff
if hasLowLimit:  u = max(u, lowLimit)
if hasHighLimit: u = min(u, highLimit)
out = 1 / (1 + exp(u))
```

缺省 markov、bias 对应项为零。temperature 缺省或为正负零时 T_eff=1，其他值按当前值计算。

## 特殊约束

- 适用于 AIC:AIV 为 1:2 的 Mix 模式，每对 AIV 仅第一个 sub-block 执行后处理。
- 权重和特征同为 FP16 或同为 BF16，temperature、bias、输出为 FP32；GM 数据连续。
- weight 为 `[K1+K2]`，特征展平为 `[B*N,K]`，bias 为跨 batch 广播的 `[N]`。
- 非空 K 使用权重全载 L1 模板；调用者检查分块所需片上容量。

## 组件类型

| 组件 | 类型 |
| --- | --- |
| ProblemShape | `asc::te::shape<int64_t,int64_t,int64_t,int64_t>`，顺序为 `(1,B*N,K1,K2)`。 |
| BlockMmad | [BlockMmad 的 ConfidenceHeadFusedPolicy 偏特化](../block/block_mmad_confidence_head_fused.md) |
| BlockScheduler | [BlockSchedulerMatmulBasic](../block/block_scheduler_matmul_basic.md)，使用 A 全载模式。 |
| BlockEpilogue | [BlockEpilogueConfidenceHeadFused](../../epilogue/block/block_epilogue_confidence_head_fused.md) |

## Params 参数结构

| 字段 | 说明 |
| --- | --- |
| problemShape | 完整问题 shape。 |
| weightGmAddr / hiddenGmAddr / markovGmAddr | 权重与两段特征的 GM 地址。 |
| mmadParams | Cube Block 参数。 |
| epilogueParams | 后处理参数。 |
| schParams | 基础调度器参数，mL1/baseM 为 1，nL1/baseN 与 Cube baseN 一致，mTailCnt/nTailCnt 为 1。 |

`mmadParams.biasGmAddr` 控制 Cube bias；`epilogueParams.biasGmAddr/hasBias/biasN` 供零 K 后处理和 batch 广播使用，两组参数须对应同一个可选 bias[N]。

## 执行流程

1. Kernel 将计算 shape 转为基础调度器的 `(M,N,K,batch)` 约定，调度器划分 N tile，AIC 将权重一次搬入 L1。
2. AIC 按 K 段搬运 hidden、markov，在首次有效 MMAD 中加入 bias，全部累加完成后将 FP32 结果写入 UB。
3. AIV 等待 Cube 就绪，执行温度、截断、Sigmoid 和 GM 写回；跨核标志保护 UB 双缓冲复用。

有 bias 时，调度 shape 为 `(1,N,K1+K2,B)`，每个 tile 限制在单个 batch 内；无 bias 时为 `(1,B*N,K1+K2,1)`，沿展平行数切分。基础调度器返回的坐标通过 `flattenedN = batchIdx*N + localN` 转为特征和输出的偏移，bias 使用 `flattenedN % N`。mTailCnt/nTailCnt 均为 1，关闭尾轮额外拆分；实际核数由启动 blockDim 和总分块数决定。

例如 B=2、N=129、baseN=128 且带 bias 时，展平起点为 0、128、129、257，有效长度为 128、1、128、1。

K1=0、K2>0 时由首轮 markov 累加初始化 bias。K1+K2=0 时跳过 Cube，以 bias 或零作为 logits 执行后处理。B=0 或 N=0 时在构造调度器前返回，不访问输入数据。

## 使用示例

```cpp
Kernel kernel;
kernel(params);
```

完整参数组装见 [ConfidenceHeadFused 示例](../../../../examples/confidence_head_fused/confidence_head_fused_mix/README.md)。
