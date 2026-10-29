# ConfidenceHeadFused

> [代码位置](../../../../../include/blaze/epilogue/tile/arch35/confidence_head_fused.h)

## 功能说明

对 FP32 logits 执行温度缩放、可选截断和 Sigmoid，结果保持 FP32。

```text
u = logits * negInvTemperature
if hasLowLimit:  u = max(u, lowLimit)
if hasHighLimit: u = min(u, highLimit)
out = 1 / (1 + exp(u))
```

## 特殊约束

- 源和目的 Tensor 均为 UB 上的 float NDExt Tensor，shape 一致，行跨度可不同。
- 仅在 AIV 上执行；调用者负责输入就绪及输出使用前的同步。
- 本层不读取 GM 温度；调用者传入预先计算的负温度倒数。
- 通过 [compute.h](../compute.md) 聚合，不依赖其他 Tile 的实现。

## 计算与精度

温度缩放和可选 clamp 完成后，本 Tile 内联执行 `Exp → Adds(1) → Div`，计算 `1 / (1 + exp(u))`。最后一步使用默认的非精确 Div；Sigmoid 使用 `preciseDiv=true`，两者不要求逐位一致。

Golden 保持 FP32 参考计算，设备结果按 [端到端示例](../../../../../examples/acl/confidence_head_fused/confidence_head_fused_mix/README.md)及 NN ST 配置的容差校验，不通过改用精确 Div 或放宽阈值消除差异。

## Params 参数结构

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| negInvTemperature | float | -1/T_eff，默认 -1。 |
| lowLimit / highLimit | float | 指数输入的截断阈值，默认 0。 |
| hasLowLimit / hasHighLimit | bool | 两侧截断是否启用，默认 false。 |

## 静态方法

```cpp
template <bool ZeroInput = false, typename SrcTensor, typename DstTensor>
__aicore__ inline static void Run(
    const SrcTensor& srcTensor, const DstTensor& dstTensor, const Params& params);
```

ZeroInput=true 时以零作为 logits，不读取源 Tensor 的数值；否则读取源 Tensor。内部按行与向量宽度遍历，以 mask 处理 N 尾部，不申请临时 UB。完整调用见 [Block Epilogue](../../block/block_epilogue_confidence_head_fused.md)。
