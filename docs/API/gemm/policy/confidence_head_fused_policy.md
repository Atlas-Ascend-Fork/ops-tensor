# ConfidenceHeadFused Policy

> [代码位置](../../../../include/blaze/gemm/policy/dispatch_policy.h)

## 功能说明

`ConfidenceHeadFusedPolicy` 通过 `ScheduleType=KernelConfidenceHeadMix` 选择 [GEMV Mix Kernel](../kernel/kernel_confidence_head_fused_mix.md)。

## 模板参数

```cpp
template <uint32_t FullLoadMode_ = A_FULL_LOAD_MODE, bool IsAtomicAdd_ = false,
          bool HasMarkov_ = false>
struct ConfidenceHeadFusedPolicy;
```

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| FullLoadMode_ | uint32_t | 全载模式，默认 A 权重全载。 |
| IsAtomicAdd_ | bool | 是否采用原子累加，默认 false。 |
| HasMarkov_ | bool | 是否包含第二段 Markov 特征，默认 false。 |

对应静态成员为 `FULL_LOAD_MODE`、`IS_ATOMIC_ADD`、`HAS_MARKOV`。当前 [BlockMmad 偏特化](../block/block_mmad_confidence_head_fused.md) 匹配 `ConfidenceHeadFusedPolicy<A_FULL_LOAD_MODE, false, HasMarkov_>`，采用 A 全载、非原子输出；bias 由运行时可选指针控制。

## Shape 下标常量

以下常量定义于 [common_utils.h](../../../../include/blaze/gemm/utils/common_utils.h)，位于 `Blaze::Gemm` 命名空间。

| 常量 | 值 | 说明 |
| --- | --- | --- |
| CONFIDENCE_HEAD_K_EMBED | 2 | ProblemShape 中 hidden 的 K 下标。 |
| CONFIDENCE_HEAD_K_MARKOV | 3 | ProblemShape 中 markov 的 K 下标。 |
