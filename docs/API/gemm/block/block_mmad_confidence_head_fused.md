# Block Mmad ConfidenceHeadFused

> [代码位置](../../../../include/blaze/gemm/block/block_mmad_confidence_head_fused.h)

## 功能说明

`BlockMmad` 针对 `ConfidenceHeadFusedPolicy<A_FULL_LOAD_MODE, false, HasMarkov_>` 的偏特化实现 M=1 的 GEMV Cube 计算。A 为一维投影权重，B 为 hidden 与可选 markov 特征的转置逻辑视图。权重全量驻留 L1，两段 K 依次累加到 FP32 L0C，完成后由 Fixpipe 写入 UB。该实现通过统一的 `BlockMmad` 模板组装到 Kernel。

## 特殊约束

- AType、BType 必须同为 half 或同为 bfloat16_t，CType 和 BiasType 为 float。
- LayoutA、LayoutC、LayoutBias 为 NDExt，LayoutB 为 DNExt；输入 GM 数据连续。
- 仅为 A 全载、非原子输出的 `ConfidenceHeadFusedPolicy` 提供此特化。
- K1+K2 须大于零；零 K 由 Kernel 分流至后处理。
- 带 bias 时，单个 N tile 不得跨越 bias[N] 的广播边界。
- 调用者须检查 L1、L0、BiasTable 和 UB 容量。

## 模板参数

```cpp
template <bool HasMarkov_, class AType_, class LayoutA_, class BType_, class LayoutB_,
          class CType_, class LayoutC_, class BiasType_, class LayoutBias_>
class BlockMmad<ConfidenceHeadFusedPolicy<A_FULL_LOAD_MODE, false, HasMarkov_>,
                AType_, LayoutA_, BType_, LayoutB_, CType_, LayoutC_, BiasType_, LayoutBias_>;
```

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| HasMarkov_ | bool | 是否包含第二段 Markov 特征。 |
| AType_ / BType_ / CType_ | class | 权重、特征和累加器的数据类型。 |
| LayoutA_ / LayoutB_ / LayoutC_ | class | 权重、特征和累加器的布局。 |
| BiasType_ | class | Bias 类型，须为 float。 |
| LayoutBias_ | class | Bias 布局，须为 NDExt。 |

组装时显式传入分发策略和统一的 Block 参数：

```cpp
using ND = asc::te::nd_ext_layout_ptn;
using DN = asc::te::dn_ext_layout_ptn;
using Policy = Blaze::Gemm::ConfidenceHeadFusedPolicy<Blaze::Gemm::A_FULL_LOAD_MODE, false, HasMarkov>;
using Mmad = Blaze::Gemm::Block::BlockMmad<Policy, half, ND, half, DN, float, ND, float, ND>;
```

## Params 参数结构

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| kEmbed / kMarkov | uint32_t | hidden、markov 的逻辑 K，不含 L1 对齐间隔。 |
| baseN / baseK | uint32_t | N/K 基本块，默认均为 128。 |
| kL1 | uint32_t | 特征 L1 的 K 分段，默认 256。 |
| l1Stages | uint32_t | 特征和 bias 的 L1 缓冲数，默认 2，最多 4。 |
| l0cStages | uint16_t | L0C 缓冲数，默认 2。 |
| biasGmAddr | GM_ADDR | 可选 FP32 bias 地址，nullptr 表示不加 bias。 |

kEmbed、kMarkov 由调用方通过 Params 显式传入，须与 Kernel 的 problemShape 一致；零值表示对应 K 段为空。

## 成员方法

```cpp
block.Init(params);
block(gmWeight, gmHiddenTile, gmMarkovTile, gmBiasTile, ubOutput, blockShape);
```

AIC 路径构造后须调用 `Init()`。权重在首次 `operator()` 调用时由 Block 内部全载至 L1，后续分块复用，搬运函数不对外公开。离开作用域时，析构函数等待片内流水事件并恢复矩阵布局状态；Kernel 在 Block 析构前完成跨核等待。

Kernel 的 `problemShape` 为 `(1,B*N,K1,K2)`，`blockShape` 为 `(1,curN,K1+K2,1)`。hidden/markov tile 的逻辑视图分别为 `[K1,curN]`、`[K2,curN]`，bias tile 为 `[1,curN]`。无 markov 时，第二个特征 Tensor 可复用第一个。

有 markov 时，第二段权重在 L1 中从 align16(K1) 开始存放，GM 中仍从 K1 开始读取。每个特征 L1 槽附带可选 FP32 bias；bias 经 GM→L1→BiasTable，仅在每个 N tile 的首次有效 MMAD 中加入，后续 K 段继续累加。

完整组装方式见 [Kernel ConfidenceHeadFused Mix](../kernel/kernel_confidence_head_fused_mix.md)。
