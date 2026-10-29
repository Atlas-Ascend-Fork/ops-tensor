# Block Epilogue Swish MX Quant

> [代码位置](../../../../include/blaze/epilogue/block/block_epilogue_swish_mx_quant.h)

## 功能与精度

`BlockEpilogueSwishMxQuant` 在 AIV 上读取 AIC 经 DualDst Fixpipe 写入 UB 的 FP32 矩阵乘结果，逐元素计算 `Swish(x, β) = x / (1 + exp(-β*x))`，输出形状为 M×N。bias 由矩阵乘阶段加入。

[Tile::Swish](../tile/arch35/swish.md) 完成 FP32 激活计算和 BF16 RINT 转换；[Tile::MxQuant](../tile/arch35/mx_quant.md) 根据 BF16 结果生成 scale，并执行 MXFP8 RINT 量化。

`scaleAlg` 支持 0（OCP）和 1（BLAS/cuBLAS）。

## 模板参数与 Params

```cpp
enum class SwishBetaMode { Scalar, PerChannel };

template <typename DataTypeOut_, typename DataTypeIn_ = float, SwishBetaMode Mode = SwishBetaMode::Scalar>
class BlockEpilogueSwishMxQuant;

struct Params {
    GM_ADDR yGmAddr{nullptr};
    GM_ADDR yScaleGmAddr{nullptr};
    uint32_t baseM{0};
    uint32_t baseN{0};
    uint32_t scaleAlg{0};
    float beta{1.0F};
};
```

`Mode` 当前仅支持 `Scalar`；`PerChannel` 保留枚举值，实例化时编译报错。`beta` 为所有元素共用的有限 FP32 标量，默认 1（SiLU）。

`DataTypeIn_` 为 `float`，`DataTypeOut_` 为 `fp8_e4m3fn_t` 或 `fp8_e5m2_t`，scale 以 E8M0 字节存储。
`baseM/baseN` 与上层 tiling 一致，当前 tile 的实际尺寸由 `blockShape` 传入。
`Init` 保存 Params 引用，其生命周期须覆盖所有 tile。

## Shape 与尾块

| 数据 | 逻辑 shape / 存储约定 |
| --- | --- |
| FP32 UB 输入 | 当前 AIV 的 `[mVec, singleN]`，行距 `Align32(singleN)`，起始 UB 偏移为 0 |
| BF16 激活结果 | 同一行距；尾部 Cast 使用 ZEROING，写回时补零到 32 元素分组边界 |
| y | `[batch..., M, N]`，仅写有效 M/N 元素 |
| yScale | `[batch..., M, ceil(N/64), 2]`，N 方向每 32 个元素一个 scale |
| scale UB 搬运布局 | 需要配对补零时使用 32 字节行距，否则按有效 scale 列数紧凑存放 |

N 可以是任意正数。非末尾 N tile 的起点和宽度须按 32 个元素对齐。有效 scale 列数为奇数时，最后一个 N tile 额外写入一个配对补零字节。全零组的 E8M0 原始字节也为 0。

两个 AIV 沿 M 分工。激活计算调用 `Tile::Swish::Run`；需要 scale 配对补零时调用 `Tile::MxQuant::PadScaleRows`。Block 负责缓冲、同步和 GM 写回。

## 公共接口

| 接口 | 作用 |
| --- | --- |
| `Init(const Params&)` | 保存 Params 引用并缓存 scale 算法和 β，初始化输出地址；AIC 侧直接返回 |
| `UpdateNextProblem(const ProblemShape&)` | 读取完整 N，计算 `yScale` 每行跨度 |
| `UpdateGlobalAddr(const BlockCoord&)` | 按 `{yOffset, yScaleOffset}` 更新 batch 输出基址 |
| `operator()(const BlockShape&, const BlockCoord&)` | 读取当前 M/N，计算激活与 MX 量化并写回；二维坐标为 `{yOffset, yScaleOffset}` |

## Kernel 集成与资源边界

配合 [QBMM MX Activation Quant Kernel](../../gemm/kernel/kernel_qbmm_mx_activation_quant.md)，使用 `KernelMmadWithScaleMxActivationQuant` 和 `L0C2UB_MODE_DUAL_DST_SPLIT_M`，`ConcatN=false`。单 batch 与多 batch 均保留完整 N；A 全载和非全载共用本组件。

```cpp
using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueSwishMxQuant<OutType, float>;
typename BlockEpilogue::Params epilogueParams{y, yScale, baseM, baseN, scaleAlg, beta};
```

ND/NZ、A/B dtype 和转置由 Cube 组件及上层 Host 校验；本 Epilogue 的 UB 输入统一为 FP32 ND。

每个 AIV 固定使用 233 KiB UB，最多处理 128×256 个元素。调用方须保证 `baseM/baseN <= 256`，并满足上述对齐约束。

本组件在 Swish 结果写入 BF16 区后执行 `NotifyCube()`，允许下一 tile 的 Fixpipe 覆盖 FP32 输入区 `[0, 131072)` 字节。每个 AIV/tile 通知一次，包括零行分支；Kernel 负责数据就绪通知及输入复用等待，详见 [Kernel 同步约定](../../gemm/kernel/kernel_qbmm_mx_activation_quant.md#输入-ub-复用通知的所有权)。

该通知仅允许复用输入区。量化输出和 scale 的搬出及复用由 `V_MTE3` / `MTE3_V` 保护，析构时等待最后一次搬出完成。

## Example 与测试

- [CSV Example](../../../../examples/acl/quant_matmul_activation_quant/quant_matmul_activation_quant/README.md)
- [Kernel UT](../../../../tests/ut/op_kernel/quant_matmul_activation_quant/test_quant_matmul_activation_quant.cpp)
