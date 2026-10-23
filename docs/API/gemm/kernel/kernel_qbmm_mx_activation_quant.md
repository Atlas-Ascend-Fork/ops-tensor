# Kernel Qbmm MX Activation Quant

> [代码位置](../../../../include/blaze/gemm/kernel/kernel_qbmm_mx_activation_quant.h)

## 功能说明

该 Kernel 是 MX 量化 Batch Matmul 的 AIC+AIV 融合实现：AIC 执行 MX MMAD，并通过
DualDst Fixpipe 将 L0C 的 FP32 结果写入 UB；AIV 执行 GELU、SwiGLU 或 Swish 激活与动态
MX 量化，输出 `y` 和 `yScale`。

本实现支持最多 4 维 Batch、A/B Batch 广播和 Batch 地址换算；单 Batch（无广播）场景
复用同一 Kernel，由调用方将 `qbmmParams` 中的 batch 维度填为默认值 1。

使用 `KernelMmadWithScaleMxActivationQuant` 调度策略，由 `MatmulWithScaleMx` 的 `ConcatN` 参数与 BlockEpilogue 配合选择激活：

- `ConcatN=false`：保留完整 N，由 BlockEpilogue 选择 GELU 或 Swish。
- `ConcatN=true`：拼接 gate/linear 两个半区，由 SwiGLU Epilogue 输出 N/2 列。

## 特殊约束

### GELU 路径

- A/B 支持同 bit-width 的 MXFP4 或 MXFP8 组合。
- `LayoutA` 支持 ND/DN，`LayoutB` 支持 ND/DN/NZ/ZN。
- 后处理使用 `BlockEpilogueGeluMxQuant`，可输出 MXFP4 或 MXFP8。

### SwiGLU 路径

- A/B 仅支持 MXFP8；ND 权重支持 A 不转置或转置，WeightNZ 产品接口要求 A 不转置。
- B 支持不转置和转置，对应 ND/DN 或 WeightNZ 的 NZ/ZN 布局。
- 原始 `N` 必须为正数且是 64 的倍数；矩阵乘结果按 `C=[gate | linear]` 等分，gate 和 linear 各占
  `N/2`，并计算 `SiLU(gate) * linear`；输出 `y` 宽度为 `N/2`。
- qbmmswigluquant 的 WeightNZ 产品组合中，B 使用 `fp8_e4m3fn_t`；A 和输出可为
  `fp8_e4m3fn_t` 或 `fp8_e5m2_t`。
- `scaleAlg=0` 使用 OCP scale，`scaleAlg=1` 使用 cuBLAS/BLAS scale。
- 后处理使用
  [`BlockEpilogueSwigluMxQuant`](../../epilogue/block/block_epilogue_swiglu_mx_quant.md)，
  输出 MXFP8。

`N/2` 是 32 的倍数。WeightNZ 的 gate/linear 半区和每个输出 tile 均从完整的
NZ/ZN 分形边界开始，Cube 分别搬运两个半区到连续的 L1/L0C 列。

### Swish 路径

标量 β 由 Epilogue Params 指定，默认 1（SiLU）。

- 使用 `ConcatN=false`，后处理替换为
  [BlockEpilogueSwishMxQuant](../../epilogue/block/block_epilogue_swish_mx_quant.md)。
- 完整 FP32 矩阵乘结果加 bias 后逐元素计算 `Swish(x, β)`，以 BF16 RINT 舍入后执行 MXFP8 RINT 量化。
- ND 路径中，A、B 可分别选用 `fp8_e4m3fn_t` 或 `fp8_e5m2_t`，并分别设置是否转置。
- WeightNZ 路径中，A 可选上述两种 FP8 类型且不转置；B 为 `fp8_e4m3fn_t`，可设置是否转置。
- 输出 dtype 由 Epilogue 指定；example 使用与 A 相同的类型。
- `scaleAlg` 仅支持 0（OCP）或 1（BLAS），输出保留完整 N，N 无 64 对齐要求。
- `epilogueParams.baseN` 传入完整 tile 宽度；单 batch、多 batch、A 全载及非全载均使用同一 Epilogue。

### Scale 与 DualDst

- `scaleAGmAddr`、`scaleBGmAddr` 的数据类型均为 `fp8_e8m0_t`。
- 令 `H=N`（GELU/Swish）或 `H=N/2`（SwiGLU），输出形状为
  `y=[batch..., M, H]`、`yScale=[batch..., M, ceil(H/64), 2]`。
- `scaleAlg=0/1` 分别选择 OCP 和 cuBLAS/BLAS；两者在 FP8 表示边界附近的差异与选择建议见
  [`MxQuant`](../../epilogue/tile/arch35/mx_quant.md#ocp-与-cublas-算法选择)。
- `BlockMmad` 的 `DispatchPolicy::L0C2UB_MODE` 必须为
  `L0C2UB_MODE_DUAL_DST_SPLIT_M`。
- AIC 写入 UB 的 M 维按 2 对齐；AIC 与 AIV 通过 Cube/Vector flag 逐 tile 同步。

### Batch

- 支持 4 维 Batch：`batchA1..4`、`batchB1..4`、`batchC1..4`。
- A/B 的每个 Batch 维度必须与 C 对应维相等或可广播。
- x1Scale/x2Scale 的 Batch 维度分别跟随 A/B。
- Bias 支持一维 N，或由 `biasThreeDim` 指定的 Batch 形式。SwiGLU 的 bias 长度仍为原始 N，
  在 gate/linear split 之前加到完整矩阵乘结果上，不能按 `N/2` 传入。

## 模板参数

```cpp
template <
    class ProblemShape,
    class BlockMmad,
    class BlockEpilogue,
    class BlockScheduler>
class GemmUniversal<...>;
```

| 参数 | 说明 |
|---|---|
| `ProblemShape` | 问题形状 `(M, N, K, Batch)` |
| `BlockMmad` | MX MMAD 组件，ScheduleType 为 `KernelMmadWithScaleMxActivationQuant` |
| `BlockEpilogue` | GELU/SwiGLU/Swish 动态 MX 量化组件，须与 `ConcatN` 匹配；同步约定见下文 |
| `BlockScheduler` | QBMM 分块调度器 |

## 数据结构

### Params

```cpp
struct Params {
    ProblemShape problemShape;
    BlockMmadParams mmadParams;
    BlockEpilogueParams epilogueParams;
    L1Params l1Params;
    BlockSchedulerParams schParams;
    QBMMTiling qbmmParams;
};
```

### QBMMTiling

```cpp
struct QBMMTiling {
    uint32_t batchA1, batchA2, batchA3, batchA4;
    uint32_t batchB1, batchB2, batchB3, batchB4;
    uint32_t batchC1, batchC2, batchC3, batchC4;
    uint32_t biasThreeDim;
    uint32_t baseM;
    uint32_t baseN;
    uint32_t baseK;
    uint32_t isBias;
    uint32_t dbL0C;
    uint32_t bMustHitL2;
};
```

`bMustHitL2` 非 0 时保持 B 的默认 L2 Cache 策略；为 0 时，Kernel 可根据当前 M/N tile
和 B 的布局关闭 B 的 L2 Cache。

## 公共接口

### operator()

```cpp
__aicore__ inline void operator()(const Params& params)
```

公开调用入口，内部执行以下流程：

1. 初始化输入地址、Batch 元数据、BlockMmad 和 BlockEpilogue。
2. SwiGLU 路径把逻辑调度宽度设置为原始 `N/2`，并把调度 `baseN` 同步减半。
3. Batch 为 1 时直接处理；Batch 大于 1 时计算各输入步长并执行 4 维广播循环。
4. 对每个 tile，AIC 完成 MX MMAD 和 Fixpipe 写入后通知 AIV；AIV 等待数据就绪，再调用 epilogue。
5. GELU/Swish 由 Epilogue 通知输入 UB 可复用，SwiGLU 由 Kernel 通知；AIC 在下一 tile 及退出前等待。

## 内部流程

### Batch 地址换算

`CalcBatchStrides` 根据逻辑 M/N/K、数据类型和 B 布局计算：

- A/B、ScaleA/ScaleB 和 Bias 的 Batch 步长。
- C/y/yScale 的 Batch 步长。
- A/B 每个 Batch 维度相对 C 的广播倍数。

`ProcessBatchLoop` 迭代 `batchC1..4`，通过 `AddBatchOffset` 更新当前 Batch 的 A/B、
Scale、Bias 和 epilogue 输出地址。

### 单 tile 处理

```text
AIC: WaitForVector（除首 tile）
     -> slice A/B/Scale/Bias
     -> BlockMmad
     -> DualDst Fixpipe 到 UB
     -> NotifyVector

AIV（GELU/Swish）:
     WaitForCube（Kernel）
     -> 激活结果写入独立 BF16 区（Epilogue）
     -> NotifyCube：释放输入 UB（Epilogue）
     -> 动态 MX 量化、提交 y/yScale 写回（Epilogue）

AIV（SwiGLU）:
     WaitForCube（Kernel）
     -> Epilogue：激活、量化、提交 y/yScale 写回
     -> NotifyCube：释放输入 UB（Kernel）
```

SwiGLU 使用逻辑输出坐标 `(M, H)`，其中 `H=N/2`；Batch 输出偏移和 `yScale`
步长也按 H 计算。

### 输入 UB 复用通知的所有权

`NotifyCube()` 的职责按 `BlockMmad::CONCAT_N` 约定：

| `ConcatN` | Epilogue | 通知执行者与时机 |
| --- | --- | --- |
| `false` | GELU/Swish | Epilogue 完成输入区的最后一次读取后通知 |
| `true` | SwiGLU | Kernel 在 Epilogue 的 `operator()` 返回后通知 |

自定义 Epilogue 须遵循同一约定：`ConcatN=false` 时自行通知，`ConcatN=true` 时由 Kernel 通知。
每个参与 tile 的 AIV 须恰好通知一次。GELU/Swish 的零行分支在返回前通知；SwiGLU 的零行分支返回后由 Kernel 通知。

GELU/Swish 在激活结果写入 BF16 区后、MX 量化前通知，允许 Fixpipe 复用每个 AIV 的 FP32 输入区 `[0, 131072)` 字节。
输出与 scale 的搬出及复用由 Epilogue 的 `V_MTE3` / `MTE3_V` 保护，析构时等待最后一次搬出完成。

Kernel 负责 `NotifyVector()` / `WaitForCube()` 数据就绪同步。AIC 首 tile 无需等待输入释放，
后续 tile（包括跨 batch）及退出前通过 `WaitForVector(..., true)` 等待两个 AIV 的通知。

## SwiGLU WeightNZ 组件组装示例

```cpp
using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
using AType = fp8_e4m3fn_t;
using BType = fp8_e4m3fn_t;
using OutType = fp8_e5m2_t;
using LayoutA = asc::te::nd_ext_layout_ptn;
using LayoutB = asc::te::nz_layout_ptn; // B 转置时使用 zn_layout_ptn
using LayoutC = asc::te::nd_ext_layout_ptn;

constexpr uint64_t FullLoadMode = Blaze::Gemm::NONE_FULL_LOAD_MODE;
using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<
    FullLoadMode,
    false,
    Blaze::Gemm::KernelMmadWithScaleMxActivationQuant,
    Blaze::Gemm::L0C2UB_MODE_DUAL_DST_SPLIT_M,
    0,
    true>; // ConcatN=true：SwiGLU 拼接 N 布局，epilogue 输出宽 N/2
using BlockMmad = Blaze::Gemm::Block::BlockMmad<
    DispatchPolicy, AType, LayoutA, BType, LayoutB,
    float, LayoutC, float, LayoutC>;
using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueSwigluMxQuant<
    OutType, float, fp8_e8m0_t>;
using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<
    ProblemShape, FullLoadMode, LayoutA, LayoutB, AType>;
using Kernel = Blaze::Gemm::Kernel::GemmUniversal<
    ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
```

WeightNZ 的 epilogue 参数按字段赋值：

```cpp
typename BlockEpilogue::Params epilogueParams{};
epilogueParams.yGmAddr = y;
epilogueParams.yScaleGmAddr = yScale;
epilogueParams.baseM = baseM;
epilogueParams.baseN = baseN / 2;
epilogueParams.scaleAlg = scaleAlg;
```

## 性能建议

- `kL1` 按 `MXFP_DIVISOR_SIZE`（64）对齐。
- `l1BufNum` 可按 L1 容量选择 2、3 或 4。
- 大 K、小 M 场景可选择 A 全载模式。
- SwiGLU 的 host tiling 将 matmul `baseN` 按 128 对齐；主半区 tile 按 64 对齐，
  32 列尾块仍满足搬运对齐。

## 适用场景

- 多 Batch 或需要 A/B Batch 广播的 QuantMatmulActivationQuant。
- MXFP4/MXFP8 GELU 融合输出量化。
- MXFP8 SwiGLU 的 ND 或 WeightNZ 权重输入。
- MXFP8 Swish 的 ND 或 WeightNZ 权重输入，输出保留完整 N。
