# Kernel Tqbmm (Mx / Mix / Fixpipe Quant)
> [代码位置](../../../../include/blaze/gemm/kernel/kernel_tqbmm.h)

## 功能说明
`kernel_tqbmm.h` 提供 TQBMM（Transpose Quant Batch MatMul）Tensor API 的三个 `GemmUniversal` 特化，
共享 TQBMM 3D batch 循环 + `perm_x1=[1,0,2]` 非连续遍历，仅 BlockMmad / BlockEpilogue / ScheduleType 不同：

| 特化 | ScheduleType | 量化格式 | 计算模式 |
|------|--------------|----------|----------|
| MX | `KernelMmadMultiBlockTQBMM` | MxFP4 / MxFP8 | 仅 AIC |
| FP8 Mix | `KernelMmadMultiBlockTQBMMMx` | FP8（`fp8_e4m3fn_t` / `fp8_e5m2_t`） | AIC + AIV（L0C→UB，AIV 反量化） |
| HIFP8 Fixpipe Quant | `KernelMmadMultiBlockTQBMMFixpipeQuant` | HIFP8（`hifloat8_t`） | 仅 AIC（fixpipe 反量化） |

> **注意**：TQBMM 与 QBMM 的核心区别在于输入/输出布局。TQBMM 使用 `perm_x1=[1,0,2]`（A 物理布局 `[m, batch, k]`），C 输出物理布局为 `[m, batch, n]`（即 `[batch, m, n]` 累加器经 `perm_y=[1,0,2]` 后存储）。

## 布局约定（与 Kernel 实现一致）

- **A（x1）**：`perm_x1=[1,0,2]` 时物理存储 `[M, B, K]`（M 轴非连续）。
  - M 步长 `aMStride = batch * k`，Batch 步长 `aBatchStride = k`。
  - MxFP4 时 stride 由 Te 布局直接按 fp4 元素（packed `fp4x2`）解释，**不得再额外右移**（历史 bug 曾对 fp4 多一次 `>>1` 造成 M>=1 错行）。
- **C（y）**：物理存储 `[M, B, N]`（`perm_y=[1,0,2]`）。
  - Batch（B）嵌在 M 平面内：B 步长 `= N`，M 步长 `= B * N`。
  - 历史 bug 曾按 `[B,M,N]` 布局（B 步长 `= M*N`）写 C，导致 B>=2 时输出 m↔2m 交错。
  - Mix 特化 epilogue 写回 offset：`offsetC = mPos * batch * n + batchIdx * n + nPos`；UB→GM 的 row pitch 同样必须用 `n * batch`（历史 bug 少乘 batch 导致 B>1 输出分块交错）。
- **x1Scale**：
  - MX：物理布局 `[M, B, scaleKLen]`（M 步长 `= B * scaleKLen`，Batch 步长 `= scaleKLen`；`scaleKLen = ceil(k/64)*2`）。
  - Mix（per-token，float）：物理布局 `[M]`（跨 batch 共享）。
- **x2Scale**：
  - MX：物理布局 `[B, scaleKLen, N]`。
  - Mix（per-channel，float）：物理布局 `[1, N]`（跨 batch 共享）。
  - Fixpipe Quant：per-channel 模式 `[1, N]`（uint64 编码，跨 batch 共享）；per-tensor 模式为标量。

## 公共约定

- **BlockScheduler**：三个特化均使用 `BlockSchedulerQuantBatchMatmulV3`，支持多 Batch 维度切分（3D batch 循环 + tail round）。
- **公共 helper**（`TqbmmDetail` 命名空间）：`MakeBatchLayout`（A/C/Scale 的 batch 布局）、`SetBL2Cache`（L2 Cache hint 动态配置）。

**参考**：[Kernel Matmul 基础框架](./kernel.md)、[Kernel Qbmm Mx](./kernel_qbmm_mx.md)、[Kernel Qbmm Mix](./kernel_qbmm_mix.md)、[Kernel Qbmm Cube](./kernel_qbmm_cube.md)

## 1. MX 特化

### 功能说明
TQBMM MX 量化 Kernel，仅支持 AIC 计算，支持 MxFP4/MxFP8 量化格式。集成 Scale 因子（x1Scale、x2Scale）处理、多 Batch 维度支持、L2 Cache 动态配置。

### 特殊约束
- **量化格式**：MxFP4（`fp4x2_e2m1_t`）、MxFP8（`fp8_e4m3fn_t`）。
- **Scale 因子**：必须提供两个 Scale 因子，均为 `fp8_e8m0_t` 类型：
  - `scaleAGmAddr`：A 矩阵缩放因子，物理布局 `[M, B, scaleKLen]`
  - `scaleBGmAddr`：B 矩阵缩放因子，物理布局 `[B, scaleKLen, N]`
- **计算模式**：仅支持 AIC 模式，不支持 AIV 计算（AIV 核直接返回）。
- **Batch 维度**：支持 4 维 Batch（batchC1/C2/C3/C4），需满足广播规则。x1Scale/x2Scale 的 Batch 维度须分别与 A 矩阵/B 矩阵一致。

### 类型定义
```cpp
// BlockMmad::DispatchPolicy::ScheduleType 为 KernelMmadMultiBlockTQBMM 时的特化
template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
class GemmUniversal;

struct TQBMMTiling {
    uint32_t batchA1, batchA2, batchA3, batchA4;
    uint32_t batchB1, batchB2, batchB3, batchB4;
    uint32_t batchC1, batchC2, batchC3, batchC4;
    uint32_t biasThreeDim;
    uint32_t baseM, baseN, baseK;
    uint32_t isBias;
    uint32_t dbL0C;
    uint32_t bMustHitL2 = 1U;
};

struct Params {
    ProblemShape problemShape;
    BlockMmadParams mmadParams;
    L1Params l1Params;
    BlockSchedulerParams schParams;
    TQBMMTiling tqbmmParams;
};
```

### 调用示例
```cpp
using AType = fp4x2_e2m1_t;
using BType = fp4x2_e2m1_t;
using CType = half;
using BiasType = float;
using LayoutA = asc::te::nd_ext_layout_ptn;
using LayoutB = asc::te::nd_ext_layout_ptn;
using LayoutC = asc::te::nd_ext_layout_ptn;

using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<
    0, false, Blaze::Gemm::KernelMmadMultiBlockTQBMM>;
using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<
    ProblemShape, 0, LayoutA, LayoutB, AType>;
using BlockMmad = Blaze::Gemm::Block::BlockMmad<
    DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC, BiasType, LayoutC>;
using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;
using TQBMMKernel = Blaze::Gemm::Kernel::GemmUniversal<
    ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

TQBMMKernel kernel;
kernel(params);
```

### 与 Kernel Qbmm Mx 的区别
| 项目 | QBMM MX | TQBMM MX |
|------|---------|----------|
| A 布局（permX1） | 无（`[batch, m, k]` 连续） | `[m, batch, k]`（M 轴非连续） |
| C 输出布局 | `[batch, m, n]` | `[m, batch, n]`（B 内嵌 M 平面，B 步长=N, M 步长=B*N） |
| x1Scale 布局 | `[batch, m, scaleKLen]` | `[m, batch, scaleKLen]`（B 内嵌 M 平面） |
| Kernel 类名 | `GemmUniversal` | `GemmUniversal`（不同特化） |
| ScheduleType | `KernelMmadWithScaleMx` | `KernelMmadMultiBlockTQBMM` |
| Tiling 结构 | `QBMMTiling` | `TQBMMTiling` |
| BlockMmad | block_mmad_qbmm_mx.h | block_mmad_qbmm_mx.h（复用） |

## 2. FP8 Mix 特化

### 功能说明
TQBMM MIX 量化 Kernel，支持 AIC+AIV 协同计算，支持 FP8（`fp8_e4m3fn_t` / `fp8_e5m2_t`）量化格式。AIC 侧完成 Cube 矩阵乘并输出 L0C→UB，AIV 侧完成反量化（per-token scale + per-channel scale）并写回 GM。

### 特殊约束
- **量化格式**：FP8（`fp8_e4m3fn_t` / `fp8_e5m2_t`）。
- **Scale 因子**（float 类型）：
  - `x2ScaleGmAddr`：per-channel 权重 scale，布局 `[1, N]`
  - `x1ScaleGmAddr`：per-token 激活 scale，布局 `[M]`
- **计算模式**：支持 AIC+AIV（MIX）模式。AIC 侧完成矩阵乘并通过 L0C→UB 输出；AIV 侧从 UB 读取中间结果，完成反量化后写回 GM。
- **AIC-AIV 同步**：使用 `common_utils.h` 中的 `NotifyVector`/`WaitForVector`/`NotifyCube`/`WaitForCube` 进行 AIC-AIV 核间同步。

### 类型定义
```cpp
// BlockMmad::DispatchPolicy::ScheduleType 为 KernelMmadMultiBlockTQBMMMx 时的特化
template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
class GemmUniversal;

struct TQBMMParams {
    uint32_t kAL1, kBL1, nBufferNum;
    uint32_t baseM, baseN, baseK;
    uint32_t dbL0C;
    uint32_t bMustHitL2 = 1U;
};

struct Params {
    ProblemShape problemShape;
    BlockMmadParams mmadParams;
    BlockSchedulerParams schParams;
    TQBMMParams tqbmmParams;
    EpilogueParams epilogueParams;
};
```

### 调用示例
```cpp
using AType = fp8_e4m3fn_t;
using BType = fp8_e4m3fn_t;
using OutType = half;
using X2ScaleType = float;
using X1ScaleType = float;
using BiasType = float;

using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMix<
    0, false, Blaze::Gemm::KernelMmadMultiBlockTQBMMMx, NON_CONTIGUOUS_TYPE>;
using BlockMmad = Blaze::Gemm::Block::BlockMmad<
    DispatchPolicy, AType, LayoutA, AscendC::Std::tuple<BType, uint64_t>, LayoutB,
    int32_t, LayoutC, BiasType, LayoutBias>;
using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueDequant<
    OutType, BiasType, X2ScaleType, X1ScaleType, float>;
using TQBMMKernel = Blaze::Gemm::Kernel::GemmUniversal<
    ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

TQBMMKernel kernel;
kernel(params);
```

### 与 Kernel Qbmm Mix 的区别
| 项目 | QBMM Mix | TQBMM Mix |
|------|----------|-----------|
| A 布局（permX1） | 无（`[batch, m, k]` 连续） | `[m, batch, k]`（M 轴非连续） |
| C 输出布局 | `[batch, m, n]` | `[m, batch, n]` |
| offsetC 计算 | `batchCOffset * m * n + mPos * n + nPos` | `mPos * batch * n + batchIdx * n + nPos` |
| ScheduleType | `KernelMmadWithScaleMix` | `KernelMmadMultiBlockTQBMMMx` |
| Tiling 结构 | `QBMMTiling` | `TQBMMParams` |
| Batch 处理 | 4 维广播（batchA/B/C 1-4） | 3D batch 循环 + tail round |
| 同步函数 | common_utils.h 公共函数 | common_utils.h 公共函数（复用） |

## 3. HIFP8 Fixpipe Quant 特化

### 功能说明
TQBMM fixpipe quant 量化 Kernel，仅支持 AIC 计算，支持 HIFP8（`hifloat8_t`）量化格式。通过 fixpipe 在 Cube 侧完成反量化（per-channel/per-tensor scale）。

### 特殊约束
- **量化格式**：HIFP8（`hifloat8_t`）。
- **Scale 因子**：
  - per-channel 模式：`scaleBGmAddr` 提供 `[1, N]` 的 uint64 编码 scale
  - per-tensor 模式：从 `scaleAGmAddr` 和 `scaleBGmAddr` 读取标量并计算 `deqScale = scaleA * scaleB`
- **计算模式**：仅支持 AIC 模式，不支持 AIV 计算（AIV 核直接返回）。

### 类型定义
```cpp
// BlockMmad::DispatchPolicy::ScheduleType 为 KernelMmadMultiBlockTQBMMFixpipeQuant 时的特化
template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
class GemmUniversal;

struct TQBMMParams {
    uint32_t x1QuantMode;
    uint32_t x2QuantMode;
    uint32_t kAL1, kBL1, nBufferNum;
    uint32_t baseM, baseN, baseK;
    uint32_t isBias;
    uint32_t dbL0C;
    uint32_t bMustHitL2 = 1U;
};

struct Params {
    ProblemShape problemShape;
    BlockMmadParams mmadParams;
    BlockSchedulerParams schParams;
    TQBMMParams tqbmmParams;
};
```

### 调用示例
```cpp
using AType = hifloat8_t;
using BType = hifloat8_t;
using X2ScaleType = uint64_t;
using CType = half;
using BiasType = float;

using DispatchPolicy = Blaze::Gemm::MatmulWithScaleFixpipeQuant<
    0, false, Blaze::Gemm::KernelMmadMultiBlockTQBMMFixpipeQuant, __gm__ int32_t*, NON_CONTIGUOUS_TYPE>;
using BlockMmad = Blaze::Gemm::Block::BlockMmad<
    DispatchPolicy, AType, LayoutA, AscendC::Std::tuple<BType, X2ScaleType>, LayoutB,
    CType, LayoutC, BiasType, LayoutBias>;
using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;
using TQBMMKernel = Blaze::Gemm::Kernel::GemmUniversal<
    ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

TQBMMKernel kernel;
kernel(params);
```

### 与 Kernel Qbmm Cube 的区别
| 项目 | QBMM Cube | TQBMM Fixpipe Quant |
|------|-----------|---------------------|
| A 布局（permX1） | 无（`[batch, m, k]` 连续） | `[m, batch, k]`（M 轴非连续） |
| C 输出布局 | `[batch, m, n]` | `[m, batch, n]` |
| ScheduleType | `KernelMmadWithScaleFixpipeQuant` | `KernelMmadMultiBlockTQBMMFixpipeQuant` |
| Tiling 结构 | `QBMMTiling` | `TQBMMParams` |
| Batch 处理 | 4 维广播（batchA/B/C 1-4） | 3D batch 循环 + tail round |
