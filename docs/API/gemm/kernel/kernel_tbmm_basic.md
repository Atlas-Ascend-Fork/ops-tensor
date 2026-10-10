# Kernel TBMM Basic
> [代码位置](../../../../include/blaze/gemm/kernel/kernel_tbmm_basic.h)

## 功能说明
TBMM（Tensor Batch Matmul）基础 Kernel，基于 Tensor API 实现，仅支持 AIC 计算，无 AIV 参与，不支持 workspace。适用于多 Batch 矩阵乘场景，在 [KernelMatmulBasic](./kernel_matmul_basic.md) 基础上扩展了 Batch 维度调度、Batch 切分（batchSplitFactor）以及 A 矩阵 Batch 维转置（PERM_X1 非连续）输入能力，集成 BlockScheduler 调度、BlockMmad 计算和 BlockEpilogueEmpty 后处理组件。

**继承自**：GemmUniversal 基础模板（特化实现）

### 特化条件
通过 SFINAE 特化 `GemmUniversal`，当 `BlockMmad::DispatchPolicy::ScheduleType` 为 `KernelMmadMultiBlockTBMM` 时启用本 Kernel：
```cpp
template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class GemmUniversal<
    ProblemShape_, BlockMmad_, BlockEpilogue_, BlockScheduler_,
    AscendC::Std::enable_if_t<
        AscendC::Std::is_same_v<KernelMmadMultiBlockTBMM,
            typename BlockMmad_::DispatchPolicy::ScheduleType>>>;
```
调度类型 `KernelMmadMultiBlockTBMM` 定义于 `dispatch_policy.h`，通过 `MatmulMultiBlockBasic` 策略的第 3 个模板参数 `KernelSchedule_` 指定。

## 特殊约束

### BlockEpilogue 限制
仅支持 `Block::BlockEpilogueEmpty`，不支持任何后处理操作。
```cpp
using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;
```

### 计算模式
仅在 AIC 核函数中执行，不支持 AIV 计算（AIV 核直接返回）。
```cpp
if ASCEND_IS_AIV {
    return;  // AIV 核直接返回，不执行任何计算
}
```

### Workspace 不支持
不支持 workspace，无法存储中间结果，适用于完整 tile 计算场景。

### ProblemShape 维度要求
不同于 [KernelMatmulBasic](./kernel_matmul_basic.md) 的 4 元 ProblemShape，本 Kernel 的 ProblemShape 为 **5 元**，第 5 维为 Batch 切分因子 `splitB`：
```cpp
using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t, int64_t>; // (m, n, k, batch, splitB)
```
- `batch`：Batch 数量，大于 1 表示多 Batch。
- `splitB`：Batch 切分因子，将 `batch` 切分为 `splitB × innerBatch`（`innerBatch = batch / splitB`）；为 1 时不切分。

### Batch 切分输出
当 `splitB > 1` 时，C 矩阵输出采用切分后的 Batch layout（`splitBatchLayoutC`），形状为 `(splitB, innerBatch, m, n)`；为 1 时退化为标准 `(batch, m, n)` layout。
```cpp
uint64_t innerBatch = batch_ / batchSplitFactor_;
if (batchSplitFactor_ > 1) {
    blockMmad(gmBlockA, gmBlockB, gmBlockBias, splitBatchGmBlockC, blockShape);  // 写入切分 C
} else {
    blockMmad(gmBlockA, gmBlockB, gmBlockBias, gmBlockC, blockShape);            // 写入标准 C
}
```

### A 矩阵 Batch 维转置（非连续）
当 `DispatchPolicy` 的 `NonContiguousType_` 设置为 `NON_CONTIGUOUS_TYPE_PERM_X1` 时，启用 A 矩阵 Batch 维转置输入（`TRANS_BATCH_A`），此时 A 的 batch stride 与 m stride 互换，适配 batch 维在 m 维内侧的排列（perm）场景。
```cpp
// TRANS_BATCH_A = true 时（perm 场景）：
batchStrideA = k_;            // batch 维 stride
mStrideA     = batch_ * k_;   // m 维 stride（batch 在 m 内侧）
// TRANS_BATCH_A = false 时（默认连续）：
batchStrideA = m_ * k_;
mStrideA     = k_;
```

## 特殊静态常量

| 常量 | 说明 |
|------|------|
| TRANS_B | B 矩阵是否转置（继承自 BlockMmad） |
| NON_CONTIGUOUS_TYPE | 非连续输入类型（继承自 BlockMmad DispatchPolicy） |
| TRANS_BATCH_A | A 矩阵 Batch 维是否转置，`NON_CONTIGUOUS_TYPE == NON_CONTIGUOUS_TYPE_PERM_X1` 时为 true |

## Params 参数结构

### 结构定义
```cpp
struct Params {
    ProblemShape problemShape;          // 问题规模 (m, n, k, batch, splitB)
    BlockMmadParams mmadParams;         // BlockMmad 参数
    BlockEpilogueParams epilogueParams; // BlockEpilogue 参数（Empty 无需设置）
    BlockSchedulerParams schParams;     // BlockScheduler 参数
};
```

### 参数详解

#### ProblemShape 参数
| 参数 | 索引 | 类型 | 说明 | 示例 |
|------|------|------|------|------|
| m | MNK_M (0) | int64_t | M 轴尺寸 | 1024 |
| n | MNK_N (1) | int64_t | N 轴尺寸 | 1024 |
| k | MNK_K (2) | int64_t | K 轴尺寸 | 512 |
| batch | MNK_B (3) | int64_t | Batch 数量 | 16 |
| splitB | MNK_SplitB (4) | int64_t | Batch 切分因子（1 表示不切分） | 1 |

#### BlockMmad 参数
详见 [BlockMmadMatmulBasic Params](../block/block_mmad_matmul_basic.md#params-参数结构)

#### BlockScheduler 参数
详见 [BlockSchedulerMatmulBasic Params](../block/block_scheduler_matmul_basic.md#params-参数结构)

#### BlockEpilogue 参数
Empty Epilogue 无需设置参数。

## 公共成员方法（Public API）

### 构造函数
```cpp
__aicore__ inline GemmUniversal()
```
功能：构造 GemmUniversal（KernelTBMMBasic）对象。

### 析构函数
```cpp
__aicore__ inline ~GemmUniversal()
```
功能：析构 GemmUniversal（KernelTBMMBasic）对象。

### operator函数
```cpp
__aicore__ inline void operator()(Params& params)
```
功能：执行 TBMM 基础矩阵乘 Kernel 计算。

执行流程：
```
AIV 核检查：直接返回
    ↓
Init：设置问题规模 (m, n, k, batch, splitB)、GM 地址
    ↓
BlockScheduler 初始化
    ↓
Block 索引检查：超出实际核数则返回
    ↓
HF32 模式设置（可选）
    ↓
BlockMmad 初始化
    ↓
MatmulProcess：构建 GM Tensor、配置 L2 Cache、遍历 tile 执行 BlockMmad
    ↓
UnsetHf32：关闭 HF32 模式
```

## MatmulProcess 执行流程

### GM Tensor 构建
使用 Tensor API 构建 A、B、C、Bias 的 GM Tensor，支持 Batch 维与切分：

```cpp
// A：Batch ND layout（TRANS_BATCH_A 时 stride 互换）
auto layoutA = MakeNDBatchLayout<AType>(batch_, m_, k_, batchStrideA, mStrideA);
// B：frame_layout_format，(batch, k, n)
auto layoutB = MakeLayoutB{}(batch_, k_, n_);
// C：Batch ND layout，(batch, m, n)
auto layoutC = MakeNDBatchLayout<CType>(batch_, m_, n_, n_, batch_ * n_);
// 切分 Batch 的 C layout：(splitB, innerBatch, m, n)
auto splitBatchLayoutC = asc::te::make_pattern_layout<
    asc::te::nd_layout_ptn,
    asc::te::layout_trait<CType, AscendC::Std::Int<asc::te::c0_element<CType>>>>(
    asc::te::make_shape(batchSplitFactor_, innerBatch, asc::te::make_shape(m_, n_)),
    asc::te::make_stride(m_ * innerBatch * n_, n_, asc::te::make_stride(innerBatch * n_, asc::te::_1{})));
// Bias：单行 (1, n)，跨 Batch 共享
auto layoutBias = MakeLayoutBias{}(1L, n_);

auto gmA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(aGmAddr_), layoutA);
auto gmB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(bGmAddr_), layoutB);
auto gmC = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(cGmAddr_), layoutC);
auto splitBatchGmC = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(cGmAddr_), splitBatchLayoutC);
auto gmBias = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(biasGmAddr_), layoutBias);
```

### Tile 循环与 Batch 切分
```cpp
for (int64_t blockIdx = curBlockIdx; blockIdx < totalBlockNums; blockIdx += coreNums) {
    auto blockShape = bs.GetBlockShape<TRANS_B, BType>(blockIdx); // (m, n, k, batch)
    auto blockCoord = bs.GetBlockCoord(blockIdx);                 // (m, n, k, batch)
    curBatchIdx_ = asc::te::get<MNK_B>(blockCoord);
    // 切出当前 tile 的 A/B/C 子 Tensor（Squeeze 掉 batch 维）
    auto gmBlockA = asc::te::squeeze<0>(gmA.slice(
        AscendC::MakeCoord(curBatchIdx_, AscendC::MakeCoord(coordM, 0L)),
        AscendC::MakeShape(1L, AscendC::MakeShape(shapeM, shapeK))));
    ...
    if (batchSplitFactor_ > 1) {
        blockMmad(gmBlockA, gmBlockB, gmBlockBias, splitBatchGmBlockC, blockShape);
    } else {
        blockMmad(gmBlockA, gmBlockB, gmBlockBias, gmBlockC, blockShape);
    }
}
```

说明：
- **总 Block 数**：`totalBlockNums = bs.GetBlockNums()`，即 M/N tile 数 × Batch 数（`blockNums_ * batch_`）
- **跨核 stride**：每个核按 `coreNums = AscendC::GetBlockNum()` 步长遍历 tile，多核并行
- **Batch 切分**：`splitB > 1` 时写入切分后的 C layout，否则写入标准 Batch C
- **无 K 切分**：当前不切 K，`kOffset = 0`
- **Bias 共享**：Bias 为跨 Batch 共享的单行向量，按 `(0, coordN)` 切片

## 调用示例

### Kernel 组装与调用

```cpp
// ============== 1. 类型定义 ==============
using AType = half;
using BType = half;
using CType = half;
using BiasType = half;
using LayoutA = asc::te::nd_layout_ptn;        // A 矩阵布局（ND/NZ）
using LayoutB = asc::te::nz_layout_ptn;        // B 矩阵布局（NZ/ND）
using LayoutC = asc::te::nd_layout_ptn;        // C 矩阵布局（ND）
using LayoutBias = LayoutC;                       // Bias 布局

// ============== 2. ProblemShape 定义（5 元）==============
// (m, n, k, batch, splitB)
using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t, int64_t>;

// ============== 3. BlockScheduler 组装 ==============
// FullLoadMode: 0=非全载（默认）, 1=A全载, 2=B全载
constexpr int64_t FULL_LOAD_MODE = 0;
using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulBasic<ProblemShape, FULL_LOAD_MODE>;

// ============== 4. BlockMmad 组装 ==============
constexpr uint64_t FUSED_OP_TYPE = 0;
// 连续输入（NonContiguousType 默认 0）：
using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockBasic<
    FULL_LOAD_MODE, FUSED_OP_TYPE, Blaze::Gemm::KernelMmadMultiBlockTBMM>;
// A 矩阵 Batch 维转置（perm）非连续场景：
// using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockBasic<
//     FULL_LOAD_MODE, FUSED_OP_TYPE, Blaze::Gemm::KernelMmadMultiBlockTBMM,
//     Blaze::Gemm::NON_CONTIGUOUS_TYPE_PERM_X1>;
using BlockMmad = Blaze::Gemm::Block::BlockMmad<
    DispatchPolicy, AType, LayoutA,
    BType, LayoutB, CType, LayoutC,
    BiasType, LayoutBias>;

// ============== 5. BlockEpilogue 组装 ==============
// TBMM Kernel 仅支持 Empty，不支持后处理
using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;

// ============== 6. Kernel 组装 ==============
using TbmmKernel = Blaze::Gemm::Kernel::GemmUniversal<
    ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

// ============== 7. Params 构造 ==============
using Params = typename TbmmKernel::Params;
Params params;

// --- ProblemShape 参数 ---
params.problemShape = {m, n, k, batch, splitB};     // (M, N, K, Batch, SplitB)

// --- BlockMmad 参数 ---
params.mmadParams.aGmAddr = aGM;                    // A 矩阵 GM 地址
params.mmadParams.bGmAddr = bGM;                    // B 矩阵 GM 地址
params.mmadParams.cGmAddr = cGM;                    // C 矩阵 GM 地址
params.mmadParams.biasGmAddr = biasGM;              // Bias GM 地址（可选，nullptr 表示无 bias）
params.mmadParams.mL1 = 256;                        // L1 M 维度尺寸
params.mmadParams.nL1 = 256;                        // L1 N 维度尺寸
params.mmadParams.kL1 = 128;                        // L1 K 维度尺寸
params.mmadParams.mL0 = 128;                        // L0 M 维度尺寸
params.mmadParams.nL0 = 128;                        // L0 N 维度尺寸
params.mmadParams.kL0 = 64;                         // L0 K 维度尺寸
params.mmadParams.l1Stages = 2;                     // L1 缓冲数量（双缓冲）
params.mmadParams.l0cStages = 1;                    // L0C 缓冲数量（单缓冲）

// --- BlockScheduler 参数 ---
params.schParams.mL1 = 256;                         // M 轴 L1 tile 尺寸
params.schParams.nL1 = 256;                         // N 轴 L1 tile 尺寸
params.schParams.kL1 = 128;                         // K 轴 L1 tile 尺寸
params.schParams.baseM = 128;                       // M 轴 L0 base 尺寸
params.schParams.baseN = 128;                       // N 轴 L0 base 尺寸
params.schParams.baseK = 64;                        // K 轴 L0 base 尺寸
params.schParams.isHf32 = 0;                        // HF32 模式标志（0=关闭）
params.schParams.l2CacheMode = Blaze::Gemm::L2_CACHE_DEFAULT;  // L2Cache 使能

// --- BlockEpilogue 参数 ---
// Empty Epilogue 无需设置参数
params.epilogueParams = {};

// ============== 8. Kernel 调用 ==============
TbmmKernel mm;
mm(params);                                         // 执行矩阵乘计算
```

### 常用配置示例

**多 Batch 场景**：
```cpp
params.problemShape = {1024, 1024, 512, 16, 1};  // batch=16，不切分
```

**Batch 切分场景**：
```cpp
params.problemShape = {1024, 1024, 512, 16, 4};  // batch=16 切为 4×4
// C 输出 layout 变为 (4, 4, m, n)
```

**A 矩阵 Batch 维转置（perm）场景**：
```cpp
using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockBasic<
    0, 0, Blaze::Gemm::KernelMmadMultiBlockTBMM,
    Blaze::Gemm::NON_CONTIGUOUS_TYPE_PERM_X1>;
```

**HF32 模式（FP16 输入 + FP32 累加）**：
```cpp
params.schParams.isHf32 = 1;  // Kernel 会在结束时自动 SetHF32Mode(0)
```

## 数据流

### 存储层次
```
GM (A/B/Bias) → BlockScheduler (M/N tile × Batch 调度) → L1 (多缓冲) → L0A/L0B (双缓冲) → L0C → GM (C，可选 Batch 切分 layout)
```

### Kernel 执行流程
```
BlockScheduler 初始化（M/N tile + Batch）
    ↓
Block 索引检查：超出实际核数则返回
    ↓
HF32 模式设置（可选）
    ↓
BlockMmad 初始化
    ↓
构建 GM Tensor（A/B/C/Bias，含 Batch 切分 C）
    ↓
SetL2Cache：配置 L2 Cache（可选）
    ↓
遍历 tile（stride = coreNums）→ 切子 Tensor → BlockMmad 执行
    ↓
清理：关闭 HF32
```

## 性能优化建议

### Batch 切分
- 当 Batch 较大且 M/N 较小导致核数利用不足时，设置 `splitB > 1` 增加 Batch 维并行度
- `splitB` 需能整除 `batch`，建议 `innerBatch = batch / splitB` 为合理值

### L2 Cache 配置
- **大矩阵/多 Batch 场景**：建议禁用 L2 Cache 避免缓存污染（`ALL_L2_CACHE_DISABLE`）
- **小矩阵场景**：可保留 L2 Cache 提升数据复用（`L2_CACHE_DEFAULT`）

### L1/L0 缓冲配置
- **小矩阵场景**：使用单缓冲（`l1Stages=1, l0cStages=1`）
- **中等矩阵场景**：使用双缓冲（`l1Stages=2, l0cStages=1`）
- **大矩阵场景**：使用四缓冲（`l1Stages=4, l0cStages=2`）

### 尾块优化
- 尾块切分（`mTailCnt`/`nTailCnt`）仅在 `batch == 1` 时生效；多 Batch 场景下 BlockScheduler 不做尾块切分
- 单 Batch 场景建议 `mTailCnt`/`nTailCnt` 取 2~4

### Bias 处理
- Bias 为跨 Batch 共享的单行向量（`(1, n)`），`biasGmAddr = nullptr` 表示无 bias

### HF32 模式
- FP16 输入 + FP32 累加输出场景建议启用 `isHf32`，Kernel 会在结束时自动关闭

### 适用场景
- **多 Batch Matmul**：Batch > 1 的批量矩阵乘
- **Batch 维转置输入**：A 矩阵 batch 维在 m 维内侧的 perm 排列
- **简单计算**：无复杂后处理需求（仅 Empty Epilogue）
