# Kernel Matmul Basic
> [代码位置](../../../../include/blaze/gemm/kernel/kernel_matmul_basic.h)

## 功能说明
基础矩阵乘 Kernel，仅支持 AIC 计算，无 AIV 参与，不支持 workspace。适用于小矩阵、简单计算场景，集成 BlockScheduler 调度、BlockMmad 计算和 BlockEpilogueEmpty 后处理组件。

**继承自**：GemmUniversal 基础模板（特化实现）

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

### 非连续输入
默认路径不支持带 stride 的非连续输入（仅支持连续 ND/NZ layout）。
当 `DispatchPolicy` 的 `NonContiguousType_` 设置为 `NON_CONTIGUOUS_TYPE_SLICE` 时，支持 A 矩阵 ND slice 非连续输入；该路径通过 `NDSliceLayoutPtn` 构造三维 GM layout，并使用 `CopySliceGM2L1` 完成 A 矩阵 GM->L1 搬运。

## 特殊静态常量

| 常量 | 说明 |
|------|------|
| TRANS_A | A 矩阵是否转置 |
| TRANS_B | B 矩阵是否转置 |
| NON_CONTIGUOUS_TYPE | 非连续类型：0=连续路径，`NoContiguousType::NON_CONTIGUOUS_TYPE_SLICE`=A 矩阵 ND slice 非连续路径 |
| IS_A_FULL_LOAD | 是否 A 全载模式（FullLoadMode_ == A_FULL_LOAD_MODE） |
| C0_SIZE_fp16 | Cube 单元大小（FP16 场景，16） |
| C0_SIZE_fp32 | Cube 单元大小（FP32 场景，8） |

## Params 参数结构

### 结构定义
```cpp
struct Params {
    ProblemShape problemShape;          // 问题规模 (m, n, k, batch)
    BlockMmadParams mmadParams;         // BlockMmad 参数
    BlockEpilogueParams epilogueParams; // BlockEpilogue 参数（Empty 无需设置）
    BlockSchedulerParams schParams;     // BlockScheduler 参数
};
```

### 参数详解

#### ProblemShape 参数
| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| m | int64_t | M 轴尺寸 | 1024 |
| n | int64_t | N 轴尺寸 | 1024 |
| k | int64_t | K 轴尺寸 | 512 |
| batch | int64_t | Batch 数量（0 或 1 为单 batch） | 1 |

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
功能：构造 GemmUniversal（KernelMatmulBasic）对象。

### 析构函数
```cpp
__aicore__ inline ~GemmUniversal()
```
功能：析构 GemmUniversal（KernelMatmulBasic）对象。

### operator函数
```cpp
__aicore__ inline void operator()(Params const& params)
```
功能：执行基础矩阵乘 Kernel 计算。
执行流程：
```
AIV 核检查：直接返回
    ↓
Init：设置问题规模、GM 地址
    ↓
BlockScheduler 初始化
    ↓
Block 索引检查：超出实际数量则返回
    ↓
HF32 模式设置（可选）
    ↓
BlockMmad 初始化
    ↓
创建 GM Tensor (ND/NZ layout)
    ↓
SetL2Cache：配置 L2 Cache（可选）
    ↓
遍历 tile → BlockMmad 执行（每个 tile 独立计算）
    ↓
UnsetHf32：关闭 HF32 模式
```

## Tile 循环策略

### Basic 特有的循环策略
```cpp
for (int64_t tileIdx = curBlockIdx; tileIdx < tileNum; tileIdx += blockNum) {
    // 每个 block 处理 tileIdx = curBlockIdx, curBlockIdx + blockNum, curBlockIdx + 2*blockNum, ...
    // 多 block 并行处理不同 tile
    // 每个 tile 独立调用 BlockMmad 执行矩阵乘
}
```

说明：
- **简单 stride 策略**：block 按固定 stride（blockNum）遍历 tile
- **无 workspace**：每个 tile 完整计算，不依赖中间结果
- **无 AIC-AIV 同步**：单核计算，无需跨核同步

## 调用示例

### Kernel 组装与调用

```cpp
// ============== 1. 类型定义 ==============
using AType = half;                              // A 矩阵数据类型
using BType = half;                              // B 矩阵数据类型
using CType = half;                              // C 矩阵输出类型
using BiasType = half;                           // Bias 数据类型（可选）
using LayoutA = asc::te::nz_layout_ptn;        // A 矩阵布局（NZ/ND）
using LayoutB = asc::te::nz_layout_ptn;        // B 矩阵布局（NZ/ND）
using LayoutC = asc::te::nd_layout_ptn;        // C 矩阵布局（ND）
using LayoutBias = LayoutC;                      // Bias 布局

// ============== 2. ProblemShape 定义 ==============
// 形状：(m, n, k, batch)，batch=0 或 1 表示单 batch
using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

// ============== 3. BlockScheduler 组装 ==============
// FullLoadMode: 0=非全载（默认）, 1=A全载, 2=B全载
constexpr int64_t FULL_LOAD_MODE = 0;
using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulBasic<ProblemShape, FULL_LOAD_MODE>;

// ============== 4. BlockMmad 组装 ==============
// DispatchPolicy: 调度策略，FusedOpType: 融合操作类型
constexpr uint64_t FUSED_OP_TYPE = 0;
using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockBasic<FULL_LOAD_MODE, FUSED_OP_TYPE>;
// A 矩阵 ND slice 非连续场景可使用：
// using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockBasic<
//     FULL_LOAD_MODE, FUSED_OP_TYPE, Blaze::Gemm::KernelMmadMultiBlockBasic, Blaze::Gemm::NON_CONTIGUOUS_TYPE_SLICE>;
using BlockMmad = Blaze::Gemm::Block::BlockMmad<
    DispatchPolicy, AType, LayoutA,
    BType, LayoutB, CType, LayoutC,
    BiasType, LayoutBias>;

// ============== 5. BlockEpilogue 组装 ==============
// Basic Kernel 仅支持 Empty，不支持后处理
using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;

// ============== 6. Kernel 组装 ==============
using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<
    ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

// ============== 7. Params 构造 ==============
using Params = typename MatmulKernel::Params;
Params params;

// --- ProblemShape 参数 ---
params.problemShape = {m, n, k, batch};          // (M, N, K, Batch)

// --- BlockMmad 参数 ---
params.mmadParams.aGmAddr = aGM;                 // A 矩阵 GM 地址
params.mmadParams.bGmAddr = bGM;                 // B 矩阵 GM 地址
params.mmadParams.cGmAddr = cGM;                 // C 矩阵 GM 地址
params.mmadParams.biasGmAddr = biasGM;           // Bias GM 地址（可选，nullptr 表示无 bias）
params.mmadParams.mL1 = 256;                     // L1 M 维度尺寸
params.mmadParams.nL1 = 256;                     // L1 N 维度尺寸
params.mmadParams.kL1 = 128;                     // L1 K 维度尺寸
params.mmadParams.mL0 = 128;                     // L0 M 维度尺寸
params.mmadParams.nL0 = 128;                     // L0 N 维度尺寸
params.mmadParams.kL0 = 64;                      // L0 K 维度尺寸
params.mmadParams.l1Stages = 2;                  // L1 缓冲数量（双缓冲）
params.mmadParams.l0cStages = 1;                 // L0C 缓冲数量（单缓冲）

// --- BlockScheduler 参数 ---
params.schParams.mL1 = 256;                      // M 轴 L1 tile 尺寸
params.schParams.nL1 = 256;                      // N 轴 L1 tile 尺寸
params.schParams.kL1 = 128;                      // K 轴 L1 tile 尺寸
params.schParams.baseM = 128;                    // M 轴 L0 base 尺寸
params.schParams.baseN = 128;                    // N 轴 L0 base 尺寸
params.schParams.baseK = 64;                     // K 轴 L0 base 尺寸
params.schParams.mTailCnt = 2;                   // M 轴尾块切分数量（Batch=1 场景）
params.schParams.nTailCnt = 2;                   // N 轴尾块切分数量（Batch=1 场景）
params.schParams.isHf32 = 0;                     // HF32 模式标志（0=关闭）
params.schParams.l2CacheMode = Blaze::Gemm::L2_CACHE_DEFAULT;  // L2Cache 使能

// --- BlockEpilogue 参数 ---
// Empty Epilogue 无需设置参数
params.epilogueParams = {};

// ============== 8. Kernel 调用 ==============
MatmulKernel mm;
mm(params);                                      // 执行矩阵乘计算
```

### 常用配置示例

**小矩阵场景**：
```cpp
params.schParams.mL1 = 128;
params.schParams.nL1 = 128;
params.schParams.kL1 = 64;
params.schParams.baseM = 64;
params.schParams.baseN = 64;
params.schParams.baseK = 32;
params.mmadParams.l1Stages = 1;  // 单缓冲
params.mmadParams.l0cStages = 1;
```

**大矩阵场景**：
```cpp
params.schParams.mL1 = 256;
params.schParams.nL1 = 256;
params.schParams.kL1 = 128;
params.schParams.baseM = 128;
params.schParams.baseN = 128;
params.schParams.baseK = 64;
params.mmadParams.l1Stages = 2;  // 双缓冲
params.mmadParams.l0cStages = 1;
params.schParams.l2CacheMode = Blaze::Gemm::ALL_L2_CACHE_DISABLE;
```

**尾块优化场景（Batch=1）**：
```cpp
params.schParams.mTailCnt = 4;  // 尾块切为 4×4 = 16 份
params.schParams.nTailCnt = 4;  // 16 个 Block 并行处理尾块
```

## 数据流

### 存储层次
```
GM (A/B/Bias) → BlockScheduler (tile 切分) → L1 (多缓冲) → L0A/L0B (双缓冲) → L0C (单缓冲或双缓冲) → GM (C)
```

### Kernel 执行流程
```
BlockScheduler 初始化
    ↓
获取 tile 数量和形状 (L1/L0 tile)
    ↓
HF32 模式设置（可选）
    ↓
BlockMmad 初始化 (设置缓冲策略)
    ↓
创建 GM Tensor (ND/NZ layout)
    ↓
配置 L2 Cache (可选)
    ↓
遍历 tile → BlockMmad 执行 (每个 tile 独立计算)
    ↓
清理 (关闭 HF32)
```

## 性能优化建议

### L2 Cache 配置
- **大矩阵场景**：建议禁用 L2 Cache 避免缓存污染（`ALL_L2_CACHE_DISABLE`）
- **小矩阵场景**：可保留 L2 Cache 提升数据复用（`L2_CACHE_DEFAULT`）

### L1/L0 缓冲配置
- **小矩阵场景**：使用单缓冲（`l1Stages=1, l0cStages=1`）
- **中等矩阵场景**：使用双缓冲（`l1Stages=2, l0cStages=1`）
- **大矩阵场景**：使用四缓冲（`l1Stages=4, l0cStages=2`）

### Bias 预处理
- **常量 bias**：可预先处理减少运行时开销
- **动态 bias**：保持实时加载

### NZ 格式优化
- **权重矩阵（B）**：优先使用 NZ 格式，提升 L1/L0 搬运效率
- **激活矩阵（A）**：使用 ND 格式即可

### K 轴切分
- `kL1` 和 `baseK` 应根据数据局部性和复用率优化
- 避免 K 轴切分过于细碎导致搬运开销增加

### 尾块优化（Batch=1 场景）
- 设置 `mTailCnt` 和 `nTailCnt` 提高尾块并行度
- 建议值：2~4

### 适用场景
- **小矩阵**：(m × n × k) 较小时，Basic Kernel 更高效
- **简单计算**：无复杂后处理需求时，Basic Kernel 足够
- **单核场景**：仅需 AIC 计算时，Basic Kernel 更简洁
