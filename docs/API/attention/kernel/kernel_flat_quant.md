# Kernel Flat Quant
> [代码位置](../../../../include/blaze/attention/kernel/kernel_flat_quant.h)

## 功能说明

FlatQuant 双矩阵乘 + AIV MX 量化 Kernel，AIC+AIV 双核协同计算。AIC 侧完成两阶段矩阵乘（A×P2→L1 中间结果，P1×中间结果→L0C→UB），AIV 侧从 UB 读取 bf16 中间结果执行 MX FP4 量化（eMax→scale→quant），输出量化后的 FP4 数据和 E8M0 scale 到 GM。通过块内同步标志（`Gemm::Sync`）实现 AIC-AIV 跨核同步。

**继承自**：[AttentionUniversal 基础框架](./kernel_universal.md)（SFINAE 特化实现，匹配 `KernelFlatQuant`）

## 特殊约束

### 双核协同模式
AIC 和 AIV 必须同时参与计算：
- **AIC**：执行双矩阵乘（Phase 1: A×P2, Phase 2: P1×temp），通过 Fixpipe 将 L0C 结果写入 UB
- **AIV**：从 UB 读取 bf16 数据，执行 MX FP4 量化，输出 int8（FP4 packed）和 scale 到 GM

### 跨核同步
使用 `Gemm::Sync::NotifyVector` / `WaitForVector`（AIC 侧）和 `Gemm::Sync::NotifyCube` / `WaitForCube`（AIV 侧）实现 AIC↔AIV 同步：
- 同步模式：`FLAT_QUANT_SYNC_MODE = 4`
- AIC→AIV 标志：`FLAT_QUANT_SYNC_AIC_AIV_FLAG = 9`（偶数轮）
- AIC→AIV 标志：`FLAT_QUANT_SYNC_AIC_AIV_FLAG + FLAT_QUANT_FLAG_ID_MAX = 25`（奇数轮）
- AIV→AIC 标志：`FLAT_QUANT_SYNC_AIV_AIC_FLAG = 8`

轮次交替使用两组 Flag ID，避免连续轮次之间的 Flag 覆盖。

### BlockMmad 限制
仅支持调度策略为 `BlockFlatQuant` 的 `BlockMmad`，即 `BlockMmad::DispatchPolicy::ScheduleType` 必须为 `KernelFlatQuant`。

### BlockScheduler 限制
使用 `BlockSchedulerFlatQuant` 调度器，按 Batch/K 迭代维度切分任务。

### 输入矩阵语义
ProblemShape `(m, n, k, b)` 在 FlatQuant 中的语义：
- **m**：M 轴尺寸（A 矩阵行数）
- **n**：N 轴尺寸（P1/P2 矩阵尺寸，即 K 和 N 维度相同）
- **k**：存储在 ProblemShape 的 B 位置，表示迭代/Batch 数量
- **b**：Batch 维度

GM 地址映射：
- `aGmAddr`：A 矩阵，形状 `(m * b, k)`，NDExt 布局
- `bGmAddr`（P1）：P1 矩阵，形状 `(m, m)`，NDExt 布局
- `cGmAddr`（P2）：P2 矩阵，形状 `(n, n)`，NDExt 布局

## 特殊常量

| 常量 | 值 | 说明 |
|------|------|------|
| FLAT_QUANT_SYNC_MODE | 4 | 块内 AIC-AIV 同步模式（SYNC_MODE_INTRA） |
| FLAT_QUANT_SYNC_AIV_AIC_FLAG | 8 | AIV→AIC 同步标志 ID |
| FLAT_QUANT_SYNC_AIC_AIV_FLAG | 9 | AIC→AIV 同步标志 ID |
| FLAT_QUANT_FLAG_ID_MAX | 16 | Flag ID 偏移量上限（奇偶轮交替） |

## 类型别名

| 类型 | 说明 |
|------|------|
| BlockMmad | BlockMmad 类型（继承自模板参数） |
| BlockScheduler | BlockScheduler 类型（继承自模板参数） |
| BlockEpilogue | BlockEpilogue 类型（继承自模板参数） |
| ProblemShape | 问题形状类型（继承自模板参数） |
| AType / BType / CType / OutType | 数据类型（继承自 BlockMmad） |
| A_T / B_T | 底层数据类型（`AType::T` / `BType::T`） |
| L0CType | L0C 累加类型，固定为 `float` |
| TupleShape | `Shape<int64_t, int64_t, int64_t, int64_t>` |
| TupleL1L0Shape | `Shape<int64_t, int64_t, int64_t, int64_t>` |
| MakeLayoutA | A 矩阵 Layout 构建器（nd_ext_layout_ptn） |
| MakeLayoutB | P1/P2 矩阵 Layout 构建器（nd_ext_layout_ptn） |

## Params 参数结构

### 结构定义
```cpp
struct Params {
    ProblemShape problemShape;          // 问题规模 (m, n, k, batch)
    BlockMmadParams mmadParams;         // BlockMmad 参数
    BlockEpilogueParams epilogueParams; // BlockEpilogue 参数
    BlockSchedulerParams schParams;     // BlockScheduler 参数
};
```

### 参数详解

#### ProblemShape 参数
| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| m | int64_t | M 轴尺寸 | 128 |
| n | int64_t | N 轴尺寸（P1/P2 矩阵尺寸） | 128 |
| k | int64_t | 存储在 B 位置，迭代/Batch 数量 | 64 |
| batch | int64_t | Batch 维度 | 1 |

#### BlockMmad 参数
详见 [BlockMmadFlatQuant Params](../block/block_mmad_flat_quant.md#params-参数结构)

#### BlockScheduler 参数
详见 [BlockSchedulerFlatQuant Params](../block/block_scheduler_flat_quant.md#params-参数结构)

#### BlockEpilogue 参数
详见 [BlockEpilogueFlatQuant Params](../../epilogue/block/block_epilogue_flat_quant.md#params-参数结构)

## 公共成员方法（Public API）

### 构造函数
```cpp
__aicore__ inline AttentionUniversal()
```
功能：构造 AttentionUniversal（KernelFlatQuant）对象。

### 析构函数
```cpp
__aicore__ inline ~AttentionUniversal()
```
功能：析构 AttentionUniversal（KernelFlatQuant）对象。

### operator函数
```cpp
__aicore__ inline void operator()(Params& params)
```
功能：执行 FlatQuant Kernel 计算。

执行流程：
```
Init：设置问题规模、GM 地址
    ↓
BlockScheduler 初始化（Batch 迭代切分）
    ↓
Block 索引检查：超出实际核数则返回
    ↓
BlockMmad 初始化（L1/L0/L0C 缓冲管理）
    ↓
BlockEpilogue 初始化（UB Tensor、GM GlobalTensor）
    ↓
创建 GM Tensor (A, P1, P2)
    ↓
Tile 循环遍历：
    ├── AIC 路径：
    │   ├── 等待 AIV→AIC Flag（非首轮）
    │   ├── Slice A 矩阵（按 batch 偏移）
    │   ├── BlockMmad 执行双矩阵乘（A×P2→L1, P1×temp→UB）
    │   └── 设置 AIC→AIV Flag
    └── AIV 路径：
        ├── 等待 AIC→AIV Flag
        ├── BlockEpilogue 执行 MX FP4 量化
        └── 设置 AIV→AIC Flag（非末轮）
```

## Tile 循环策略

### FlatQuant 特有的循环策略
```cpp
for (int64_t tileIdx = curBlockIdx; tileIdx < blockNums; tileIdx += coreNums) {
    // roundIdx = tileIdx / coreNums  —— 当前迭代轮次
    // iterBatch = 当前 tile 的 batch 迭代数
    // batchOffset = 当前 tile 的 batch 起始偏移

    if ASCEND_IS_AIC {
        // 非首轮等待 AIV 完成上一轮的量化
        if (roundIdx > 0) {
            Gemm::Sync::WaitForVector<SYNC_MODE, PIPE_FIX>(AIV_AIC_FLAG [+ FLAG_ID_MAX if even round]);
        }
        // AIC 双矩阵乘
        blockMmadOp(gmBlockA, gmP1, gmP2, blockShape, isFirstRound);
        // 通知 AIV 数据已准备好
        Gemm::Sync::NotifyVector<SYNC_MODE, PIPE_FIX>(AIC_AIV_FLAG [+ FLAG_ID_MAX if odd round]);
    }
    if ASCEND_IS_AIV {
        // AIV 等待 AIC 完成
        Gemm::Sync::WaitForCube<SYNC_MODE, PIPE_V>(AIC_AIV_FLAG);
        // AIV MX 量化
        epilogueOp(batchOffset, iterBatch);
        // 非末轮通知 AIC 可以继续
        if (tileIdx + coreNums < CeilAlign(blockNums, coreNums)) {
            Gemm::Sync::NotifyCube<SYNC_MODE, PIPE_MTE3>(AIV_AIC_FLAG);
        }
    }
}
```

说明：
- **stride 策略**：block 按 `coreNums` stride 遍历 tile
- **轮次交替 Flag**：偶数轮用 `AIC_AIV_FLAG`，奇数轮用 `AIC_AIV_FLAG + FLAG_ID_MAX`，避免 Flag 覆盖
- **AIV subBlock 分发**：`roundIdx & 1 == GetSubBlockIdx()` 控制子块处理
- **isFirstRound**：`tileIdx < coreNums` 时为 true，控制 P1/P2 首次加载到 L1

## 调用示例

### Kernel 组装与调用

```cpp
// ============== 1. 类型定义 ==============
using AType = bfloat16_t;
using BType = bfloat16_t;
using CType = bfloat16_t;
using OutType = bfloat16_t;
using LayoutA = asc::te::nd_ext_layout_ptn;
using LayoutB = asc::te::nd_ext_layout_ptn;
using LayoutC = asc::te::nd_ext_layout_ptn;
using LayoutOut = LayoutC;

// ============== 2. ProblemShape 定义 ==============
using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

// ============== 3. DispatchPolicy 组装 ==============
using DispatchPolicy = Blaze::Attention::BlockFlatQuant<>;

// ============== 4. BlockScheduler 组装 ==============
using BlockScheduler = Blaze::Attention::Block::BlockSchedulerFlatQuant<ProblemShape>;

// ============== 5. BlockMmad 组装 ==============
using BlockMmad = Blaze::Attention::Block::BlockMmad<
    DispatchPolicy, AType, LayoutA,
    BType, LayoutB, OutType, LayoutC, CType, LayoutOut>;

// ============== 6. BlockEpilogue 组装 ==============
using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueFlatQuant<
    bfloat16_t, int8_t, uint8_t>;

// ============== 7. Kernel 组装 ==============
using FlatQuantKernel = Blaze::Attention::Kernel::AttentionUniversal<
    ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

// ============== 8. Params 构造 ==============
using Params = typename FlatQuantKernel::Params;
Params params;

// --- ProblemShape 参数 ---
params.problemShape = {m, n, k, batch};

// --- BlockMmad 参数 ---
params.mmadParams.aGmAddr = aGM;       // A 矩阵 GM 地址
params.mmadParams.bGmAddr = p1GM;      // P1 矩阵 GM 地址
params.mmadParams.cGmAddr = p2GM;      // P2 矩阵 GM 地址
params.mmadParams.problemShape = {m, n, k, batch};
params.mmadParams.tileL1 = {mL1, nL1, kL1, iterBatch};
params.mmadParams.tileL0 = {0, 0, baseK, 0};
params.mmadParams.hasP2 = true;

// --- BlockScheduler 参数 ---
params.schParams.iterBatch = iterBatch;
params.schParams.dstTypeMax = 6.0f;       // 量化目标最大值
params.schParams.invDstTypeMax = 1.0f / 6.0f;

// --- BlockEpilogue 参数 ---
params.epilogueParams.outGmAddr = outGM;    // 量化输出 GM 地址
params.epilogueParams.scaleGmAddr = scaleGM; // Scale 输出 GM 地址
params.epilogueParams.problemShape = {m, n, k, batch};
params.epilogueParams.dstTypeMax = 6.0f;
params.epilogueParams.invDstTypeMax = 1.0f / 6.0f;

// ============== 9. Kernel 调用 ==============
FlatQuantKernel kernel;
kernel(params);
```

## 数据流

### 存储层次
```
GM (A, P1, P2) → L1 (AIC Phase1: A×P2) → L0C → L1 (temp)
                                                         ↓
                       L1 (P1, temp) → L0 (AIC Phase2: P1×temp) → L0C → UB
                                                                          ↓
                                              Gemm::Sync (AIC→AIV)
                                                                          ↓
                                              AIV: UB → MX FP4 Quant → GM (Output + Scale)
```

### AIC-AIV 同步时序
```
Round 0:
  AIC: 双矩阵乘 → SetFlag(AIC_AIV_FLAG=9)
  AIV: WaitFlag(AIC_AIV_FLAG=9) → MX量化 → SetFlag(AIV_AIC_FLAG=8)

Round 1:
  AIC: WaitFlag(AIV_AIC_FLAG=8) → 双矩阵乘 → SetFlag(AIC_AIV_FLAG=25)
  AIV: WaitFlag(AIC_AIV_FLAG=25) → MX量化 → SetFlag(AIV_AIC_FLAG=8)

Round 2:
  AIC: WaitFlag(AIV_AIC_FLAG=8) → 双矩阵乘 → SetFlag(AIC_AIV_FLAG=9)
  ...
```

## 性能优化建议

### iterBatch 配置
- `iterBatch` 控制每个 tile 处理的 batch 迭代数，影响 L1 空间占用和并行度
- 较大的 `iterBatch` 减少调度开销但增加 L1 压力
- 建议 `iterBatch` 使 `m * iterBatch` 对齐到 `BLOCK_CUBE`（16）

### baseK 配置
- `baseK` 控制 L0 K 轴切分粒度，影响 L0A/L0B 双缓冲效率
- 建议 `kL1 / baseK` 为整数，减少尾块开销

### AIC-AIV 流水线
- 偶数/奇数轮交替 Flag 确保 AIC 和 AIV 流水线并行
- 首轮 P1/P2 加载到 L1 后复用，避免重复搬运

### 适用场景
- Attention 场景中的双矩阵乘 + 在线量化
- 需要 AIC 矩阵乘与 AIV 向量量化融合的场景
