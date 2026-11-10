# Block Scheduler Flat Quant
> [代码位置](../../../../include/blaze/attention/block/block_scheduler_flat_quant.h)

## 功能说明

FlatQuant 调度器，按 Batch 维度切分任务。每个 Block 处理 `iterBatch` 个 Batch，支持尾块处理（当 Batch 总数不能被 `blockNum * iterBatch` 整除时）。与 Gemm 模块的调度器不同，FlatQuant 调度器不使用 Z 型扫描，而是按 Batch 维度线性切分。

**继承自**：无（独立类）

## 模板参数

| 参数 | 类型 | 说明 |
|------|------|------|
| ProblemShape_ | `Shape<int64_t, int64_t, int64_t, int64_t>` | 问题规模 `(m, n, k, b)`，FlatQuant 中 k == n；b 存储在 B 位置表示 Batch 数量 |

## 类型别名

| 类型 | 说明 |
|------|------|
| BlockShape | `Shape<int64_t, int64_t, int64_t, int64_t>` (mL1, nL1, kL1, iterBatch) |
| BlockL1L0Shape | `Shape<int64_t, int64_t, int64_t, int64_t>` (mL1, nL1, kL1, iterBatch) |
| ProblemShape | 问题规模类型（模板参数） |

## ProblemShape 语义

FlatQuant 中 ProblemShape `(m, n, k, b)`（k == n）的语义与 Gemm 不同：
- **m**：M 轴尺寸（A 矩阵行数；P1 变换矩阵为 `(m, m)`）
- **n**：N 轴尺寸（P2 变换矩阵为 `(n, n)`），同时作为 kL1（L1 K 轴，`kL1_ = n_`）
- **k**（K 位置，`MNK_K`）：矩阵乘 K 维，FlatQuant 中 k == n（P2 为 n×n 矩阵）；调度器不读取该维度
- **b**（B 位置，通过 `MNK_B` 获取）：Batch 数量，调度器按此维度切分

```cpp
m_ = Get<MNK_M>(shape);  // M 轴
n_ = Get<MNK_N>(shape);  // N 轴
k_ = Get<MNK_B>(shape);  // Batch 数量（调度维度）
kL1_ = n_;               // L1 K 轴 = N 轴尺寸
```

## Params 参数结构

### 结构定义
```cpp
struct Params {
    int64_t iterBatch = 1;     // 每个 tile 处理的 Batch 数量
    float dstTypeMax = 0.0f;   // 预留字段，调度器当前不读取；量化阈值须设置在 epilogueParams.dstTypeMax
    float invDstTypeMax = 0.0f;// 预留字段，调度器当前不读取；量化阈值须设置在 epilogueParams.invDstTypeMax
};
```

### 参数详解

| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| iterBatch | int64_t | 每个 tile 处理的 Batch 数量 | 2 |
| dstTypeMax | float | 预留字段，调度器不读取、不传递；量化阈值由 BlockEpilogueFlatQuant 从自身 Params（`epilogueParams.dstTypeMax`）读取并用于 scale 算法分派 | 无需设置 |
| invDstTypeMax | float | 预留字段，调度器不读取、不传递；量化阈值由 BlockEpilogueFlatQuant 从自身 Params（`epilogueParams.invDstTypeMax`）读取 | 无需设置 |

## 构造函数

```cpp
__aicore__ inline BlockSchedulerFlatQuant(
    const ProblemShape& shape,  // 问题规模 (m, n, k, b)，FlatQuant 中 k == n
    int64_t blockNum,           // 核数量
    const Params& params)       // 调度参数
```

### 参数说明
| 参数 | 类型 | 说明 |
|------|------|------|
| shape | ProblemShape | 问题规模 `(m, n, k, b)`，FlatQuant 中 k == n |
| blockNum | int64_t | 核数量（`AscendC::GetBlockNum()`） |
| params | Params | 调度参数 |

### 执行流程
```
1. 设置问题规模：m_, n_, k_（Batch 数量）
2. 设置 L1 形状：mL1_ = m_ * iterBatch, kL1_ = n_, nL1_ = n_
3. 计算主循环轮数：mainBatchLoop_ = k_ / iterBatch / blockNum
4. 计算尾块参数：
   - remainderBatch = k_ - mainBatchLoop_ * blockNum_ * iterBatch_
   - mainTailBatch_ = CeilDiv(remainderBatch, blockNum_)  —— 尾块每核迭代数
   - mainTailBlock_ = remainderBatch % blockNum_           —— 需要少处理 1 个迭代的核数
```

## 公共成员方法（Public API）

### GetBlockNums
```cpp
__aicore__ inline int64_t GetBlockNums()
```
功能：返回总 tile 数量。
返回值：`CeilAlign(CeilDiv(k_, iterBatch_), blockNum_)`

### GetCoreNums
```cpp
__aicore__ inline int64_t GetCoreNums(int64_t blockNum)
```
功能：返回实际需要的核数量（不超过 Batch 数量）。
返回值：`min(k_, blockNum)`（当 `k_ < blockNum` 时返回 `k_`）

### GetBlockShape
```cpp
__aicore__ inline BlockL1L0Shape GetBlockShape(int64_t tileIdx)
```
功能：返回当前 tile 的形状 `(mL1, nL1, kL1, iterBatch)`。

参数说明：
| 参数 | 类型 | 说明 |
|------|------|------|
| tileIdx | int64_t | tile 索引 |

返回值说明：
- 主循环阶段：`{mL1, nL1, kL1, iterBatch}`
- 尾块阶段（`mainTailBatch_ > 0`）：
  - `mainTailIdx < mainTailBlock_`：`{mL1 / iterBatch * mainTailBatch_, nL1, kL1, mainTailBatch_}`
  - `mainTailIdx >= mainTailBlock_`：`{mL1 / iterBatch * (mainTailBatch_ - 1), nL1, kL1, mainTailBatch_ - 1}`

### GetBlockCoord
```cpp
__aicore__ inline int64_t GetBlockCoord(int64_t tileIdx, int64_t curBlockIdx)
```
功能：返回当前 tile 的 Batch 起始偏移（用于计算 A 矩阵 GM 地址偏移）。

参数说明：
| 参数 | 类型 | 说明 |
|------|------|------|
| tileIdx | int64_t | tile 索引 |
| curBlockIdx | int64_t | 当前 block 索引 |

返回值说明：
- 主循环阶段：`tileIdx * iterBatch`
- 尾块阶段：
  - `curBlockIdx < mainTailBlock_`：`mainBatchTotal + curBlockIdx * mainTailBatch_`
  - `curBlockIdx >= mainTailBlock_`：`mainBatchTotal + mainTailBlock_ * mainTailBatch_ + (curBlockIdx - mainTailBlock_) * (mainTailBatch_ - 1)`

## 尾块处理

### 尾块场景
当 `k_`（Batch 数量）不能被 `blockNum * iterBatch` 整除时，产生尾块：

```
示例：k_=10, blockNum=4, iterBatch=2

mainBatchLoop_ = 10 / 2 / 4 = 1（主循环 1 轮）
remainderBatch = 10 - 1 * 4 * 2 = 2
mainTailBatch_ = CeilDiv(2, 4) = 1（尾块每核 1 个迭代）
mainTailBlock_ = 2 % 4 = 2（前 2 个核处理 1 个迭代，后 2 个核处理 0 个迭代）

Block 0: round 0 → iterBatch=2 (batch 0-1), round 1 → mainTailBatch=1 (batch 8)
Block 1: round 0 → iterBatch=2 (batch 2-3), round 1 → mainTailBatch=1 (batch 9)
Block 2: round 0 → iterBatch=2 (batch 4-5), round 1 → mainTailBatch-1=0 (跳过)
Block 3: round 0 → iterBatch=2 (batch 6-7), round 1 → mainTailBatch-1=0 (跳过)
```

## 调用示例

### 组件组装
```cpp
using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
using BlockScheduler = Blaze::Attention::Block::BlockSchedulerFlatQuant<ProblemShape>;
```

### 参数准备
```cpp
BlockScheduler::Params params = {
    .iterBatch = 2
};
// 注意：dstTypeMax / invDstTypeMax 为预留字段，调度器不读取、也不会传递给 Epilogue。
// 量化阈值须设置在 Kernel::Params 的 epilogueParams（即 BlockEpilogueFlatQuant::Params）：
//   kernelParams.epilogueParams.dstTypeMax = 6.0f;
//   kernelParams.epilogueParams.invDstTypeMax = 1.0f / 6.0f;
```

### 组件初始化
```cpp
ProblemShape shape{m, n, k, b};   // FlatQuant 中 k == n
int64_t blockNum = AscendC::GetBlockNum();
BlockScheduler scheduler(shape, blockNum, params);
```

### 获取 tile 信息
```cpp
int64_t curBlockIdx = Gemm::GetCurrentBlockIdx();
int64_t blockNums = scheduler.GetBlockNums();
int64_t coreNums = scheduler.GetCoreNums(blockNum);

for (int64_t tileIdx = curBlockIdx; tileIdx < blockNums; tileIdx += blockNum) {
    auto blockShape = scheduler.GetBlockShape(tileIdx);
    int64_t iterBatch = Get<3>(blockShape);  // 当前 tile 的迭代数
    int64_t batchOffset = scheduler.GetBlockCoord(tileIdx, curBlockIdx);
    // 使用 batchOffset 计算 A 矩阵 GM 偏移，调用 BlockMmad
}
```

## 数据流

### 调度流程
```
问题规模 (m, n, k, b)（k == n）
    ↓
iterBatch 切分：mL1 = m * iterBatch
    ↓
主循环轮数计算：mainBatchLoop = k_ / iterBatch / blockNum
    ↓
尾块参数计算：mainTailBatch, mainTailBlock
    ↓
Tile 遍历：
    ├── GetBlockShape → (mL1, nL1, kL1, iterBatch)
    └── GetBlockCoord → batchOffset（A 矩阵偏移）
    ↓
BlockMmad 执行（使用 batchOffset 和 iterBatch）
```

## 适用场景

| 场景 | 配置建议 |
|------|----------|
| 均匀切分 | k_ 可被 blockNum * iterBatch 整除 |
| 尾块场景 | k_ 不能整除时自动处理尾块 |
| 大 Batch | 增大 iterBatch 减少调度开销 |
| 小 Batch | iterBatch=1，每个 tile 处理 1 个迭代 |
