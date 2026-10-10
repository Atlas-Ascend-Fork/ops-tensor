# Block Scheduler 公共框架
> 基础接口定义，所有 BlockScheduler 实现需遵循此框架

## 概述
BlockScheduler 负责 tile 切分、Block 分配、Z 型扫描、尾块处理等调度任务。不同 Kernel 类型使用不同的 Scheduler 实现：
详见：[README.md](./README.md) 查看 API 清单和实现对比。

## 模板参数

### 通用模板参数
```cpp
template <class ProblemShape_, ...>
class BlockScheduler;
```

| 参数 | 类型 | 说明 |
|------|------|------|
| ProblemShape_ | Shape<int64_t, int64_t, int64_t, int64_t> | 问题规模 `(m, n, k, batch)` |

### 特有模板参数
| Scheduler | 特有参数 | 说明 |
|-----------|----------|------|
| BlockSchedulerMatmulBasic | FullLoadMode_, IsFp32_, IsNdFormat_ | 全载模式（0/1/2）、是否 FP32、是否 ND 格式 |
| BlockSchedulerStreamK | - | - |
| BlockSchedulerQuantBatchMatmulV3 | FullLoadMode_, LayoutA_, LayoutB_, AType_ | 全载模式、布局、数据类型 |

## 类型别名

### 通用类型
```cpp
using BlockShape = Shape<int64_t, int64_t, int64_t, int64_t>;
using ProblemShape = ProblemShape_;

// BlockSchedulerMatmulBasic / BlockSchedulerStreamK
using BlockCoord = Coord<int64_t, int64_t, int64_t, int64_t>;

// BlockSchedulerQuantBatchMatmulV3
using BlockCoord = Coord<int64_t, int64_t>;
```

| 类型 | 说明 |
|------|------|
| BlockShape | Block 形状 `(m, n, k, batch)` |
| BlockCoord | Block 坐标，Matmul Basic/StreamK 为 `(mTileIdx, nTileIdx, kTileIdx, batchIdx)`；QuantBatchMatmulV3 为 `Coord<int64_t, int64_t>`，仅包含 `(mTileIdx, nTileIdx)` |
| ProblemShape | 问题规模类型 |

## Params 结构体

### BlockSchedulerMatmulBasic::Params
详见 [BlockSchedulerMatmulBasic 参数详解](./block_scheduler_matmul_basic.md#params-参数结构)


## 构造函数

### 通用构造函数接口
```cpp
__aicore__ inline BlockScheduler(const ProblemShape& shape, const Params& params)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| shape | ProblemShape | 问题规模 `(m, n, k, batch)` |
| params | Params | 调度参数 |

## 通用成员方法

### GetBlockNums / GetCoreNums
```cpp
__aicore__ inline int64_t GetBlockNums();    // BlockSchedulerMatmulBasic, BlockSchedulerMatmulStreamK
__aicore__ inline int64_t GetCoreNums();     // BlockSchedulerMatmulBasic, BlockSchedulerMatmulStreamK
```
功能：
- `GetBlockNums()`: 返回总 block 数量（含 batch）。
- `GetCoreNums()`: 返回实际需要的核数量（不超过 block 总数）。

### GetBlockShape
```cpp
template <bool TransB_ = false, class BType_>
__aicore__ inline BlockShape GetBlockShape(int64_t blockIdx);       // BlockSchedulerMatmulBasic
__aicore__ inline BlockShape GetBlockShape(int64_t blockIdx);       // BlockSchedulerMatmulStreamK
template <QuantMode AQuantMode_, QuantMode BQuantMode_, bool WeightNz_ = false>
__aicore__ inline BlockShape GetBlockShape(BlockCoord blockCoord);      // BlockSchedulerQuantBatchMatmulV3
```
功能：返回当前 block 的形状。

QuantBatchMatmulV3 的 `BlockShape` 第 3、4 个字段用于携带 M/N 尾块切分偏移。

### GetTileIdx / GetBlockCoord
```cpp
__aicore__ inline BlockCoord GetBlockCoord(int64_t blockIdx);            // BlockSchedulerMatmulBasic
__aicore__ inline BlockCoord GetBlockCoord(int64_t blockIdx);       // BlockSchedulerMatmulStreamK
__aicore__ inline bool GetTileIdx(BlockCoord& blockCoord);              // BlockSchedulerQuantBatchMatmulV3
```
功能：返回当前 block 的坐标。

### GetTileCoord（QuantBatchMatmulV3）
```cpp
__aicore__ inline void GetTileCoord(BlockCoord blockCoord, int64_t& mPos, int64_t& nPos);
```
功能：QuantBatchMatmulV3 根据 2D Block 坐标和 Scheduler 内部记录的尾块切分偏移计算 GM 地址偏移。

## Z 型扫描

### 扫描策略
所有 Scheduler 使用 Z 型扫描策略：
- **WINDOW_LEN = 4**：扫描窗口大小
- **正向扫描**：偶数行（rowIdx % 2 == 0）
- **反向扫描**：奇数行（rowIdx % 2 != 0）

### 扫描示意
```
Z 型扫描示意（mTileNum=4, nTileNum=4）

     N轴 →
   +--+--+--+--+
   |0 |1 |2 |3 |  Row 0（正向）
M  +--+--+--+--+
轴  |7 |6 |5 |4 |  Row 1（反向）
↓   +--+--+--+--+
   |8 |9 |10|11|  Row 2（正向）
   +--+--+--+--+
   |15|14|13|12|  Row 3（反向）
   +--+--+--+--+

扫描顺序：0→1→2→3→7→6→5→4→8→9→10→11→15→14→13→12
```

### 扫描实现
```cpp
// 奇数行反向扫描
if (rowIdx % 2 != 0) {
    nTileIdx = nTileNum - 1 - nTileIdx;
}
```

## 尾块处理

### 尾块判断
| Scheduler | 尾块判断 |
|-----------|----------|
| BlockSchedulerMatmulBasic | `mTileIdx >= mL1NormCnt_` 或 `nTileIdx >= nL1NormCnt_` |
| BlockSchedulerStreamK | `mTileIdx == (mTileNum - 1)` 或 `nTileIdx == (nTileNum - 1)` |
| BlockSchedulerQuantBatchMatmulV3 | `mIdx >= mBaseNormCnt_` 或 `nIdx >= nBaseNormCnt_` |

### 尾块切分
| Scheduler | 尾块切分支持 |
|-----------|--------------|
| BlockSchedulerMatmulBasic | 支持（mTailCnt, nTailCnt） |
| BlockSchedulerStreamK | 支持（K 轴尾块） |
| BlockSchedulerQuantBatchMatmulV3 | 支持（mTailTile, nTailTile） |


## 数据流

### 通用调度流程
```
问题规模 (m, n, k, batch)
    ↓
tile 切分 (mTileNum, nTileNum, kTileNum)
    ↓
尾块参数计算
    ↓
Block 分配 (blockIdx, blockNum)
    ↓
Z 型扫描 (rowIdx, mTileIdx, nTileIdx)
    ↓
Block 形状/坐标 (BlockShape, BlockCoord)
    ↓
BlockMmad 执行
```

## 调用示例

### Matmul Basic Scheduler
```cpp
using ProblemShape = Shape<int64_t, int64_t, int64_t, int64_t>;
using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulBasic<ProblemShape, 0>;

BlockScheduler::Params params = {
    .mL1 = 256, .nL1 = 256, .kL1 = 128,
    .baseM = 128, .baseN = 128, .baseK = 64,
    // ...
};

ProblemShape shape{m, n, k, batch};
BlockScheduler scheduler(shape, params);

int64_t blockIdx = AscendC::GetBlockIdx();
int64_t coreNums = scheduler.GetCoreNums();

for (int64_t blockIdx = blockIdx; blockIdx < scheduler.GetBlockNums(); blockIdx += coreNums) {
    auto blockShape = scheduler.GetBlockShape<TRANS_B, BType>(blockIdx);
    auto blockCoord = scheduler.GetBlockCoord(blockIdx);
    // BlockMmad 计算
}
```
