# Block Scheduler Matmul Basic
> [代码位置](../../../../include/blaze/gemm/block/block_scheduler_matmul_basic.h)

## 功能说明
MatmulBasic 调度器，支持 tile 切分、Z 型扫描、尾块切分、FP32 SplitK 切分等。适用于 Basic Kernel 场景。

**继承自**：无（独立类）

## 模板参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| ProblemShape_ | Shape<int64_t, int64_t, int64_t, int64_t> | - | 问题规模 `(m, n, k, batch)` |
| FullLoadMode_ | int64_t | 0 | 全载模式：0=非全载, 1=A全载, 2=B全载 |
| IsFp32_ | bool | false | 是否为 FP32 类型 |
| IsNdFormat_ | bool | true | 是否为 ND 格式 |

## 全载模式

| 值 | 说明 | 适用场景 |
|----|------|----------|
| 0 | 非全载模式（默认） | 通用场景，支持 SplitK |
| 1 | A 矩阵全载 | A 矩阵较小，可完全载入 L1 |
| 2 | B 矩阵全载 | B 矩阵较小，可完全载入 L1 |

## Params 参数结构

### 结构定义
```cpp
struct Params {
    uint32_t mL1 = 0;                                        // M 轴 L1 tile 尺寸
    uint32_t nL1 = 0;                                        // N 轴 L1 tile 尺寸
    uint32_t kL1 = 0;                                        // K 轴 L1 tile 尺寸
    uint32_t baseM = 0;                                      // M 轴 L0 base 尺寸
    uint32_t baseN = 0;                                      // N 轴 L0 base 尺寸
    uint32_t baseK = 0;                                      // K 轴 L0 base 尺寸
    uint32_t mTailCnt = 0;                                   // M 轴尾块切分数量
    uint32_t nTailCnt = 0;                                   // N 轴尾块切分数量
    uint32_t mBaseTailSplitCnt = 1;                          // M 轴 L1 尾块切分数量
    uint32_t nBaseTailSplitCnt = 1;                          // N 轴 L1 尾块切分数量
    uint32_t mTailMain = 1;                                  // M 轴 L1 尾块主尺寸
    uint32_t nTailMain = 1;                                  // N 轴 L1 尾块主尺寸
    uint8_t isHf32 = 0;                                      // HF32模式标志 (0=关闭, 1=开启)
    uint32_t l2CacheMode = L2_CACHE_DEFAULT;                 // L2Cache默认使能
    uint32_t sliceM;                                         // 非连续场景m轴
    uint32_t srcNdStride;                                    // 非连续场景m轴stride
    uint32_t innerBatch = 1;                                 // 非连续transpose场景内轴batch值
};
```

### 参数详解

#### 1. L1 Tile 形状 (mL1, nL1, kL1)

**作用**：将大矩阵切分为多个 L1 tile，每个 Block 处理一个 tile。

**传值建议**：
| 参数 | 建议值 | 说明 |
|------|--------|------|
| mL1 | 128~256 | M 轴 tile 尺寸，建议与 baseM 成倍数关系 |
| nL1 | 128~256 | N 轴 tile 尺寸，建议与 baseN 成倍数关系 |
| kL1 | 64~128 | K 轴 tile 尺寸，建议与 baseK 成倍数关系 |

**常用配置**：
```
// 小矩阵场景
mL1=128, nL1=128, kL1=64

// 大矩阵场景
mL1=256, nL1=256, kL1=128
```

#### 2. L0 Base 形状 (baseM, baseN, baseK)

**作用**：L1 tile 进一步切分为 L0 block，每次 Mmad 计算一个 L0 block。

**传值建议**：
| 参数 | 建议值 | 说明 |
|------|--------|------|
| baseM | 64~128 | M 轴 L0 尺寸，建议 mL1/baseM 为整数 |
| baseN | 64~128 | N 轴 L0 尺寸，建议 nL1/baseN 为整数 |
| baseK | 32~64 | K 轴 L0 尺寸，建议 kL1/baseK 为整数 |

**常用配置**：
```
// 配合小 L1 tile
baseM=64, baseN=64, baseK=32

// 配合大 L1 tile
baseM=128, baseN=128, baseK=64
```

#### 3. 尾块切分 (mTailCnt, nTailCnt)

**作用**：Batch=1 场景下，最后一个 tile 的尾块进一步切分给多个 Block 处理，提高并行度。

**触发条件**：
- `batch_ == 1`
- `tileIdx / blockNum_ == (perCoreBlockNum_ - 1)`（最后一个 tile）

**传值建议**：
| 参数 | 建议值 | 说明 |
|------|--------|------|
| mTailCnt | 1~4 | M 轴尾块切分数量，建议不超过 4 |
| nTailCnt | 1~4 | N 轴尾块切分数量，建议不超过 4 |

**示例**：
```
// 不切分（默认）
mTailCnt=0, nTailCnt=0  // 实际会被设为 1

// 2x2 切分
mTailCnt=2, nTailCnt=2  // 尾块切为 4 份，4 个 Block 并行处理
```

#### 4. L1 尾块切分 (mBaseTailSplitCnt, nBaseTailSplitCnt, mTailMain, nTailMain)

**作用**：当矩阵 M/N 轴不能被 mL1/nL1 整除时，尾块区域进一步切分。

**传值建议**：
| 参数 | 建议值 | 说明 |
|------|--------|------|
| mBaseTailSplitCnt | 1 | M 轴 L1 尾块切分数量，建议为 1（不切分） |
| nBaseTailSplitCnt | 1 | N 轴 L1 尾块切分数量，建议为 1（不切分） |
| mTailMain | 1 | M 轴尾块主尺寸（当切分数量>1 时使用） |
| nTailMain | 1 | N 轴尾块主尺寸（当切分数量>1 时使用） |

#### 5. HF32 模式 (isHf32)

**作用**：启用 HF32 计算模式，用于特定精度场景。

**传值建议**：
| 值 | 说明 |
|----|------|
| 0 | 关闭 HF32 模式（默认） |
| 1 | 启用 HF32 模式 |

#### 6. L2 Cache 配置 (l2CacheMode)

**作用**：控制 A/B 矩阵的 L2 Cache 行为。

**可选值**：
| 常量 | 说明 | 适用场景 |
|------|------|----------|
| L2_CACHE_DEFAULT | L2 Cache 使能（默认） | 通用场景 |
| A_L2_CACHE_DISABLE | 禁用 A 矩阵 L2 Cache | A 矩阵复用少的场景, 如A全载 |
| B_L2_CACHE_DISABLE | 禁用 B 矩阵 L2 Cache | B 矩阵复用少的场景, 如B全载 |
| ALL_L2_CACHE_DISABLE | 禁用所有 L2 Cache | A B 矩阵都复用少的场景, 避免L2替换开销 |

#### 7. 非连续场景参数 (sliceM, srcNdStride, innerBatch)

**作用**：处理非连续 ND 格式的矩阵数据。

**传值建议**：
| 参数 | 说明 | 使用场景 |
|------|------|----------|
| sliceM | 非连续 Slice 场景 M 轴切分大小  | 非连续 Slice 场景 |
| srcNdStride | 非连续 Slice 场景 M 轴 stride |  非连续 Slice 场景 |
| innerBatch |非连续 Transpose 场景内轴 batch | 非连续 Transpose 场景 |

**判断逻辑**：
```
isSlice_ = (srcNdStride != 1 && sliceM != 0)
```

slice 路径不新增调度器接口，Kernel 通过 `Params::sliceM`、`Params::srcNdStride` 和 `ProblemShape` 构造 A 矩阵三维 GM layout：shape 为 `[m / sliceM, [sliceM, k]]`，stride 为 `[srcNdStride, [k, 1]]`。

## 类型别名

| 类型 | 说明 |
|------|------|
| BlockShape | Block 形状：`Shape<int64_t, int64_t, int64_t, int64_t>` |
| BlockCoord | Block 坐标：`Coord<int64_t, int64_t, int64_t, int64_t>` (mOffset, nOffset, kOffset, batchIdx) |
| ProblemShape | 问题规模类型（模板参数） |

## 构造函数

```cpp
__aicore__ inline BlockSchedulerMatmulBasic(
    const ProblemShape& shape,  // 问题规模 (m, n, k, batch)
    const Params& params)       // 参数
```

### 参数说明
| 参数 | 类型 | 说明 |
|------|------|------|
| shape | ProblemShape | 问题规模 `(m, n, k, batch)` |
| params | Params | 调度参数 |

### 执行流程
```
1. 设置问题规模：k_, batch_, innerBatch_
2. 设置 L1/L0 形状：mL1_, nL1_, kL1_
3. 计算 block 数量：mBlockNums_, nBlockNums_, blockNums_
4. 计算 L1 尾块参数：mL1NormCnt_, mL1TailMain_, mL1TailLast_, nL1NormCnt_, nL1TailMain_, nL1TailLast_
5. 判断非连续场景：isSlice_
6. 计算尾块切分：mTailCnt_, nTailCnt_, tailCnt_（batch=1 场景）
7. 计算扫描窗口：mainWindow_, mainRow_, tailWindow_
```

## 公共成员方法（Public API）

### GetBlockNums
```cpp
__aicore__ inline int64_t GetBlockNums()
```
功能：返回总 block 数量（`blockNums_ * batch_`）。

### GetCoreNums
```cpp
__aicore__ inline int64_t GetCoreNums()
```
功能：返回实际需要的核数量（不超过 block 总数）。
返回值：`min(blockNums_ * batch_, blockNum_)`

### GetBlockShape
```cpp
template <bool TransB_ = false, class BType_>
__aicore__ inline BlockShape GetBlockShape(int64_t blockIdx)
```
功能：返回当前 block 的形状。
返回值：`BlockShape {mL1, nL1, k, batch}`

参数说明：
| 参数 | 类型 | 说明 |
|------|------|------|
| blockIdx | int64_t | block 索引 |

模板参数说明：
| 参数 | 说明 |
|------|------|
| TransB_ | B 矩阵是否转置（默认 false） |
| BType_ | B 矩阵数据类型 |

### GetBlockCoord
```cpp
__aicore__ inline BlockCoord GetBlockCoord(int64_t blockIdx)
```
功能：返回当前 block 的坐标。
返回值：`BlockCoord {mOffset, nOffset, kOffset, batchIdx}`

参数说明：
| 参数 | 类型 | 说明 |
|------|------|------|
| blockIdx | int | block 索引 |

## 调用示例

### 组件组装
```cpp
using ProblemShape = Shape<int64_t, int64_t, int64_t, int64_t>;
constexpr int64_t FULL_LOAD_MODE = 0;  // 非全载模式
using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulBasic<ProblemShape, FULL_LOAD_MODE>;
```

### 参数准备
```cpp
BlockScheduler::Params params = {
    // L1 tile 形状
    .mL1 = 256,
    .nL1 = 256,
    .kL1 = 128,

    // L0 base 形状
    .baseM = 128,
    .baseN = 128,
    .baseK = 64,

    // 尾块切分（Batch=1 场景）
    .mTailCnt = 2,
    .nTailCnt = 2,

    // L1 尾块切分
    .mBaseTailSplitCnt = 1,
    .nBaseTailSplitCnt = 1,
    .mTailMain = 1,
    .nTailMain = 1,

    // HF32 模式
    .isHf32 = 0,

    // L2 Cache
    .l2CacheMode = L2_CACHE_DEFAULT,

    // 非连续场景（连续 ND 格式不需要设置）
    .sliceM = 0,
    .srcNdStride = 1,
    .innerBatch = 1
};
```

### 组件初始化
```cpp
ProblemShape shape{m, n, k, batch};
BlockScheduler scheduler(shape, params);
```

### 获取 block 数量
```cpp
int64_t blockNums = scheduler.GetBlockNums();
int64_t coreNums = scheduler.GetCoreNums();
for (int64_t blockIdx = blockIdx; blockIdx < blockNums; blockIdx += coreNums) {
    // 处理 block
}
```

### 获取 Block 形状
```cpp
using BType_ = half;
bool TransB = false;
auto blockShape = scheduler.GetBlockShape<TransB, BType_>(blockIdx);
int64_t mL1 = Get<0>(blockShape);
int64_t nL1 = Get<1>(blockShape);
int64_t kL1 = Get<2>(blockShape);
int64_t batch = Get<3>(blockShape);
```

### 获取 Block 坐标
```cpp
auto blockCoord = scheduler.GetBlockCoord(blockIdx);
int64_t mOffset = Get<0>(blockCoord);
int64_t nOffset = Get<1>(blockCoord);
int64_t batchIdx = Get<3>(blockCoord);
```

## 数据流

### Z 型扫描
```
// 奇数行反向扫描
if (rowIdx % 2 != 0) {
    nBlockIdx_ = nBlockNums_ - 1 - nBlockIdx_;
}
```

**示意图**：
```
Z 型扫描示意（mBlockNums_=4, nBlockNums_=4）

     N轴 →
   ┌──┬──┬──┬──┐
   │0 │1 │2 │3 │  ← Row 0（正向）
M  ├──┼──┼──┼──┤
轴 │7 │6 │5 │4 │  ← Row 1（反向）
↓  ├──┼──┼──┼──┤
   │8 │9 │10│11│  ← Row 2（正向）
   ├──┼──┼──┼──┤
   │15│14│13│12│  ← Row 3（反向）
   └──┴──┴──┴──┘

扫描顺序：0→1→2→3→7→6→5→4→8→9→10→11→15→14→13→12
```

### 窗口扫描
```
mainWindow_ = 4  （窗口长度）
mainRow_ = mBlockNums_ / mainWindow_ - 1
tailWindow_ = mBlockNums_ - mainRow_ * mainWindow_
```

## 适用场景

| 场景 | 配置建议 |
|------|----------|
| Basic Kernel | FullLoadMode=0，默认配置 |
| 尾块优化 | batch=1，设置 mTailCnt/nTailCnt |
