# Block Scheduler StreamK
> [代码位置](../../../../include/blaze/gemm/block/block_scheduler_matmul_streamk.h)

## 功能说明
StreamK 调度器，支持 DP+SK 混合策略。将问题规模切分为 DP（Data Parallel）模式和 SK（StreamK）模式的 tile，适用于 StreamK Kernel 的 AIC+AIV 双核协同计算场景。

**继承自**：[Block Scheduler 公共框架](./block_scheduler.md)

## 模板参数

### 模板参数列表
```cpp
template <typename ProblemShape_>
class BlockSchedulerMatmulStreamK;
```

### 参数详解

| 参数 | 类型约束 | 默认值 | 说明 |
|------|----------|--------|------|
| ProblemShape_ | `Shape<int64_t, int64_t, int64_t, int64_t>` | - | 问题规模 `(m, n, k, batch)` |

### ProblemShape 参数详解

**类型定义**：
```cpp
using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
```

**参数组成**：
| 索引 | 参数 | 类型 | 说明 | 示例值 |
|------|------|------|------|--------|
| 0 | m | int64_t | M 轴尺寸（矩阵 A 的行数） | 1024 |
| 1 | n | int64_t | N 轴尺寸（矩阵 B 的列数） | 1024 |
| 2 | k | int64_t | K 轴尺寸（矩阵 A 的列数 = 矩阵 B 的行数） | 512 |
| 3 | batch | int64_t | Batch 数量（0 或 1 表示单 batch） | 1 |

**使用示例**：
```cpp
// 定义 ProblemShape 类型
using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

// 创建 ProblemShape 实例
ProblemShape problemShape{1024, 1024, 512, 1};  // (M, N, K, Batch)

// 获取各维度值
int64_t m = Get<0>(problemShape);
int64_t n = Get<1>(problemShape);
int64_t k = Get<2>(problemShape);
int64_t batch = Get<3>(problemShape);
```

**参数约束**：
- `m > 0`：M 轴尺寸必须大于 0
- `n > 0`：N 轴尺寸必须大于 0
- `k > 0`：K 轴尺寸必须大于 0
- `batch >= 0`：Batch 数量，0 或 1 表示单 batch
- `m`, `n`, `k` 建议与 `baseM`, `baseN`, `baseK` 成倍数关系，减少尾块开销

**特殊说明**：
- **单 Batch 场景**：`batch = 0` 或 `batch = 1`，不进行 Batch 循环
- **多 Batch 场景**：`batch > 1`，Kernel 层会进行 Batch 循环处理
- StreamK 调度器会根据 ProblemShape 计算 tile 切分和 DP+SK 混合策略

## 准入条件
StreamK 调度器的 DP+SK 划分依赖核心不变量：**`tailMNBlockNums != 0`（存在真实的 SK 块）**。
`CheckIsSkScene` 以"最后一核轮 = 尾部 SK 块"作为判定依据，当 MN 块数可整除核数时该假设不成立，tiling 侧必须保证不进入此场景。

| 准入条件 | 表达式 | 违反后果 |
|----------|--------|----------|
| 非整除约束（必需） | `(mBlockNums × nBlockNums) % usedCoreNum != 0` 且 `mBlockNums × nBlockNums >= usedCoreNum` | 整除时不存在 SK 块，末轮 DP 块被误判为 SK（结果写 workspace 且 AIV 拒绝回写），输出整片全零（无兜底） |
| 多 batch 口径（必需） | `(mBlockNums × nBlockNums × batch) % usedCoreNum != 0`（SK 块存在时） | AIV 侧 `lastLoopTotalCnt` 按**含 batch** 口径计算，整除时 AIV 全部早退，workspace 结果不被汇聚回写 |
| 尾轮长度约束（DPSK 推荐） | `totalMNCnt % aicNum <= aicNum / 2` | 尾部 SK 轮长度超过 DP 轮一半，负载失衡，性能劣化 |

两个推荐准入窗口：

- **纯 SK 场景（MN 小、K 大切 K）**：
  - `totalMNCnt = mBlockNums × nBlockNums <= aicNum / 2`（单轮约束，多轮直接退回非 StreamK 模板）
  - `kCnt = aicNum / totalMNCnt >= 2`，`singleCoreK = CeilDiv(k, kCnt)`
  - 该窗口下 `usedCoreNum > totalMNCnt` 恒成立，`tailMNBlockNums = totalMNCnt != 0`
- **DP+SK 混合场景（MN 大）**：
  - `totalMNCnt >= aicNum`
  - `totalMNCnt % aicNum != 0` 且 `totalMNCnt % aicNum <= aicNum / 2`
  - `kCnt = aicNum / (totalMNCnt % aicNum)`（依赖余数非零，整除会导致除零）

### DP+SK 混合策略
StreamK 调度器将 tile 分为两种模式：
- **DP（Data Parallel）模式**：完整 tile，每个核独立处理完整的 (m, n) tile，结果直接输出到 GM
- **SK（StreamK）模式**：K 轴切分 tile，多个核协同处理一个 (m, n) tile 的不同 K 切分，结果输出到 workspace

### DP 模式 block
- **block 数量**：`totalMNBlockNumsInDP_ = mBlockNums_ × nBlockNums_ - tailMNBlockNums`
- **每个核处理**：完整的 (m, n) block，K 轴不切分
- **输出目标**：GM（通过 BlockMmad 输出）

### SK 模式 block
- **block 数量**：`tailMNBlockNums × skBlockNums`
- **每个核处理**：一个 (m, n) block 的部分 K 切分
- **K 切分数量**：`skBlockNums_ = CeilDiv(k_, skSingleCoreK_)`
- **输出目标**：workspace（通过 BlockMmad 输出）

### block 索引分配
```
blockIdx 判断：
DP 模式：CeilDiv((blockIdx + 1), usedCoreNums_) < CeilDiv(blockNums_, usedCoreNums_)
SK 模式：CeilDiv((blockIdx + 1), usedCoreNums_) == CeilDiv(blockNums_, usedCoreNums_)
```

### Z 型扫描
使用 Z 型扫描策略：
- **正向扫描**：偶数行（rowIdx % 2 == 0）
- **反向扫描**：奇数行（rowIdx % 2 != 0）


## Params 参数结构

### 结构定义
```cpp
struct Params {
    int64_t usedCoreNum{0};                                   // 使用的核数
    int64_t baseM{0};                                         // L0 M 维度 base 大小
    int64_t baseN{0};                                         // L0 N 维度 base 大小
    int64_t baseK{0};                                         // L0 K 维度 base 大小
    int64_t singleCoreK{0};                                   // SK 模式下单核处理的 K 大小
    int64_t kL1{0};                                           // L1 K 维度大小
    uint8_t isHf32{0};                                        // HF32模式标志 (0=关闭, 1=开启)
    uint32_t l2CacheMode = L2_CACHE_DEFAULT; // L2 Cache 配置
};
```

说明：
- `usedCoreNum`：参与计算的 AIC 核数量
- `baseM` / `baseN`：L0 M/N 维度 base 形状，同时作为 L1 M/N 维度
- `baseK`：L0 K 维度 base 形状
- `singleCoreK`：SK 模式下单核处理的 K 维大小（用于 K 轴切分）
- `kL1`：L1 K 维度大小
- `isHf32`：HF32 模式标志（uint8_t 类型，0=关闭，1=开启）
- `l2CacheMode`：L2 Cache 配置模式（uint32_t 类型）

## 特殊成员变量

| 变量 | 说明 |
|------|------|
| usedCoreNums_ | 使用的核数 |
| skBlockNums_ | SK 模式 K 轴 block 数量 |
| blockNums_ | 总 block 数量（DP block + SK block） |
| totalMNBlockNumsInDP_ | DP 模式 block 数量 |
| mBlockIdx_ | 当前 M 轴 block 索引 |
| nBlockIdx_ | 当前 N 轴 block 索引 |
| kBlockIdx_ | 当前 K 轴切分索引（SK 模式） |
| curKBlockNums_ | 当前 K 轴 block 数量（DP=1, SK=skBlockNums_） |
| skSingleCoreK_ | SK 模式单核 K 大小 |

## 特殊成员方法

### 构造函数
```
__aicore__ inline BlockSchedulerMatmulStreamK(const ProblemShape& shape, const Params& params)
```
功能：初始化 BlockSchedulerMatmulStreamK，计算 DP+SK 混合 tile 切分。
参数说明：
| 参数 | 类型 | 说明 |
|------|------|------|
| shape | ProblemShape | 问题规模 `(m, n, k, batch)` |
| params | Params | 调度参数（usedCoreNum, baseM, baseN, baseK, singleCoreK, kL1, isHf32, l2CacheMode） |

执行流程：
1. 设置问题规模：`m_`, `n_`, `k_`, `batch_`
2. 设置 L1/L0 形状：`mL1_ = baseM`, `nL1_ = baseN`, `skSingleCoreK_ = singleCoreK`, `kL1_`, `baseK_`
3. 计算 block 数量：`mBlockNums_ = CeilDiv(m_, mL1_)`, `nBlockNums_ = CeilDiv(n_, nL1_)`, `skBlockNums_ = CeilDiv(k_, skSingleCoreK_)`
4. 计算 DP+SK block：
   - `tailMNBlockNums = (mBlockNums_ × nBlockNums_) % usedCoreNums_`（SK 模式 block 数量）
   - `totalMNBlockNumsInDP_ = mBlockNums_ × nBlockNums_ - tailMNBlockNums`（DP 模式 block 数量）
   - `blockNums_ = totalMNBlockNumsInDP_ + tailMNBlockNums × skBlockNums_`（总 block 数量）
5. 设置 HF32 和 L2 Cache 模式：`isHf32_`, `l2CacheMode_`

### GetBlockNums
```cpp
__aicore__ inline int64_t GetBlockNums()
```
功能：返回总 block 数量（`blockNums_ × batch_`）。

### GetCoreNums
```
__aicore__ inline int64_t GetCoreNums()
```
功能：返回实际需要的核数量（不超过 block 总数）。

### GetBlockShape
```
__aicore__ inline BlockShape GetBlockShape(int64_t blockIdx)
```
功能：返回当前 block 的单核形状。
参数说明：
| 参数 | 类型 | 说明 |
|------|------|------|
| blockIdx | int64_t | block 索引 |

返回值：`BlockShape {blkM, blkN, blkK, 0}`
特殊逻辑：
- **尾块判断**：`mBlockIdx_ == (mBlockNums_ - 1)` 或 `nBlockIdx_ == (nBlockNums_ - 1)`
- **K 切分尾块**：`kBlockIdx_ == (curKBlockNums_ - 1)`
- **DP 模式**：`blkK = k_`（完整 K）
- **SK 模式**：`blkK = skSingleCoreK_` 或 `tailSingleCoreK`

### GetBlockCoord
```
__aicore__ inline BlockCoord GetBlockCoord(int64_t blockIdx)
```
功能：返回当前 block 的单核坐标。
参数说明：
| 参数 | 类型 | 说明 |
|------|------|------|
| blockIdx | int64_t | block 索引 |

返回值：`BlockCoord {mBlockIdx_, nBlockIdx_, kBlockIdx_, 0}`
说明：K 轴索引 `kBlockIdx_` 仅在 SK 模式有效（DP 模式为 0）。

### CheckIsSkScene
```cpp
__aicore__ inline bool CheckIsSkScene(int64_t blockIdx)
```
功能：判断当前 block 是否为 SK 模式。
参数说明：
| 参数 | 类型 | 说明 |
|------|------|------|
| blockIdx | int64_t | block 索引 |

返回值：
- **true**：SK 模式（K 轴切分）
- **false**：DP 模式（完整 block）

判断逻辑：
```
CeilDiv((blockIdx + 1), usedCoreNums_) == CeilDiv(blockNums_, usedCoreNums_)
```


## 调用示例

### 组件组装
```
using ProblemShape = Shape<int64_t, int64_t, int64_t, int64_t>;
using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulStreamK<ProblemShape>;
```

### 参数准备
```
BlockScheduler::Params params = {
    usedCoreNum,     // 使用的核数（如 8）
    baseM,           // L0 M 维度 base（如 256）
    baseN,           // L0 N 维度 base（如 256）
    baseK,           // L0 K 维度 base（如 32）
    singleCoreK,     // SK 模式单核 K 大小（如 k_ / 4）
    kL1,             // L1 K 维度（如 baseK）
    isHf32,          // HF32 模式（uint8_t，0 或 1）
    l2CacheMode      // L2 Cache 配置
};
```

### 组件初始化
```
ProblemShape shape{m, n, k, batch};
BlockScheduler scheduler(shape, params);
```

### 获取 block 数量
```
int64_t blockNums = scheduler.GetBlockNums();
int64_t coreNums = scheduler.GetCoreNums();
for (int64_t blockIdx = GetBlockIdx(); blockIdx < blockNums; blockIdx += coreNums) {
    // 处理 block
}
```

### 判断 DP/SK 模式
```
bool isSkScene = scheduler.CheckIsSkScene(blockIdx);
if (isSkScene) {
    // SK 模式：输出到 workspace
} else {
    // DP 模式：输出到 GM
}
```

### 获取单核形状
```
auto singleCoreShape = scheduler.GetBlockShape(blockIdx);
int64_t blkM = Get<0>(singleCoreShape);
int64_t blkN = Get<1>(singleCoreShape);
int64_t blkK = Get<2>(singleCoreShape);
```

### 获取单核坐标
```
auto singleCoreCoord = scheduler.GetBlockCoord(blockIdx);
int64_t mBlockIdx = Get<0>(singleCoreCoord);
int64_t nBlockIdx = Get<1>(singleCoreCoord);
int64_t kBlockIdx = Get<2>(singleCoreCoord);  // SK 模式有效
```

## 数据流

### DP+SK 混合策略流程
```
问题规模 (m, n, k, batch)
    ↓
tile 切分 (mBlockNums, nBlockNums, skBlockNums)
    ↓
DP block 数量 = mBlockNums × nBlockNums - tailMNBlockNums
    ↓
SK block 数量 = tailMNBlockNums × skBlockNums
    ↓
总 block 数量 = DP block + SK block
    ↓
block 索引判断 (CheckIsSkScene)
    ↓
DP 模式：完整 block，输出到 GM
SK 模式：K 轴切分，输出到 workspace
```

### DP 模式流程
```
blockIdx 判断：CeilDiv((blockIdx + 1), usedCoreNums) < CeilDiv(blockNums, usedCoreNums)
    ↓
curKBlockNums = 1（不切分 K）
    ↓
kBlockIdx = 0
    ↓
GetBlockShape：blkK = k_（完整 K）
    ↓
BlockMmad：输出到 GM
```

### SK 模式流程
```
blockIdx 判断：CeilDiv((blockIdx + 1), usedCoreNums) == CeilDiv(blockNums, usedCoreNums)
    ↓
curKBlockNums = skBlockNums（K 轴切分）
    ↓
kBlockIdx = (blockIdx % usedCoreNums) % curKBlockNums
    ↓
GetBlockShape：blkK = skSingleCoreK_ 或 tailSingleCoreK
    ↓
BlockMmad：输出到 workspace
    ↓
BlockEpilogue（AIV）：workspace 汇聚 → GM
```

### Z 型扫描流程
```
mnIdxInCurLoop（DP/SK 模式的 MN 索引）
    ↓
rowIdx = mnIdxInCurLoop / nBlockNums / mainWindow
    ↓
rowIdx < mainRow：mBlockIdx = rowIdx × mainWindow + mnIdxInCurLoop % mainWindow
    ↓
rowIdx == mainRow：尾窗口计算
    ↓
rowIdx % 2 != 0：反向扫描（nBlockIdx = nBlockNums - 1 - nBlockIdx)
```

## 性能优化建议

### usedCoreNums 配置
- **建议值**：根据实际 AIC 核数量设置（如 8、16）
- **SK 模式比例**：`tailMNBlockNums = (mBlockNums × nBlockNums) % usedCoreNums`
- **准入检查**：`mBlockNums × nBlockNums`（多 batch 时含 batch）须不能整除 `usedCoreNums`，否则不满足 StreamK 准入条件（详见[准入条件](#准入条件)）
- **优化**：调整 usedCoreNums 以减少 SK 模式 block 数量

### singleCoreK 配置
- **建议值**：约为 `k_ / 4`，平衡 K 轴切分数量
- **SK block 数量**：`skBlockNums = CeilDiv(k_, singleCoreK)`
- **优化**：调整 singleCoreK 以减少 K 轴切分数量

### DP+SK 比例配置
- **DP 模式**：`totalMNBlockNumsInDP_ = mBlockNums × nBlockNums - tailMNBlockNums`
- **SK 模式**：`tailMNBlockNums × skBlockNums`
- **优化**：调整 mBlockNums, nBlockNums, usedCoreNums 以增加 DP 模式比例

### block 形状配置
- **mL1 = baseM**：L1 M 维度等于 L0 base（如 256）
- **nL1 = baseN**：L1 N 维度等于 L0 base（如 256）
- **kL1**：L1 K 维度（如 baseK 或更大）
- **优化**：使用性能最优的 block 形状

### HF32 模式配置
- **isHf32 = 1**：启用 HF32 计算模式 0=关闭, 1=开启
- **适用场景**：需要高精度计算的 FP32 场景

### 适用场景
- **StreamK Kernel**：AIC + AIV 双核协同
- **大矩阵场景**：(m × n × k) 较大，需要多核并行
- **K 轴切分场景**：K 维度远大于 M/N
- **高并行度场景**：需要充分利用 AIC 和 AIV 双核
