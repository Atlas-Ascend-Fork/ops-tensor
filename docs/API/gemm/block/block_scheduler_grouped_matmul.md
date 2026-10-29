# Block Scheduler Grouped Matmul（非量化 GMM 调度器）

> [代码位置](../../../../include/blaze/gemm/block/block_scheduler_grouped_matmul.h)

## 功能说明

`BlockSchedulerGmmNoQuant` 是非量化 Grouped Matmul Kernel 的调度组件，承担两类状态：

1. **组间偏移管理**：按 group 逐次推进 x/weight/bias/y 的存储偏移（连续单 tensor 场景）或 tensor 索引（多 tensor 场景），并向 Kernel 返回本组的 `GroupCoord`
2. **组内块调度**：把本组 `[m, n]` 划分为 `baseM × baseN` 的 tile，采用 SWAT 波前扫描，跨组延续物理核编号以保证负载连续；末组任务不足时按 mTailCnt/nTailCnt 拆分尾部 tile

**框架参考**：[Block Scheduler 公共框架](./block_scheduler.md)

## 特殊约束

- 由 `GemmUniversal` 的 `KernelGroupedMmadNoQuant` 特化使用（[kernel_grouped_matmul](../kernel/kernel_grouped_matmul.md)）
- 与通用调度器不同，本组件**持有逐组可变状态**（组偏移累计、组内波前位置），一次构造实例跨全部 group 复用，Kernel 每组调用 `UpdateNextGroup` 推进
- Kernel 侧按“虚拟 block 区间 `[groupStartBlock_, groupEndBlock_)`”编址：物理核通过 `GetFirstBlockIdx` 把自己的编号映射进当前组的虚拟区间，使跨组起始核不重置

## 数据结构

### Params

| 参数 | 说明 |
|------|------|
| `baseM` / `baseN` | 组内 M/N 轴基础 tile 大小 |
| `mTailCnt` / `nTailCnt` | 末组尾部 tile 的 M/N 拆分份数（1 = 不拆分） |
| `mTailAlign` / `nTailAlign` | 拆分后子块的最小对齐粒度（N 轴 NZ 场景为 c0） |
| `groupType` | -1 / 0 / 2，决定偏移推进与稀疏门控 |
| `groupNum` | matmul 组数 |
| `initialM` | tiling 给出的初始 M，用于单组 split-M 的 tail 判定 |
| `singleX` / `singleWeight` / `singleY` | 各 tensor 是否单 tensor；**单 tensor 时返回累计偏移，多 tensor 时偏移恒 0（Kernel 改用 tensor 索引）** |
| `transB` / `weightNz` / `weightElementSize` | weight 布局信息，参与 NZ 对齐尺寸计算 |
| `groupListType` | 0/1/2；=2 且 groupType=0 时启用稀疏定位 |

### GroupCoord（Kernel 消费的组偏移）

| 槽位 | 含义 |
|------|------|
| MNK_M | x 组内起始偏移（单 x 时有效） |
| MNK_N | weight 组内起始偏移（单 weight 时有效；稀疏时为实际组索引 × WeightSize(n, k)，NZ 布局取对齐尺寸） |
| MNK_K | bias 组内起始偏移（单 weight 时有效；稀疏时为实际组索引 × n） |
| MNK_B | y 组内起始偏移（单 y 时有效） |

## 公共接口

| 接口 | 说明 |
|------|------|
| `UpdateNextGroup(problemShape)` | 推进组偏移并重建组内块调度，返回本组 `GroupCoord`。**空组也推进**（shape 按 0 截断），保证偏移与调度同步 |
| `UpdateNextOutputOffset(problemShape)` | 仅推进 y 偏移（AIV 空 K 组补零路径使用），不动块调度状态 |
| `SetGroupIdx(idx)` | Kernel 每组回填组索引：稀疏（`groupListType==2 && groupType==0`）时由 groupList 第一列实际组索引覆写后传入，激活稀疏直接定位；dense 场景 Kernel 不调用，`groupIdx_` 不被消费 |
| `GetBlockNums()` / `GetCoreNums()` | 当前组虚拟任务总数 / 参与核数（`min(groupEndBlock_, blockNum_)`） |
| `GetFirstBlockIdx(blockIdx)` | 物理核 → 当前组虚拟区间首个任务号 |
| `GetBlockShape<TransB, BType>(taskIdx)` / `GetBlockCoord(taskIdx)` | 任务号 → 块形状（含 tail split 子块）/ 块坐标 |

## 组偏移推进规则（UpdateGroupOffset）

```text
单 x:      nextAOffset_ += m × k           （x 多 tensor 时 Kernel 用 tensor 索引，偏移 0）
单 weight: dense  → nextBOffset_ += WeightSize(n, k)，nextBiasOffset_ += n
           sparse → B 偏移 = groupIdx × WeightSize(n, k)
                    bias 偏移 = groupIdx × n               （直接定位，不累计）
单 y:      nextCOffset_ += m × n
```

**稀疏门控**：稀疏定位仅在 `groupListType == 2 && groupType == 0` 时生效。tiling 将 m-m-s 归一化为 groupType = -1，此时即使 groupListType = 2 也走 dense 累计路径（groupIdx 不被消费）。

**WeightSize**：ND 为 `n × k`；NZ 为 `CeilAlign(n, c0) × CeilAlign(k, 16)`（c0 = 32 / weightElementSize）。

## 组内块调度（UpdateNextProblem → UpdateBlockInfo）

```text
mTileNum = CeilDiv(m, baseM)，nTileNum = CeilDiv(n, baseN)
logicalTileNum = mTileNum × nTileNum
tailWaveBase   = 最后一个不满波前的起点
  波前 = logicalTileNum % blockNum_ ≠ 0 且 (mTailCnt > 1 || nTailCnt > 1) 时
         尾部 wave 按 mTailCnt × nTailCnt 展开为多任务
mainWindow_ = min(WINDOW_LEN, mTileNum)       ← SWAT 窗口（WINDOW_LEN = 4）
组区间: groupStartBlock_ → groupEndBlock_ = groupStartBlock_ + totalTileNum
        nextGroupStartBlock_ = groupEndBlock_ % blockNum_   ← 跨组延续起始核
```

- `GetTileCoord`：SWAT 扫描——窗口内按行主序，奇数行 N 轴反向，提升 weight 局部性
- `GetSplitBlockInfo`：tail split 任务拆出 `mSplit/nSplit` 子块及块内偏移
- `groupStartBlock_` 由上一组的 `nextGroupStartBlock_` 继承，物理核编号跨组不重置

## 与 Kernel 的协作时序

```text
每组:
    Kernel 主循环 sparse 分支 → 读 groupList 第一列实际组索引覆写 groupIdx，
                                SetGroupIdx 回填（dense 场景不调用）
    Kernel.PrepareGroup(groupIdx, splitValue) → 调 UpdateNextGroup(shape) → 得 GroupCoord + 组内调度就绪
    Kernel 逐 taskIdx: GetBlockShape/GetBlockCoord → 切片 → BlockMmad
AIV 补零路径(split-K k=0 组):
    Kernel 调 UpdateNextOutputOffset(shape) → 仅取 y 偏移
```

## 使用示例

见 `examples/acl/grouped_matmul/grouped_matmul_no_quant/`（全部场景矩阵的 tiling 构造与启动）。
