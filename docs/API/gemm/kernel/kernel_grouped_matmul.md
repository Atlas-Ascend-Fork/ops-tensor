# Kernel Grouped Matmul（非量化 Grouped Matmul）

> [代码位置](../../../../include/blaze/gemm/kernel/kernel_grouped_matmul.h)

## 功能说明

`GemmUniversal` 针对 `KernelGroupedMmadNoQuant` 调度策略的特化实现，是**非量化 Grouped Matmul**（x/weight/out 均为 FLOAT16/BFLOAT16/FLOAT32）的 Tensor API Kernel。每个 group 执行 `y = x @ weight (+ bias)`，group 间按 groupType 切分 M 轴或 K 轴，或各自独立（不分组）。

支持的核心能力：

| 能力 | 说明 |
|------|------|
| groupType = -1（不分组） | x/weight/y 均为多 tensor，各组独立 matmul，groupList 不参与形状与寻址 |
| groupType = 0（M 轴分组） | s-s-s / s-m-s / m-m-s 三种 tensor 组合；M 分布由 groupList 给出 |
| groupType = 2（K 轴分组） | s-s-s（y 为 `[G, M, N]`）/ s-m-m（y 为多 tensor）；x 必须转置为 `[K, M]` |
| groupListType = 0/1/2 | 累积和 / 各组大小 / 稀疏 `[E, 2]` 对（仅 M 轴分组） |
| weight ND / NZ | NZ 权重按 `[n/c0, k/16, 16, c0]` 帧排布，偏移走 NZ 对齐尺寸 |
| AIC+AIV 协同 | MIX_AIC_1_2 启动；AIV 仅在 K 轴分组时对空 K 组（k=0）补零输出 |

**框架参考**：[Kernel 基础框架](./kernel.md)

## 模板参数

| 参数 | 说明 |
|------|------|
| `ProblemShape_` | `asc::te::shape<int64_t, int64_t, int64_t, int64_t>`，即 (m, n, k, batch) |
| `BlockMmad_` | 计算组件，DispatchPolicy 的 ScheduleType 必须为 `KernelGroupedMmadNoQuant` |
| `BlockEpilogue_` | 后处理组件，通常为 `BlockEpilogueEmpty` |
| `BlockScheduler_` | 调度组件，固定为 [BlockSchedulerGmmNoQuant](../block/block_scheduler_grouped_matmul.md) |

由 `BlockMmad_` 派生的编译期常量：

| 常量 | 说明 |
|------|------|
| `TRANS_A` | x 转置（K 轴分组时为 true，x 数据排布为 `[K, M]`） |
| `TRANS_B` | weight 转置 |
| `WEIGHT_NZ` | weight 为 FRACTAL_NZ 格式 |
| `ENABLE_INPLACE_ADD` | DispatchPolicy 的 OUTPUT_MODE 为 INPLACE_ADD 时启用 FP32 原子加输出（本特化当前由 OVERWRITE 路径使用） |

## 数据结构

### GMMTiling

```cpp
struct GMMTiling {
    uint32_t groupNum;        // matmul 组数
    int32_t  groupType;       // -1 不分组 / 0 M 轴分组 / 2 K 轴分组
    uint32_t groupListType;   // 0 累积和 / 1 各组大小 / 2 稀疏对
    uint64_t singleX;         // x 是否单 tensor（1 单 / 0 多）
    uint64_t singleWeight;    // weight 是否单 tensor
    uint64_t singleY;         // y 是否单 tensor
    uint32_t hasBias;         // 是否携带 bias
    uint32_t weightNoL2Cache; // weight 是否禁用 L2 缓存
};
```

### Params

| 字段 | 说明 |
|------|------|
| `problemShape` | 初始问题规模；单 tensor 场景的 N/K 基准值（M 或 K 每组按 groupType 被 splitValue 覆盖） |
| `mmParams` | `BlockMmad::Params`：aGmAddr/bGmAddr/cGmAddr/biasGmAddr/groupListGmAddr 均为 **tensor list 地址**（ListTensorDesc 编码），另含 baseM/baseN/kL1 等 L1/L0 tiling |
| `epilogueParams` | 后处理参数 |
| `schedulerParams` | 调度器参数，见 [BlockSchedulerGmmNoQuant](../block/block_scheduler_grouped_matmul.md) |
| `gmmParams` | `GMMTiling` |

## 场景矩阵与寻址语义

> “单/多”指 aclTensorList 的元素个数。tiling 层可能将部分组合归一化（见下表备注）。

| groupType | x | weight | y | groupList | 寻址语义 |
|:---:|:---:|:---:|:---:|:---|:---|
| -1 | 多 | 多 | 多 | 不消费 | 各组独立：`x_i @ weight_i → y_i`，M/N/K 全部来自各 tensor 描述符 |
| 0 | 单 | 单 | 单 | 必传 | y 单 tensor 按 `Σm_g×n` 顺序堆叠；weight 单 tensor `[G, K, N]` 按组偏移（sparse 时按实际组索引直接定位） |
| 0 | 单 | 多 | 单 | 必传 | weight tensor list 按组索引（sparse 时按实际组索引）选 tensor |
| 0 | 多 | 多 | 单 | 可选 | **tiling 归一化为 groupType = -1**：groupList 不参与形状与寻址，M 取自各 x 描述符，dense 寻址 |
| 2 | 单(转置) | 单 | 单 `[G,M,N]` | 必传 | weight `[K, N]` 按 K 行段偏移；y 按组连续 |
| 2 | 单(转置) | 多 | 多 | 可选 | 各 weight 只存本组 K 段 `[k_g, N]`；y 各组独立 |

### 稀疏 groupList（groupListType = 2）语义

- 形状 `[E, 2]`，每行 `[实际组索引, 组大小]`，**非零组前置**，空组（size=0）后置
- 仅 M 轴分组（attr 层 groupType=0）支持；m-m-s 归一化为 -1 后按 dense 语义处理（groupList 不参与形状与寻址）
- 主循环逐条目解析，**两个索引分离**：
  - `loopIdx`（条目位置）——读取第二列作为该组 M 大小（splitValue）
  - `groupIdx`（实际组索引）——被第一列**覆写**，用于 s-m-s 的 weight/bias tensor list **选 tensor**，并经 `SetGroupIdx` 喂调度器完成 s-s-s 的 weight/bias 单 tensor 存储**直接定位**
- x/y 与 weight/bias 寻址基准不同：x/y 单 tensor 只含非空组数据、按条目出现顺序堆叠，偏移走**条目序累加**（dense/sparse 一致）
- 遇到首个空组（M ≤ 0）即 break——非零组前置契约保证后续全空
- 支持域（attr 层）：详见算子对外接口文档（aclnnGroupedMatmulV5，Ascend 950 非量化约束）

## 执行流程

```text
operator()
    ├─ 参数校验（IsValidGroupParams）
    ├─ 按 groupListType 展开 groupList GM 布局（sparse 为 [2E]）
    ├─ AIV: ProcessEmptyGroups —— 仅 split-K，对 k=0 组并发清零 y 输出段
    └─ AIC: 逐条目循环（loopIdx）
         ├─ sparse + M 分组：读 groupList 第一列实际组索引覆写 groupIdx，
         │    并 SetGroupIdx 喂调度器（单 weight 存储直接定位）
         ├─ GetSplitValueFromGroupList(gmGroupList, loopIdx) —— 按条目位置读取
         │    （cumsum 差分 / count 直读 / sparse 第二列）
         ├─ PrepareGroup(groupIdx, splitValue)
         │    ├─ SetProblemShape: 组内问题规模（M/K 轴被 splitValue 覆盖）
         │    └─ scheduler.UpdateNextGroup: 组偏移 + 组内块调度状态推进
         ├─ M/N/K 任一 ≤ 0 → continue（sparse+M分组 且 M≤0 → break）
         └─ ProcessSingleGroup
              ├─ 组内 shape 逐块切片（ND/NZ 布局）
              ├─ weight L2 hint（weightNoL2Cache && mL1 > m 时禁用）
              └─ blockMmad_(blockA, blockB, blockBias, blockC, blockShape) 逐 tile 计算
```

### SetProblemShape 的 shape 来源

| 条件 | M 来源 | N/K 来源 |
|------|--------|----------|
| 非 single-tensor（任一 x/weight 为多 tensor） | x 描述符第一维（TRANS_A 时第二维） | weight 描述符（sparse 时按实际组索引选取） |
| single-tensor + split-K 且 groupIdx == 0 | x 描述符（覆盖 tiling 初值） | tiling |
| 其它 single-tensor | 上一组值 → 被 splitValue 覆盖（M 或 K） | tiling |

### ProcessEmptyGroups（AIV 补零）

仅当 `groupType == SPLIT_K` 且 groupList 非空时工作：MIX_AIC_1_2 下按 `logicalCore × taskRatio` 将空组（k=0）的 `M×N` 输出区间切给各 AIV 子核并发写零，保证输出完整。

## 特殊约束

- **校验规则**（IsValidGroupParams，OVERWRITE 路径）：groupType ∈ {-1, 0, 2}；groupListType ∈ {0, 1, 2}；groupType ≠ -1 时必须携带 groupList。tiling 层负责 attr 层契约（sparse 仅 M 轴分组）与 m-m-s 归一化，kernel 对归一化后的 (-1, 2) 组合按 dense 语义放行
- **实现注记（groupList 读取）**：主循环对每个条目调用 `GetSplitValueFromGroupList` 读取 groupList；因此 groupNum > 0 时 host 须提供合法 groupList GM buffer
- **INPLACE_ADD 路径**仅接受 dense groupListType {0, 1} + split-K + 单 tensor + 无 bias
- **NZ weight**：要求 `k % 16 == 0` 且 `n % c0 == 0`（c0 = 32/元素大小）；组偏移按 `CeilAlign(n, c0) × CeilAlign(k, 16)` 对齐
- **x 转置**（TRANS_A）仅用于 K 轴分组，数据排布 `[K, M]`

## 使用示例

完整 CSV 驱动的端到端样例（数据生成/tiling 构造/ACL 启动/精度验证）见
`examples/acl/grouped_matmul/grouped_matmul_no_quant/`，覆盖全部场景矩阵。
