# BlockAttnResPrepare 新增接口说明

本文说明 ops-tensor 为 `BlockAttnResPrepare` Phase 1 AIC+AIV 模板新增的公开接口。算子原型、Host
Tiling 与 Kernel 入口属于 ops-transformer；ops-transformer 负责把私有 TilingData 显式转换为下述组件参数。

## 1. 组件组合

```cpp
using BlockAttnResPrepareProblemShape =
    asc::te::shape<int64_t, int64_t, int64_t, int64_t>; // [S, N, D, T]
using BlockAttnResPrepareMmadTuple =
    AscendC::Std::tuple<BlockAttnResPrepareMm1, BlockAttnResPrepareMm2>;
using KernelBlockAttnResPrepare = AttentionUniversal<
    BlockAttnResPrepareProblemShape,
    BlockAttnResPrepareMmadTuple,
    BlockAttnResPrepareBlockEpilogue,
    BlockAttnResPrepareBlockScheduler>;
```

`BlockAttnResPreparePolicy::ScheduleType` 为 `KernelBlockAttnResPrepareSchedule`。该类型只用于编译期选择
`AttentionUniversal` 特化和校验组件组合，不携带运行时状态。

MM1 使用 `BlockAttnResPrepareMm1MmadPolicy`，通过
`NoContiguousType::NON_CONTIGUOUS_TYPE_BATCHED_B` 显式启用 batched-DN B Tensor 沿 BL1 N 方向拼接；MM2 使用
默认的 `BlockAttnResPrepareMm2MmadPolicy`。因此新增搬运不会根据 Tensor 维数隐式改变普通 `BlockMmad` 通路。

## 2. Kernel 参数

`KernelBlockAttnResPrepare::Params` 与仓内 `GemmUniversal` 的组合方式一致：顶层只组合问题形状和各组件的
`Params`，不再维护一份算子专用的扁平 Tiling 参数。

```cpp
struct Params {
    ProblemShape problemShape;              // [S, N, D, T]
    Mm1Params mm1Params;                    // Q * V^T
    Mm2Params mm2Params;                    // E * V
    BlockEpilogueParams epilogueParams;     // RMS、softmax、空输入和 workspace
    BlockSchedulerParams schedulerParams;   // T/S 切分和逻辑核分配
};
```

调用接口为：

```cpp
__aicore__ inline void operator()(const Params& params);
```

Kernel 不校验 Host 已经保证的容量和形状约束，也不从其他组件参数中重新推导 MMAD 参数。

## 3. MM1/MM2 参数

MM1 和 MM2 直接复用 `Gemm::Block::BlockMmad::Params`。ops-transformer 必须显式填写地址、L1/L0
切分和 stage。

| 参数组 | A / B / C | 矩阵含义 | 关键切分 |
| --- | --- | --- | --- |
| `mm1Params` | `pseudoQuery` / `blockResidual` / dot workspace | `[blockS,D] * [D,blockT*validN]` | `mL1=sAlign`、`nL1=mm1NAlign`、`kL1=baseDAlign` |
| `mm2Params` | E workspace / `blockResidual` / `numerator` | `[blockS,validN] * [validN,validD]` | `mL1=sAlign`、`nL1=baseDAlign`、`kL1=nAlign` |

两组参数的 `biasGmAddr` 保持空指针。MM2 通过 Fixpipe 直接写最终 GM 输出。

## 4. BlockEpilogue 参数与接口

```cpp
struct BlockEpilogueBlockAttnResPrepare::Params {
    GM_ADDR validBlocksGmAddr;
    GM_ADDR softmaxMaxGmAddr;
    GM_ADDR weightedOutputGmAddr;
    GM_ADDR softmaxSumGmAddr;
    GM_ADDR workspaceGmAddr;
    uint64_t totalD;
    uint32_t baseD;
    uint32_t baseDAlign;
    uint32_t dTileNum;
    uint32_t sAlign;
    uint8_t vUbBufferNum;
    uint64_t eWorkspaceElems;
    uint64_t vUbElems;
    uint64_t dotUbElems;
    uint64_t reduceUbElems;
    uint64_t softmaxUbElems;
    uint64_t workspacePerCoreElems;
    float epsilon; // 默认 1.0e-6F
};
```

公开阶段接口：

| 接口 | 作用 |
| --- | --- |
| `Init(params)` | 保存 Epilogue 参数，建立 UB 区域布局并计算 `1 / D` |
| `ReduceV(vTensor)` | 分 D tile 搬入 `V[validN,D]`，累计每个 N 行的平方和 |
| `FinalizeSoftmax(dotTensor, eWorkspaceTensor, maxTensor, sumTensor)` | RMS 归一化 dot，计算 softmax，并写 E/max/sum |
| `ProcessEmptyInput(outputTensor, maxTensor, sumTensor)` | `validN == 0` 时把 numerator/max/sum 全部写 0 |

这些接口只接收带 Layout 的 Tensor；调用方不传 UB offset、长度或 stride 标量。

## 5. BlockScheduler 参数与接口

```cpp
struct BlockSchedulerBlockAttnResPrepare::Params {
    uint32_t totalWorkUnits;
    uint32_t usedCoreNum;
    uint32_t baseT;
    uint32_t baseS;
    uint32_t sTileNum;
    uint32_t mm1NAlign;
};
```

构造接口：

```cpp
BlockSchedulerBlockAttnResPrepare(
    const ProblemShape& problemShape,
    const Params& params,
    uint32_t validN,
    uint32_t logicalCoreIndex);
```

Kernel 负责运行时 token 合并，并把物理 AIC/AIV index 映射为共享 workspace 的 `logicalCoreIndex`；
Scheduler 只负责 block 分配、`BlockInfo` 生成和 AIV 行切分。

| 接口 | 返回值 |
| --- | --- |
| `GetCoreNums()` | Host 选择的逻辑核数 |
| `GetBlockNums()` | Kernel 传入的运行时 block 总数 |
| `GetNextBlock(BlockInfo&)` | 当前逻辑核的下一个 `[S,N,D,T]` block |
| `GetAivRowRange(blockShape)` | 当前 AIV 的 `{rowStart,rowCount}` |

## 6. Tile 接口

新增后处理 Tile 头位于 `include/blaze/epilogue/tile/`，通过 arch 分发机制调用 arch35 实现：

- `ReduceSquare<FirstTile>::Run(vTensor, sumSquareTensor)`：按行累计平方和；
- `RmsSoftmax::Run(sumSquareTensor, dotTensor, maxTensor, sumTensor, reciprocalD, epsilon)`：RMS 归一化和 softmax；
- `InitializeEmptySoftmax::Run(maxTensor, sumTensor)` 将空输入的 max 初始化为 FP32 最小有限值，将 sum 初始化为 1。
- `FillUb<float>::FillWithValue(tensor, 0.0F)`：复用通用 Tile，把空输入输出 Tensor 的有效区域置 0。

Tile API 均以 Tensor 为参数，不暴露 UB 地址布局。

## 7. 输入输出契约

- `blockResidual`、`pseudoQuery`、`numerator`、`logitMax`、`expSum` 均为 FP32；
- `validBlocks` 为 UINT64，并裁剪到 `[0,N]`；
- `0 < N <= VECTOR_REG_WIDTH / sizeof(float)`，当前 Ascend 950 为 `N <= 64`；
- `D > 0`、`baseD > 0`、`dTileNum > 0`，且 `vUbBufferNum` 必须为 2；
- `validBlocks == 0` 时跳过 Cube，`numerator` 输出全 0，`logitMax` 输出 FP32 最小有限值，`expSum` 输出全 1。
- 非空输入执行 `Q * V^T -> RMS/softmax -> E * V`。
