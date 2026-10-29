# BlockAttnResPrepare Kernel

> [代码位置](../../../../include/blaze/attention/kernel/kernel_block_attn_res_prepare.h)

`KernelBlockAttnResPrepare` 是以下组件的公开组合别名：

```cpp
AttentionUniversal<ProblemShape, BlockMmadTuple, BlockEpilogue, BlockScheduler>
```

`ProblemShape` 按 `[S,N,D,T]` 解释，`BlockMmadTuple` 的元素 0/1 分别为 MM1 和 MM2。编译期通过
`KernelBlockAttnResPrepareSchedule` 选择对应的 `AttentionUniversal` 特化。

## 参数组织

```cpp
struct Params {
    ProblemShape problemShape;
    Mm1Params mm1Params;
    Mm2Params mm2Params;
    BlockEpilogueParams epilogueParams;
    BlockSchedulerParams schedulerParams;
};
```

这种组织与 `GemmUniversal` 一致。ops-transformer 显式组装每个组件的参数；Tensor Kernel 不再接收专用
扁平 Tiling 结构，也不在 Kernel 内重新计算 MM1/MM2 的 L1/L0 参数。

## 职责

- 把组件参数中的 GM 地址绑定为强类型 Tensor；
- 读取并把 `validBlocks` 裁剪到 `[0,N]`；
- 把物理 AIC/AIV index 映射为共享 workspace 的逻辑核 index；
- 构造 Scheduler，并消费其 `BlockInfo` 和 `AivRowRange`；
- 构造 residual、dot、E、max、sum、output 等语义化 Tensor；
- 编排 MM1、AIV Epilogue、MM2 和 mode-4 跨核同步。

Host Tiling 负责形状、容量和切分合法性，Kernel 不重复执行 `HasSafeStorageContract` 一类运行时校验。

## 调用示例

完整可编译、可运行并带三路输出 golden 校验的示例见
[block_attn_res_prepare](../../../../examples/acl/block_attn_res_prepare/block_attn_res_prepare/README.md)。

以下代码展示在算子 Kernel entry 中如何把 host tiling 和 GM 地址完整映射到公开组合别名。
该 Kernel 使用一个 AIC 和两个 AIV 的 MIX 任务组，因此入口必须声明
`KERNEL_TYPE_MIX_AIC_1_2`。示例中的 `tiling` 字段均由 host tiling 计算得到。

```cpp
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

using Kernel = Blaze::Attention::Kernel::KernelBlockAttnResPrepare;
constexpr uint32_t MM1_L0_K_MAX = 64U;
constexpr uint32_t SINGLE_STAGE = 1U;

Kernel::Params params{};
params.problemShape = {
    static_cast<int64_t>(tiling.totalS),
    static_cast<int64_t>(tiling.totalN),
    static_cast<int64_t>(tiling.totalD),
    static_cast<int64_t>(tiling.totalT)};

params.mm1Params.aGmAddr = effectiveQueryGM;
params.mm1Params.bGmAddr = blockResidualGM;
params.mm1Params.cGmAddr = workspaceGM;
params.mm1Params.workspaceGmAddr = workspaceGM;
params.mm1Params.mL1 = tiling.sAlign;
params.mm1Params.nL1 = tiling.mm1NAlign;
params.mm1Params.kL1 = tiling.baseDAlign;
params.mm1Params.mL0 = tiling.sAlign;
params.mm1Params.nL0 = tiling.mm1NAlign;
params.mm1Params.kL0 = tiling.baseD < MM1_L0_K_MAX ? tiling.baseD : MM1_L0_K_MAX;
params.mm1Params.l1Stages = tiling.mm1L1Stages;
params.mm1Params.l0cStages = SINGLE_STAGE;

params.mm2Params.aGmAddr = workspaceGM;
params.mm2Params.bGmAddr = blockResidualGM;
params.mm2Params.cGmAddr = weightedOutputGM;
params.mm2Params.workspaceGmAddr = workspaceGM;
params.mm2Params.mL1 = tiling.sAlign;
params.mm2Params.nL1 = tiling.baseDAlign;
params.mm2Params.kL1 = tiling.nAlign;
params.mm2Params.mL0 = tiling.sAlign;
params.mm2Params.nL0 = tiling.baseDAlign;
params.mm2Params.kL0 = tiling.nAlign;
params.mm2Params.l1Stages = SINGLE_STAGE;
params.mm2Params.l0cStages = SINGLE_STAGE;

params.epilogueParams.validBlocksGmAddr = validBlocksGM;
params.epilogueParams.softmaxMaxGmAddr = softmaxMaxGM;
params.epilogueParams.weightedOutputGmAddr = weightedOutputGM;
params.epilogueParams.softmaxSumGmAddr = softmaxSumGM;
params.epilogueParams.workspaceGmAddr = workspaceGM;
params.epilogueParams.totalD = tiling.totalD;
params.epilogueParams.baseD = tiling.baseD;
params.epilogueParams.baseDAlign = tiling.baseDAlign;
params.epilogueParams.dTileNum = tiling.dTileNum;
params.epilogueParams.sAlign = tiling.sAlign;
params.epilogueParams.vUbBufferNum = tiling.vUbBufferNum;
params.epilogueParams.eWorkspaceElems = tiling.eWorkspaceElems;
params.epilogueParams.vUbElems = tiling.vUbElems;
params.epilogueParams.dotUbElems = tiling.dotUbElems;
params.epilogueParams.reduceUbElems = tiling.reduceUbElems;
params.epilogueParams.softmaxUbElems = tiling.softmaxUbElems;
params.epilogueParams.workspacePerCoreElems = tiling.workspacePerCoreElems;
params.epilogueParams.epsilon = tiling.epsilon;

params.schedulerParams.totalWorkUnits = tiling.totalWorkUnits;
params.schedulerParams.usedCoreNum = tiling.usedCoreNum;
params.schedulerParams.baseT = tiling.baseT;
params.schedulerParams.baseS = tiling.baseS;
params.schedulerParams.sTileNum = tiling.sTileNum;
params.schedulerParams.mm1NAlign = tiling.mm1NAlign;

Kernel kernel;
kernel(params);
```

`workspaceGM` 同时承载 MM1 dot、AIV 临时数据、E 和 MM2 输入。各区域大小及
`workspacePerCoreElems` 必须与 host tiling 的布局完全一致，不能在 Kernel entry 中重新推导。

## 流程

```text
AIC: MM1(Q * V^T) -> dot workspace
                  |
                  v
AIV: reduce(V^2) -> RMS normalize(dot) -> softmax -> E workspace
                  |
                  v
AIC: MM2(E * V) -> numerator GM
```

同一逻辑组的一个 AIC 和两个 AIV 共享一段 per-core workspace。Kernel 使用三组握手：dot ready、E ready、
E buffer free。MM2 通过 Fixpipe 直接写最终 GM。

`validBlocks == 0` 时跳过 Cube，仅由 AIV0 把 `numerator`、`logitMax`、`expSum` 写 0。
