# Kernel Matmul With Scale Add
> [代码位置](../../../../include/blaze/gemm/kernel/kernel_matmul_with_scale_add.h)

## 功能说明
兼容头文件保留原入口，并转发到公共 [Kernel Matmul With Epilogue](./kernel_matmul_with_epilogue.md)；原调度标签、分块与后处理计算接口保持兼容。

FusedMatMul scale_add场景的AIC+AIV融合Kernel，按`KernelMmadFmmWithScaleAdd`调度策略对`GemmUniversal`进行特化，实现：

$$ y = alpha \times (x1@x2) + beta \times x3 $$

AIC执行矩阵乘并通过Fixpipe将fp32累加结果搬到UB，AIV使用`BlockEpilogueFmmWithScaleAdd`完成缩放、相加、类型转换和写回。

**继承自**：GemmUniversal基础模板，按`KernelMmadFmmWithScaleAdd`调度策略特化。

**适用场景**：arch35上bf16/fp16、连续ND、非转置、无Bias的Batch Matmul + scale_add场景。

## 与相关Kernel对比

| 特性 | KernelMatmulWithScaleAdd | KernelMatmulFixpipeOpti |
|------|--------------------------|-------------------------|
| 计算公式 | alpha × (x1@x2) + beta × x3 | x1@x2 |
| BlockMmad | BlockMmadMatmulFixpipeOpti | BlockMmadMatmulFixpipeOpti |
| BlockEpilogue | BlockEpilogueFmmWithScaleAdd | BlockEpilogueFixpipe |
| AIV参与 | 是，1个AIC对应2个AIV | 是 |
| Batch | 支持，x1/x2/x3 Batch一致 | 由具体组装决定 |

## 调度策略

```cpp
using DispatchPolicy = MatmulMultiBlockFixpipeOpti<
    ND_ALIG_1V2_FIXPIPE,
    0,
    KernelMmadFmmWithScaleAdd>;
```

## 特殊约束

### AIC-AIV同步
```cpp
构造: AIV SetFlag(AIV_SYNC_AIC_FLAG) × 2
析构: AIC WaitFlag(AIV_SYNC_AIC_FLAG) × 4
```

每个N tile由AIC发送ready标志，AIV完成后处理后发送free标志。析构时AIC等待两个AIV的最后一组free标志，保证输出完成后再退出。

### splitM子块
```cpp
if ASCEND_IS_AIV {
    if (!params.mmadParams.splitM && AscendC::GetSubBlockIdx() > 0) {
        return;
    }
    curBlockIdx /= AscendC::GetTaskRation();
}
```

FusedMatMul scale_add使用1:2 Fixpipe输出并开启splitM，两个AIV按M轴分工。

### UB配置
```cpp
if ASCEND_IS_AIC {
    auto mmParams = params.mmadParams;
    mmParams.ubDB = 1;
    blockMmad.Init(mmParams);
}
```

x3/output使用fp32累加器后的UB空间，因此Kernel固定关闭累加器UB ping-pong。

## Params参数结构

```cpp
struct Params {
    ProblemShape problemShape;          // (m, n, k, batch)
    BlockMmadParams mmadParams;         // BlockMmad参数，包含splitM和ubDB
    BlockEpilogueParams epilogueParams; // 地址、缩放、problemShape、splitM、baseM/baseN
    BlockSchedulerParams schParams;     // BlockScheduler参数
};
```

## 使用示例

```cpp
using Layout = asc::te::nd_ext_layout_ptn;
using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockFixpipeOpti<
    Blaze::Gemm::ND_ALIG_1V2_FIXPIPE, 0, Blaze::Gemm::KernelMmadFmmWithScaleAdd>;
using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulBasic<
    ProblemShape, Blaze::Gemm::NONE_FULL_LOAD_MODE, false, true>;
using BlockMmad = Blaze::Gemm::Block::BlockMmad<
    DispatchPolicy, half, Layout, half, Layout, float, Layout, half, Layout>;
using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueFmmWithScaleAdd<DispatchPolicy, half>;
using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<
    ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

using Params = typename MatmulKernel::Params;
Params params = {
    {m, n, k, batch},
    {x1GM, x2GM, nullptr, nullptr, nullptr, workspaceGM, ...},
    {x3GM, yGM, alpha, beta},
    {...}
};

params.epilogueParams.problemShape = params.problemShape;
params.epilogueParams.splitM = params.mmadParams.splitM != 0;
params.epilogueParams.baseM = params.schParams.baseM;
params.epilogueParams.baseN = params.schParams.baseN;
MatmulKernel kernel;
kernel(params);
```

## 组件组装

```text
KernelMatmulWithScaleAdd
    → BlockSchedulerMatmulBasic
    → BlockMmadMatmulFixpipeOpti
    → BlockEpilogueFmmWithScaleAdd
```

详见：
- [BlockMmadMatmulFixpipeOpti](../block/block_mmad_matmul_fixpipe_opti.md)
- [BlockEpilogueFmmWithScaleAdd](../../epilogue/block/block_epilogue_fmm_with_scale_add.md)
