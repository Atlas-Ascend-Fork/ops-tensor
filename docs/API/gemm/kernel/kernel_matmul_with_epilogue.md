# Kernel Matmul With Epilogue

> [代码位置](../../../../include/blaze/gemm/kernel/kernel_matmul_with_epilogue.h)

## 功能说明

`GemmUniversal` 的公共 AIC/AIV 调度实现：AIC 执行矩阵乘，将完整 K 轴累加后的 FP32 结果通过 Fixpipe 搬到 UB；AIV 调用模板参数 `BlockEpilogue` 完成后处理和输出写回。

sigmoid 与 scale_add 共用此 Kernel，通过具体的 `BlockEpilogue` 选择计算：

| 后处理 | BlockEpilogue | 计算公式 |
| --- | --- | --- |
| sigmoid | `BlockEpilogueMatmulWithSigmoid` | `sigmoid(x1 @ x2 + bias)`，bias 可选 |
| scale_add | `BlockEpilogueFmmWithScaleAdd` | `alpha * (x1 @ x2) + beta * x3` |

## 调度策略

公共特化使用 `is_same_v` 精确匹配 `BlockMmad::DispatchPolicy::ScheduleType`：新组装使用 `KernelMmadWithEpilogue`；原有 `KernelMmadFmmWithScaleAdd` 标签继续保留并路由到同一实现。sigmoid 无需专属调度标签。

```cpp
using SigmoidDispatchPolicy = Blaze::Gemm::MatmulMultiBlockFixpipeOpti<
    Blaze::Gemm::ND_ALIG_1V2_FIXPIPE, 10, Blaze::Gemm::KernelMmadWithEpilogue>;
using ScaleAddDispatchPolicy = Blaze::Gemm::MatmulMultiBlockFixpipeOpti<
    Blaze::Gemm::ND_ALIG_1V2_FIXPIPE, 0, Blaze::Gemm::KernelMmadFmmWithScaleAdd>;
```

## 组件组装

调用方显式包含公共 Kernel 头与所选后处理头。例如 sigmoid：

```cpp
#include "blaze/epilogue/block/block_epilogue_matmul_with_sigmoid.h"
#include "blaze/gemm/block/block_scheduler_matmul_basic.h"
#include "blaze/gemm/kernel/kernel_matmul_with_epilogue.h"

using DispatchPolicy = SigmoidDispatchPolicy;
using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
using LayoutC = asc::te::nd_ext_layout_ptn;
using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulBasic<
    ProblemShape, Blaze::Gemm::NONE_FULL_LOAD_MODE, false, true>;
using BlockMmad = Blaze::Gemm::Block::BlockMmad<
    DispatchPolicy, ElementType, LayoutA, ElementType, LayoutB,
    float, LayoutC, BiasType, LayoutC>;
using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueMatmulWithSigmoid<
    DispatchPolicy, ElementType>;
using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<
    ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
```

兼容入口 `blaze/gemm/kernel/kernel_matmul_with_scale_add.h` 继续聚合原有 scale_add epilogue 与公共 Kernel，不包含独立实现。直接使用公共接口时，scale_add 组装显式包含 `blaze/epilogue/block/block_epilogue_fmm_with_scale_add.h` 与公共 Kernel 头，选择 `ScaleAddDispatchPolicy` 和 `BlockEpilogueFmmWithScaleAdd<DispatchPolicy, ElementType>`。输入布局、数据类型及分块参数由对应场景配置。

## Epilogue 调用接口

`KernelMmadWithEpilogue` 路径按 `mmadParams.nL0` 将调度块切成实际 Fixpipe N tile；每个 tile 同时切片 B 和 bias，并让 AIC/AIV 使用相同的形状与坐标。

入口负责填充后处理初始化参数。sigmoid 入口填写：

```cpp
params.epilogueParams.problemShape = params.problemShape;
params.epilogueParams.splitM = params.mmadParams.splitM != 0;
```

scale_add 入口在此基础上填写分块配置：

```cpp
params.epilogueParams.baseM = params.schParams.baseM;
params.epilogueParams.baseN = params.schParams.baseN;
```

公共 Kernel 直接使用完整的 `epilogueParams`，初始化不再按后处理类型分支：

```cpp
epilogueOp.Init(params.epilogueParams);
epilogueOp(ubTensor, tileShape, tileCoord);
```

原 `KernelMmadFmmWithScaleAdd` 保留既有分块方式。

公共 Kernel 统一使用三参后处理接口：`tileShape` 提供当前 tile 的真实 M/N/K，`tileCoord` 提供全局 M/N/K 与 batch 坐标，由 Epilogue 计算输出地址。

scale_add新增的三参入口计算偏移后转调原多参接口，原有M裁剪、N分块、UB分段和同步实现保持不变。SYRK继续使用原多参接口及显式staging窗口。

## Params 参数结构

```cpp
struct Params {
    ProblemShape problemShape;         // (M, N, K, batch)
    BlockMmadParams mmadParams;         // A/B/bias 地址和分块参数
    BlockEpilogueParams epilogueParams; // 由具体 BlockEpilogue::Params 定义
    BlockSchedulerParams schParams;    // 调度参数
};
```

- sigmoid 的 `epilogueParams` 包含 `outputGmAddr`、`problemShape`、`splitM`。
- scale_add 的 `epilogueParams` 包含 `x3GmAddr`、`outputGmAddr`、`alpha`、`beta`、`problemShape`、`splitM`、`baseM`、`baseN`。
- 入口填写完整的后处理参数：`problemShape` 与外层 Kernel 参数一致，`splitM` 与 BlockMmad 一致；scale_add 的 `baseM/baseN` 与 BlockScheduler 一致。SYRK 在组装入口填写 `epilogueParams.problemShape`，同样使用单参数 `Init`。

## 约束

- 公共 Kernel 固定 `mmadParams.ubDB = 1`，后处理使用 FP32 累加器之后的 UB 空间。
- 当前两种组装使用 `splitM` 与 1:2 Fixpipe 输出；两个 AIV 按 M 轴分工，没有有效行的 AIV 仍参与同步。后处理必须遵守与 BlockMmad 一致的 ready/free 同步协议。
- sigmoid 在完整 K 轴累加后执行。当前 FusedMatMul sigmoid 组装使用二维 FP16/BF16 输入、同类型输出，`batch = 1`；通过 `nd_ext_layout_ptn` / `dn_ext_layout_ptn` 支持 NN、NT、TN、TT。
- sigmoid 的 bias 地址为空时不相加；类型为输入类型或 FP32，调用方检查 `[N]` 或 `[1,N]` 元数据，BlockMmad 沿 M 轴广播。
- 当前 FusedMatMul scale_add 组装面向 arch35、FP16/BF16、连续 ND、非转置且无 bias 的输入，支持输入和输出 batch 一致的批量矩阵乘。其他限制由具体 BlockMmad、BlockScheduler 和 BlockEpilogue 决定。

详见：

- [BlockMmadMatmulFixpipeOpti](../block/block_mmad_matmul_fixpipe_opti.md)
- [BlockEpilogueMatmulWithSigmoid](../../epilogue/block/block_epilogue_matmul_with_sigmoid.md)
- [BlockEpilogueFmmWithScaleAdd](../../epilogue/block/block_epilogue_fmm_with_scale_add.md)
- [fused_mat_mul_sigmoid 样例](../../../../examples/fused_mat_mul/fused_mat_mul_sigmoid/README.md)
