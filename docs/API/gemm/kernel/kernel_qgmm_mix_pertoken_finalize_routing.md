# KernelQgmmMixPertokenFinalizeRouting

## 功能与接口

本组件为 arch35 pertoken 全量化 GroupedMatmulFinalizeRouting 的 `GemmUniversal` 特化。AIC 执行量化矩阵乘并写出 split-M UB 结果，AIV 执行 per-channel/per-token 反量化、可选 BF16 bias、logit 加权和路由写回。

```cpp
using Policy = Blaze::Gemm::MatmulWithScaleMix<
    0, false, Blaze::Gemm::KernelQgmmMixPertokenFinalizeRouting>;
using Mmad = Blaze::Gemm::Block::BlockMmad<Policy, AType, LayoutA,
    AscendC::Std::tuple<BType>, LayoutB, AccType, LayoutC, bfloat16_t, LayoutBias>;
using Epilogue = Blaze::Epilogue::Block::BlockEpilogueFinalizeRouting<
    OutType, AccType, float, RowIndexType, WeightScaleType, IsDeterministic,
    Blaze::Epilogue::Block::FinalizeRoutingMode::PERTOKEN>;
using Kernel = Blaze::Gemm::Kernel::GemmUniversal<
    ProblemShape, Mmad, Epilogue, Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit>;
// Kernel kernel; kernel(params);
```

代码中的类型由调用方声明。`ProblemShape` 是 `(M,N,K,1)` 的四维 shape；调度入口由 Policy 的 `ScheduleType` 选择。

## 类型和布局

A/B 的模板准入类型为 INT8、FP8 E4M3FN/E5M2、HIFLOAT8，实际组合还须满足底层 MMAD 的类型配对要求。算子集成常用 INT8×INT8、E4M3FN×E4M3FN、HIFLOAT8×HIFLOAT8。

- A：`nd_ext_layout_ptn`。
- B：`nd_ext_layout_ptn`、`dn_ext_layout_ptn`、`nz_layout_ptn`、`zn_layout_ptn`。
- INT8 输入的 AccType 为 INT32；FP8/HIFLOAT8 为 FP32。Epilogue 的 DataTypeIn 必须与 Mmad::L0CType 一致。
- OutType 为 FP32 或 BF16；logit 为 FP32；WeightScaleType 为 FP32 或 BF16；rowIndex 为 INT32 或 INT64。
- 本 Kernel 不处理 INT32 Cube bias，BF16 bias 由 Epilogue 在 scale 乘法之后相加。

## Params

| 字段 | 含义 |
| --- | --- |
| `problemShape` | 全局逻辑 `(M,N,K,1)` |
| `mmadParams` | `Mmad::Params`；输入 A/B 的 GM 地址，具体分块参数由 Kernel 组装 |
| `sharedInputParams` | `residualGm,yGmAddr,sharedInputOffset,sharedInputLen,n,batch,residualScale`；offset/len 按行计 |
| `epilogueParams` | 输出、scale、bias、logit、rowIndex 和可选确定性 workspace 地址 |
| `groupListGmAddr` | INT64 分组列表 |
| `gmmParams` | 下述 `GmmTiling` |

`GmmTiling` 字段按声明顺序为：

```cpp
uint32_t groupNum, batch, sharedInputOffset, sharedInputLen;
float residualScale;
uint32_t baseM, baseN, baseK, kAL1, kBL1, l1BufferNum;
uint8_t hasBias, dbL0C, groupListType;
uint32_t deterWorkspaceSize, coreNum;
```

baseM/baseN/baseK 和 kAL1/kBL1 均以矩阵元素数计，deterWorkspaceSize 以字节计；coreNum 是实际启动的 AIC 核数。hasBias 与 bias 指针须一致。ScaleB、logit、rowIndex 为必需数据，ScaleA 为 nullptr 时跳过逐 token scale。

## 输出初始化和计算顺序

调用方须在每次调用前清零输出 y。Kernel 在开始矩阵乘前，将 shared input 区间按 residualScale 写入 y。随后按 group 和调度 tile 计算，AIC/AIV 使用成对同步保护结果 UB。

每个 INT8 累加结果先转换为 FP32；依次乘 ScaleB、可选 ScaleA，再加可选 bias，最后乘 logit 并路由到 y。具体转换和写回规则见 [Epilogue](../../epilogue/block/block_epilogue_finalize_routing.md)。

## 确定性模式

`IsDeterministic=false` 使用 Epilogue 原子写回。`IsDeterministic=true` 将每个输入行贡献先写入 workspace，再按输入行次序归并。输出行按 `rowIndex % (2*coreNum)` 分配给 AIV；并行调用的 coreNum 必须与实际启动配置一致。

调用方必须设置 `gmmParams.deterWorkspaceSize` 和 `epilogueParams.workspaceGmAddr`。地址指向用户 workspace，不含系统 workspace 偏移。有效行窗口为：

```text
windowRows = floor(deterWorkspaceSize / (N * sizeof(OutType)) / baseM) * baseM
```

windowRows 至少为 baseM。每个专家的 M 按当前窗口剩余行数分段，所有核按相同的分段边界同步；窗口写满后先按输入行顺序归并，再复用 workspace。单个专家可以跨越多个窗口。workspace 可写且大小不少于 deterWorkspaceSize，调用方须避免与输出/输入重叠。该组件只提供 pertoken 的确定性处理，不为 MX 或 A8W4 提供该模式。

## 相关资料

[组件总览](./grouped_matmul_finalize_routing.md)；[实现](../../../../include/blaze/gemm/kernel/kernel_qgmm_mix_pertoken_finalize_routing.h)。
