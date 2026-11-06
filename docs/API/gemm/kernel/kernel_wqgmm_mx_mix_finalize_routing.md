# GmmWqgmmMxFinalizeRoutingKernel

## 功能与类型

本组件在 arch35 执行 FP8 E4M3FN 激活与 packed FP4 权重的分组矩阵乘，并完成 logit 加权与 FP32 路由累加。AIV 将 packed FP4 转换为 FP8 并写入 L1；AIC 通过公共 MX BlockMmad 执行矩阵乘，结果经 split-M Fixpipe 写入 UB。

AType 为 `fp8_e4m3fn_t`；BType 为 `fp4x2_e2m1_t` 或组件模板允许的 `fp4x2_e1m2_t`。算子集成使用 E2M1。ScaleA/ScaleB 均为 E8M0，bias/shared input 为 BF16，logit/y 为 FP32，rowIndex 为 INT64。A 使用 ND，B 使用转置 WeightNZ（`zn_layout_ptn`）；不提供非转置 B 或 BF16 输出入口。

## 组件组装

```cpp
using Policy = Blaze::Gemm::GroupedMatmulFinalizeRoutingWithWeightQuantMx;
// Mmad 的 A/B 参数分别是 (数据类型, E8M0)，layout 参数是对应二元 tuple。
using Mmad = Blaze::Gemm::Block::BlockMmad<Policy, ATypes, ALayouts, BTypes, BLayouts,
    float, asc::te::nd_ext_layout_ptn, bfloat16_t, asc::te::nd_ext_layout_ptn>;
using Kernel = Blaze::Gemm::Kernel::GmmWqgmmMxFinalizeRoutingKernel<
    ProblemShape, Mmad, Scheduler, WeightPrologue, Epilogue, OutputInit>;
```

`ProblemShape` 为三维 `(K,N,groupNum)`，不能与全量化 Kernel 的 `(M,N,K,1)` 混用。Scheduler 使用 `BlockSchedulerWqgmmNResplit`；WeightPrologue、OutputInit 分别使用本头文件的 `GmmWqgmmFrWeightPrologue`、`GmmWqgmmFrOutputInit`；Epilogue 的 Mode 为 A8W4。

## Params

| 字段 | 含义 |
| --- | --- |
| `problemShape` | `(K,N,groupNum)` |
| `mmad` | ScaleA/ScaleB、bias、A、y 等 GM 地址；L1 分区及动态 K 由 Kernel 配置 |
| `scheduler` | N 主块/两级尾块数量与宽度、核数、baseM、N |
| `prologue.ptrB` | 连续 packed FP4 权重的基址 |
| `epilogue` | 输出、logit、rowIndex 地址；`nSize` 是完整 N，不是 tile N |
| `outputInit` | 输出与 shared input 的初始化范围 |
| `ptrGroupList` | INT64 分组列表 |
| `groupListType` | 0 为累计行数，1 为每组行数 |
| `hasBias` | 是否读取 BF16 bias；与 bias 地址一致 |

`outputInit` 的字段按顺序为：

```cpp
float* ptrY;
bfloat16_t* ptrSharedInput;
uint64_t nSize, sharedInputOffset, sharedInputLen;
float sharedInputWeight;
```

GM 指针具有 `__gm__` 地址空间。nSize 是列数，sharedInputOffset/sharedInputLen 以行为单位。无 shared input 时 len 为 0，ptrSharedInput 可为 nullptr。

## 存储与执行约束

- WeightNZ 的 K0=32、N0=16。K 必须为 32 的倍数，N 必须为 16 的倍数；每组 packed FP4 权重占 `K*N/2` 字节，各组连续存放。逻辑转置权重 `[N,K]` 存为 `[K/32,N/16,16,32]` 的分形排列，不支持通过扩大组间存储步长补齐不对齐的 K/N。
- ScaleA 为 `[M,ceil(K/64),2]`；ScaleB 为 `[E,N,ceil(K/64),2]`。分组 A/ScaleA/logit/rowIndex 的行偏移按 groupList 推进。
- tile M/N 不超过 256；scheduler 的 coreNum 与实际启动 AIC 核数相同。空专家不生成计算 tile。
- kernel 为每个 tile 选择 ka/kb/k0，并保证 AIC/AIV 使用相同的权重 K 分块；公共 BlockMmad 由 AIC 搬运 ScaleB。
- 输出/权重转换共用部分 UB，必须使用配套组件的缓冲与同步协议。rowIndex 预取使用 242 KiB 起的两个 1 KiB 槽，logit 使用 244 KiB 起的两个 512 B 槽。
- AIV 在当前 tile 的前两段权重处理后预取当前 tile 的路由参数，再消费上一 tile。Fixpipe 写 UB 前，AIC 等待两个 AIV 释放输出区域。
- FP32 累加结果经 Epilogue 乘以 64，再乘 logit。该因子用于恢复权重转换的计算比例，调用方不应再乘一次。
- 非负 rowIndex 必须小于 batch；负 rowIndex 跳过对应行的路由贡献。写回采用 FP32 原子加，不提供确定性模板入口。

## 使用方法

`Kernel kernel; kernel(params);` 必须位于 `KERNEL_TYPE_MIX_AIC_1_2` 入口内。调用方须在每次调用前清零 y；OutputInit 仅在 shared input 区间写入缩放值。最终输出是该初值与所有路由贡献之和。

完整组装示例见 [量化直调示例](../../../../examples/acl/grouped_matmul/grouped_matmul_finalize_routing/README.md)。[实现文件](../../../../include/blaze/gemm/kernel/kernel_wqgmm_mx_mix_finalize_routing.h)。
