# GroupedMatmulFinalizeRouting 组件使用指南

## 适用范围

本组组件面向 Ascend950、DAV_3510（arch35），在分组矩阵乘之后执行 logit 加权和路由累加。组件属于 device 侧 C++ 接口，不解析 aclTensor，不执行算子 Host 的类型、形状和 workspace 校验。调用方负责提供合法的模板组合、存储格式和 tiling。

| 数据流 | 输入与权重 | 计算组件 | Kernel 入口 |
| --- | --- | --- | --- |
| pertoken 全量化 | INT8、FP8 E4M3FN、HIFLOAT8 的同类型输入；A 为 ND，B 可按组件能力配置 ND/DN/NZ/ZN | `BlockMmad<MatmulWithScaleMix<...>>` | `GemmUniversal<ProblemShape, Mmad, Epilogue, Scheduler>` |
| MX 全量化 | 同为 MXFP8 或同为 MXFP4；A 为 ND，B 可配置 ND/DN/NZ/ZN | `BlockMmad<GroupedMatmulWithScaleMx<...>>` | `GemmUniversal<ProblemShape, Mmad, tuple<Prologue,Epilogue>, Scheduler>` |
| MX A8W4 | FP8 E4M3FN × packed FP4 E2M1；A 为 ND，B 为转置 WeightNZ | `BlockMmad<GroupedMatmulFinalizeRoutingWithWeightQuantMx,...>` | `GmmWqgmmMxFinalizeRoutingKernel` |

以上 pertoken 列列出算子集成使用的组合；组件模板还允许 E5M2 等类型，其模板准入不等同于 aclnn 接口支持范围。MX WeightNZ 的数据类型限制见 [MX Kernel](./kernel_qgmm_mx_mix_finalize_routing.md)。

## 张量语义

逻辑输入 A 为 `[M,K]`，分组数为 E，每组逻辑权重为 `[K,N]`，输出为 `[batch,N]`。groupList、logit 和 rowIndex 分别描述专家分组、输入行权重和输出行映射。

- `groupListType=0`：INT64 累计行数列表，非递减，末项为 M；相邻值相等表示空组。
- `groupListType=1`：INT64 行数列表，每项非负，总和为 M。
- logit 含 M 个元素；rowIndex 含 M 个元素。除 A8W4 的负索引跳过规则外，rowIndex 必须落在 `[0,batch)`。
- shared input 为 BF16，逻辑形状为 `[sharedInputLen,N]`。其第 0 行写入 y 的 `sharedInputOffset` 行，先乘 `residualScale`，随后与路由贡献累加。
- `sharedInputOffset`、`sharedInputLen`、batch 的单位均为行；`0 <= offset` 且 `offset+len <= batch`。len 为 0 时不读取 shared input。
- pertoken 的 ScaleB 为每组每列一个元素，逻辑 `[E,N]`；可选 ScaleA 为每输入行一个 FP32 元素 `[M]`。ScaleB 为 FP32 或 BF16，bias 为可选 BF16 `[E,N]`。
- MX ScaleA/ScaleB 为 E8M0，每 32 个 K 元素共享一个 scale；尾 K 的 scale 存储须满足 K64 对齐。ScaleA 为 `[M,ceil(K/64),2]`；非转置 ScaleB 为 `[E,ceil(K/64),N,2]`，转置 ScaleB 为 `[E,N,ceil(K/64),2]`。

非转置 B 的逻辑存储为 `[E,K,N]`；转置 B 为 `[E,N,K]`。NZ/ZN 的 padding 和组跨度由权重 dtype、布局共同确定，不能用逻辑元素数代替存储字节数。FP4 两个元素占一个字节。

## 调用次序

1. 选择 Kernel、Mmad、Epilogue 和 Scheduler 的模板组合。
2. 填入当前设备的核数、分组参数、合法的 M/N/K 分块及 GM 地址；FP4 地址偏移以 packed 存储换算。
3. 分配并清零输出，分配可选 workspace，构造 Kernel 的 `Params`。
4. 通过 `KERNEL_TYPE_MIX_AIC_1_2` 启动，一个 AIC 配两个 AIV，调用 Kernel 的 `operator()`。
5. 等待 stream 完成后读取输出。包含原子累加的路径不承诺不同调度之间逐位一致。

三个数据流均要求调用方在每次启动前清零输出 y。Kernel 的 shared input 处理仅覆盖对应窗口，随后 epilogue 将路由贡献累加到 y。单独调用 epilogue 时，也须由调用方准备正确的输出初值。

## Tiling 与资源

baseM/baseN/baseK、kAL1/kBL1、MX scaleKL1 均按原始维度元素数计量；scaleKL1 不是 E8M0 元素个数。对齐、L0/L1/UB 容量和事件编号必须与选定组件一致。MX scale 窗口应覆盖完整 K64 块，并与操作数 K 分块配套。

MX 和 pertoken epilogue 的缓冲按最大 M=128、N=256 规划，调用方不能超出对应布局容量。A8W4 使用专用的 M≤256、N≤256 tile 和固定 UB 分区，其 kernel 配套构造 L1 参数。相同核上的组件不得在缓冲生命周期内占用同一 UB 地址或相同事件编号。

确定性 pertoken 调用须为 Kernel 和 Epilogue 同时配置 workspace、deterWorkspaceSize 和实际 AIC 核数，详见 [pertoken Kernel](./kernel_qgmm_mix_pertoken_finalize_routing.md)。

## 使用示例与测试

- [GMMFR 直调示例](../../../../examples/acl/grouped_matmul/grouped_matmul_finalize_routing/README.md)：统一目标覆盖 pertoken、MX WeightND、MX WeightNZ 和 A8W4，包含 FP32/BF16 输出。
- [Kernel UT](../../../../tests/ut/op_kernel/grouped_matmul_finalize_routing/README.md)：组件组装和 CPU 仿真执行检查。
- 三种量化模式共用 [BlockEpilogueFinalizeRouting](../../epilogue/block/block_epilogue_finalize_routing.md)。Pertoken 和 A8W4 使用 [ProcessSharedInput](../../epilogue/tile/arch35/process_shared_input.md)。
