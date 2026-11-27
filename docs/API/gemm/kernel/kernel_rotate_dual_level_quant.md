# RotateDualLevelQuant Kernel

## 功能说明

`KernelMmadRotateDualLevelQuant` 对输入沿尾轴分块做旋转矩阵乘法，并融合可选 clamp、一级归一化和二级 MX 量化。该 `GemmUniversal` 特化面向 Ascend 950（DAV_3510），采用一个 AIC 配合两个 AIV 的 Mix 执行方式。AIC 计算旋转结果，AIV 输出 FP4/FP8 量化数据、FP32 一级尺度和 E8M0 二级尺度。

对原始第 `m` 行的第 `b` 个 `K` 元素块，旋转结果为

$$
Z_{m,b}=\operatorname{cast}_{\mathrm{BF16}}(X_{m,b}R_{e(m),b}).
$$

`e(m)` 为 M 轴分组编号，普通模式按单组处理。若启用 clamp，先在 BF16 域按每 32 个元素限制绝对值，阈值为该块最大绝对值乘以 BF16 系数 `alpha`，乘积保留为 BF16。

一级归一化沿每行的原始尾轴分为 512 元素块。每块按有限值的最大绝对值计算一个 FP32 尺度，输入除以这个尺度后转换为 BF16。FP4 的一级分母为 6，采用 FP32 倒数乘法生成尺度。FP8 的一级分母为 448 或 57344，采用 FP32 除法。二级量化对归一化结果按每 32 个元素生成共享 E8M0 尺度，用对应的内部 BF16 乘数缩放，再将 BF16 乘积转换为目标类型。一级分块可包含多个旋转块，末尾不足 512 时按有效列写回。

## 数据布局与约束

`ProblemShape` 的四个字段为 `(M, N, K, rotationsPerGroup)`。`M` 为合并前导维后的原始行数，`N` 为原始尾维，`K` 为旋转矩阵阶数，支持 32、64、128、256。`N` 须为 `K` 的整数倍。调用方准备连续行优先的 ND 数据，并处理空输入。

| 数据 | GM 布局 | 数据类型 |
| --- | --- | --- |
| `x` | `(M, N)` | FP16 或 BF16 |
| `rotation` | 普通模式为 `(K,K)` 或 `(N/K,K,K)`。分组模式为 `(E,K,K)` 或 `(E,N/K,K,K)` | 与 `x` 相同 |
| `groupListGmAddr` | 分组模式为 `(E,)`，普通模式传空指针 | INT64 |
| `y` | 逻辑形状 `(M,N)`。FP4 每行占 `N/2` 字节，FP8 每行占 `N` 字节 | E2M1 FP4、E4M3FN FP8、E5M2 FP8 |
| 一级尺度 | `(M, ceil(N/512))` | FP32 |
| 二级尺度 | `(M, ceil(N/64), 2)` | E8M0，每尺度占 1 字节 |

- 共享矩阵时 `rotationsPerGroup=1`，按列块配置矩阵时取 `N/K`。
- ACLNN 对接采用 `1≤E≤1024` 的分组约束。`groupListType=0` 表示非负、单调不减的累计行数，最后一个值不超过 `M`。`groupListType=1` 表示各组的非负行数，总和不超过 `M`。空组仍占用旋转矩阵的组下标。仅覆盖部分行时，三路输出在未覆盖行中的原始 GM 内容保留。
- FP4 每字节的低 4 位对应前一个逻辑元素。二级尺度每行有 `ceil(N/32)` 个有效编码，按两个一组存放。有效编码数为奇数时，行末填充字节写入编码 0，对应 E8M0 数值 $2^{-127}$，使用时须跳过该填充。
- 一级尺度小于 FP32 最小正规值时写为正零，一级归一化保留其输入中零元素的符号。二级尺度的内部乘数在边界上由算法单独确定，例如 FP8 的 `scaleAlg=0` 在共享指数编码为 0 时将乘数清零。

## 模板参数

| 参数 | 取值 |
| --- | --- |
| `ProblemShape` | `asc::te::shape<int64_t, int64_t, int64_t, int64_t>` |
| `BlockMmad` | 普通 Fixpipe `BlockMmad`，策略为 `MatmulMultiBlockFixpipeOpti<ND_ALIG_1V2_FIXPIPE, OP_TYPE_QUANT, KernelMmadRotateDualLevelQuant, 0>`，输出类型为 `bfloat16_t` |
| `BlockScheduler` | 普通模式用 `BlockSchedulerMatmulBasic<SchedulerShape>`，其 shape 为五维 INT64。分组模式用 `BlockSchedulerGmmNoQuant` |
| `BlockEpilogue` | `BlockEpilogueRotateQuant<bfloat16_t, YType, ScaleType, DefaultFusion<YType, bfloat16_t>, true>`，`ScaleType` 占 1 字节 |

## 数据结构

`GemmUniversal::Params` 包含以下字段。

| 字段 | 设置方式 |
| --- | --- |
| `problemShape` | 传入原始形状和每组物理旋转矩阵数 |
| `mmadParams` | 传入输入和 rotation 的 GM 地址。`oriK`、`nL1`、`nL0` 均为 `K`。`mL1=mL0` 为 Cube 行块，`kL1=K`，`kL0` 根据 L0B 双缓冲容量选取。设置 `splitM=1`，用 `ubPitchGran` 指定输入行跨度的对齐粒度。 |
| `schedulerParams` | 按原始行及 512 列一级块调度，Scheduler 的 N 切块填 512。分组模式设置 `groupType=0`、`groupNum=E`、`initialM=M`，打开单 X、单 Weight、单 Y，使用连续 ND 权重。 |
| `groupListGmAddr` | 分组列表的 GM 地址，仅分组模式使用 |
| `epilogueParams` | `mx` 配置二级量化，`level0` 配置一级归一化和可选 clamp |

### Epilogue 参数

- `mx` 的 `outGmAddr`、`scaleGmAddr` 分别指向量化数据和二级尺度。`problemShape` 与 Kernel 一致，`axis` 选择尾轴，`ubDB` 与 `mmadParams` 一致，`splitM=true`。Clamp 放在 `level0`，`mx.needClamp` 设为 `false`。
- `level0.scaleGmAddr` 指向 FP32 尺度，`scalesPerRow=ceil(N/512)`，`rotK=K`。`tileRows` 取单 AIV 所需行容量。`alpha` 为 BF16 标量，取值在 `(0,1)` 内时将 `needClamp` 设为 `true`。
- `roundMode` 的内部编码为 `4`（rint）、`0`（round）、`1`（floor）。FP4 支持三种模式，FP8 仅支持 rint。
- `scaleAlg=0` 基于最大绝对值指数选取共享尺度，适用于 FP4/FP8。`scaleAlg=1` 适用于 FP8，块最大绝对值乘以目标最大有限值的 FP32 倒数，再向上选择二次幂尺度。`scaleAlg=2` 适用于 FP4，`dstTypeMax` 取 `[6,12]` 中的有限值。分母为 6 或 7 时使用 BF16 尾数规则，其余值使用 FP32 除法。算法 0、1 的 `dstTypeMax` 设为 0。

### UB 布局

`inputPitch` 以 BF16 元素为单位，取 `K` 按 `ubPitchGran` 对齐后的值。`workspaceBase` 是每个 AIV 私有归一化区的 UB 字节偏移。`collectUbOffset` 和 `level0ScaleUbOffset` 则是相对于 `workspaceBase` 的字节偏移。

以下为直调样例采用的规划，其中 `r` 表示每 AIV 的行容量。

| 区域 | 起点 | 容量 |
| --- | --- | --- |
| Fixpipe 接收槽 0 | UB 偏移 0 | `align32(r * inputPitch * 2 + 512)` 字节，含读保护 |
| 私有 MX 区 | `workspaceBase`，紧接接收槽 0 | `NORMALIZED_WORKSPACE_BYTES`（54 KiB） |
| K 宽片段收集区 | `workspaceBase + collectUbOffset` | `r * 512 * 2 + 512` 字节 |
| FP32 一级尺度暂存 | `workspaceBase + level0ScaleUbOffset` | 每行 32 字节 |
| Fixpipe 接收槽 1 | 半个 UB 的起点，仅 `ubDB=2` 时启用 | 与接收槽 0 相同 |

`r` 不超过 `MAX_NORMALIZED_ROWS`（32）。启用双槽时，槽 0 和全部私有区必须放入前半 UB，槽 1 必须放入后半 UB。私有区始终使用同一地址。计算中间值留在 UB，直调 Kernel 无需 GM workspace。

## 公共接口与执行流程

```cpp
__aicore__ inline void operator()(const Params& params);
```

1. 两个 AIV 为每个 Fixpipe 槽发出初始可写通知。Scheduler 给各核分配原始行和 512 列任务，分组模式按组表更新任务范围。
2. AIC 逐次计算 `K` 列旋转块，通过 Fixpipe 将 BF16 结果按行分给两个 AIV。AIV 收集片段后归还对应槽的可写通知，空行的 AIV 也参与握手。
3. AIV 收齐当前一级块后执行 clamp、一级归一化及二级 MX 量化，随后写回三个输出。AIC 在退出前等待各槽的最后一次归还。

## 调用示例

[RotateDualLevelQuant 直调样例](../../../../examples/acl/rotate_dual_level_quant/rotate_dual_level_quant/README.md) 给出完整的组件组装、UB 参数、Kernel 启动和输出校验。

[Kernel 源码](../../../../include/blaze/gemm/kernel/kernel_rotate_dual_level_quant.h) · [Epilogue 源码](../../../../include/blaze/epilogue/block/block_epilogue_rotate_quant.h)
