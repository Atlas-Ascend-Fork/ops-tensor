# kernel_wqgmm_mix_weight_prologue

## 功能说明

[kernel_wqgmm_mix_weight_prologue.h](../../../../include/blaze/gemm/kernel/kernel_wqgmm_mix_weight_prologue.h)
提供 arch35 MX A8W4 Grouped Matmul 的通用 Kernel 编排。AIV 将 packed FP4 Weight
转换并搬入共享 L1，AIC 使用 FP8 E4M3 激活和 E8M0 ScaleA/ScaleB 完成 MX MMAD，
并将结果写为 FP16 或 BF16。

Kernel 层拥有 group/block 调度、AIV Prologue、AIC MMAD 调用和同步生命周期；算子侧负责
选择具体数据类型、Layout、BlockMmad、BlockScheduler、BlockPrologue，并把 host tiling
映射到各级 Params。ops-tensor 不提供算子专用 tiling DTO 或一键运行适配函数。

该 Kernel 支持：

- Weight 为 packed FP4 E2M1 或 E1M2；
- 输出和可选 Bias 为 FP16 或 BF16；
- groupListType 为 0（累计 Offset）、1（当前组 Count）或 2（Sparse）；
- 连续多 expert Weight 或每 expert 一个 Tensor 的 TensorList；
- N 方向 main、first-tail、second-tail 三段调度。

## MX Pipeline 策略约束

`Blaze::Gemm::Kernel::MxPipeline::GmmWeightQuantMxKernel` 的 `PipelineStages_` 必须与
`BlockMmad::DispatchPolicy`、`BlockPrologue::DispatchPolicy` 对应同一个
`MxWeightPipelineConfig`，否则编译期报错，避免生产者和消费者采用不同的缓冲布局与同步协议。

`MxPipelineMode` 与模式派发语义定义在 `blaze/gemm/policy/dispatch_policy.h` 中的
`MxWeightPipelinePolicy`；物理缓冲布局和搬运配置仍由 Block 层的
`MxWeightPipelineConfig` 管理。原有 Config 名称、模式编号和默认路径保持不变。

调用方通过 `Params` 和 `operator()` 进入 Kernel；不再提供未使用的 `RunSssUnfolded` 入口。

## 公共接口

~~~cpp
#include "blaze/gemm/kernel/kernel_wqgmm_mix_weight_prologue.h"

using KernelImpl = Blaze::Gemm::Kernel::GmmWeightQuantMxKernel<
    ProblemShape, BlockMmad, BlockScheduler, void, BlockPrologue,
    IsSingleMultiSingle>;

KernelImpl kernel;
kernel(params);
~~~

模板参数：

| 参数 | 说明 |
| :--- | :--- |
| ProblemShape | 四维 shape，依次保存占位 M、K、N、groupNum |
| BlockMmad | `GroupedMatmulWithWeightQuantMx` 策略下的共享 MX Weight-Prologue Block MMAD 特化 |
| BlockScheduler | Kernel 层 N-resplit 调度器 |
| BlockEpilogue | 当前路径保留为 void |
| BlockPrologue | Kernel 层 FP4→FP8、ScaleB、Bias 前处理 |
| IsSingleMultiSingle | false 表示连续 Weight；true 表示 TensorList |

Kernel Params 包含 ProblemShape、BlockMmad/BlockScheduler/BlockPrologue Params、groupList 地址、
groupListType、hasBias 和 bias 地址。具体 Params 的值必须由算子仓根据自己的 tiling contract 组装。

注意：GroupedMatmul 算子侧 singleWeight 与模板布尔值语义相反。singleWeight=true 使用连续存储并
实例化 IsSingleMultiSingle=false；singleWeight=false 使用 TensorList 并实例化
IsSingleMultiSingle=true。

## 输入、输出与布局

| 参数 | 类型与布局 | 逻辑 shape | 地址语义 |
| :--- | :--- | :--- | :--- |
| x | fp8_e4m3fn_t，ND | [totalM, K] | 所有 expert 沿 M 连续拼接 |
| weight | packed FP4，ZN | 每 expert [K, N] | 连续数据首地址或 TensorList 描述符 |
| xScale | fp8_e8m0_t，ScaleA ND | [totalM, ceil(K/64)*2] | 与 x 的 M 顺序一致 |
| weightScale | fp8_e8m0_t，ScaleB DN | 每 expert [ceil(K/64)*2, N] | 连续数据首地址或 TensorList 描述符 |
| bias | FP16/BF16，ND | 每 expert [N] | hasBias=0 时可为空；否则与 Weight 使用相同存储模式 |
| groupList | int64_t，ND | grouplisttype 0/1：[groupNum]；grouplisttype 2：[groupNum, 2] | 原始数组地址，不是 TensorList |
| y | FP16/BF16，ND | [totalM, N] | 所有 expert 沿 M 连续拼接 |

每个字节承载两个 FP4 元素。连续存储模式下，相邻 expert 的 Weight 字节步长为
K*N/2，ScaleB 元素步长为 N*ceil(K/64)*2，Bias 元素步长为 N。

TensorList 描述符须覆盖 groupNum 项。Weight、ScaleB 和非空 Bias 使用相同的 expert 顺序。

## groupList 与调度约束

groupListType=0 时，第 i 项为前 i+1 个 expert 的累计 M；序列须非递减，最后一项等于
totalM。groupListType=1 时，第 i 项为该 expert 的 M count；所有值之和等于 totalM。
两种模式都允许值为 0 的空 expert。

当 groupListType 为 2 时，groupList 的 shape 为 `[E, 2]`，E 为分组数（groupNum），
数据排布为 `[[groupIdx0, groupSize0], [groupIdx1, groupSize1], ...]`。
其中，groupIdx 表示分组索引，groupSize 表示 M 轴上每组的大小，groupList 中所有数值均须非负。
groupSize 非零的分组按 groupIdx 升序排列在前，groupSize 为零的分组按 groupIdx 升序排列在后。
例如，`[[1, 7], [3, 10], [0, 0], [2, 0]]`。

N 三段 block 必须无遗漏覆盖 N。Kernel 对每个 expert 按 baseM 继续切 M，并在逻辑 Cube core
间轮转起始 block，保持跨 expert 的负载均衡。AIC 和 AIV 使用同一调度器实例语义。

## 调用约束

- global kernel 必须使用 KERNEL_TYPE_MIX_AIC_1_2；
- 当前路径面向 Ascend 950（DAV_3510）；
- 所有 expert 共用 K、N、WeightType 和输出类型；
- hasBias 必须与 bias 地址是否有效一致；
- packed FP4 和 FP8 UB 布局必须按 Align16(N) 建立物理尾块，不能按 N/8 独立分裂；
- BlockMmad 和 BlockPrologue 必须使用同一编译期 grouped schedule policy 和 ready/free 协议。

## BlockMmad 参数和 K 流水

BlockMmad 的接口见 [MX Weight-Prologue 文档](../block/block_mmad_weight_prologue_mx.md)。
算子或 example 通过以下字段指定 A、ScaleA、ScaleB、Bias 和 C 的地址：

```cpp
typename BlockMmad::Params mmParams{};
mmParams.aGmAddr = xAddr;
mmParams.scaleAGmAddr = perTokenScale;
mmParams.scaleBGmAddr = scaleBAddr;
mmParams.biasGmAddr = biasAddr;
mmParams.cGmAddr = yAddr;
```

Kernel 负责 BlockMmad 的完整构造参数，包括 GM 地址、L1 分区、缓冲容量及同步配置。
AIC 计算实例通过带参构造函数初始化；每个 tile 的 ka/kb/k0 由 Kernel 根据实际 M/N 和缓冲容量确定。
AIC 使用 ka/kb/k0 执行计算，AIV prologue 使用相同分块规则确定的 kb 转换 Weight 和 ScaleB 并写入 L1。
AIC 在 ka 窗口内跨 kb 块复用 A，每个 kb 块完成后释放 B 槽位，
并在 ka 窗口结束后切换 A 槽位。
AIC 与 AIV 使用同一份 L1 地址配置，动态 K 的物理容量约束取自该配置中的 A 缓冲容量。
地址配置的生命周期覆盖 AIV prologue 的执行过程；ready/free 标志遵循配套协作协议。

### 动态 K 分块选择

Kernel 私有方法 `CalcDynamicKBlock` 根据当前 tile 的 M/N 和缓冲容量选择 ka/kb/k0，
BlockMmad 根据传入的分块执行计算。以下为 Kernel 内部调用示意：

```cpp
uint64_t kaL1Size = 0;
uint64_t kbL1Size = 0;
uint64_t l0KSize = 0;
CalcDynamicKBlock(mL1Size, nL1Size, kaL1Size, kbL1Size, l0KSize);
blockMmad(tensorBlockAGm, tensorBlockScaleAGm, tensorScaleBGm, tensorBlockCGm,
          kaL1Size, kbL1Size, l0KSize);
```

GMM 以当前 scheduler tile 的实际 M/N 选择 K 分块：M≤160 且 N≤128 时 kb=512，否则 kb=256；
M≤128 且 N≤128 时 k0=256，否则 k0=128。ka 是 kb 的整数倍，按 N 与 Align16(M) 的比例选择复用深度，
同时按 `sizeof(AType)` 将模型容量与实际 A buffer 容量换算为元素数，取两者允许的最小复用深度；
模型容量为 80 KiB，实际容量为 128 KiB 减去可选 bias 的 4 KiB。
A 的槽位独立于 B 槽位，在同一 ka 窗口内跨多个 kb 块复用。
ScaleA 使用独立的 Scale 窗口，窗口长度为 4096 个 K 元素，不随 ka 窗口切换。
AIV 使用同一辅助函数计算 kb，确保 producer 和 consumer 的 K 序列一致。

复用深度按如下公式计算，其中 M/N 是当前非空 tile 的实际尺寸，配套 scheduler 保证 M/N≤256：

```text
alignedM = Align16(M)
depth = max(1, CeilDiv(N, 2 * alignedM))
modelMaxDepth = (80 * 1024 / sizeof(AType)) / (alignedM * kb)
actualMaxDepth = ((128 * 1024 - (hasBias ? 4 * 1024 : 0)) / sizeof(AType)) / (alignedM * kb)
maxDepth = max(1, min(modelMaxDepth, actualMaxDepth))  // 整数除法
ka = min(depth, maxDepth) * kb
```

例如 M=16、N=256 时，kb=256、k0=128、ka=2048；同一 A 窗口可供 8 个 B 块使用。
每个 B 块完成后释放自己的槽位，A 槽位在 ka 窗口结束后切换。Scale 窗口独立推进，GMM 的窗口长度为 4096。

## 仓内验证

[grouped_matmul_mx_a8w4 example](../../../../examples/acl/grouped_matmul/grouped_matmul_mx_a8w4/README.md)
使用 example-local 适配代码直接实例化通用 Kernel，保留确定性数据生成、NPU 执行和原精度阈值。
Transformer 的生产适配和 host tiling 映射位于 ops-transformer 的
`gmm_weight_quant_tensor_api_mx_kernel.h`。


## V3 A8W4 SwiGLU 后处理分支

### 融合计算与入口

`GmmWeightQuantMxKernel` 在同一个模板类内，根据是否配置Epilogue，通过 `if constexpr` 选择 `RunGm` 或 `RunEpilogue`，不再单独定义后处理类特化。非空Epilogue分支将MX A8W4 Grouped Matmul、mode 2 SwiGLU、可选expert smooth和MXFP8量化组装为AIC:AIV=1:2的TensorAPI Kernel。中间矩阵结果由Fixpipe从L0C写入UB，不落GM。

`Params` 按编译期配置选择原GM参数或UB后处理参数，调用方仍通过 `Kernel::Params` 和 `operator()` 传入参数；分支专用成员也按编译期配置选择。

原GM入口按参数准备、Cube/Vector分组和tile执行拆分，专家输入地址共用 `GetGmExpertAddress` 处理TensorList与连续存储两种形式。Prologue构造和阶段恢复共用 `ResumeVectorState` 初始化事件，事件顺序不变。

```cpp
using Prologue = Blaze::Gemm::Kernel::GroupedWeightPrologueMx<
    fp8_e4m3fn_t, fp4x2_e2m1_t, bfloat16_t, true>;
using Epilogue = Blaze::Epilogue::Block::BlockEpilogueSwigluMxQuant<
    fp8_e4m3fn_t, float, fp8_e8m0_t, false, true, true>;
using Kernel = Blaze::Gemm::Kernel::GmmWeightQuantMxKernel<asc::te::shape<int64_t, int64_t, int64_t, int64_t>, BlockMmad, Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit, Epilogue, Prologue, true>;
```

`BlockMmad`采用专用的`GroupedMatmulSwigluWithWeightQuantMx`策略，提供`ComputeContinuousToL0c`和`CopyContinuousResultToUb`。Prologue的第四个模板参数`V3A8w4Fast`开启V3 A8W4路径，默认构造函数复用同一套L1参数和事件初始化。旧GMM调用仍使用三个模板参数，并通过MMAD参数构造Prologue。Epilogue的最后一个模板参数继续显式开启V3 A8W4路径。

ScaleA/ScaleB的L1→L0搬运和MMAD参数更新复用`gemm/tile/compute.h`中的公共实现，纯GMM/QBMM与融合块共用这部分代码。融合分支保留成对权重同步、持续L0C结果和延迟Fixpipe输出；原GM输出入口继续保留在同一个Block类中，由GM策略使用。

当前进一步共用A/B描述符、矩阵切片搬运、scale读取和MMAD发射。L1物理分区统一引用
`MxWeightQuantGmmL1Layout`。`ReadMxGroupRows` 为原GMM与融合路径共用的groupList解析，
保留累计偏移、逐专家计数及原GMM的稀疏分组语义；融合入口仍只接受type 0/1。
两处Prologue到Epilogue的阶段交接统一由 `ConsumePrevious` 执行。
`RunContinuousPairRange` 只接收实际使用的weight和分块参数，不再构造或传递未使用的scale/bias描述符。

复用`BlockSchedulerGmmSwatWithTailSplit`分配任务，并按expert的实际M进行BalanceM。基本分块上限为M=256、半宽N=128，半宽N尾块为64。

### 支持范围

- X为FP8 E4M3 ND，Weight为FP4 E2M1 FRACTAL_NZ；X不转置，Weight仅支持转置ZN视图，ScaleB为ScaleB-DN。
- FP4源存储C0=64，转换后按FP8计算布局使用C0=32。所有expert权重在一个连续缓冲中。
- K和完整N按128对齐；M支持非对齐及零token expert。
- `groupListType=0/1`分别表示累计偏移和逐expert计数。
- mode 2，`scaleAlg=0/1`；无GMM bias，`gluBias`为激活公式的标量偏置。
- 输出为FP8 E4M3，输出scale为E8M0；可选smooth为FP32 `[E,N/2]`。

### 调用顺序

调用方填充`Kernel::Params`的输入地址、M/N/K、expert数量、groupList类型、scheduler及epilogue参数，然后调用`kernel(params)`。

1. AIC创建MMAD对象，AIV创建Prologue和Epilogue对象；这些对象在整次Kernel调用期间复用。
2. Scheduler依次分配expert和tile，按当前expert的M调整M分块。
3. `RunContinuousPairRange`分别读取激活分支与gate分支权重，一个VF调用内完成两段FP4→FP8转换，再写入配套L1区域；不再保留旧concat拼接搬运的分支和状态。
4. AIC直接将weightScale搬入L1，复用A/xScale的K遍历，执行MMAD，结果保留在L0C。
5. AIC等待输入UB可复用，随后Fixpipe输出到AIV的UB；AIV处理SwiGLU、smooth及MX量化。

前一tile的输出处理与下一tile的权重阶段按显式握手交接，不能删除`FinalizeVectorCompute → epilogue → ResumeVectorState`顺序。

### 数值顺序

```text
left  = converted_left_result  * inputScale
right = converted_right_result * inputScale
a     = min(left, clampLimit)
out   = a / (1 + exp(-gluAlpha * a))
out   = out * (clip(right, -clampLimit, clampLimit) + gluBias)
out   = out * smoothScale[expert, channel]  // smooth为空时略过
v     = BF16_RNE(out)
y, yScale = MxQuant(v, scaleAlg)
```

Kernel把局部epilogue参数的`inputScale`设为64，恢复FP4转换的数值比例。smooth乘法在FP32激活之后、BF16舍入之前。调用方的原始参数不会被改写。

### 缓冲与同步

新路径权重输入区为64 KiB、转换输出区为128 KiB；Epilogue的FP32输入从64 KiB处开始，与权重暂存区按阶段复用。量化输出和打包scale使用独立双缓冲；smooth行位于246 KiB处，最多128个FP32元素。总地址范围不超过248 KiB UB。

- K循环内使用L1双缓冲和AIC/AIV生产消费通知。
- 结果就绪使用mode 4的2/18；输入UB可复用通知使用3/19。
- Epilogue新路径使用独立本地事件5/6，保护两组输出槽位。
- 模式和第二颗AIV的flag偏移复用公共sync常量；保留原pipe和等待顺序。

同一expert、同一通道tile的smooth行会被缓存；切换expert或通道后重新搬入。关闭smooth时不搬运该输入。

### 兼容性边界

主线GMM/QBMM的公共MMAD和默认Epilogue路径不因此切换到新快速实现。参数合法性由上层ACLNN/Tiling校验；Kernel为设备侧`void`模板，不返回ACLNN错误码。
