# Block Epilogue GELU(tanh) MX Quant
> [代码位置](../../../../include/blaze/epilogue/block/block_epilogue_gelu_tanh_mx_quant.h)

## 功能说明

`BlockEpilogueGeluTanhMxQuant` 在 AIV 上读取 UB 中的 FP32 矩阵乘结果，执行
GELU（tanh）和动态 MX 量化，将 FP8/FP4 输出 `y` 与 E8M0 scale 写回 GM。

本组件的计算流程为：

~~~text
L0C(float) → UB(float) → GELU(tanh, fp32) → bf16
           → MX quantization → y(FP8/FP4) + yScale(E8M0)
~~~

GELU 的 tanh 近似公式为：

$$
\operatorname{GELU}(x)=0.5x\left(1+\tanh\left(\sqrt{2/\pi}\left(x+0.044715x^3\right)\right)\right)
$$

实现采用以下指数形式计算：

$$
\operatorname{GELU}(x)=\frac{x}{1+\exp\left(-\sqrt{8/\pi}\left(x+0.044715x^3\right)\right)}
$$

激活计算以 fp32 完成，结果存储为 bf16，后续 MX scale 计算和 FP8/FP4
量化均基于该 bf16 激活结果进行。

## 支持范围

- 输入累加类型：仅支持 `float`。
- 输出类型：`fp8_e4m3fn_t`、`fp8_e5m2_t`、`fp4x2_e2m1_t`、`fp4x2_e1m2_t`。
- 激活函数：仅支持 GELU（tanh），不支持 GELU（erf）。
- yScale 类型：`fp8_e8m0_t`，以 E8M0 shared exponent 表示。
- 执行位置：AIV，配合 AIC:AIV=1:2 的 MIX Kernel。
- 上游 Kernel：`KernelGroupedMmadWithScaleMxActivationQuant`，即 GMMAQ/QGMM
  的 AIC+AIV 融合路径。

## 动态 MX 量化

量化沿 N 轴进行，每 32 个元素计算一个 shared exponent。对一组激活值
`{v_i}`，OCP 量化的概念流程为：

$$
shared\_exp=\left\lfloor\log_2\left(\max_i |v_i|\right)\right\rfloor-e_{max}
$$

$$
scale=2^{shared\_exp},\qquad q_i=\operatorname{cast\_to\_dst\_type}(v_i/scale)
$$

其中 `e_max` 由目标输出类型决定。`scale` 写入 E8M0 yScale，`q_i` 写入
目标 FP8/FP4 输出。

### 量化算法

| `scaleAlg` | 算法 | 支持范围与说明 |
|------|------|------|
| `0` | OCP | 支持 FP8、`FLOAT4_E2M1` 和 `FLOAT4_E1M2` 输出，按 MX shared exponent 规则量化 |
| `1` | cuBLAS | 仅支持 FP8 输出；根据每个量化块的最大绝对值计算 E8M0 scale |
| `2` | FP4 动态 dtype range | 仅支持 `FLOAT4_E2M1`；使用 `dstTypeMax` 调整目标 dtype 范围，必要时回退到通用 cuBLAS scale 计算 |

cuBLAS 类路径先计算量化块的最大绝对值，再按目标 dtype 的最大可表示值
得到浮点 scale，并向上取整到 E8M0 可表示的 scale，保证量化结果不溢出。

### `dstTypeMax`

`dstTypeMax` 主要用于 `scaleAlg=2`：

- `0.0`：使用默认目标范围。
- `[6.0, 12.0]`：使用指定的 FP4 E2M1 目标最大值。
- `0.0`、`6.0` 和 `7.0` 使用动态范围优化路径；其他合法值使用通用
  cuBLAS scale 计算。

## 输出 scale 布局

每 32 个 N 元素参与一次 shared exponent 归约，但 yScale 按每 64 个 N
元素存放两个 E8M0 值。对于一个完整问题，单行 scale 的元素数为：

~~~text
scaleN = ceil(N / 64) * 2
~~~

因此 yScale 的逻辑布局等价于：

~~~text
(M, ceil(N / 64), 2)
~~~

实际存储按行连续排列。FP4 输出为两个元素打包一个字节，因此 FP4 的 y
地址偏移、拷贝长度和行间 stride 均按 2 个元素/字节进行换算；yScale
仍按 E8M0 字节存储。

## 特殊类型别名

| 类型 | 说明 |
|------|------|
| `DataTypeOut` | 输出类型，必须为 FP8 或 FP4 类型 |
| `DataTypeIn` | 输入类型，固定为 `float` |
| `BlockShape` | 当前 AIC/AIV tile 的形状 `(M, N, K, batch)` |
| `ProblemShape` | 当前 group 的问题形状 `(M, N, K, batch)` |
| `OutputOffsets` | 当前 group 的 y 和 yScale 输出偏移 |

## 参数结构

### `OutputOffsets`

~~~cpp
struct OutputOffsets {
    int64_t yOffset;
    int64_t yScaleOffset;
};
~~~

`OutputOffsets` 由 QGMM scheduler 为当前 group 和当前 block 计算。`yOffset`
按输出元素计数，`yScaleOffset` 按 E8M0 scale 元素计数；FP4 的 y 偏移由
epilogue 内部转换为打包后的字节偏移。

### `Params`

~~~cpp
struct Params {
    GM_ADDR yGmAddr;
    GM_ADDR yScaleGmAddr;
    uint32_t baseM;
    uint32_t baseN;
    uint32_t scaleAlg;
    float dstTypeMax;
};
~~~

| 参数 | 说明 |
|------|------|
| `yGmAddr` | 量化输出 y 的 GM 地址 |
| `yScaleGmAddr` | E8M0 yScale 的 GM 地址 |
| `baseM` | 基础 M tile 大小，用于 UB 临时空间规划 |
| `baseN` | 基础 N tile 大小，用于 UB 临时空间规划 |
| `scaleAlg` | MX scale 算法，支持 OCP、cuBLAS 和 FP4 动态范围路径 |
| `dstTypeMax` | FP4 动态范围路径使用的目标 dtype 最大值 |

## 接口

### `Init`

~~~cpp
__aicore__ inline void Init(const Params& params)
~~~

仅 AIV 执行初始化：

1. 根据 `DataTypeOut` 设置目标类型的最大指数和默认最大值。
2. 按最大 tile 规模规划输入、激活、scale、量化输出和临时 UB 空间。
3. 保存 y/yScale 的 GM 基地址。

AIC 侧不分配和初始化 epilogue 的 vector 临时资源。

### `UpdateNextProblem`

~~~cpp
__aicore__ inline void UpdateNextProblem(const ProblemShape& problemShape)
~~~

切换到下一个 group 时更新当前 group 的 N 维度和 yScale 行跨度：

~~~text
n_      = problemShape.N
scaleN_ = ceil(N / 64) * 2
~~~

### `UpdateGlobalAddr`

~~~cpp
__aicore__ inline void UpdateGlobalAddr(const OutputOffsets& baseOffsets)
~~~

根据当前 group 的输出偏移刷新 y/yScale 的 GM 地址。对于 FP4 输出，y 的
元素偏移会在内部右移一位，以匹配两个 FP4 元素共用一个字节的存储格式。

### `operator()`

~~~cpp
__aicore__ inline void operator()(
    const BlockShape& blockShape,
    const OutputOffsets& outputOffsets)
~~~

处理一个 AIC/AIV tile：

1. 两个 AIV 按 M 维均分当前 tile；没有分配到有效行的 AIV 直接返回。
2. 从 UB 地址 0 读取 AIC 写入的 float L0C 结果。
3. 计算 GELU（tanh），并将激活结果转换为 bf16。
4. 按 32 个元素计算 MX shared exponent 和 reciprocal scale。
5. 将激活结果量化为 FP8 或 FP4。
6. 转换 yScale 的行布局，并将 y 与 yScale 写回 GM。

## GMMAQ/QGMM 组合方式

典型组合如下：

~~~cpp
using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

using DispatchPolicy = Blaze::Gemm::GroupedMatmulWithScaleMx<
    0, false, Blaze::Gemm::KernelGroupedMmadWithScaleMxActivationQuant>;

using BlockMmad = Blaze::Gemm::Block::BlockMmad<
    DispatchPolicy, AType, LayoutA, BType, LayoutB,
    float, LayoutC, float, LayoutC>;

using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueGeluTanhMxQuant<
    OutputType, float>;

using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit;

using GmmaqKernel = Blaze::Gemm::Kernel::GemmUniversal<
    ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
~~~

QGMM Kernel 按 group list 逐组更新问题形状和 GM 偏移：

1. scheduler 计算当前 group 的 A/B/Scale/Bias 输入偏移和 y/yScale 输出偏移。
2. AIC 使用 `BlockMmad` 完成 MX Grouped Matmul，并通过 DualDst fixpipe
   将 float L0C 结果写到 UB。
3. AIC 调用 `NotifyVector()`，AIV 调用 `WaitForCube()` 后执行本 epilogue。
4. AIV 完成 y/yScale 写回后调用 `NotifyCube()`，允许 AIC 复用 UB 和继续
   下一个 tile。

对应的上游实现为：

- [Kernel QGMM MX](../../gemm/kernel/kernel_qgmm_mx_basic.md)
- [BlockMmad QGMM MX](../../gemm/block/block_mmad_qgmm_mx.md)
- [QGMM MX ActivationQuant 代码](../../../../include/blaze/gemm/kernel/kernel_qgmm_mx_activation_quant.h)

## 特殊约束

### 类型约束

- `DataTypeIn` 必须为 `float`。
- `DataTypeOut` 必须为 `fp8_e4m3fn_t`、`fp8_e5m2_t`、`fp4x2_e2m1_t`
  或 `fp4x2_e1m2_t`。
- yScale 固定使用 E8M0 表示。

### Tile 与 UB 约束

- 一个 AIC 对应两个 AIV；每个 AIV 处理当前 tile 的一个 M 子块。
- Host tiling 需要满足：

  ~~~text
  ceil(baseM / 2) * baseN <= 128 * 256
  ~~~

- AIC 写入 UB 的 float 行跨度按 `Align32(N)`，M 方向按 2 对齐。
- N 维量化以 32 个元素为基本归约粒度，yScale 以 64 个 N 元素为地址
  粒度。

### Group 与 tail 约束

- `UpdateNextProblem` 和 `UpdateGlobalAddr` 必须在切换 group 时同步更新。
- QGMM ActivationQuant 路径不能把最后一个 group 的 N tail 拆成小于
  64 元素 scale 粒度且跨越 scale group 边界的子块，否则相邻子块可能
  覆盖同一个 yScale group。
- 当前融合路径对最后一个 group 保持不拆分 tail 的调度语义。

### AIC/AIV 同步约束

- AIV 必须在读取 UB 前等待 AIC 完成 L0C→UB。
- AIV 完成 y/yScale 写回后必须通知 AIC，AIC 才能复用 UB。
- Kernel 的 `BlockMmad` 必须使用支持 DualDst 的 ActivationQuant
  dispatch policy。

## 与 QBMMAQ Epilogue 的关系

本组件与 [Block Epilogue GELU MX Quant](./block_epilogue_gelu_mx_quant.md)
共享 AIC+AIV 融合、GELU 后 bf16 中间结果和动态 MX 量化的基本数据流，但接口
和适用 Kernel 不同：

- QBMMAQ epilogue 支持 GELU（tanh）/GELU（erf）、float/bf16 输入，以及
  `geluAlg`、`quantAlg`、`fp4RoundMode` 参数。
- GMMAQ/QGMM epilogue 仅支持 float 输入和 GELU（tanh），使用
  `scaleAlg`、`dstTypeMax` 参数。
- QBMMAQ 处理 Batch Matmul 的 batch 偏移；GMMAQ/QGMM 处理 group list、
  group shape 和每个 group 的 y/yScale 偏移。

因此，两者的文档结构和量化原理可以复用，但不能直接复用对方的参数、调度
和输入类型说明。

## 适用场景

- Ascend950 上的 GMMAQ/QGMM MX Grouped Matmul 融合后处理。
- 需要在单个 MIX Kernel 中完成 Grouped Matmul、GELU（tanh）和 MXFP8/MXFP4
  activation quant 的推理场景。
- 通过 AIC 计算与 AIV 向量后处理流水重叠，降低独立 activation quant
  Kernel 的额外访存和调度开销。
