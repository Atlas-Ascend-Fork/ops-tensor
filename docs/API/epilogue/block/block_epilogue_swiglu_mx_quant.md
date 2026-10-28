# Block Epilogue SwiGLU MX Quant

> [代码位置](../../../../include/blaze/epilogue/block/block_epilogue_swiglu_mx_quant.h)

## 功能说明

`BlockEpilogueSwigluMxQuant` 在 AIV 上读取 AIC 通过 DualDst Fixpipe 写入 UB 的两路 FP32 矩阵乘
结果，执行 SwiGLU、转换为 bf16、按 N 轴每 32 个元素生成一个 E8M0 scale，并
输出 MXFP8 `y` 与 `yScale`。一个 AIC 对应两个 AIV，两个 AIV 沿 M 轴分工。

输入 UB 的逻辑布局为 `[M, 2*N]`，前 `N` 列为 gate，后 `N` 列为 linear；输出布局为 `[M, N]`。

## 模式边界

支持以下模式：

- `swigluMode=0`：
  `left / (1 + exp(-left)) * right`。
- `swigluMode=2`：
  `min(left, clampLimit) / (1 + exp(-gluAlpha * min(left, clampLimit)))`
  再乘 `clamp(right, -clampLimit, clampLimit) + gluBias`。

V3 接口仅支持 mode 2；mode 0 仅供 V2 内部兼容使用。

## 模板参数

`DataTypeOut_` 为量化输出类型（`fp8_e4m3fn_t` 或 `fp8_e5m2_t`）；
`DataTypeIn_` 为矩阵乘中间结果类型，当前为 `float`；
`DataTypeScale_` 为输出 scale 类型，当前为 `fp8_e8m0_t`。

## 参数结构

```cpp
struct Params {
    GM_ADDR yGmAddr{nullptr};
    GM_ADDR yScaleGmAddr{nullptr};
    uint32_t baseM{0};
    uint32_t baseN{0};
    int64_t swigluMode{0};
    float clampLimit{7.0F};
    float gluAlpha{1.702F};
    float gluBias{1.0F};
    uint32_t scaleAlg{0};
    float dstTypeMax{0.0F};
};
```

| 参数 | 说明 |
| --- | --- |
| `yGmAddr` | MXFP8 输出 GM 地址，必须非空 |
| `yScaleGmAddr` | E8M0 输出 scale GM 地址，必须非空 |
| `baseM/baseN` | 与 Kernel tiling 保持一致的兼容字段 |
| `swigluMode` | 共享组件内部支持 V2 mode 0 与 V3 mode 2；V3 必须为 2 |
| `clampLimit` | mode 2 裁剪上限；V3 要求有限且大于 0 |
| `gluAlpha` | mode 2 sigmoid 输入缩放系数；V3 要求为有限 float |
| `gluBias` | mode 2 gate 偏置；V3 要求为有限 float |
| `scaleAlg` | V3 公开 0（OCP）或 1（cuBLAS） |
| `dstTypeMax` | V3 MXFP8 固定为 0；当前 V3 计算不使用其他值 |

`OutputOffsets` 的 `yOffset` 和 `yScaleOffset` 分别是对应输出中的元素偏移，
用于定位当前 group 和 tile 的写回位置。

本 Block 复用 [Tile::MxQuant](../tile/arch35/mx_quant.md) 的 `GroupMaxExp`、
`GenScale`、`Quantize` 和 `TransScaleLayout`，只选择 OCP 或 cuBLAS 算法，
没有动态 dtype range 分支。调用方必须校验 `scaleAlg=0/1` 和 `dstTypeMax=0`，
不能把共享 Tile 的 `DYN_DTYPE_RANGE` 能力作为 V3 WeightNZ 的公开能力。

OCP 使用 `zeroScaleOnZeroExp=false`。

## Shape 与输出布局

设当前 group 的矩阵乘完整宽度为 `NFull=2*NOut`：

| 数据 | shape / 行距 |
| --- | --- |
| L0C→UB 输入 | 逻辑 `[blockM, 2*blockN]`，两个 N 半区连续 |
| y | `[M, NOut]`，当前 tile 仅写有效 `blockM*blockN` |
| yScale | `[M, ceil(NOut/64), 2]` |
| 激活中间值 | bf16，单行按 `Align64(blockN)` 存放 |
| 每行有效 scale | `ceil(blockN/32)`，按 32B 行块写回 |

Split-M Fixpipe 要求复制行数为偶数。组件会为奇数 M tile 创建偶数行的 UB
视图，AIV 仍按原始逻辑行数计算并忽略 padding 行。

## 公共接口

### `GetConcatL0c2UbTensor`

```cpp
auto GetConcatL0c2UbTensor(int64_t rows, int64_t cols);
```

返回供 AIC 写入的 `[ceil_align(rows,2), 2*cols]` float UB Tensor。

### `Init`

```cpp
void Init(const Params& params);
```

AIV 保存输出地址和量化参数并规划 UB；AIC 直接返回。

### `UpdateNextProblem`

```cpp
void UpdateNextProblem(const ProblemShape& problemShape);
```

更新当前输出 `NOut` 和每行 `yScale` stride。

### `UpdateGlobalAddr`

```cpp
void UpdateGlobalAddr(const OutputOffsets& baseOffsets);
```

按当前 group 的基础偏移更新 `y/yScale` GM 地址。

### `operator()`

```cpp
void operator()(const BlockShape& blockShape, const OutputOffsets& outputOffsets);
```

执行当前 tile 的 SwiGLU、MX 量化、scale 布局转换和 GM 写回。

`operator()` 返回后，由 Kernel 执行 `NotifyCube()`，零行 AIV 也遵循此流程。
输出与 scale 的搬出由本组件的流水事件保护。详见 [QBMM Kernel 同步约定](../../gemm/kernel/kernel_qbmm_mx_activation_quant.md#输入-ub-复用通知的所有权)。

## 资源与校验边界

每个输入半区最多预留 `64*256` 个 float 元素；tiling 必须保证每个 AIV
负责的 `M*Align64(N)` 不超过该容量。Blaze Epilogue 是设备侧 `void`
组件，不返回 ACLNN 状态码。必选输出地址及所有属性的合法性由上层
ACLNN/tiling 校验；必选空指针由上层按接口规范返回 `161001`。
