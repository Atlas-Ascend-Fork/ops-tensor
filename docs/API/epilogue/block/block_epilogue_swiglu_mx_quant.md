# Block Epilogue SwiGLU MX Quant

> [代码位置](../../../../include/blaze/epilogue/block/block_epilogue_swiglu_mx_quant.h)

## 功能说明

`BlockEpilogueSwigluMxQuant` 在 AIV 上读取 AIC 通过 DualDst Fixpipe 写入 UB 的两路 FP32 矩阵乘
结果，执行 SwiGLU、转换为 bf16、按 N 轴每 32 个元素生成一个 E8M0 scale，并
输出 MXFP8 `y` 与 `yScale`。一个 AIC 对应两个 AIV，两个 AIV 沿 M 轴分工。

输入 UB 的逻辑布局为 `[M, 2*N]`，前 `N` 列为激活分支 left，后 `N` 列为右分支 right；
输出布局为 `[M, N]`。可选的 expert smooth 在 FP32 SwiGLU 后、BF16RNE 前相乘。

## 模式边界

支持以下模式：

- `swigluMode=0`：
  `left / (1 + exp(-left)) * right`。
- `swigluMode=2`：
  `min(left, clampLimit) / (1 + exp(-gluAlpha * min(left, clampLimit)))`
  再乘 `clamp(right, -clampLimit, clampLimit) + gluBias`。

V3 接口仅支持 mode 2；mode 0 仅供 V2 内部兼容使用。

## 模板参数

```cpp
template <typename DataTypeOut_, typename DataTypeIn_ = float,
          typename DataTypeScale_ = fp8_e8m0_t,
          bool DirectBf16Output_ = false, bool EnableSmoothScale_ = false,
          bool A8W4V3Fast_ = false>
class BlockEpilogueSwigluMxQuant;
```

`DataTypeOut_` 为量化输出类型（`fp8_e4m3fn_t` 或 `fp8_e5m2_t`）；
`DataTypeIn_` 为矩阵乘中间结果类型，当前为 `float`；
`DataTypeScale_` 为输出 scale 类型，当前为 `fp8_e8m0_t`。

第四参数 `DirectBf16Output_` 保留默认 false；为 true 时要求 BF16 输出，跳过 MX
量化和 yScale 写回。第五参数 `EnableSmoothScale_` 默认为 false，旧实例行为不变。
为 true 时启用激活前的 inputScale 恢复及可选 expert smooth；不能与
`DirectBf16Output_=true` 组合。MX A8W4 融合路径使用
`<fp8_e4m3fn_t,float,fp8_e8m0_t,false,true,true>`。
第六参数`A8W4V3Fast_`仅由新V3 A8W4路径显式开启，启用独立UB布局、输出双缓冲、smooth缓存及合并量化步骤；默认false的全量化调用保持原路径。

## 参数结构

```cpp
struct Params {
    GM_ADDR yGmAddr{nullptr};
    GM_ADDR yScaleGmAddr{nullptr};
    uint32_t baseM{0};
    uint32_t baseN{0};
    uint32_t scaleAlg{0};
    uint32_t swigluMode{0};
    float clampLimit{7.0F};
    float gluAlpha{1.702F};
    float gluBias{1.0F};
    float dstTypeMax{0.0F};
    GM_ADDR smoothScaleGmAddr{nullptr};
    float inputScale{1.0F};
    Params() = default;
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
| `smoothScaleGmAddr` | 尾部新增，可选 FP32 `[E,NOut]` 地址；启用 smooth 模板后读取，空地址等价全 1 |
| `inputScale` | 尾部新增，默认 1；启用 smooth 模板时在激活前乘左右两分支，MX A8W4 新 Kernel 内部自动设置局部值为 64 |

`scaleAlg` 仍为第五字段，原有字段顺序和默认值不变；两个新字段追加在 Params 尾部。
`EnableSmoothScale_=false` 时忽略 smooth 地址和 inputScale。
`GmmWeightQuantMxKernel::RunEpilogue` 在AIV初始化时复制 epilogue Params，将局部
`inputScale` 设为 64 后调用 `Init`，随后跨tile复用对象，wrapper 无需手填补偿；调用方原始 Params
和共享 epilogue 的默认值 1 保持不变。
启用时即使 smooth 地址为空，仍执行 inputScale 恢复。恢复后再计算
`a=min(left,clampLimit)`，分子和 `exp(-gluAlpha*a)` 都使用裁剪后的 a。
FP32 SwiGLU 结果乘 expert/channel smooth 后才转为 BF16RNE，不可先舍入再乘 smooth。

`OutputOffsets` 的 `yOffset` 和 `yScaleOffset` 分别是对应输出中的元素偏移，
用于定位当前 group 和 tile 的写回位置。

本 Block 复用 [Tile::MxQuant](../tile/arch35/mx_quant.md) 的 `GroupMaxExp`、
`GenScale`、`Quantize` 和 `TransScaleLayout`，只选择 OCP 或 cuBLAS 算法，
没有动态 dtype range 分支。调用方必须校验 `scaleAlg=0/1` 和 `dstTypeMax=0`，
不能把共享 Tile 的 `DYN_DTYPE_RANGE` 能力作为 V3 WeightNZ 的公开能力。

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

### `GetAccumulatorTensor`

```cpp
auto GetAccumulatorTensor(const AccumulatorTensorParams& params) const;
```

返回供 AIC 写入的 `[ceil_align(params.rows,2), 2*params.cols]` float UB Tensor。默认路径从UB偏移0开始，A8W4快速路径从64 KiB处开始。

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

### `SetSmoothScaleOffset`

```cpp
void SetSmoothScaleOffset(int64_t offset);
```

选择 smooth 的 expert 行和通道 tile，offset 按 FP32 元素计；初值为 0。
对 `[E,NOut]` smooth，调用方在执行 tile 前传入
`expert*NOut+tileNOffset`，不加 token 行偏移。`UpdateGlobalAddr` 只改变输出基址，
不会自动更新 smooth 偏移。默认smooth路径在 `operator()` 中搬入当前有效通道的一行 FP32 smooth，
等待 MTE2→V 后在每个有效 M 行广播复用；空 smooth 地址不触发搬运。
快速路径还提供`PrefetchSmoothScale(columns)`：缓存记录已加载的列数。同一expert和通道偏移下，请求宽度不超过缓存范围时直接复用；请求宽度增大时重新搬入，避免64列缓存被误用于128列计算。偏移改变时清空缓存范围。

### `operator()`

```cpp
void operator()(const BlockShape& blockShape, const OutputOffsets& outputOffsets);
```

执行当前 tile 的 SwiGLU、MX 量化、scale 布局转换和 GM 写回。

`operator()` 返回后，由所属 Kernel 完成输入 UB 复用通知，零行 AIV 也遵循此流程。
既有 QBMM 路径执行 `NotifyCube()`，详见 [QBMM Kernel 同步约定](../../gemm/kernel/kernel_qbmm_mx_activation_quant.md#输入-ub-复用通知的所有权)。
新 [GmmWeightQuantMxKernel 后处理分支](../../gemm/kernel/kernel_wqgmm_mix_weight_prologue.md) 则由Prologue阶段交接保护共享输入UB，并持续复用Epilogue对象；独立的输出槽位通过本地事件5/6保护。
输出与 scale 的搬出由本组件的流水事件保护。

## 资源与校验边界

每个输入半区最多预留 `64*256` 个 float 元素；tiling 必须保证每个 AIV
负责的 `M*Align64(N)` 不超过该容量。smooth 行缓冲从 `246 * 1024` 字节处开始，
每个 AIV 最多使用 128 个 FP32 元素，即 512 字节，结束位置为 252416 字节，
不超过 248 KiB UB。该区域与 prologue 及 epilogue 的其他缓冲均不重叠。
MX A8W4快速路径的Prologue使用64 KiB输入区和128 KiB转换输出区，Epilogue从64 KiB处读取FP32输入，并使用212 KiB开始的FP8双缓冲、196 KiB开始的打包scale双缓冲。共享输入区遵守Prologue暂停、Epilogue处理、Prologue恢复的阶段交接顺序。

Blaze Epilogue 是设备侧 `void`
组件，不返回 ACLNN 状态码。必选输出地址及所有属性的合法性由上层
ACLNN/tiling 校验；必选空指针由上层按接口规范返回 `161001`。

本文描述当前源码契约，不代表新增 smooth 路径已通过设备编译或 NPU 验证。

旧 `GetL0c2UbTensor` 和 `GetConcatL0c2UbTensor` 保留兼容，按主线标记废弃；新调用使用 `GetAccumulatorTensor`。OCP 使用 `zeroScaleOnZeroExp=false`。
