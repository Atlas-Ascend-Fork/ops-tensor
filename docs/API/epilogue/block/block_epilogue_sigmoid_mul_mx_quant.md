# Block Epilogue Sigmoid Mul MX Quant

> [代码位置](../../../../include/blaze/epilogue/block/block_epilogue_sigmoid_mul_mx_quant.h)

## 功能与精度

`BlockEpilogueSigmoidMulMxQuant` 在 Ascend 950 的 AIV 上读取 Fixpipe 写入 UB 的 FP32 矩阵乘结果 G，并加载 BF16 乘法输入 B，计算以下流程。B 与输出的逻辑 shape 均为 `[M, N]`，不支持 B 广播。

```text
S = sigmoid(G)                         // FP32
C = RINT_BF16(S * FP32(B))              // BF16
(Y, YScale) = DynamicMxQuant(C)         // FP8 + E8M0
```

sigmoid 与乘法在同一个 VF 中使用 FP32；仅乘积转换一次 BF16。该精度路径与逐个执行 BF16 sigmoid、mul 的中间舍入边界不同。bias 由矩阵乘阶段处理。

MX 量化复用 [Tile::MxQuant](../tile/arch35/mx_quant.md)，支持 `scaleAlg=0`（OCP）和 `scaleAlg=1`（BLAS/cuBLAS），FP8 量化使用 RINT。

## 模板参数与 Params

```cpp
template <typename DataTypeOut_, typename DataTypeIn_ = float, typename DataTypeMul_ = bfloat16_t>
class BlockEpilogueSigmoidMulMxQuant;

struct Params {
    GM_ADDR yGmAddr{nullptr};
    GM_ADDR yScaleGmAddr{nullptr};
    GM_ADDR mulGmAddr{nullptr};
    uint32_t scaleAlg{0};
};
```

| 模板参数 | 含义 | 当前支持类型 |
| --- | --- | --- |
| `DataTypeOut_` | 量化输出 Y | `fp8_e4m3fn_t`、`fp8_e5m2_t` |
| `DataTypeIn_` | Fixpipe 写入 UB 的门控矩阵 G | `float`（默认值） |
| `DataTypeMul_` | 从 GM 加载的乘法输入 B | `bfloat16_t`（默认值） |

类内分别公开 `DataTypeOut`、`DataTypeIn` 和 `DataTypeMul` 类型别名。省略第三个参数时，原有 `<OutType, MatmulOutType>` 调用仍使用 BF16 乘法输入。

sigmoid 与乘法的内部计算固定使用 `float`，量化前的乘积固定使用 `bfloat16_t`。乘法输入 `DataTypeMul` 当前仅支持 BF16，与乘积复用同一 UB 区域；其他输入类型由编译期断言拒绝。

`mulGmAddr` 指向 BF16 ND 乘法输入张量；输出 scale 使用 E8M0 字节存储。`Init` 保存 Params 引用，调用方必须保证其生命周期覆盖所有 tile。

## Shape、资源与同步

| 数据 | 逻辑 shape / 存储约定 |
| --- | --- |
| FP32 UB 输入 | 当前 AIV 的 `[mVec, singleN]`，行距 `Align32(singleN)`，UB 偏移为 0 |
| BF16 B / C | 共享同一 UB 区域；每个向量读取 B 后再写入 C，不改写 GM 中的 B |
| Y | `[batch..., M, N]`，只写有效元素 |
| YScale | `[batch..., M, ceil(N/64), 2]`，每 32 个 N 元素一个 scale |

两个 AIV 沿 M 分工，每个 AIV 固定使用 233 KiB UB，最多处理 128×256 个元素。调用方须保证 `baseM/baseN <= 256`；非末尾 N tile 的起点和宽度须按 32 个元素对齐。末尾不足 32 元素的量化组由 BF16 ZEROING cast 补零，奇数 scale 列由最后一个 N tile 补齐配对字节。

`V_MTE2` / `MTE2_V` 保护 B/C 区的搬入与复用。乘积写入 BF16 区后通知 Cube，允许下一次 Fixpipe 覆盖 FP32 输入区；量化输出和 scale 的搬出复用由 `V_MTE3` / `MTE3_V` 保护。

## 公共接口与集成

| 接口 | 作用 |
| --- | --- |
| `Init(const Params&)` | 保存参数引用、scale 算法和当前 AIV 编号，初始化 GM 地址 |
| `UpdateNextProblem(const ProblemShape&)` | 根据完整 N 计算乘法输入 B 的行距、scale 行跨度及配对补齐列数 |
| `UpdateGlobalAddr(const BlockCoord&)` | 按 `{outputElementOffset, scaleByteOffset}` 更新 B、Y 和 YScale 的 batch 基址 |
| `operator()(const BlockShape&, const BlockCoord&)` | 加载当前 B tile，执行 sigmoid、乘法、量化与输出搬运 |

配合 [QBMM MX Activation Quant Kernel](../../gemm/kernel/kernel_qbmm_mx_activation_quant.md) 的 `KernelMmadWithScaleMxActivationQuant` 使用，采用 `L0C2UB_MODE_DUAL_DST_SPLIT_M`、`ConcatN=false`。矩阵输入 dtype、ND/NZ、转置、batch 广播及可选 bias 由上层 Host 和 Cube 组件处理；B 必须覆盖完整输出 shape。

乘法输入 B 的 GM→UB 搬运采用保守的 L2 策略：仅当实际 GM 起点和行跨度 `N * sizeof(DataTypeMul)` 均按 128B 对齐，且本次单行搬运长度 `nSize * sizeof(DataTypeMul)` 按 512B 对齐时，使用 `cache_mode::disable` 开启 bypass；其余使用 `cache_mode::normal`。尾块和尾块重切分按实际 `nSize` 判断，不以 `baseN` 或补齐后的 UB 行跨度代替。

该条件依据不启用 HBM 复用的性能数据选取，用于减少窄块和非对齐场景的退化，不是硬件对 bypass 的使用限制，也不保证所有 shape 最优；不设置乘法输入 B 的总大小阈值。

## 集成示例

保留已有 `BlockMmad`、scheduler 和 `KernelMmadWithScaleMxActivationQuant` 策略，将 epilogue 类型和参数替换为：

```cpp
using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueSigmoidMulMxQuant<fp8_e4m3fn_t, float, bfloat16_t>;
using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
typename Kernel::Params params{};
params.problemShape = {m, n, k, 1};
params.epilogueParams = {yGm, yScaleGm, mulInputGm, 1U}; // BLAS; mulInputGm: BF16 [M,N]
// 按上述 Kernel 的契约填写 mmadParams / scale 地址、l1Params、schParams 和 qbmmParams。
Kernel kernel;
kernel(params);
```

完整可运行入口见 [CSV example](../../../../examples/acl/quant_matmul_activation_quant/quant_matmul_activation_quant/README.md#sigmoid-激活与逐元素乘法用例)。`mulInputGm` 是独立乘法输入张量地址，与矩阵乘的权重 B 区分；不得将它与 y/yScale 的 GM 存储重叠。

## 验证入口

- [Kernel UT](../../../../tests/ut/op_kernel/quant_matmul_activation_quant/test_quant_matmul_activation_quant.cpp) 检查默认/显式模板参数的一致性、新 epilogue 与通用 Kernel 的类型/参数契约，以及 CPU-debug 下初始化、batch 地址更新和重复构造析构。该测试不执行 VF 数值计算。
- [CSV 用例](../../../../examples/acl/quant_matmul_activation_quant/quant_matmul_activation_quant/quant_matmul_activation_quant.csv) 中的 `sigmoid_*` 在真实设备执行融合计算，与独立 CPU golden 比较 Y/scale；覆盖 ND/NZ、FP8 组合、转置、OCP/BLAS、bias、全载、尾块以及不同乘法输入矩阵大小和对齐组合。
- example 对 Y/scale 预填 `0xFF`，逐字节检查 scale 配对补零，验证 GM 乘法输入不变，并在相同输入分配上再次启动后比较输出/scale。重复启动用于正确性验证，不作为性能测量。
- 生产头文件登记在 `classify_rule.yaml` 的 `matmul-quant@ops-tensor/src/release/style`；UT、example 和文档由已有 `tests/`、`examples/`、`docs/` 的 `unrelease/test_code` 目录规则覆盖。
