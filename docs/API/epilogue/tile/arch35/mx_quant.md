# MxQuant（Tile 级 MX 动态量化）
> [代码位置](../../../../../include/blaze/epilogue/tile/arch35/mx_quant.h)

## 功能说明

`MxQuant` 将 UB 中的 BF16 数据按每 32 个元素一组量化为 FP8/FP4，并生成 E8M0 scale。各接口可独立调用：

- **GroupMaxExp**：提取每组最大指数（OCP）或最大绝对值（cuBLAS / DynDtypeRange），以 `uint16_t` 位编码输出。
- **GenScale**：生成 E8M0 scale 和 BF16 倒数，倒数以 `uint16_t` 位编码存储。
- **Quantize**：将输入乘以对应组的倒数，转换并打包为 FP8/FP4。
- **TransScaleLayout**：将紧凑 scale 重排为 32 字节行距。
- **PadScaleRows**：按 32 字节行距重排 scale，并清零指定的尾列。
- **TransFp4OutLayout**：将 FP4 输出从 `Align16(n/2)` 字节行距重排为 `Align32(n/2)` 字节行距。

公共接口接收 UB Tensor，校验类型和布局后调用 VF；Block 负责缓冲、同步和 GM 写回。

## 特殊约束

### 架构
面向 dav-3510，经 [compute.h](../compute.md) 引入。

### 数据类型
`DataTypeOut` 支持 `fp8_e4m3fn_t` / `fp8_e5m2_t` / `fp4x2_e2m1_t` /
`fp4x2_e1m2_t`（通过类级 `static_assert` 校验）。FP8 与 FP4 量化路径由 `DataTypeOut`
编译期分派（`if constexpr`），FP4 舍入模式（RINT / FLOOR / ROUND）由
`Quantize` 的运行时参数分派到模板 Vf。

### 运行时算法配置（MxQuantConfig）
```cpp
enum class MxScaleAlg : uint8_t { OCP = 0, CUBLAS = 1, DYN_DTYPE_RANGE = 2 };

struct MxQuantConfig {
    MxScaleAlg alg{MxScaleAlg::OCP};
    uint16_t fpEmax{0};             // E4M3:0x0400  E5M2:0x0780  E2M1:0x0100  E1M2:0x0000
    float invDstTypeMax{1.0f};      // cuBLAS: 1/dstTypeMax（host 可覆盖）
    uint16_t addValueBits{0x003f};  // Dyn: dstTypeMax==7 ? 0x001f : 0x003f
    bool zeroScaleOnZeroExp{false}; // OCP zeroMask 判据（见下）
};
```
Block 在 `Init` 中保存标量配置，在计算函数内构造 `MxQuantConfig`。
主机编译时 `compute.h` 不引入 Tile 类型，因此 Block 的成员声明使用标量类型。

### OCP 与 cuBLAS 算法选择

- `OCP` 根据组内最大指数生成 2 的整数次幂 scale。归一化结果超过目标类型范围时，由后续 FP8 转换处理。
- `CUBLAS` 根据组内最大绝对值和目标最大有限值计算 scale，并将 E8M0 指数向上取整，降低边界值溢出的风险。
- 两者均按 32 个元素分组，区别在于 scale 的计算方式。

### OCP 零值处理（zeroScaleOnZeroExp）

`sharedExp = max(maxExp, fpEmax) - fpEmax`，其中 `maxExp`、`fpEmax` 均为 BF16 指数域编码。

| 配置 | 清零条件与范围 |
| --- | --- |
| `false` | `sharedExp == 0` 时将 reciprocal 置零 |
| `true` | 原始 `maxExp == 0` 时将 yScale 和 reciprocal 置零 |

当 `0 < maxExp <= fpEmax` 时，两种配置生成的 reciprocal 编码分别为 `0` 和 `0x7f00`。
此时 yScale 的 E8M0 编码均为 `0x00`，表示 `2^-127`。
各 Block 的配置见下文。DynDtypeRange 忽略此参数，在 `sharedExp == 0` 时将 reciprocal 置零。

### 张量约束
- **元素类型**：src 为 `bfloat16_t`；maxExp / reciprocal 为 `uint16_t`；
  yScale / y / 转置输入输出为 `int8_t`
- **内存位置**：所有张量必须为 UB
- **排布**：所有张量必须为 `nd_ext_layout_ptn`（仅支持 ND）
- **连续存储**：量化阶段按 totalCount / totalScale 连续扫描，行距由 Block 保证。
  GELU/Swish 激活区行距为 Align32(n)，SwiGLU 为 Align64(n)
- **maxExp 每组一个**：`totalScale = CeilDiv(totalCount, 32)`；totalCount 可为非 32 倍数，尾组通过 mask 限定有效元素。
- **yScale 打包**：`DIST_PACK_B16`（每寄存器 vlHalf/2 条），调用方 buffer 按
  totalScale 字节预留
- **y 打包**：`DIST_PACK4_B32`（每块 64 元素），FP4 输出的 N 尾部无效元素由调用方清零。
- **TransScaleLayout**：src 每行紧凑存储 scaleBlockN 个 scale，dst 行距为 32 字节。
  接口只写有效列；若后续搬运包含行尾补齐字节，调用方须先将这些字节清零。
- **PadScaleRows**：src/dst 为基址 32B 对齐且不重叠的 UB 缓冲，行距分别为 `scaleCols` 和 32 字节。
  列数及读取范围要求见 [接口说明](#mxquantpadscalerows)。

## 特殊类型

### MxQuant
```cpp
template <typename DataTypeOut_>
class MxQuant {
public:
    template <typename SrcTensor, typename MaxExpTensor>
    __aicore__ inline void GroupMaxExp(const SrcTensor& srcTensor, const MaxExpTensor& maxExpTensor,
                                         uint32_t totalCount, bool useAbs);

    template <typename MaxExpTensor, typename ScaleTensor, typename ReciprocalTensor>
    __aicore__ inline void GenScale(const MaxExpTensor& maxExpTensor, const ScaleTensor& yScaleTensor,
                                        const ReciprocalTensor& reciprocalTensor, const MxQuantConfig& cfg,
                                        uint32_t totalScale);

    template <typename SrcTensor, typename ReciprocalTensor, typename YTensor>
    __aicore__ inline void Quantize(const SrcTensor& srcTensor, const ReciprocalTensor& reciprocalTensor,
                                    const YTensor& yTensor, uint32_t totalCount,
                                    MxQuantFp4RoundMode fp4RoundMode = MxQuantFp4RoundMode::RINT);

    template <typename ScaleTensor, typename ScaleBlockTensor>
    __aicore__ inline void TransScaleLayout(const ScaleTensor& srcTensor, const ScaleBlockTensor& dstTensor,
                                            uint16_t mSize, uint16_t scaleBlockN);

    template <typename ScaleTensor, typename ScaleBlockTensor>
    __aicore__ inline void PadScaleRows(const ScaleTensor& srcTensor, const ScaleBlockTensor& dstTensor,
                                        uint16_t mSize, uint32_t scaleCols, uint32_t scaleCopyCols);

    template <typename YTensor, typename YBlockTensor>
    __aicore__ inline void TransFp4OutLayout(const YTensor& srcTensor, const YBlockTensor& dstTensor,
                                             uint16_t mSize, uint16_t nSize);
};
```

### MxQuant::GroupMaxExp
| 参数 | 类型 | 说明 |
|------|------|------|
| srcTensor | SrcTensor | 输入 UB Tensor，bfloat16_t，连续 totalCount 元素 |
| maxExpTensor | MaxExpTensor | 输出 UB Tensor，uint16_t，totalScale = CeilDiv(totalCount, 32) 组 |
| totalCount | uint32_t | 连续有效元素数，可为非 32 倍数 |
| useAbs | bool | false → 指数位掩码（OCP）；true → 绝对值掩码（cuBLAS/Dyn） |

### MxQuant::GenScale
| 参数 | 类型 | 说明 |
|------|------|------|
| maxExpTensor | MaxExpTensor | 输入 UB Tensor，uint16_t |
| yScaleTensor | ScaleTensor | 输出 UB Tensor，int8_t（E8M0 打包写入） |
| reciprocalTensor | ReciprocalTensor | 输出 UB Tensor，uint16_t |
| cfg | MxQuantConfig | 算法配置（alg 三选一分派） |
| totalScale | uint32_t | 组数（= CeilDiv(totalCount, 32)） |

### MxQuant::Quantize
| 参数 | 类型 | 说明 |
|------|------|------|
| srcTensor | SrcTensor | 输入 UB Tensor，bfloat16_t，连续 totalCount 元素 |
| reciprocalTensor | ReciprocalTensor | 输入 UB Tensor，uint16_t（GenScale 输出） |
| yTensor | YTensor | 输出 UB Tensor，int8_t（DataTypeOut 打包写入） |
| totalCount | uint32_t | 连续元素总数 |
| fp4RoundMode | MxQuantFp4RoundMode | FP4 舍入（RINT/FLOOR/ROUND，FP8 忽略） |

### MxQuant::TransScaleLayout / TransFp4OutLayout
| 参数 | 类型 | 说明 |
|------|------|------|
| srcTensor | SrcTensor | 输入 UB Tensor，int8_t |
| dstTensor | DstTensor | 输出 UB Tensor，int8_t |
| mSize | uint16_t | 行数 |
| scaleBlockN | uint16_t | 每行有效 scale 数（TransScaleLayout） |
| nSize | uint16_t | 每行元素数（TransFp4OutLayout，行距内部推导） |

### MxQuant::PadScaleRows

| 参数 | 类型 | 说明 |
|------|------|------|
| srcTensor | ScaleTensor | 输入 UB Tensor，int8_t，mSize 行紧凑存储，每行 scaleCols 字节 |
| dstTensor | ScaleBlockTensor | 输出 UB Tensor，int8_t，预留 mSize × 32 字节 |
| mSize | uint16_t | 行数 |
| scaleCols | uint32_t | 每行有效 scale 字节数，同时作为源行距 |
| scaleCopyCols | uint32_t | 每行写入及后续复制到 GM 的字节数；配对补零时为 scaleCols + 1 |

要求 `0 < scaleCols <= scaleCopyCols <= 32`。每行复制 `[0, scaleCols)`，
清零 `[scaleCols, scaleCopyCols)`，其余字节保持原值。Block 负责缓冲、同步和 GM 写回。

有效数据 mask 与写出 mask 分开设置，避免将下一行的 scale 误写到本行 padding。
当 `mSize > 0` 时，从源基址开始至少须有以下字节数的 UB 可读：

```text
readableBytes = AlignDown((mSize - 1) * scaleCols, 32) + 256 + 32
```

[LoadUnAlign](https://asc.gitcode.com/api/SIMD-API/basic_api/reg_vector_compute/reg_data_load/LoadUnAlign_continuous.html)
读取 256B 向量和 32B 缓存块，mask 不限制源读取范围。

## 使用示例

```cpp
#include "blaze/epilogue/tile/compute.h"

// Block 层持有 UB 字节偏移（geluResUbOffset_ / maxExpUbOffset_ ...）
const uint32_t alignedN = Gemm::CeilAlign(nSize, AscendC::ONE_BLK_SIZE);
const uint32_t totalData = mSize * alignedN;
const uint32_t totalScale = Gemm::CeilDiv(totalData, AscendC::ONE_BLK_SIZE);

auto dataLayout = Gemm::MakeNDExtLayout<int8_t>(1, totalData, totalData);
auto scaleLayout = Gemm::MakeNDExtLayout<int8_t>(1, totalScale, totalScale);
auto actTensor = asc::te::make_tensor(
    asc::te::make_mem_ptr<asc::te::location::ub, bfloat16_t>(geluResUbOffset_), dataLayout);
auto maxExpTensor = asc::te::make_tensor(
    asc::te::make_mem_ptr<asc::te::location::ub, uint16_t>(maxExpUbOffset_), scaleLayout);
// ... yScaleTensor / reciprocalTensor / yTensor 同理

MxQuantConfig cfg{};              // 函数体内组装（禁止作为 Block 类成员）
cfg.alg = MxScaleAlg::OCP;
cfg.fpEmax = fpEmax_;
cfg.zeroScaleOnZeroExp = false;

Blaze::Epilogue::Tile::MxQuant<DataTypeOut> mx;
mx.GroupMaxExp(actTensor, maxExpTensor, totalData, false);
mx.GenScale(maxExpTensor, yScaleTensor, reciprocalTensor, cfg, totalScale);
mx.Quantize(actTensor, reciprocalTensor, yTensor, totalData);
mx.TransScaleLayout(yScaleTensor, yScaleBlockTensor, mSize, scaleBlockN);
```

## 数据流

```
激活结果（bf16, UB, totalCount = mSize × Align32(n)）
    │
    ↓ GroupMaxExp（每 32 元素组取最大指数位/绝对值）
maxExp（uint16, totalScale = CeilDiv(totalCount, 32) 组）
    │
    ↓ GenScale（OCP / cuBLAS / DynDtypeRange 三算法之一）
yScale（E8M0, DIST_PACK_B16 打包）+ reciprocal（uint16, BF16 倒数位编码）
    │                                   │
    ↓ Quantize（×reciprocal → SAT cast） │
y（FP8/FP4, DIST_PACK4_B32 打包）        ↓ TransScaleLayout（ceil(N/32) 有效值
                                         │   → 每行 32B 行块布局）
                                         yScale（GM 搬运布局）
```

## 与 Block 层的关系

调用组件及其配置：

- [BlockEpilogueGeluTanhMxQuant](../../block/block_epilogue_gelu_tanh_mx_quant.md)：
  三算法全支持，`zeroScaleOnZeroExp = false`
- [BlockEpilogueGeluMxQuant](../../block/block_epilogue_gelu_mx_quant.md)：
  三算法 + FP4 输出转置（`TransFp4OutLayout`），`zeroScaleOnZeroExp = false`
- [BlockEpilogueFlatQuant](../../block/block_epilogue_flat_quant.md)：
  三算法全支持（dstTypeMax 0/6,7/其他 分派），`zeroScaleOnZeroExp = true`，
  eMax 的 abs/exp 路径由 dstTypeMax ∈ [6,12] 区间判定

- [BlockEpilogueSwigluMxQuant](../../block/block_epilogue_swiglu_mx_quant.md)：
  OCP/cuBLAS，`zeroScaleOnZeroExp = false`。

- [BlockEpilogueSwishMxQuant](../../block/block_epilogue_swish_mx_quant.md)：
  OCP/cuBLAS、FP8 RINT，`zeroScaleOnZeroExp = false`。

Block 根据输出类型和 Host 参数选择配置，管理 UB 布局、同步、尾部补零和 GM 写回。
