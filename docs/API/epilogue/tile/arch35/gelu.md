# Gelu（Tile 级 GELU 激活）
> [代码位置](../../../../../include/blaze/epilogue/tile/arch35/gelu.h)

## 功能说明

Tile 级 GELU 激活组件，提供两种算法，供多个 Block Epilogue 复用：

- **GeluTanh**：tanh 近似，`0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`，
  等价 sigmoid 形式 `x / (1 + exp(-√(8/π)·(x + 0.044715·x³)))`，纯寄存器指令实现
- **GeluErf**：erf 精确形式，`0.5·x·(1 + erf(x/√2))`，`ErfCallee` 分段多项式
  近似（寄存器级移植自 erf_3510_impl.h）+ 寄存器组装，同样纯寄存器实现

采用两层设计：
- **公共接口（`__aicore__`）**：接收 `make_tensor` 构造的 UB 张量，内部经
  `.data().get()` 提取 `__ubuf__` 指针后委托 Vf；入口含 static_assert
  类型/排布门禁与 `if ASCEND_IS_AIC` 早退；`nSize % sizePerRepeat == 0` 时
  以 `FULL_REPEAT` 形式启动 Vf，省去逐 repeat 的 `UpdateMask`
- **私有实现（`__simd_vf__` / `__simd_callee__`）**：Reg API 寄存器级计算。
  两种算法各有一个模板化 Vf 入口——`GeluTanhVf` / `GeluErfVf`，加载头由
  共享的 `GeluImplBase::GeluLoadCallee`（fp32 直接加载 / 16F unpack+加宽，
  `if constexpr` 按输入类型选择）+ 各自的 `GeluTanhCoreCallee` /
  `GeluErfCoreCallee` 数学链 + 共享的 `GeluImplBase::GeluStoreCallee` 缩窄尾段
  组成。整块 tile 一次 `asc_vf_call` 完成，无 temp buffer、无高层 API 调用

## 特殊约束

### 架构
仅 dav-3510（`__NPU_ARCH__ == 3510`），经
[compute.h](../compute.md) 派发引入，Block 层禁止直接 include 本头文件。

### 数据类型
输入/输出元素类型**从张量推导**（`get_attribute_element_type`），各自支持
`float`、`bfloat16_t`、`half`，**任意 3×3 组合**。类型转换策略：

| 输入 | 计算 | 输出 | 缩窄策略 |
|------|------|------|---------|
| float | fp32 | float | 直接存储，无缩窄 |
| float | fp32 | bfloat16_t | `CT_32F_TO_16F`（NO_SAT，bf16 与 fp32 指数域相同不会溢出） |
| float | fp32 | half | `CT_32F_TO_16F_SAT`（SAT，超过 65504 钳位而非 inf） |
| bfloat16_t / half | 先 `CT_16F_TO_32F` 加宽到 fp32 | 同上 | 同上 |

缩窄舍入模式均为 `CAST_RINT`（就近舍入），由按输出类型选择的
`GeluImplBase::GeluOutCastTrait` 编译期决定。

### 张量契约（static_assert 门禁）
- **元素类型**：src/dst 元素各自须为 `float` / `bfloat16_t` / `half`
  （类型即计算类型，无需显式模板参数）
- **形状**：`mSize` / `nSize` 从 srcTensor 的 layout shape 读取
  （`((1, rows), (1, cols))`），不作为 `Run` 参数传递
- **内存位置**：所有张量必须为 UB
- **排布**：所有张量必须为 `nd_ext_layout_ptn`（仅支持 ND）
- **行距**：src/dst 的 rowPitch 均须为 `Gemm::Align32(n)` 元素
  （Vf 内按 `mIdx * nAligned` 计算行偏移）

### MX 尾组补零（kPadToMxGroup）
MX 量化 epilogue 要求最后一个 MX 组（32 元素）的尾段是确定的零值而非
UB 残留数据。两个特化的 `Run` 均提供 `kPadToMxGroup` 模板参数：
为 true 时 16F 输出的 store mask 从 `nSize` 加宽到 `nAligned`，配合缩窄
cast 的 ZEROING 合并模式把尾段写成 0。**仅允许 `bfloat16_t` 输出**
（fp32 直存路径没有 ZEROING cast，static_assert 拦截）。
调用侧若多行在行距下天然连续（`nSize == nAligned`），可将 tile 展平成
单行 layout（cols = mSize × nAligned）以减少逐行循环开销——形状经
layout 传入，`Run` 接口不变。

## 特殊类型

### Gelu
tag 分派的全特化（与 dequant.h 同一组织方式）：tag 选择算法，
类型与排布跟随张量；主模板仅声明，不支持的 tag 编译期失败。
两个特化共享的内部实现（张量门禁、Vf 参数打包、cast traits、
load/store callee）收敛在 `GeluImplBase` 基类中——成员全部
`protected`、特化私有继承，外部代码无法直接使用这些内部件。

```cpp
struct GeluTanhTag {};  // tanh 近似
struct GeluErfTag {};   // erf 精确

template <class GeluTag>
class Gelu;  // 主模板：仅声明

template <>
class Gelu<GeluTanhTag> {
public:
    // kPreciseDiv：tanh 专属的除法模式（true = 误差补偿除法）；
    // kPadToMxGroup：MX 尾组补零（仅 bf16 输出）；
    // tile 形状 [mSize, nSize] 从 srcTensor 的 layout 读取
    template <bool kPreciseDiv = false, bool kPadToMxGroup = false,
              typename SrcTensor, typename DstTensor>
    __aicore__ inline static void Run(const SrcTensor& srcTensor, const DstTensor& dstTensor);
};

template <>
class Gelu<GeluErfTag> {
public:
    template <bool kPadToMxGroup = false, typename SrcTensor, typename DstTensor>
    __aicore__ inline static void Run(const SrcTensor& srcTensor, const DstTensor& dstTensor);
};
```

### Gelu\<GeluTanhTag\>::Run
| 参数 | 类型 | 说明 |
|------|------|------|
| srcTensor | SrcTensor | 输入 UB Tensor，形状 [mSize, nSize] 编码在 layout 中，rowPitch = Align32(n) |
| dstTensor | DstTensor | 输出 UB Tensor，与 src 同形同行距 |

执行流程：fp32 输入直接 `DataCopy` 加载；16F 输入 `DIST_UNPACK_B16` 加载 +
`CT_16F_TO_32F` 加宽 → `GeluTanhCoreCallee`（Mul→Mul→Axpy→Muls→Exp→Adds→Div
七指令数学链 → 按输出类型缩窄 → store）。

### Gelu\<GeluErfTag\>::Run
| 参数 | 类型 | 说明 |
|------|------|------|
| srcTensor | SrcTensor | 输入 UB Tensor，形状 [mSize, nSize] 编码在 layout 中 |
| dstTensor | DstTensor | 输出 UB Tensor，与 src 同形同行距 |

执行流程（单次 Vf 启动，逐行逐 repeat）：加载 x →（16F 输入先 unpack+加宽）→
`GeluErfCoreCallee`（`Muls(x, 1/√2)` → `ErfCallee` 分段多项式近似 →
`(1+erf)·(0.5·x)` 组装 → 缩窄 store）。

## 使用示例

```cpp
#include "blaze/epilogue/tile/compute.h"

// Block 层持有 UB 字节偏移（geluResUbOffset_ ...）
const uint32_t nAligned = Gemm::Align32(static_cast<uint32_t>(nSize));
auto layout = Gemm::MakeNDExtLayout(static_cast<int64_t>(mSize),
                                    static_cast<int64_t>(nSize),
                                    static_cast<int64_t>(nAligned));
auto srcTensor = asc::te::make_tensor(
    asc::te::make_mem_ptr<asc::te::location::ub, float>(0), layout);
auto dstTensor = asc::te::make_tensor(
    asc::te::make_mem_ptr<asc::te::location::ub, bfloat16_t>(geluResUbOffset_), layout);

namespace Tile = Blaze::Epilogue::Tile;

Tile::Gelu<Tile::GeluTanhTag>::Run(srcTensor, dstTensor);   // tanh 近似
// 或 erf 精确形式（无需 temp buffer）：
// Tile::Gelu<Tile::GeluErfTag>::Run(srcTensor, dstTensor);
// tanh 误差补偿除法（GELU (tanh) MX 血缘）：
// Tile::Gelu<Tile::GeluTanhTag>::Run</*kPreciseDiv=*/true>(srcTensor, dstTensor);
// MX 尾组补零（MX 量化 epilogue，bf16 输出）：
// Tile::Gelu<Tile::GeluTanhTag>::Run</*kPreciseDiv=*/false, /*kPadToMxGroup=*/true>(srcTensor, dstTensor);
// Tile::Gelu<Tile::GeluErfTag>::Run</*kPadToMxGroup=*/true>(srcTensor, dstTensor);
```

## 数据流

```
UB（src, rowPitch=Align32(n)）
    ↓ [16F] unpack + CT_16F_TO_32F 加宽        ↓ [fp32] 直接加载
    └──────────────┬───────────────────────────┘
                   ↓
    GeluTanhCoreCallee / GeluErfCoreCallee（fp32 寄存器计算）
                   ↓
    CT_32F_TO_OUT 缩窄（half→SAT / bf16→NO_SAT / float→直存，
    由 GeluImplBase::GeluOutCastTrait<DataTypeOut> 选择）
                   ↓
UB（dst, rowPitch=Align32(n)；kPadToMxGroup 时末组尾段写 0）
```

## 与 Block 层的关系

`Gelu` 是首个从 Block 内联实现抽取的 Epilogue Tile，当前被三个 Block 复用：
- [BlockEpilogueGeluMxQuant](../../block/block_epilogue_gelu_mx_quant.md)：
  `Gelu<GeluErfTag>::Run</*kPadToMxGroup=*/true>` /
  `Gelu<GeluTanhTag>::Run<false, true>` 按 `geluAlg` 运行时选择；
  行连续时把 tile 展平成单行 layout 传入
- [BlockEpilogueGeluTanhMxQuant](../../block/block_epilogue_gelu_tanh_mx_quant.md)：
  固定 `Gelu<GeluTanhTag>::Run</*kPreciseDiv=*/true>`，输出 bfloat16 激活结果供 MX 量化
- BlockEpilogueGeluFixpipe：fixpipe 场景，固定 `Gelu<GeluTanhTag>` 默认除法模式

Block 层负责 UB 偏移管理、AIC/AIV 跨核同步与 slot 双缓冲，Tile 层只做纯计算。
