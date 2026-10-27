# Dequant（Tile 级反量化）

> [代码位置](../../../../../include/blaze/epilogue/tile/arch35/dequant.h)

## 功能说明

Tile 级反量化组件，在**两个轴**上完成编译期分发：

1. **类模板参数（量化对象 tag）**——选择反量化的对象与流水线阶段：
   - `Fp4ToFp8Tag`：prologue，packed FP4 权重 × per-group scale → FP8，MMAD **前**，喂 L1 分形布局；
   - `B32ToFpTag`：epilogue，任意 int32/fp32 输入 × x2Scale × x1Scale + bias → fp32/bf16/fp16（不限 MMAD 累加结果）。
2. **Run 模板参数（量化粒度 `DequantMode`）**——每个家族支持一个子集，由 Run 入口的
   static_assert 强制；dtype 与 layout 差异由 Run 的张量参数推导，不由 tag 决定。

各家族流水线位置与 guard 约定**互不相同，不要跨家族统一**：W4 由 AIV 侧权重
prologue 发起（无 AIC guard）；B32 为 AIV-only（保留 AIC 早退 guard）。

## 特殊约束

### 架构

仅 dav-3510（`__NPU_ARCH__ == 3510`），经
[compute.h](../compute.md) 派发引入；上层组件依赖 `blaze/epilogue/tile/compute.h`，
不直接 include 本头文件。

### 接口形态

函数模板不能偏特化，因此第一轴扩展点定义为类模板 `Dequant<DequantTag>`（DequantTag
为量化对象 tag），主模板只有声明，不支持的 tag 在实例化时 static_assert 报错；`Run` 为静态成员
函数模板，第二轴（粒度）走其模板参数：

- 量化对象 + 流水线阶段 → 类模板参数（tag）+ 类模板偏特化
- 量化粒度（`DequantMode`）→ `Run` 的模板参数，家族内不支持的组合由入口 static_assert 拦截
- dtype/layout/存储位置 → 从 `Run` 的张量参数类型经 `static_assert` + `if constexpr` 门禁与分派

### 数据类型和布局门禁（static_assert）

**W4 家族**（`Dequant<Fp4ToFp8Tag>`）：
- 仅支持 `Gran == DequantMode::PER_GROUP`，且 `GroupSize == 32`；
- 三个 Tensor 均须为 **UB** tensor；
- 输出 `fp8_e4m3fn_t`，输入 packed FP4，scale `half` / `bfloat16_t`；
- 输入 layout 仅支持 `dn_ext_layout_ptn`（ND-trans 路径）与 `nz_layout_ptn`（NZ 路径）。

**B32 家族**（`Dequant<B32ToFpTag>`）：
- `x2Gran` 仅 PER_TENSOR / PER_CHANNEL；`x1Gran` 仅 NONE / PER_TENSOR / PER_TOKEN；
- 输入 `int32_t` / `float`（任意来源，不限 MMAD 累加结果）；输出 `float` / `bfloat16_t` / `half`；
- x2Scale `float` / `bfloat16_t` / `half`；bias `float` / `bfloat16_t` / `half`；
- x1Scale 张量固定 fp32（per-token 的 `DIST_BRC_B32` 广播加载按 fp32 寄存器相乘）；
- 五个 Tensor 均须为 **UB** tensor 且均为 `nd_ext_layout_ptn`。

## 特殊数据结构

### `DequantMode`

```cpp
enum class DequantMode : uint8_t {
    NONE = 0,        // 该轴不带 scale（仅 B32 x1 轴）
    PER_TENSOR = 1,  // 全局标量
    PER_CHANNEL = 2, // N 轴向量 scale（B32 x2 轴 / 权重 N 维）
    PER_TOKEN = 3,   // M/行轴向量 scale（Acc x1 轴）
    PER_GROUP = 4,   // K 组轴 scale（W4 权重；group size 走 Run 的 GroupSize）
};
```

### 量化对象 tag

```cpp
struct Fp4ToFp8Tag {}; // prologue: packed FP4 weight -> FP8, MMAD 前
struct B32ToFpTag {};      // int32/fp32 -> fp32/bf16/fp16（×x2/x1 scale + bias，任意来源）
```

### `A8W4TcgDequantParams`

```cpp
struct A8W4TcgDequantParams {
    uint32_t validK{0};
    __ubuf__ uint8_t* scaleMaskAddr{nullptr};
};
```

| 字段            | 说明                                                                                                                                                                  |
| :-------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `validK`        | 运行时 K 有效长度。NZ 路径按`validK / GROUP_SIZE` **floor** 取组数：`k % 32 != 0` 时尾组不转换——这是任何 layout 都无法表达的 floor 语义，故放在运行时参数                |
| `scaleMaskAddr` | NZ 路径的 VF scale 掩码（一个完整 VF 掩码寄存器，每 64bit 字重复 32bit-on/32bit-off 模式），由调用方的 UB storage 持有；VF 调用边界不保持寄存器，故每次经参数重装        |

## 特殊成员方法

### `Dequant<Fp4ToFp8Tag>::Run`

```cpp
template <DequantMode Gran, uint32_t GroupSize, class InputTensor, class ScaleTensor, class OutTensor>
__aicore__ inline static void Run(const InputTensor& input, const ScaleTensor& scale,
                                  const OutTensor& output, const A8W4TcgDequantParams& params);
```

按输入 layout 分派到两条路径，三 Tensor 的布局契约如下：

| 路径     | 输入（packed FP4）                                                     | scale                                                        | 输出（FP8）                                                    |
| :------- | :--------------------------------------------------------------------- | :----------------------------------------------------------- | :------------------------------------------------------------- |
| ND-trans | `dn_ext_layout_ptn` (k, n)，物理 n 行 pitch 64 对齐后 Slice            | `dn_ext_layout_ptn` (kGroup, n)，n 行 pitch 32B 对齐后 Slice | `Weight8BitDnToZnUbLayoutPtn`                                  |
| NZ       | `nz_layout_ptn` (k, n) 显式 stride（紧凑 pitch，n1 间距 = 32 × kLen） | `nd_ext_layout_ptn` (kGroup, n)，行 pitch 为 32 对齐的 N     | `NzRowPaddingLayoutPtn`（256 元素 chunk + BL1 ping-pong 交织） |

scale/输出的布局族见
[layout_struct.h](../../../../../include/blaze/gemm/utils/layout_struct.h)；
两条输出布局同时是 UB→L1 搬运（[CopyPaddedUBToL1](../../../gemm/tile/copy_ub_to_l1.md) 转换权重族）
的源契约，VF 输出布局与 UB→L1 搬运共用同一个 layout 定义（单一事实源）。

### `Dequant<B32ToFpTag>::Run`

```cpp
template <DequantMode x2Gran, DequantMode x1Gran, bool hasBias,
          class AccTensor, class DstTensor, class X2ScaleTensor, class X1ScaleTensor, class BiasTensor>
__aicore__ inline static void Run(const AccTensor& accTensor, const DstTensor& dstTensor,
                                  const X2ScaleTensor& x2ScaleTensor, const X1ScaleTensor& x1ScaleTensor,
                                  const BiasTensor& biasTensor, float x2ScaleScalar, float x1ScaleScalar,
                                  uint16_t mSize, uint16_t nSize, uint32_t nSrcAligned, uint32_t nDstAligned);
```

单次 VF launch、寄存器驻留全流程；行距（`nSrcAligned` / `nDstAligned`）保持显式参数——
它们是 Block 侧的 buffer 布局关注点（L0C 32B 对齐与 workspace GetVecLen 对齐并存）。
参数语义：

| 参数            | 说明                                                                                                                              |
| :-------------- | :-------------------------------------------------------------------------------------------------------------------------------- |
| `accTensor`     | 累加器（int32/fp32），UB，行距 = nSrcAligned（Block 侧布局）                                                                       |
| `dstTensor`     | 输出（fp32/bf16/fp16），UB ping-pong，行距 = nDstAligned                                                                           |
| `x2ScaleTensor` | `PER_CHANNEL` → 单行向量（跨行复用，按 vfBlock 偏移索引）；`PER_TENSOR` → 占位张量（不解引用），数值走 `x2ScaleScalar`                 |
| `x1ScaleTensor` | `PER_TOKEN` → fp32 线性 mSize 项（Block 已套用行偏移）；否则占位张量                                                                |
| `biasTensor`    | `hasBias` → fp32/bf16/fp16 单行向量（跨行复用，元素类型即 bias 计算类型，由 Tile 从张量推导）；否则占位张量                     |
| 标量槽          | 未启用的 stage 不读取；`PER_TENSOR` 的值经对应 scalar 传入，禁用槽传 `1.0f`                                                          |

**StreamK 无 bias 的「x2*x1 预合并 + 0xFFFFE000 尾数掩码」属于 Block 侧 Init 关注点**：
Block 把（可能已合并的）标量经 `x2ScaleScalar` 传入并调用 `Run<PER_TENSOR, NONE>`，
Tile 本身不感知。

## 计算流程

### W4 家族

两条路径共用同一条 VF 数学链，差别在加载排布与 store 模式：

1. `DIST_UNPACK4_B8` 从相距 64 字节（128 个 packed FP4 元素）的两列加载并 unpack；
2. `CastWeightF4ToF16`：F4 → F16（scale 为 bf16 时经 bf16 中转再转 F16）；
3. `Mul` per-group scale（ND 路径 scale 经 `Interleave` 交织成两列；NZ 路径从相距
   `BLOCK_CUBE`(16) 元素的两个 scale bank 做相同分布的块加载，用 `scaleMaskAddr` 掩码
   `Select` 逐 lane 选bank）；
4. F16 → F32 → FP8 两轮 `Cast`（RegLayout 0/2 交错），`Select` + `DeInterleave`（ND）
   或 `Select`（NZ）重排 lane；
5. store：ND 路径 `DATA_BLOCK_COPY` 按 `Weight8BitDnToZnUbLayoutPtn` 的 slab 推进；
   NZ 路径 `DIST_PACK_B16` 按 `NzRowPaddingLayoutPtn` 的 chunk/vl/n1 三层 stride 推进。

### B32 家族

五段 stage 与 Block 侧传入的编译期参数一一对应（`if constexpr` 消歧）：

1. **加载 + 累加器 cast**：`int32 → fp32`（fp32 累加器为寄存器直拷）；
2. **× x2Scale**：`PER_TENSOR` 单条 `Muls` / `PER_CHANNEL` 加载 + 展宽（bf16/fp16 经
   ZERO/ONE 寄存器对 + `Interleave` 1:2 展宽）+ `Mul`；
3. **× x1Scale**：`PER_TENSOR` 单条 `Muls` / `PER_TOKEN` `DIST_BRC_B32` 广播 + `Mul` /
   `NONE` 寄存器直拷；
4. **+ bias**：fp32 直加 / 16F 展宽后加 / 无 bias 直拷；
5. **输出 cast + store**：fp32 `DIST_NORM_B32` / 16F `DIST_PACK_B32`。

16F 输出收窄为 **NO_SAT**（dequant 谱系），与 Gelu 的 half-SAT 策略**刻意不同**——
各家族保留各自的精度基线，不要跨家族统一。

## 与 Block 层的关系

`Dequant` 只做纯计算：一次 `Run` 处理调用方传入的一个 tile，Tensor 准备、UB slot 生命周期
和多 tile 编排由上层负责。当前调用方：

| 家族                  | 调用方                                                                                                                     |
| :-------------------- | :------------------------------------------------------------------------------------------------------------------------- |
| `Dequant<Fp4ToFp8Tag>` | [Kernel Wqmm Mix Pergroup](../../../gemm/kernel/kernel_wqmm_mix_pergroup.md) 的 AIV 权重 prologue（`KernelPergroupWeightPrologue`）：prologue 构造好满足上表契约的 UB tensor view 后委托本组件，转换结果随后由 UB→L1 搬运读走 |
| `Dequant<B32ToFpTag>`      | [BlockEpilogueDequant](../../block/block_epilogue_dequant.md)（MIX dequant 后处理）与 [BlockEpilogueQbmmPertensorStreamK](../../block/block_epilogue_qbmm_pertensor_streamk.md)（StreamK 归约后处理）：Block 负责 GM→UB staging、runtime 模式分发与张量构造（禁用槽位传占位张量），计算链委托本组件 |

## 使用示例

### W4 权重反量化（prologue）

```cpp
#include "blaze/epilogue/tile/compute.h"

using A8W4Dequant = Blaze::Epilogue::Tile::Dequant<Blaze::Epilogue::Tile::Fp4ToFp8Tag>;
using W4Gran = Blaze::Epilogue::Tile::DequantMode;
Blaze::Epilogue::Tile::A8W4TcgDequantParams params;
params.validK = static_cast<uint32_t>(bubKLen);
params.scaleMaskAddr = scaleMaskUb;   // 调用方 UB storage 持有的掩码

A8W4Dequant::Run<W4Gran::PER_GROUP, 32>(weightInUb, scaleUb, weightOutUb, params);
```

### int32/fp32 反量化（epilogue）

```cpp
#include "blaze/epilogue/tile/compute.h"

using B32ToFpDequant = Blaze::Epilogue::Tile::Dequant<Blaze::Epilogue::Tile::B32ToFpTag>;
using DequantMode = Blaze::Epilogue::Tile::DequantMode;

// accTensor: int32 UB (m, n, nSrcAligned)；dstTensor: bf16 UB ping-pong；
// x2ScaleTensor: per-channel 单行向量；x1ScaleTensor: fp32 per-token 向量；
// biasTensor: bf16 单行向量；禁用槽位传同 dtype 占位张量、标量传 1.0f。
B32ToFpDequant::Run<DequantMode::PER_CHANNEL, DequantMode::PER_TOKEN, true, bfloat16_t>(
    accTensor, dstTensor, x2ScaleTensor, x1ScaleTensor, biasTensor,
    x2ScaleScalar, 1.0f, mSize, nSize, nSrcAligned, nDstAligned);
```
