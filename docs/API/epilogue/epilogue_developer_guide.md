# Epilogue 层开发指导

> Epilogue 层在 AIV（向量核）上执行 MMAD 计算后的后处理：反量化、激活函数、量化、写回 GM。

## 1. 核心定位

```
AIC（Cube 核）                    AIV（向量核）
    |                                  |
    |-- MMAD 计算 → L0C               |
    |-- L0C → UB（Fixpipe）----------→|
    |                                  |-- Epilogue：读 UB → 后处理 → 写 GM
```

Epilogue 运行在 AIV 上，与 AIC 的 MMAD 计算**并行**。两者通过块内 AIC-AIV 同步标志（`Gemm::Sync`）协同。

## 2. 统一对外接口（新增文件必须遵守）

新增 `include/blaze/epilogue/block/block_epilogue_*.h` 后处理文件时，必须遵循本节规范。通用 tile 后处理的标准形式如下：

```cpp
#include "blaze/epilogue/block/block_epilogue_types.h"

class BlockEpilogueExample {
public:
    struct Params {
        GM_ADDR outGmAddr{nullptr};
        // 其他初始化参数，使用默认成员初始化。
    };

    using AccumulatorTensorParams = Blaze::Epilogue::Block::AccumulatorTensorParams;

    __aicore__ inline BlockEpilogueExample();
    __aicore__ inline void Init(const Params& params);
    __aicore__ inline auto GetAccumulatorTensor(const AccumulatorTensorParams& params) const;
    __aicore__ inline void operator()(const BlockShape& blockShape, const BlockCoord& blockCoord);

    // 仅 9.2.0 已存在的文件需要保留兼容接口；其余文件不添加旧式散参接口。
    // Compatibility interfaces retained for existing callers.

private:
    // 新增内部辅助函数、常量和状态成员。
};

// 成员函数在类外实现。
```

### 2.1 接口职责

| 接口或类型 | 是否必需 | 规范 |
|------------|----------|------|
| `Params` | 必需 | 聚合初始化参数，只保存对象初始化所需配置；成员提供默认值 |
| `Init(const Params&)` | 必需 | 一次性初始化，绑定 GM 地址、保存问题规模并配置 UB 布局 |
| `operator()` | 必需 | 每个 tile 的执行入口；Tensor、形状、坐标和偏移入参沿用具体场景的既有形式 |
| `AccumulatorTensorParams` | 接收 L0C 输出时必需 | 使用 `block_epilogue_types.h` 中的公共类型，字段为 `rows`、`cols`、`slot`、`batch` |
| `GetAccumulatorTensor(const AccumulatorTensorParams&) const` | 接收 L0C 输出时必需 | 返回接收 L0C 输出、供后处理读取的 UB Tensor，由 Epilogue 决定 UB 地址和 Layout |
| `UpdateGlobalAddr()` / `UpdateNextProblem()` | 按场景提供 | 多 batch/group 场景的状态更新接口；偏移参数优先使用 `OutputOffsets` 等聚合类型 |

`GetAccumulatorTensor` 的语义是“获取累加结果 Tensor”，不是通用的 UB 空间分配接口。该接口返回 Tensor，不要求调用方向 Epilogue 传入 UB 指针；AIC 将 L0C 结果写入该 Tensor，AIV 随后在同一地址执行后处理。不从 L0C 接收输入的 Epilogue（例如 Empty、StreamK 或专用多阶段后处理）无需提供该接口。

### 2.2 类声明和实现布局

- 成员函数统一先在类内声明，再在类外实现，默认实参只写在声明中。
- 访问级别按 `public` 在前、`private` 在后排列，新增内部辅助函数和状态集中放入 `private`；`9.2.0` 已公开的成员保持原有可见性。
- `public` 区域中，类型和参数结构之后集中放置新接口；为源码兼容保留的旧接口和数据成员统一放在新接口之后，并添加 `Compatibility interfaces` 前缀的注释。
- `9.2.0` 已有文件保持既有 `operator()` 入参不变；本次重构不引入 `ExecParams`，初始化与 UB Tensor 获取通过编译期检测适配。
- 头文件可能被单独包含，必须自行包含所需依赖，不能依赖其他头文件的包含顺序。
- 新增头文件必须登记到 `classify_rule.yaml`，并补充对应 API 文档。

专用多阶段后处理可以保留 `ReduceV`、`FinalizeSoftmax` 等业务语义明确的公开方法，但仍须遵守参数聚合、接口分组和类外实现规则。

## 3. 主要 Epilogue 类型

| 类型 | 代表 | 数据流 |
|------|------|--------|
| **空操作** | `BlockEpilogueEmpty` | 无（占位符） |
| **Fixpipe 直写** | `BlockEpilogueFixpipe` | UB → 可选 ReLU → GM |
| **反量化** | `BlockEpilogueDequant` | int32→fp32→×scale→+bias→cast→GM |
| **MX 量化+激活** | `BlockEpilogueGeluMxQuant` | fp32→GELU→bf16→MaxExp→Scale→FP8/FP4→GM |
| **SwiGLU+MX 量化** | `BlockEpilogueSwigluMxQuant` | fp32→SwiGLU→bf16→MX 量化→GM |
| **Swish+MX 量化** | [BlockEpilogueSwishMxQuant](./block/block_epilogue_swish_mx_quant.md) | fp32→Swish(x, β)→bf16→MXFP8 量化→GM，保留完整 N |
| **StreamK 归约** | `BlockEpilogueMatmulStreamK` | workspace GM→UB→K 轴 Add→cast→GM |
| **Per-TOKEN Scale** | `BlockEpiloguePerTokenScale` | UB→×perTokenScale→cast→GM |
| **Scale+Add** | `BlockEpilogueFmmWithScaleAdd` | UB→×alpha+beta×x3→GM |

## 4. 关键实现模式

### 4.1 AIC/AIV 同步（Fixpipe 路径）

AIC 写 L0C→UB 后设 flag，AIV 等 flag 后读 UB 处理：

```cpp
// AIV 侧
Gemm::Sync::WaitForCube<AIC_SYNC_AIV_MODE_4, PIPE_V>(AIC_SYNC_AIV_FLAG + slot);
// ... 处理 UB 数据 ...
Gemm::Sync::NotifyCube<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIV_SYNC_AIC_FLAG + slot);
```

### 4.2 UB 布局

大多数 Epilogue 手动按字节偏移划分 UB：

```cpp
// Init 中计算各区域偏移
uint64_t inputOffset = 0;
uint64_t outputOffset = inputOffset + inputSize;
uint64_t scaleOffset = outputOffset + outputSize;
// 通过 MakeMemPtr<UB>(offset) 创建 tensor
```

### 4.3 VF 向量计算

Epilogue 的核心计算通过 `__simd_vf__` 函数在向量单元执行：

```cpp
static __simd_vf__ inline void DequantVf(__ubuf__ float* input, __ubuf__ float* output, ...) {
    // RegTensor 寄存器操作：Cast, Mul, Add, Exp, Reduce...
}

// 调用
asc_vf_call<DequantVf>(inputAddr, outputAddr, ...);
```

### 4.4 Ping-Pong 输出

M 维度分块处理，奇偶块交替使用 UB 的两个输出区域，使向量计算和 MTE3 写回重叠：

```cpp
for (int mChunk = 0; mChunk < mChunks; ++mChunk) {
    int outOffset = (mChunk & 1) ? pongOffset_ : pingOffset_;
    // 计算写入 outOffset，上一块的 MTE3 同时写另一块
}
```

### 4.5 变体分支（`if constexpr`）

```cpp
if constexpr (DispatchPolicy::FUSED_OP_TYPE == OP_TYPE_RELU) { /* 启用 ReLU */ }
if constexpr (is_same_v<OutType, bfloat16_t>) { /* 跳过 ReLU（硬件限制） */ }
if constexpr (is_same_v<OutType, fp8_e4m3fn_t>) { /* FP8 量化路径 */ }
if constexpr (is_same_v<OutType, fp4x2_e2m1_t>) { /* FP4 量化路径 */ }
```

## 5. 与 Kernel 层的交互

Kernel 通过模板参数注入 Epilogue：

```cpp
// Kernel 模板
template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class GemmUniversal { ... };

// Kernel 内部
BlockEpilogue epilogueOp;
epilogueOp.Init(params.epilogueParams);

// L0C → UB 路径：由 Epilogue 返回接收累加结果的 Tensor。
auto ubC = epilogueOp.GetAccumulatorTensor({baseM, baseN});
blockMmad(gmA, gmB, ubC, blockShape);

// AIV 侧沿用具体 Epilogue 的既有散参形式执行后处理。
epilogueOp(blockShape, offsetC, splitM, baseM, baseN);
```

`9.2.0` 已有 Kernel 通过 `block_epilogue_compat.h` 检测单参数 `Init` 和 `GetAccumulatorTensor`：新 Epilogue 使用聚合初始化和 Epilogue 提供的 UB Tensor，旧自定义 Epilogue 沿用原 UB 构造与初始化方式。所有路径的 `operator()` 入参保持不变。`BlockEpilogueMatmulStreamK` 和 `BlockEpilogueQbmmPertensorStreamK` 暂保留原有的七参数 `Init`，由对应 StreamK Kernel 直接调用。

接入 [QBMM MX Activation Quant Kernel](../gemm/kernel/kernel_qbmm_mx_activation_quant.md#输入-ub-复用通知的所有权) 时，
`ConcatN=false` 的 Epilogue 须自行调用 `NotifyCube()`；`ConcatN=true` 时由 Kernel 在 Epilogue 返回后通知。
每个 AIV/tile 须恰好通知一次，包括零行分支。输出和 scale 的搬出由 Epilogue 自行同步。

## 6. 检查清单

- [ ] 提供 `Params`、`Init(const Params&)` 和与具体场景匹配的 `operator()`，不引入 `ExecParams`
- [ ] 接收 L0C 输出时提供 `GetAccumulatorTensor`，不向调用方索要 UB 指针
- [ ] 成员函数类内声明、类外实现，`public` 在前、`private` 在后，`9.2.0` 已公开成员保持原有可见性
- [ ] 新接口集中放置，既有兼容接口和数据成员置于其后并添加兼容注释
- [ ] 头文件可独立包含，新增头文件已登记到 `classify_rule.yaml`
- [ ] UB 布局在 `Init` 中一次性计算
- [ ] 块内 AIC-AIV 同步 flag ID 与 AIC 侧一致
- [ ] VF 函数用 `__simd_vf__`，可组合子函数用 `__simd_callee__`
- [ ] 向量写操作后 `LocalMemBar` 保证可见性
- [ ] `static_assert` 校验元素类型、内存位置、Layout
