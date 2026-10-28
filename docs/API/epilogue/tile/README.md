# Epilogue Tile 层组件概述

## API 清单

| 组件名 | 描述 |
| :----------------------------------------------------------- | :------: |
| [compute](./compute.md) | Tile 计算组件的架构派发入口，Block 层唯一的 include 路径 |
| [arch35/gelu](./arch35/gelu.md) | Tile 级 GELU 激活，支持 tanh、erf 两种算法 |
| [arch35/confidence_head_fused](./arch35/confidence_head_fused.md) | FP32 logits 的温度/clamp/sigmoid，仅公开静态 Run |
| [arch35/sigmoid](./arch35/sigmoid.md) | Tile 级 sigmoid，FP32 输入与 float/FP16/BF16 输出，源和目的可使用不同的行跨度 |
| [arch35/swish](./arch35/swish.md) | FP32 Swish → BF16 RINT，尾部补零到 32 元素边界 |
| [arch35/dequant](./arch35/dequant.md) | Tile 级反量化（双轴 API：类模板 tag 选量化对象、Run 模板 `DequantMode` 选粒度）：Fp4ToFp8Tag per-group 权重 → FP8；B32ToFpTag int32/fp32 ×scale + bias → fp32/bf16/fp16 |
| [arch35/mx_quant](./arch35/mx_quant.md) | Tile 级 MX 动态量化链（eMax → scale → FP8/FP4 量化 → 布局转置） |
| `arch35/initialize_empty_softmax.h` | BlockAttnResPrepare 空输入场景的 softmax 统计量初始化 |
| `arch35/reduce_square.h` | Tile 级 V 平方和归约，BlockAttnResPrepare 依赖 |
| `arch35/rms_softmax.h` | Tile 级 RMS-softmax 计算，BlockAttnResPrepare 依赖 |

## 核心组件关系

```
BlockEpilogue（Block 层）
    └── #include "blaze/epilogue/tile/compute.h"（架构派发入口）
            └── arch35/（dav-3510 实现）
                    ├── Gelu                     → BlockEpilogueGeluMxQuant / BlockEpilogueGeluTanhMxQuant
                    ├── Sigmoid                  → BlockEpilogueMatmulWithSigmoid
                    ├── Swish                    → BlockEpilogueSwishMxQuant
                    ├── Dequant<Fp4ToFp8Tag> → Gemm Kernel Wqmm Mix Pergroup 的 AIV prologue
                    ├── Dequant<B32ToFpTag>      → BlockEpilogueDequant
                    │                              / BlockEpilogueQbmmPertensorStreamK
                    ├── MxQuant                  → BlockEpilogueGeluMxQuant / BlockEpilogueGeluTanhMxQuant
                    │                              / BlockEpilogueSwishMxQuant / BlockEpilogueSwigluMxQuant
                    │                              / BlockEpilogueFlatQuant
                    ├── InitializeEmptySoftmax   ┐
                    ├── ReduceSquare             ├→ BlockEpilogueBlockAttnResPrepare
                    └── RmsSoftmax              ┘
```

## 使用流程

1. **选择 Tile**：在 `blaze/epilogue/tile/arch35/` 下找到所需的计算组件（如 `Gelu`）
2. **经由 compute.h 引入**：Block 层只 `#include "blaze/epilogue/tile/compute.h"`，禁止直接
   include arch35 头文件
3. **构造 Tensor**：使用 `asc::te::make_tensor` + `Gemm::MakeNDExtLayout` 构造 UB 侧
   NDExt 张量
4. **调用 Tile**：实例化 Tile 类并调用公共方法（`__aicore__` 入口自带 static_assert
   类型/排布门禁与 AIC 早退）

## 与 Block 层的关系

Tile 层是 Block 层的可复用计算组件，职责边界：
- **Tile 层**：纯粹的逐元素/逐行向量计算，不持有 UB 生命周期，不负责 AIC/AIV 跨核同步，
  也不感知 slot/ping-pong 等流水线状态
- **Block 层**：管理 UB 布局与字节偏移、跨核同步、GM 输出，将构造好的 Tensor 传给 Tile

同一 Tile 可被多个 Block 复用（如 `Gelu` 同时服务
[BlockEpilogueGeluMxQuant](../block/block_epilogue_gelu_mx_quant.md) 与
[BlockEpilogueGeluTanhMxQuant](../block/block_epilogue_gelu_tanh_mx_quant.md)）。

## 扩展规范

新增 Tile 时：
1. 实现放在 `include/blaze/epilogue/tile/arch35/<name>.h`
2. 在 `include/blaze/epilogue/tile/compute.h` 的 `defined(__NPU_ARCH__)` 守卫内注册
   （include 列表保持字母序）
3. 对外函数开头增加 static_assert 基础校验（数据类型、UB 位置、NDExt 排布）
