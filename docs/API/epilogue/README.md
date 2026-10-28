# Gemm/Epilogue 类模板概述

## 开发指导

新 Epilogue 开发请参阅：[Epilogue 层编写指导](./epilogue_developer_guide.md)

## API 清单

| 组件名 | 描述 |
| :----------------------------------------------------------- | :------: |
| [block_epilogue_empty](./block/block_epilogue_empty.md) | 空后处理组件，用于不支持后处理的 Kernel |
| [block_epilogue_streamk](./block/block_epilogue_matmul_streamk.md) | StreamK 后处理组件，支持 workspace 汇聚、类型转换、ReLU |
| [block_epilogue_dequant](./block/block_epilogue_dequant.md) | MIX 模板 dequant 向量后处理，AIV 侧 int32→fp32 × scale + bias → bf16/fp16/fp32 |
| [block_epilogue_qbmm_pertensor_streamk](./block/block_epilogue_qbmm_pertensor_streamk.md) | QBMM per-tensor StreamK 专用后处理，归约 raw partial 后应用 scale/bias |
| [block_epilogue_per_token_scale](./block/block_epilogue_per_token_scale.md) | 应用 per-token scale 和可选 row-sum offset 修正 |
| [block_epilogue_finalize_routing](./block/block_epilogue_finalize_routing.md) | GroupedMatmulFinalizeRouting 通用后处理，完成 logit 融合、类型转换和 rowIndex 原子写回 |
| [block_epilogue_fmm_with_scale_add](./block/block_epilogue_fmm_with_scale_add.md) | FusedMatMul scale_add 向量后处理，执行 alpha × (x1@x2) + beta × x3 |
| [block_epilogue_confidence_head_fused](./block/block_epilogue_confidence_head_fused.md) | FP32 温度/clamp/sigmoid、K=0 bias 输入及 UB ping/pong 写回 |
| [block_epilogue_matmul_with_sigmoid](./block/block_epilogue_matmul_with_sigmoid.md) | FusedMatMul sigmoid 向量后处理，执行 sigmoid(x1@x2 + bias) |
| [block_epilogue_flat_quant](./block/block_epilogue_flat_quant.md) | FlatQuant AIV 侧 MX FP4 量化后处理，bf16→eMax→E8M0 scale→FP4 量化 |
| [block_epilogue_gelu_mx_quant](./block/block_epilogue_gelu_mx_quant.md) | GELU 激活 + 动态 MXFP8/MXFP4 量化 |
| [block_epilogue_swish_mx_quant](./block/block_epilogue_swish_mx_quant.md) | Swish 激活 + 动态 MXFP8 量化 |
| [block_epilogue_swiglu_mx_quant](./block/block_epilogue_swiglu_mx_quant.md) | SwiGLU 激活 + 动态 MXFP8 量化 |
| [block_epilogue_gelu_tanh_mx_quant](./block/block_epilogue_gelu_tanh_mx_quant.md) | GELU（tanh）激活 + 动态 MXFP8/MXFP4 量化 |
| [block_epilogue_block_attn_res_prepare](./block/block_epilogue_block_attn_res_prepare.md) | BlockAttnResPrepare AIV 后处理，执行 V 平方和、RMS-softmax、空输入处理和结果搬运 |
| [block_epilogue_iterbatch](./block/block_epilogue_iterbatch.md) | IterBatch AIV 后处理组件，双 sub-block 奇偶消费 AIC fixpipe 写入的 UB 帧并 ND 写回 GM（ND_FIXPIPE_1_2） |
| [tile](./tile/README.md) | Tile 级可复用计算组件（compute.h 架构派发 + arch35 实现：Gelu、Sigmoid、RmsSoftmax 等） |

## 公共框架

BlockEpilogue 组件遵循公共分层约定，通常包含：
- 类型别名
- 数据结构（Params）
- 与具体计算阶段匹配的核心方法。通用 GEMM 后处理通常提供 `Init`、`Run` 或 `operator()`；
  BlockAttnResPrepare 后处理提供 `ReduceV`、`FinalizeSoftmax` 和 `ProcessEmptyInput`。

详见：[block_epilogue.md](./block/block_epilogue.md)

## 核心组件关系

```
BlockEpilogue
    ├── BlockShape (Block 形状)
    ├── BlockCoord (Block 坐标)
    ├── Params (参数结构)
    └── 核心方法
            ├── Init (初始化)
            ├── Run (执行后处理)
            └── operator() (调用接口)
```

## 实现差异对比

| Epilogue 类型 | 计算内容 | 计算位置 | Workspace | 类型转换 | ReLU | 适用场景 |
|--------------|---------|---------|-----------|---------|-----|---------|
| BlockEpilogueEmpty | 无（空实现） | 无 | 不支持 | 无 | 不支持 | Basic Kernel |
| BlockEpilogueMatmulStreamK | workspace 汇聚、Add、Cast、ReLU | AIV 核 | 支持 | 支持 float → half/bf16 | 可选支持 | StreamK Kernel |
| BlockEpilogueDequant | dequant：int32→fp32 × x2Scale [× x1Scale] + bias | AIV 核 | 不支持 | 支持 fp32 → bf16/fp16/fp32 | 不支持 | QBMM MIX Kernel |
| BlockEpilogueQbmmPertensorStreamK | workspace 归约 + dequant + bias | AIV 核 | 支持 | 支持 int32/fp32 → bf16/fp16/fp32 | 不支持 | QBMM per-tensor StreamK |
| BlockEpilogueFmmWithScaleAdd | alpha × (x1@x2) + beta × x3 | AIV 核 | 不支持 | 支持 fp32 → bf16/fp16 | 不支持 | FusedMatMul scale_add Kernel |
| BlockEpilogueMatmulWithSigmoid | sigmoid(x1@x2 + bias) | AIV 核 | 不支持 | 支持 fp32 → bf16/fp16 | 不支持 | FusedMatMul sigmoid Kernel |
| BlockEpilogueFlatQuant | MX FP4 量化：eMax→E8M0 scale→FP4 quant | AIV 核 | 不支持 | 支持 bf16 → FP4(int8) | 不支持 | Attention FlatQuant Kernel |
| BlockEpilogueGeluMxQuant | GELU + MXFP8/MXFP4 量化 | AIV 核 | 不支持 | 支持 float → bf16 → FP8/FP4 | GELU | QBMM MX ActivationQuant Kernel |
| BlockEpilogueSwishMxQuant | Swish(x, β) + MXFP8 量化 | AIV 核 | 不支持 | float → Swish → bf16 → FP8 | Swish | QBMM MX ActivationQuant Kernel |
| BlockEpilogueSwigluMxQuant | SwiGLU + MXFP8 量化 | AIV 核 | 不支持 | 支持 float → bf16 → FP8 | SwiGLU | QBMM/QGMM MIX Kernel |
| BlockEpilogueGeluTanhMxQuant | GELU（tanh） + OCP/cuBLAS MXFP8 或 MXFP4 量化 | AIV 核 | 不支持 | float → GELU（tanh） → FP8/FP4 | GELU（tanh） | QGMM MX ActivationQuant MIX Kernel |
| BlockEpilogueBlockAttnResPrepare | V 平方和、RMS-softmax、空输入处理和结果搬运 | AIV 核 | 支持 | FP32 | 不支持 | BlockAttnResPrepare Kernel |
| BlockEpilogueIterbatch | 无融合计算，AIV 消费 AIC fixpipe 写入的 UB 帧，ND 搬出 GM | AIV 核 | 不支持 | 不支持 | 不支持 | IterBatch ND_FIXPIPE_1_2 Kernel |

## 使用流程

1. **查看公共框架**：了解类型别名和核心接口 → [block_epilogue.md](./block/block_epilogue.md)
2. **选择具体实现**：根据 Kernel 类型选择 Empty 或 StreamK
3. **组装组件**：在 Kernel 模板参数中定义 BlockEpilogue 类型
4. **初始化**：调用 Init 设置参数（StreamK 需要额外参数）
5. **执行后处理**：调用 Run 或 operator 执行（Empty 无实际效果）

## 设计说明

### 为什么需要 BlockEpilogueEmpty
1. **模板参数要求**：Kernel 模板需要 BlockEpilogue 参数
2. **接口一致性**：保持与其他 Epilogue 组件相同的接口
3. **扩展性**：未来可替换为实际的后处理组件
4. **零开销**：空实现不会引入额外计算开销

### BlockEpilogue 在 Kernel 中的作用
- **Basic Kernel**：使用 BlockEpilogueEmpty，无后处理
- **StreamK Kernel**：使用 BlockEpilogueMatmulStreamK，在 AIV 核执行后处理
  - 从 workspace 读取 AIC 计算的中间结果
  - 执行 Add 汇聚（K 轴切分）
  - 执行类型转换（float → half/bf16）
  - 执行可选的 ReLU 激活
  - 输出最终结果到 GM
