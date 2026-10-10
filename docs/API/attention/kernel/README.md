# Attention/Kernel 模板概览

## API 清单

| 组件名 | 说明 |
| :--- | :---: |
| [kernel_universal](./kernel_universal.md) | AttentionUniversal 基础框架，SFINAE 未匹配时编译报错 |
| [kernel_flat_quant](./kernel_flat_quant.md) | FlatQuant 双矩阵乘 + AIV MX 量化 Kernel，AIC 完成双阶段矩阵乘后通过 L0C→UB 传递给 AIV 执行 MX FP4 量化 |

## 公共框架

所有 Kernel 组件均基于 [kernel_universal.md](./kernel_universal.md) 公共框架实现，统一包含：
- 模板参数
- 数据结构，如 `Params`
- 核心方法，如 `Init`、`operator()`

详见：[kernel_universal.md](./kernel_universal.md)

## 核心组件关系

```text
AttentionUniversal
    -> BlockSchedulerFlatQuant
    -> BlockMmadFlatQuant (AIC)
    -> BlockEpilogueFlatQuant (AIV)
```

## 实现差异

| Kernel 类型 | 计算模式 | 双矩阵乘 | 后处理 | AIC-AIV 同步 | BlockScheduler | 适用场景 |
|------------|---------|---------|--------|-------------|---------------|---------|
| KernelFlatQuant | AIC + AIV 双核 | 支持（A×P2→L1, P1×temp→UB） | MX FP4 量化 | 块内 AIC-AIV 同步（`Gemm::Sync`，PIPE_FIX/PIPE_V/PIPE_MTE3） | BlockSchedulerFlatQuant | Attention 双矩阵乘 + 在线量化 |

## 使用流程

1. **查看公共框架**：了解模板参数和核心接口 → [kernel_universal.md](./kernel_universal.md)
2. **选择具体实现**：当前提供 FlatQuant 路径
3. **查看特殊约束**：了解 AIC-AIV 同步机制和 Batch 迭代策略
4. **组装组件**：定义 ProblemShape、BlockMmad、BlockEpilogue、BlockScheduler
5. **准备参数**：构造 Params 结构体
6. **执行 Kernel**：实例化并调用 operator()
