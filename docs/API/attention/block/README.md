# Attention/Block 类模板概述

## API 清单

### BlockMmad（矩阵乘计算）
| 组件名 | 描述 |
| :----------------------------------------------------------- | :------: |
| [block_mmad](./block_mmad.md) | BlockMmad 基础框架，SFINAE 未匹配时编译报错 |
| [block_mmad_flat_quant](./block_mmad_flat_quant.md) | FlatQuant 双矩阵乘 Block，AIC 侧完成 A×P2→L1 和 P1×temp→L0C→UB 两阶段计算 |

### BlockScheduler（任务调度）
| 组件名 | 描述 |
| :----------------------------------------------------------- | :------: |
| [block_scheduler_flat_quant](./block_scheduler_flat_quant.md) | FlatQuant 调度器，按 Batch 维度切分任务、尾块处理 |

## 公共框架

### BlockMmad 公共框架
所有 BlockMmad 组件基于 [block_mmad.md](./block_mmad.md) 公共框架实现，通过 SFINAE 对 `DispatchPolicy` 进行特化选择。

详见：[block_mmad.md](./block_mmad.md)

## 核心组件关系

```
BlockMmadFlatQuant (AIC)
    ├── DispatchPolicy (BlockFlatQuant)
    ├── Phase 1: A × P2 → L0C → Fixpipe → L1 (temp)
    ├── Phase 2: P1 × temp → L0C → Fixpipe → UB
    └── BufferManager (L1/L0/L0C 缓冲管理)

BlockSchedulerFlatQuant
    ├── Batch 维度切分
    ├── 尾块处理（mainTailBatch / mainTailBlock）
    └── Block 坐标计算
```

## 实现差异对比

| Block 类型 | 调度策略 | 计算模式 | 双矩阵乘 | L1 缓冲 | L0C 缓冲 | 输出目标 | 适用场景 |
|-----------|---------|---------|---------|---------|-----------|---------|---------|
| BlockMmadFlatQuant | BlockFlatQuant | AIC | 支持（Phase1 + Phase2） | 双缓冲 | PingPong | UB（供 AIV 消费） | FlatQuant Kernel |

## 使用流程

1. **查看公共框架**：了解模板参数和 SFINAE 机制 → [block_mmad.md](./block_mmad.md)
2. **选择具体实现**：当前提供 FlatQuant 双矩阵乘实现
3. **定义调度策略**：选择 `BlockFlatQuant`
4. **组装组件**：定义数据类型、布局类型
5. **初始化**：调用 Init 设置 tile 形状、缓冲策略
6. **执行计算**：调用 operator 执行双矩阵乘
