# Attention Universal 基础框架
> 公共接口说明

## 概述

Attention Kernel 基础框架，提供统一的模板参数和 SFINAE 分发机制。与 Gemm 模块的 `GemmUniversal` 类似，`AttentionUniversal` 是一个主模板，通过 `BlockMmad_::DispatchPolicy::ScheduleType` 进行 SFINAE 特化选择。未匹配的 DispatchPolicy 会在编译期触发 `static_assert` 报错。

详见：[README.md](./README.md) 查看 API 清单和实现对比。

## 类模板概述

### 模板定义

```cpp
template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_, typename Enable_ = void>
class AttentionUniversal;
```

### 模板参数

| 参数 | 说明 |
|------|------|
| ProblemShape_ | 问题形状类型，通常为 `asc::te::shape<int64_t, int64_t, int64_t, int64_t>` (m, n, k, batch) |
| BlockMmad_ | BlockMmad 类，矩阵乘计算组件 |
| BlockEpilogue_ | BlockEpilogue 类，后处理组件 |
| BlockScheduler_ | BlockScheduler 类，任务调度组件 |
| Enable_ | SFINAE 使能参数，默认为 `void`，由特化版本使用 `enable_if_t` 匹配 |

### SFINAE 分发机制

主模板（未匹配）会触发编译期报错：

```cpp
template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_, typename Enable_ = void>
class AttentionUniversal {
    static_assert(Gemm::always_false_v<BlockEpilogue_> && Gemm::always_false_v<BlockMmad_>,
                  "AttentionUniversal is not implemented for this BlockEpilogue or BlockMmad");
};
```

特化版本通过 `enable_if_t<is_same_v<ScheduleType, KernelFlatQuant>>` 匹配，例如：

```cpp
template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class AttentionUniversal<ProblemShape_, BlockMmad_, BlockEpilogue_, BlockScheduler_,
                         AscendC::Std::enable_if_t<AscendC::Std::is_same_v<
                             KernelFlatQuant, typename BlockMmad_::DispatchPolicy::ScheduleType>>> {
    // FlatQuant 特化实现
};
```

### 特化版本

| ScheduleType | 特化文件 | 说明 |
|-------------|---------|------|
| `KernelFlatQuant` | [kernel_flat_quant.md](./kernel_flat_quant.md) | 双矩阵乘 + AIV MX 量化 |

## 公共约束

1. **模板参数要求**：
   - ProblemShape 必须为 `asc::te::shape<int64_t, int64_t, int64_t, int64_t>` 类型，分别表示 **m n k b** 维度大小
   - BlockMmad 必须提供 `DispatchPolicy::ScheduleType` 类型，用于 SFINAE 匹配
   - BlockEpilogue 必须与 Kernel 类型匹配
   - BlockScheduler 必须提供 tile 切分和调度功能

2. **SFINAE 匹配**：
   - 未匹配的 `ScheduleType` 会触发编译期 `static_assert` 报错
   - 匹配成功后进入对应特化实现

## 调用示例

### 组件组装模板

```cpp
using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
using DispatchPolicy = Blaze::Attention::BlockFlatQuant<>;
using BlockMmad = Blaze::Attention::Block::BlockMmad<
    DispatchPolicy, QType, LayoutQ, KType, LayoutK, VType, LayoutV, OutType, LayoutOut>;
using BlockScheduler = Blaze::Attention::Block::BlockSchedulerFlatQuant<ProblemShape>;
using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueFlatQuant<bfloat16_t, fp4x2_e2m1_t, uint8_t>;

using Kernel = Blaze::Attention::Kernel::AttentionUniversal<
    ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
```

### Kernel 执行模板

```cpp
using Params = typename Kernel::Params;
Params params;
// ... 设置参数 ...
Kernel kernel;
kernel(params);
```
