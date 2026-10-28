# Tile Mmad ConfidenceHeadFused

> [代码位置](../../../../include/blaze/gemm/tile/tile_trait.h)

## 功能说明

`MmadTraitConfidenceHeadGemv` 为 `asc::te::make_mmad` 提供 M=1 的原生 GEMV 配置，关闭 M 补齐，保留向量乘矩阵的计算方式。

## 特殊约束

- MMAD 的逻辑 M 为 1，N/K 为当前有效长度。
- 普通累加与首次带 BiasTable 的累加共用该 Trait。
- 特征搬运、两段 K 累加和 Fixpipe 布局由 [Cube Block](../block/block_mmad_confidence_head_fused.md) 管理；本层不处理温度和激活。

## 使用示例

```cpp
#include "blaze/gemm/tile/tile_trait.h"

auto atom = asc::te::make_mmad(
    asc::te::mmad_operation{}, Blaze::Gemm::Tile::MmadTraitConfidenceHeadGemv{});
```
