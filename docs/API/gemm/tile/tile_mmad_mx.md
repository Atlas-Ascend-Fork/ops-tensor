# Tile Mmad MX
> [代码位置](../../../../include/blaze/gemm/tile/tile_trait.h)

## 功能说明
MX 量化矩阵乘 Tile，定义 MmadTraitMX trait，用于支持 MxFP4/MxFP8 量化矩阵乘计算。

## 特殊约束

### 数据类型支持
仅支持 MxFP4 和 MxFP8 量化数据类型：
- **MxFP4**：`fp4x2_e2m1_t`、`fp4x2_e1m2_t`
- **MxFP8**：`fp8_e5m2_t`、`fp8_e4m3fn_t`

### Trait 定义
```
constexpr asc::te::mmad_trait MX_MMAD_TRAIT = asc::te::mmad_trait{0, false, false, true, asc::te::mmad_type::mx};
```

说明：
- `asc::te::mmad_type::mx`：标识 MX 量化计算模式
- 自动应用于 Mmad 计算

## 特殊类型

### MmadTraitMX
```
struct MmadTraitMX {
    using TraitType = asc::te::mmad_trait;
    static constexpr const TraitType value = MX_MMAD_TRAIT;
};
```

功能：MX Mmad Trait 定义，用于量化矩阵乘。

## 使用示例

### 在 BlockMmadMX 中使用
```
// Mmad 计算（自动使用 MmadTraitMX）
asc::te::mmad(
    asc::te::mmad_atom<
        asc::te::mmad_traits<
            asc::te::mmad_operation,
            Blaze::Gemm::Tile::MmadTraitMX>>{},
    tensorL0C, tensorAL0, tensorBL0);
```

说明：
- `MmadTraitMX` 自动应用于 MX 量化计算
- 无需手动指定 trait 参数

## 数据流

```
L0A（量化数据） + L0B（量化数据）
    ↓
Mmad（MmadTraitMX）
    ↓
L0C（float 结果）
```

说明：量化数据在 Mmad 计算时自动反量化为 float。
