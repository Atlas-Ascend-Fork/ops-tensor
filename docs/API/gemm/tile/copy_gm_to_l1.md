# Copy GM To L1
> [代码位置](../../../../include/blaze/gemm/tile/arch35/copy_gm_to_l1.h)

说明：调用方应包含聚合头 `tile/datamove.h`（`tile/copy_gm_to_l1.h` 为兼容保留的转发入口）。

## 功能说明
A 矩阵 ND slice 非连续场景，以及 SwiGLU MX 左右 B/ScaleB 半区拼接场景的 GM->L1 搬运 Tile。

该组件将三维 GM layout `[ndNum, [sliceM, curK]]` 按 ND2NZ 方式搬运到 L1 NZ layout，GM stride 为 `[srcNdStride, [k, 1]]`。

## 特殊约束

### 调用场景
`CopySliceGM2L1` 仅用于 A 矩阵 ND slice 非连续输入。普通连续输入继续使用默认 `CopyGM2L1`。
`CopyConcatGM2L1` 则服务于 SwiGLU 的 B/ScaleB 左右半区拼接。

### 架构支持
当前实现位于 `tile/arch35/copy_gm_to_l1.h`，仅在 `__NPU_ARCH__ == 3510` 时引入。调用方应包含聚合头
`tile/datamove.h`；本头文件为兼容保留的转发入口。

### 数据类型
`CopySliceGM2L1` 支持 `half` 和 `float` 数据类型。

`CopyConcatGM2L1` 还支持 QGMM MX 使用的 FP8/FP4 B 与 E8M0 ScaleB。

## 特殊类型

### CopySliceGM2L1
```cpp
struct CopySliceGM2L1 {
    template <typename Tp, const Tp& traits, typename T, typename U>
    __aicore__ inline static void Copy(const T& dst, const U& src);
};
```

功能：将 A 矩阵 slice 后的三维 GM Tensor 搬运到 L1 Tensor。

## 使用方式

```cpp
auto copyGM2L1Slice = asc::te::make_copy(Blaze::Gemm::Tile::CopySliceGM2L1{});
asc::te::copy(copyGM2L1Slice, tensorAL1, gmTileASlice);
```

说明：
- `gmTileASlice` 的 shape 为 `[curM / sliceM, [sliceM, curK]]`
- `tensorAL1` 的 layout 为 A 矩阵 L1 NZ layout
- 该路径由 `BlockMmadMatmulBasic` 在 `NON_CONTIGUOUS_TYPE_SLICE` 场景下自动选择

### CopyConcatGM2L1

```cpp
struct CopyConcatGM2L1Params {
    uint64_t n;  // SwiGLU 拆分前完整 N
    uint64_t k;  // 完整 K
};

struct CopyConcatGM2L1 {
    // 前导 <Tp, traits> 遵循 copy_traits 检测约定（直调时用默认值），拼接路径不使用
    template <typename Tp = asc::te::gm_to_l1_trait,
              const Tp& traits = asc::te::gm_to_l1_trait_default::value,
              typename T, typename U>
    __aicore__ inline static void Copy(
        const T& dst, const U& src, const CopyConcatGM2L1Params& params);
};

```

`Copy` 通过通用 ND2NZ/DN2NZ 路径把 SwiGLU 的左右两个 N 半区拼接到同一 L1 Tensor。
B 使用完整 N/K 的 GM stride；ScaleB 仍使用 `scaleb_nd_layout_ptn` 或
`scaleb_dn_layout_ptn`，每个 K/64 分组保存两个 scale。所有调用统一经
`asc::te::copy(make_copy(CopyConcatGM2L1{}).with(params), dst, src)` 发起。

`BlockMmadQGmmMx` 的普通 NZ/ZN 双源 Tensor 使用 tensor_api `copy_gm_to_l1`
分别搬运左右半区，不经过 `CopyConcatGM2L1`；搬运前按需补零，K 已对齐时无需额外填充。
