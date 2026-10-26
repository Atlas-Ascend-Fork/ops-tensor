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
支持 `half` 和 `float` 数据类型。

`CopyConcatGM2L1` 的通用路径还支持 QGMM MX 使用的 FP8/FP4 B 与 E8M0 ScaleB；
`TryCopyWeightNz` 快速路径仅支持物理 NZ/ZN 的 1 字节 FP8，不支持 FP4。

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
    template <typename T, typename U>
    __aicore__ inline static bool TryCopyWeightNz(
        const T& dst, const U& src, const CopyConcatGM2L1Params& params);

    template <typename T, typename U>
    __aicore__ inline static void Copy(
        const T& dst, const U& src, const CopyConcatGM2L1Params& params);
};
```

`Copy` 把 SwiGLU 的左右两个 N 半区拼接到同一 L1 Tensor。B 使用完整 N/K 的 GM stride；
ScaleB 仍使用 `scaleb_nd_layout_ptn` 或 `scaleb_dn_layout_ptn`，每个 K/64 分组保存两个 scale。

`TryCopyWeightNz` 是 NZ/ZN FP8 的可选快速路径。调用时 `src` 必须已经是左半区、当前 K window
的 slice。仅当以下条件全部满足时发起 DMA 并返回 `true`：

- 源布局为物理 NZ/ZN、元素宽度为 1 字节且不是 FP4；
- 完整 `K` 是 128 的倍数，完整 `N` 是 64 的倍数；
- 当前 K window 非空且按 128 对齐，目标 L1 K 与该 window 相等；
- 当前 N 宽度非空，NZ 按 32、ZN 按 16 对齐，且不超过 `N / 2`；
- DMA 的 burst/byte 计数通过实现中的范围检查；调用方仍须确保目标 SDK 的指令参数限制和 L1 容量约束。

任一条件不满足时，函数在发起 DMA 前返回 `false`。调用方必须保留普通 Tensor Copy 回退，
分别把 left/right slice 写入拼接 L1 的两个半区。
