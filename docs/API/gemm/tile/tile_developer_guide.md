# Tile 层编写指导

> Tile 层是 Blaze 三层架构（Kernel → Block → Tile）中最底层的组件，提供细粒度的数据搬运和计算原语。本文档以 `CopySliceGM2L1` 等现有 Tile 为参考，介绍如何编写一个新的 Tile。

## 1. Tile 分类

| 类别 | 典型代表 | 注册方式 | 调用方式 | `.with()` |
|------|---------|---------|---------|-----------|
| **Copy Tile（委托基类式）** | `CopySliceGM2L1`, `CopyGM2UBWeight` | CopyTraits 继承框架基类 | `MakeCopy → Copy` | 否 |
| **Copy Tile（自定义式）** | `CopyConcatGM2L1`, `CopyMatmulSliceGM2L1` | CopyTraits 完全自定义 | `MakeCopy → .with(params) → Copy` | 是 |
| **Compute Tile** | `PadMxKAL1`, `FillUb`, `ScaleMxBias` | 无 | `X::Method(args...)` | 否 |
| **Trait-Only Tile** | `MmadTraitMX`, `CopyL0C2GMTraitRelu` | MmadTraits 特化 | 作为模板参数传入 `MmadAtom` | 通过 MmadAtom 的 `.with()` |

> Epilogue Tile（`ReduceSquare`、`RmsSoftmax`、`Gelu` 等）的编写模式请参阅 [Epilogue 层编写指导](../../epilogue/epilogue_developer_guide.md)。

## 2. 文件结构与统一 Include

### 2.1 目录结构

```
include/blaze/gemm/tile/
├── compute.h              # 统一入口：计算/变换类 Tile
├── datamove.h             # 统一入口：数据搬运类 Tile
├── tile_trait.h           # Trait 定义（独立）
├── copy_gm_to_l1.h        # 以下 5 个为兼容保留的转发头，仅 include 聚合头；
├── copy_gm_to_ub.h        # 新代码请勿包含它们，也不得再新增同类转发头
├── copy_weight_ub_to_l1.h
├── pad_mx_kl1.h
├── shift_w4_to_w8.h
└── arch35/                # 架构实现
    ├── copy_gm_to_l1.h
    ├── copy_gm_to_ub.h
    ├── copy_mx_scale.h
    ├── copy_ub_to_l1.h
    ├── copy_weight_ub_to_l1.h
    ├── fill_ub.h
    ├── pad_mx_kl1.h
    ├── scale_mx_bias.h
    └── shift_w4_to_w8.h
```

### 2.2 统一 Include 模式

Tile 层对外提供两个统一入口头文件，**使用者只需 include 这两个文件**，无需关心具体架构：

```cpp
// 数据搬运类（CopyGM2L1, CopyGM2UB, CopyUB2L1, CopyMxScale 等）
#include "blaze/gemm/tile/datamove.h"

// 计算/变换类（PadMxKL1, FillUb, ScaleMxBias, ShiftW4ToW8 等）
#include "blaze/gemm/tile/compute.h"
```

每个统一入口内部按架构条件转发：

```cpp
// compute.h 示例
#pragma once
#if (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510)
#include "blaze/gemm/tile/arch35/fill_ub.h"
#include "blaze/gemm/tile/arch35/pad_mx_kl1.h"
#include "blaze/gemm/tile/arch35/scale_mx_bias.h"
#include "blaze/gemm/tile/arch35/shift_w4_to_w8.h"
#endif
```

### 2.3 新增 Tile 的注册步骤

1. 在 `arch35/` 下创建实现文件
2. 根据类别将其 `#include` 添加到 `compute.h` 或 `datamove.h`
3. 无需创建单独的顶层转发头文件（旧有的 `copy_gm_to_l1.h` 等仅为向后兼容保留）

## 3. Copy Tile 编写

Copy Tile 的核心是定义一个包含 `static void Copy(...)` 的结构体，并注册 `CopyTraits` 特化将其接入框架。根据是否需要调用方传入运行时参数，CopyTraits 的注册方式分为两种：**委托基类式**和**自定义式**。

### 3.1 两种 CopyTraits 注册模式对比

| | 委托基类式 | 自定义式 |
|---|---|---|
| **代表** | `CopySliceGM2L1`, `CopyGM2UBWeight` | `CopyConcatGM2L1`, `CopyMatmulSliceGM2L1` |
| **CopyTraits 实现** | 继承框架 4 参数基类 | 完全自定义，不继承基类 |
| **`.with()` 支持** | 不支持 | 支持 |
| **`CopyUnpack`** | 由框架基类提供 | 自行实现，转发存储的参数 |
| **额外参数传递** | 函数入参直接传 | 通过 `.with()` 存储，`CopyUnpack` 转发 |
| **Copy 方法签名** | `<Tp, traits, T, U>` 标准 4 参数 | `<T, U>` + params 结构体 |
| **适用场景** | 不需要运行时参数，或参数可从 Layout 推导 | 需要调用方显式传入运行时参数 |

### 3.2 定义 Tile 结构体（两种模式共用）

以 `CopySliceGM2L1` 为参考模板，展示 Tile 结构体的通用编写方式：

```cpp
#include "tensor_api/tensor.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"

namespace Blaze::Gemm::Tile {
using AscendC::Te::C0_ELEMENT;

struct CopySliceGM2L1 {
    template <typename Tp, const Tp& traits, typename T, typename U>
    __aicore__ inline static void Copy(const T& dst, const U& src)
    {
        using srcType = typename U::elementType;
        auto layoutGm = src.Layout(); // shape: [ndNum, [sliceM, curK]], stride: [oriM * k, [k, 1]]
        auto layoutL1 = dst.Layout(); // l1 shape: [mL1, kL1] ==> NZ: ((m0, m1), (k0, k1))

        // 1. 从 layout 提取 shape 信息（使用 MNK_M 等语义化常量）
        auto m0 = AscendC::Te::Get<MNK_M>(AscendC::Te::Get<MNK_M>(layoutL1.Shape()));
        auto m1 = AscendC::Te::Get<1>(AscendC::Te::Get<MNK_M>(layoutL1.Shape()));
        uint32_t mL1 = m1 * m0; // curML1
        uint16_t ndNum = static_cast<uint16_t>(AscendC::Te::Get<0>(layoutGm.Shape()));
        uint16_t nValue = static_cast<uint16_t>(AscendC::Te::Get<0>(AscendC::Te::Get<1>(layoutGm.Shape())));
        uint32_t dValue = static_cast<uint32_t>(AscendC::Te::Get<1>(AscendC::Te::Get<1>(layoutGm.Shape())));

        // 2. 计算 stride
        uint64_t srcDValue = AscendC::Te::Get<0>(AscendC::Te::Get<1>(layoutGm.Stride()));
        uint32_t dstNzC0Stride = AscendC::Std::ceil_align(mL1, AscendC::BLOCK_CUBE);
        uint32_t dstNzMatrixStride = nValue * C0_ELEMENT<srcType>;
        uint8_t cacheMode = src.Engine().GetCacheMode();

        // 3. 按数据类型分发到底层 DMA 指令
        if constexpr (sizeof(srcType) == sizeof(half)) {
            CopyGmToCbufMultiNd2nz((__cbuf__ half*)(dst.Data().Get()),
                                   (__gm__ half*)(src.Data().Get()), ndNum, ...);
        } else if constexpr (sizeof(srcType) == sizeof(float)) {
            CopyGmToCbufMultiNd2nz((__cbuf__ float*)(dst.Data().Get()),
                                   (__gm__ float*)(src.Data().Get()), ndNum, ...);
        }
    }

private:
    template <typename T>
    __aicore__ inline static void CopyGmToCbufMultiNd2nz(__cbuf__ T* dst, __gm__ T* src,
                                                          uint16_t ndNum, ...)
    {
        if ASCEND_IS_AIV { return; }  // MTE2 操作仅在 Cube 核执行
        if constexpr (AscendC::Te::CURRENT_ARCH_VERSION == AscendC::Te::ArchVersion::V3510) {
            // 配置 MTE2 NZ 参数并调用 DMA 指令
            uint64_t mte2NzPara = /* 组装 loop4/loop3/loop2 dst stride + ndNum */;
            AscendC::Te::SetMTE2NzPara(mte2NzPara);
            asc_copy_gm2l1_nd2nz(dst, src, ...);
        }
    }
};

} // namespace Blaze::Gemm::Tile
```

**关键要点：**

- **模板签名**：委托基类式使用 `<typename Tp, const Tp& traits, typename T, typename U>` 标准 4 参数；自定义式只需 `<typename T, typename U>` + params 参数
- **Layout 信息从 Tensor 运行时提取**：通过 `.Layout()` 获取 shape/stride，不硬编码。NZ layout 的 M 轴使用 `MNK_M` 语义常量索引
- **流水线守卫**：MTE2/MTE3 搬运操作需要 `if ASCEND_IS_AIV { return; }` 跳过 Vector 核
- **架构守卫**：使用 `if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3510)` 保护架构相关代码
- **MTE2 NZ 参数组装**：通过位运算将 loop2/3/4 dst stride 和 ndNum 打包到 64 位 `mte2NzPara`，调用 `SetMTE2NzPara` 写入 CCE 寄存器

### 3.3 委托基类式 CopyTraits（不支持 `.with()`）

CopyTraits 通过继承框架的 4 参数基类完成注册，框架自动提供 `CopyUnpack` 等机制。适用于**不需要调用方传入额外运行时参数**的场景。

```cpp
namespace AscendC {
namespace Te {

// 泛型 Traits 委托：支持用户传入自定义 Traits
template <typename Traits>
struct CopyTraits<Blaze::Gemm::Tile::CopySliceGM2L1, Traits>
    : public CopyTraits<Blaze::Gemm::Tile::CopySliceGM2L1, Traits,
                        Blaze::Gemm::Tile::CopySliceGM2L1, Traits> {};

// 默认 Traits 特化：无参调用时使用
template <>
struct CopyTraits<Blaze::Gemm::Tile::CopySliceGM2L1>
    : public CopyTraits<Blaze::Gemm::Tile::CopySliceGM2L1, CopyGM2L1TraitDefault> {};

} // namespace Te
} // namespace AscendC
```

**工作原理：** 框架基类 `CopyTraits<Tile, Traits, Tile, Traits>` 内部已实现 `CopyUnpack`，它会直接以标准 4 参数形式调用 `Tile::Copy<Tp, traits>(dst, src)`，不需要额外的参数转发。

**调用方式：**

```cpp
auto copyGM2L1Slice = AscendC::Te::MakeCopy(Blaze::Gemm::Tile::CopySliceGM2L1{});
AscendC::Te::Copy(copyGM2L1Slice, tensorAL1, gmTileA);
```

**Traits 基类选择**（根据搬运方向）：

| 搬运方向 | 默认 Trait 基类 |
|---------|----------------|
| GM → L1 | `CopyGM2L1TraitDefault` |
| GM → UB | `CopyGM2UBTraitDefault` |
| UB → L1 | `CopyUB2L1TraitDefault` |
| L0C → UB | `CopyL0C2UBTraitDefault` |
| L0C → GM | `CopyL0C2GMTraitDefault` |

### 3.4 自定义式 CopyTraits（支持 `.with()` 传参）

当 Tile 需要调用方**显式传入运行时参数**时，CopyTraits 必须完全自定义：自行定义 `.with()` 存储参数、`CopyUnpack` 转发参数。

自定义式内部根据参数形式又分为两种子模式：

| | 结构体参数式 | 标量参数式 |
|---|---|---|
| **代表** | `CopyConcatGM2L1` | `CopyMatmulSliceGM2L1` |
| **参数形式** | Params 结构体（`n`, `k`） | 标量（`sliceM`） |
| **Copy 签名** | `<T, U>` + params（去掉 Tp/traits） | `<Tp, traits, T, U>` + 标量（保留标准 4 参数） |
| **CopyUnpack** | 显式 `<T, U>` 类型 | 可变参数包 `Args...` 转发 |
| **CopyTraits 特化** | `template <>` 默认特化 | `template <>` 针对特定 TraitDefault 特化 |

#### 模式 A：结构体参数式 — `CopyConcatGM2L1`

**步骤 1：定义 Params 结构体**

```cpp
namespace Blaze::Gemm::Tile {

struct CopyConcatGM2L1Params {
    uint64_t n;
    uint64_t k;
};

} // namespace Blaze::Gemm::Tile
```

**步骤 2：Tile 结构体 — Copy 方法去掉 Tp/traits，直接接受 params**

```cpp
struct CopyConcatGM2L1 {
    template <typename T, typename U>
    __aicore__ inline static void Copy(const T& dst, const U& src,
                                       const CopyConcatGM2L1Params& params)
    {
        using SrcLayoutPtn = AscendC::Te::GetLayoutPattern<typename U::layoutType>;
        constexpr bool isScaleB = AscendC::Std::is_one_of_v<SrcLayoutPtn,
            AscendC::Te::ScaleBNDLayoutPtn, AscendC::Te::ScaleBDNLayoutPtn>;
        if constexpr (isScaleB) {
            CopyScaleB(dst, src, params);
        } else {
            CopyB(dst, src, params);
        }
    }

private:
    // ...内部实现省略
};
```

注意：`Copy` 方法签名**去掉了** `Tp`/`traits` 模板参数，因为 `CopyUnpack` 会手动解包并转发。

**步骤 3：CopyTraits — `template <>` 默认特化，CopyUnpack 显式类型**

```cpp
namespace AscendC {
namespace Te {

template <>
struct CopyTraits<Blaze::Gemm::Tile::CopyConcatGM2L1> {
    using TraitType = typename CopyGM2L1TraitDefault::TraitType;
    static constexpr const TraitType defaultTrait = CopyGM2L1TraitDefault::value;

    __aicore__ inline constexpr CopyTraits with(
        const Blaze::Gemm::Tile::CopyConcatGM2L1Params& copyParams) const
    {
        return {copyParams};
    }

    template <const TraitType& trait = defaultTrait, typename T, typename U>
    __aicore__ inline void CopyUnpack(const T& dst, const U& src) const
    {
        (void)trait;
        Blaze::Gemm::Tile::CopyConcatGM2L1::Copy(dst, src, params);
    }

    Blaze::Gemm::Tile::CopyConcatGM2L1Params params{};
};

} // namespace Te
} // namespace AscendC
```

**步骤 4：调用方式**

```cpp
auto copyConcatGM2L1 = AscendC::Te::MakeCopy(Blaze::Gemm::Tile::CopyConcatGM2L1{});
AscendC::Te::Copy(
    copyConcatGM2L1.with(Blaze::Gemm::Tile::CopyConcatGM2L1Params{n_, k_}),
    tensorBL1, gmBlockB);
```

#### 模式 B：标量参数式 — `CopyMatmulSliceGM2L1`

与模式 A 的关键区别：Copy 方法**保留标准 4 参数签名** `<Tp, traits, T, U>`，额外参数是标量而非结构体，CopyUnpack 使用**可变参数包**转发。

**步骤 1：Tile 结构体 — Copy 方法保留标准 4 参数 + 额外标量**

```cpp
namespace Blaze::Gemm::Tile {

struct CopyMatmulSliceGM2L1 {
    template <typename Tp, const Tp& traits, typename T, typename U>
    __aicore__ inline static void Copy(const T& dst, const U& src, uint64_t sliceM = 8)
    {
        using srcType = typename T::elementType;
        auto layoutGm = src.Layout();
        auto layoutL1 = dst.Layout();

        // 使用 sliceM 计算 ndNum、nValue 等
        auto m0 = AscendC::Te::Get<0>(AscendC::Te::Get<0>(layoutL1.Shape()));
        auto m1 = AscendC::Te::Get<1>(AscendC::Te::Get<0>(layoutL1.Shape()));
        uint32_t mL1 = m1 * m0;
        uint16_t ndNum = mL1 / sliceM;
        uint16_t nValue = sliceM;
        uint32_t dValue = /* curKL1 from layout */;

        // ... 计算 stride 并调用 DMA 指令（与 CopySliceGM2L1 类似）
    }
};

} // namespace Blaze::Gemm::Tile
```

**步骤 2：CopyTraits — 针对特定 TraitDefault 特化，CopyUnpack 用可变参数包**

```cpp
namespace AscendC {
namespace Te {

template <>
struct CopyTraits<Blaze::Gemm::Tile::CopyMatmulSliceGM2L1,
                  AscendC::Te::CopyGM2L1TraitDefault> {
    using TraitType = typename AscendC::Te::CopyGM2L1TraitDefault::TraitType;
    static constexpr const TraitType defaultTrait =
        AscendC::Te::CopyGM2L1TraitDefault::value;

    template <const TraitType& trait = defaultTrait, typename... Args>
    __aicore__ inline void CopyUnpack(const Args&... args) const
    {
        Blaze::Gemm::Tile::CopyMatmulSliceGM2L1::Copy<TraitType, trait, Args...>(
            args..., sliceM);
    }

    uint64_t sliceM = 0;
};

// 无参版本：继承 4 参数基类，使 MakeCopy(X{}) 不带 .with() 也能编译
template <>
struct AscendC::Te::CopyTraits<Blaze::Gemm::Tile::CopyMatmulSliceGM2L1>
    : public CopyTraits<
          AscendC::Te::CopyGM2L1, AscendC::Te::CopyGM2L1TraitDefault,
          Blaze::Gemm::Tile::CopyMatmulSliceGM2L1,
          AscendC::Te::CopyGM2L1TraitDefault> {
};

} // namespace Te
} // namespace AscendC
```

注意与模式 A 的区别：
- 特化目标是 `<CopyMatmulSliceGM2L1, CopyGM2L1TraitDefault>`（针对特定 Trait），而非 `<CopyMatmulSliceGM2L1>`（默认特化）
- `CopyUnpack` 使用 `typename... Args` 可变参数包，而非显式 `<T, U>` 类型
- 额外提供一个无参的默认特化继承 4 参数基类，使不带 `.with()` 的调用也能编译（此时 `sliceM` 使用默认值 8）

**步骤 3：调用方式**

```cpp
// 调用方先 Slice GM Tensor，再通过 .with(sliceM) 传入参数
auto gmTileA = gmA.Slice(
    AscendC::Te::MakeCoord(0, AscendC::Te::MakeCoord(0, iter0 * kL1_)),
    AscendC::Te::MakeShape(1, AscendC::Te::MakeShape(curM, curKL1)));

auto copyGM2L1Slice = AscendC::Te::MakeCopy(Blaze::Gemm::Tile::CopyMatmulSliceGM2L1{});
AscendC::Te::Copy(copyGM2L1Slice.with(sliceM_), tensorAL1, gmTileA);
```

#### `.with()` 的工作流程（两种模式通用）

```
调用方 .with(params)
  → CopyTraits 拷贝一份新实例，存储 params/sliceM
    → 框架调用 CopyUnpack(dst, src)
      → CopyUnpack 将存储的参数转发给 Tile::Copy(...)
```

#### 机制差异详解（FAQ）

开发新 Tile 时，容易对模式 A 和模式 B 的写法差异产生困惑。以下逐一解答。

##### Q1：为什么模式 B（`CopyMatmulSliceGM2L1`）无需自己定义 `with()`，而模式 A（`CopyConcatGM2L1`）必须自己写？

**答案：取决于默认特化是否继承了框架 4 参数基类。**

模式 B 的默认特化继承了框架 4 参数基类，`with()` 由基类提供：

```cpp
// 模式 B：继承 4 参数基类 → 基类已实现 with()，无需自己写
template <>
struct AscendC::Te::CopyTraits<Blaze::Gemm::Tile::CopyMatmulSliceGM2L1>
    : public CopyTraits<
          AscendC::Te::CopyGM2L1, AscendC::Te::CopyGM2L1TraitDefault,
          Blaze::Gemm::Tile::CopyMatmulSliceGM2L1,
          AscendC::Te::CopyGM2L1TraitDefault> {
};
```

完整调用链：

```
MakeCopy(CopyMatmulSliceGM2L1{})
  → 命中默认特化（继承基类，基类提供 with()）
    → .with(sliceM_)
      → 基类 with() 返回带 sliceM 的 <CopyMatmulSliceGM2L1, CopyGM2L1TraitDefault> 实例
        → 框架调用该实例的 CopyUnpack(dst, src)
          → CopyUnpack 将 sliceM 追加到参数末尾，转发给 Copy<Tp, trait, T, U>(dst, src, sliceM)
```

模式 A 的默认特化**没有继承任何基类**，是完全自定义的，所以必须自己定义 `with()`：

```cpp
// 模式 A：没有继承基类 → 必须自己写 with()
template <>
struct CopyTraits<Blaze::Gemm::Tile::CopyConcatGM2L1> {
    // 自己定义 with()
    __aicore__ inline constexpr CopyTraits with(
        const CopyConcatGM2L1Params& copyParams) const
    {
        return {copyParams};
    }
    // 自己定义 CopyUnpack
    template <const TraitType& trait = defaultTrait, typename T, typename U>
    __aicore__ inline void CopyUnpack(const T& dst, const U& src) const
    {
        (void)trait;
        CopyConcatGM2L1::Copy(dst, src, params);
    }
    CopyConcatGM2L1Params params{};
};
```

**核心规则：继承基类 = 基类提供 `with()`；不继承 = 自己写 `with()`。**

##### Q2：为什么模式 A 的 Copy 去掉了 `Tp`/`traits`，而模式 B 保留了？

**答案：取决于 `CopyUnpack` 是否将 trait 转发给 `Copy`。**

模式 A 的 `CopyUnpack` **丢弃了 trait**，直接调 `Copy(dst, src, params)`：

```cpp
// 模式 A：CopyUnpack 消费了 trait，不传给 Copy
template <const TraitType& trait = defaultTrait, typename T, typename U>
__aicore__ inline void CopyUnpack(const T& dst, const U& src) const
{
    (void)trait;  // ← trait 在这里被丢弃
    CopyConcatGM2L1::Copy(dst, src, params);  // ← Copy 只收到 (dst, src, params)
}
```

`Copy` 不需要 `Tp`/`traits`，因为 `CopyUnpack` 已经帮它消费了 trait。

模式 B 的 `CopyUnpack` **显式传递了 `Tp`/`traits`** 给 `Copy`：

```cpp
// 模式 B：CopyUnpack 将 trait 转发给 Copy
template <const TraitType& trait = defaultTrait, typename... Args>
__aicore__ inline void CopyUnpack(const Args&... args) const
{
    // ← 显式传 <TraitType, trait>
    CopyMatmulSliceGM2L1::Copy<TraitType, trait, Args...>(args..., sliceM);
}
```

`Copy` 必须保留 `Tp`/`traits`，因为 `CopyUnpack` 会转发给它。

**核心规则：`CopyUnpack` 丢弃 trait → `Copy` 不需要 `Tp`/`traits`；`CopyUnpack` 转发 trait → `Copy` 必须保留 `Tp`/`traits`。**

##### 总结对照表

| | 模式 A：`CopyConcatGM2L1` | 模式 B：`CopyMatmulSliceGM2L1` |
|---|---|---|
| 默认特化 | 不继承基类，完全自定义 | 继承 4 参数基类 |
| `with()` 来源 | 自己定义 | 基类提供 |
| `CopyUnpack` 对 trait | `(void)trait` 丢弃 | 显式传给 `Copy<Tp, trait, ...>` |
| `Copy` 模板参数 | `<T, U>`（去掉 `Tp`/`traits`） | `<Tp, traits, T, U>`（保留标准 4 参数） |
| `CopyUnpack` 参数 | 显式 `<T, U>` 类型 | 可变参数包 `Args...` |

### 3.5 如何选择注册模式

```
需要调用方传入运行时参数？
├── 否 → 委托基类式（3.3）
│         CopyTraits 继承框架基类，代码量少
│
└── 是 → 自定义式（3.4）
          CopyTraits 完全自定义，实现 .with() + CopyUnpack
          ├── 参数是结构体（如 n, k）→ 模式 A：Copy 去掉 Tp/traits，
          │   CopyUnpack 显式 <T, U> 类型
          └── 参数是标量（如 sliceM）→ 模式 B：Copy 保留标准 4 参数，
              CopyUnpack 用 Args... 可变参数包转发
```

## 4. Compute Tile 编写

Compute Tile 不需要注册 CopyTraits，直接通过静态方法调用。以 `PadMxKAL1` 为参考。

### 4.1 基本结构

```cpp
namespace Blaze::Gemm::Tile {

struct PadMxKAL1 {
    template <typename T, typename U>
    __aicore__ inline static void PadZero(const T& tensorL1, const U& tensorGm)
    {
        using type = typename T::elementType;
        static_assert(IsMxFp4<type>() || IsMxFp8<type>(), "Only support mxfp4/mxfp8!");

        auto layoutL1 = tensorL1.Layout();
        auto layoutGm = tensorGm.Layout();

        // 提取 K 轴维度
        auto kAxis = /* 从 layoutGm 提取 */;
        auto kAxisL1Align = /* 从 layoutL1 提取 */;

        if (kAxis == kAxisL1Align) { return; }

        // 使用 Slice 定位需要补零的区域
        auto sliceTensor = tensorL1.Slice(
            AscendC::Te::MakeCoord(0, kAxisND2NZAlign),
            AscendC::Te::MakeShape(mAlign, kAxisL1Align - kAxisND2NZAlign));

        // 调用底层 fill 指令
        asc_fill_l1((__cbuf__ half*)sliceTensor.Data().Get(), half(0), config);
    }
};

} // namespace Blaze::Gemm::Tile
```

### 4.2 调用方式

```cpp
Blaze::Gemm::Tile::PadMxKAL1::PadZero(tensorAL1, gmTileA);
```

## 5. Trait-Only Tile 编写

仅定义 constexpr Trait 值，不包含任何方法实现。以 `MmadTraitMX` 为参考。

```cpp
namespace Blaze::Gemm::Tile {

constexpr AscendC::Te::MmadTrait MX_MMAD_TRAIT = AscendC::Te::MmadTrait{
    0, false, false, true, AscendC::Te::MmadType::MX};

struct MmadTraitMX {
    using TraitType = AscendC::Te::MmadTrait;
    static constexpr const TraitType value = MX_MMAD_TRAIT;
};

} // namespace Blaze::Gemm::Tile
```

调用方式（作为模板参数传入 MmadAtom）：

```cpp
AscendC::Te::Mmad(
    AscendC::Te::MmadAtom<
        AscendC::Te::MmadTraits<AscendC::Te::MmadOperation,
                                Blaze::Gemm::Tile::MmadTraitMX>>{}.with(mmadParams),
    tensorL0C, tensorAL0, tensorBL0);
```

## 6. Layout 信息提取速查

Tile 内部通过 Tensor 的 `.Layout()` 获取运行时元信息：

```cpp
auto layout = tensor.Layout();

// NZ Layout shape: ((m0, m1), (k0, k1))
auto m0 = AscendC::Te::Get<0>(AscendC::Te::Get<0>(layout.Shape()));
auto m1 = AscendC::Te::Get<1>(AscendC::Te::Get<0>(layout.Shape()));
auto k0 = AscendC::Te::Get<0>(AscendC::Te::Get<1>(layout.Shape()));
auto k1 = AscendC::Te::Get<1>(AscendC::Te::Get<1>(layout.Shape()));

// ZN Layout shape: ((k0, k1), (n0, n1))
auto k0 = AscendC::Te::Get<0>(AscendC::Te::Get<0>(layout.Shape()));
auto n1 = AscendC::Te::Get<1>(AscendC::Te::Get<1>(layout.Shape()));

// 3D ND Layout shape: (ndNum, (sliceM, curK))
auto ndNum = AscendC::Te::Get<0>(layout.Shape());
auto sliceM = AscendC::Te::Get<0>(AscendC::Te::Get<1>(layout.Shape()));
auto curK = AscendC::Te::Get<1>(AscendC::Te::Get<1>(layout.Shape()));

// 总行/列维度（自动展平）
uint64_t totalRow = AscendC::Te::GetTotalRowShape(layout);
uint64_t totalCol = AscendC::Te::GetTotalColumnShape(layout);

// Stride 提取方式与 Shape 相同
auto stride0 = AscendC::Te::Get<0>(layout.Stride());

// Layout Pattern 编译期判断
constexpr bool isNZ = AscendC::Te::IsSatisfiedPtnFormatV<Tensor, AscendC::Te::NZLayoutPtn>;
constexpr bool isZN = AscendC::Te::IsSatisfiedPtnFormatV<Tensor, AscendC::Te::ZNLayoutPtn>;

// 内存位置
using MemLoc = AscendC::Te::GetMemLocation<Tensor>;
constexpr bool isUB = AscendC::Std::is_same_v<MemLoc, AscendC::Te::Location::UB>;
```

## 7. 常用工具函数

| 函数/常量 | 头文件 | 说明 |
|----------|--------|------|
| `BLOCK_CUBE` (16) | `common_utils.h` | 分形块维度 |
| `C0_SIZE_B8` (32), `C0_SIZE_B4` (64) | `common_utils.h` | 8-bit/4-bit C0 维度 |
| `C0_ELEMENT<T>`, `C0_SIZE<T>` | `legacy_type.h` | 类型 T 的 C0 元素数/字节数 |
| `CeilDiv(a, b)` | `common_utils.h` | 向上取整除法 |
| `CeilAlign(a, b)` | `common_utils.h` | 向上对齐 |
| `Align16/32/64(x)` | `common_utils.h` | 2 的幂对齐 |
| `MXFP_DIVISOR_SIZE` (64) | `common_utils.h` | MXFP K 轴对齐粒度 |
| `AscendC::Std::ceil_align(v, align)` | 框架 | 对齐到 align 的倍数 |
| `AscendC::BLOCK_CUBE` | 框架 | 框架侧 BLOCK_CUBE 常量 |

## 8. 编写检查清单

### 文件结构
- [ ] 实现文件放在 `tile/arch35/my_tile.h`
- [ ] 在 `compute.h`（计算/变换类）或 `datamove.h`（数据搬运类）中添加 `#include`
- [ ] 不新增顶层转发头（旧有的 5 个仅为兼容保留）
- [ ] 使用正确的命名空间：`Blaze::Gemm::Tile`

### Copy Tile
- [ ] 结构体包含 `static void Copy(...)` 方法
- [ ] 模板签名 `<typename Tp, const Tp& traits, typename T, typename U>`（无参）或 `<typename T, typename U>`（带参）
- [ ] 从 `src.Layout()` / `dst.Layout()` 提取 shape/stride，不硬编码
- [ ] MTE2/MTE3 操作包含 `if ASCEND_IS_AIV { return; }` 守卫
- [ ] 架构相关代码包含 `if constexpr (CURRENT_ARCH_VERSION == ...)` 守卫
- [ ] 在 `namespace AscendC::Te` 中注册 `CopyTraits` 特化
- [ ] 带参 Tile 实现 `.with()` 返回新 CopyTraits 实例，`CopyUnpack` 转发存储的参数

### Compute Tile
- [ ] 使用 `static_assert` 校验元素类型、内存位置、Layout Pattern
- [ ] 方法名使用语义化命名（`PadZero`、`FillWithValue`、`Run`）
- [ ] SIMD 函数使用 `__simd_vf__`，可组合子函数使用 `__simd_callee__`
- [ ] 向量写操作后添加 `LocalMemBar` 保证可见性

### 通用
- [ ] 包含必要的头文件：`tensor_api/tensor.h`、`common_utils.h`、`layout_utils.h`
- [ ] 添加文件头版权信息和 `\file`、`\brief` 注释
- [ ] 数据类型分发使用 `if constexpr (sizeof(T) == ...)` 或 `is_one_of_v`
- [ ] 使用者只需 `#include "blaze/gemm/tile/compute.h"` 或 `"blaze/gemm/tile/datamove.h"`，无需 include 架构子目录
