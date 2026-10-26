# Pad MX K L1
> [代码位置](../../../../include/blaze/gemm/tile/arch35/pad_mx_kl1.h)

说明：调用方应包含聚合头 `tile/compute.h`（`tile/pad_mx_kl1.h` 为兼容保留的转发入口）。

## 功能说明
MX 量化 K 轴 Padding Tile，用于 L1 缓冲区的 K 轴尾部补零对齐。支持 NZ/ZN 布局，确保 K 轴对齐到 C0_SIZE。

## 特殊约束

### 数据类型支持
仅支持 MxFP4 和 MxFP8 量化数据类型：
- **MxFP4**：`fp4x2_e2m1_t`、`fp4x2_e1m2_t`
- **MxFP8**：`fp8_e5m2_t`、`fp8_e4m3fn_t`

### 布局支持
支持 NZ 和 ZN 布局：
- **nz_layout_ptn**：非转置布局
- **zn_layout_ptn**：转置布局

### C0_SIZE 对齐
K 轴需对齐到 C0_SIZE：
- **FP4**：C0_SIZE = 64
- **FP8**：C0_SIZE = 32

### MXFP 对齐要求
- K 轴需对齐到 `MXFP_DIVISOR_SIZE`（64）
- 使用 ND2NZ 指令时，内轴自动补零，外轴需手动补零

## 特殊类型

### PadMxKL1Base
```
struct PadMxKL1Base {
    template <typename T>
    __aicore__ inline static void PadZero(const T& tensorL1, uint64_t repeatTimes, uint64_t blockNum, uint64_t dstGap);

    template <typename type>
    __aicore__ inline static constexpr bool IsMxFp4();

    template <typename type>
    __aicore__ inline static constexpr bool IsMxFp8();
};
```

功能：MX K 轴 Padding 基类。

### PadMxKAL1
```
struct PadMxKAL1 : public PadMxKL1Base {
    template <typename T, typename U>
    __aicore__ inline static void PadZero(const T& tensorL1, const U& tensorGm);
};
```

功能：A 矩阵 L1 K 轴 Padding。

### PadMxKBL1
```
struct PadMxKBL1 : public PadMxKL1Base {
    template <typename T, typename U>
    __aicore__ inline static void PadZero(const T& tensorL1, const U& tensorGm);
};
```

功能：B 矩阵 L1 K 轴 Padding。

## 特殊成员方法

### PadMxKL1Base::PadZero
```
template <typename T>
__aicore__ inline static void PadZero(const T& tensorL1, uint64_t repeatTimes, uint64_t blockNum, uint64_t dstGap)
```
功能：对 L1 Tensor 进行补零。
参数说明：
| 参数 | 类型 | 说明 |
|------|------|------|
| tensorL1 | T | L1 Tensor（slice 后） |
| repeatTimes | uint64_t | 重复次数 |
| blockNum | uint64_t | 每次补零的 block 数量 |
| dstGap | uint64_t | 目标间隔 |

说明：使用 `asc_fill_l1` 指令补零。

### PadMxKL1Base::IsMxFp4
```
template <typename type>
__aicore__ inline static constexpr bool IsMxFp4()
```
功能：判断是否为 MxFP4 数据类型。

### PadMxKL1Base::IsMxFp8
```
template <typename type>
__aicore__ inline static constexpr bool IsMxFp8()
```
功能：判断是否为 MxFP8 数据类型。

### PadMxKAL1::PadZero
```
template <typename T, typename U>
__aicore__ inline static void PadZero(const T& tensorL1, const U& tensorGm)
```
功能：对 A 矩阵 L1 K 轴进行补零。
参数说明：
| 参数 | 类型 | 说明 |
|------|------|------|
| tensorL1 | T | A 矩阵 L1 Tensor |
| tensorGm | U | A 矩阵 GM Tensor（用于获取实际 K 维度） |

执行流程：
**nz_layout_ptn 场景**：
1. 判断是否为 FP4：FP4 不需补零（ND2NZ 自动处理）
2. 判断补零范围：`kAxisL1Align - kAxis >= C0_SIZE`
3. 计算 slice 坐标：从 `kAxisND2NZAlign` 开始
4. 执行补零：`PadMxKL1Base::PadZero(sliceTensor, 1, mAlign, 0)`

**zn_layout_ptn 场景**：
1. 判断补零范围：`kAxis != kAxisL1Align`
2. 计算迭代次数：`m1`（M 轴大分形个数）
3. 计算 dstGap：`dstRowStride / c0_element - kAxisL1Align + kAxis`
4. 执行补零：`PadMxKL1Base::PadZero(sliceTensor, m1, kAxisL1Align - kAxis, dstGap)`

### PadMxKBL1::PadZero
```
template <typename T, typename U>
__aicore__ inline static void PadZero(const T& tensorL1, const U& tensorGm)
```
功能：对 B 矩阵 L1 K 轴进行补零。
参数说明：
| 参数 | 类型 | 说明 |
|------|------|------|
| tensorL1 | T | B 矩阵 L1 Tensor |
| tensorGm | U | B 矩阵 GM Tensor（用于获取实际 K 维度） |

执行流程：
**nz_layout_ptn 场景**：
1. 判断补零范围：`kAxis != kAxisL1Align`
2. 计算迭代次数：`n1`（N 轴大分形个数）
3. 计算 slice 坐标：从 `(kAxis, 0)` 开始
4. 执行补零：`PadMxKL1Base::PadZero(sliceTensor, n1, kAxisL1Align - kAxis, kAxis)`

**zn_layout_ptn 场景**：
1. 判断是否为 FP4：FP4 不需补零
2. 判断补零范围：`kAxisL1Align - kAxis >= C0_SIZE`
3. 计算 slice 坐标：从 `(kAxisND2NZAlign, 0)` 开始
4. 执行补零：`PadMxKL1Base::PadZero(sliceTensor, 1, nAlign, 0)`

## 使用示例

### A 矩阵 L1 Padding
```
using LayoutA = asc::te::nz_layout_ptn;
using AType = fp4x2_e2m1_t;

auto tensorAL1 = asc::te::make_tensor(...);  // L1 Tensor
auto gmTileA = gmA.slice(...);                  // GM Tensor（slice 到当前 tile）

Blaze::Gemm::Tile::PadMxKAL1::PadZero(tensorAL1, gmTileA);
```

### B 矩阵 L1 Padding
```
using LayoutB = asc::te::zn_layout_ptn;
using BType = fp8_e5m2_t;

auto tensorBL1 = asc::te::make_tensor(...);  // L1 Tensor
auto gmTileB = gmB.slice(...);                  // GM Tensor（slice 到当前 tile）

Blaze::Gemm::Tile::PadMxKBL1::PadZero(tensorBL1, gmTileB);
```

## 数据流

### NZ 布局补零流程
```
GM → L1（ND2NZ）
    ↓
内轴自动补零（ND2NZ 指令）
    ↓
外轴手动补零（PadMxKL1）
    ↓
L1（K 轴对齐到 C0_SIZE）
```

### ZN 布局补零流程
```
GM → L1（ND2NZ）
    ↓
外轴手动补零（PadMxKL1）
    ↓
L1（K 轴对齐到 C0_SIZE）
```

说明：ND2NZ 指令仅支持内轴补零，外轴需通过 PadMxKL1 手动补零。
