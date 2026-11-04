# Block Mmad Qgmm Mx
> [代码位置](../../../../include/blaze/gemm/block/block_mmad_qgmm_mx.h)

## 功能说明
MX 量化 Grouped Matmul 的 Block 组件，基于 Tensor API 实现，仅支持 AIC 计算。
组件负责单核 block 内的 A/B、ScaleA/ScaleB 搬运与 L0 tile 级 MX Mmad 计算，并处理 bias；
融合 SwiGLU 路径还可同时计算权重左右两个 N 半区，并将两个结果拼接写入 UB。

**继承思路参考**：[Block Mmad 公共框架](./block_mmad.md)

## 特殊约束

### 调度策略限制
仅支持 `GroupedMatmulWithScaleMx` 调度策略，不用于 Basic、QBMM 或 StreamK 路径。

### 量化数据类型支持
支持以下典型 MX 量化输入类型：
- `fp4x2_e2m1_t`
- `fp4x2_e1m2_t`
- `fp8_e5m2_t`
- `fp8_e4m3fn_t`

### Scale 类型
ScaleA 和 ScaleB 固定使用 `fp8_e8m0_t`。

### 计算模式
仅支持 AIC 模式，不支持 AIV 计算。

### 输出目标
默认将结果直接输出到 GM；当 `TensorC` 位于 UB 时，使用 MIX Split-M copy trait 将 float
L0C 结果写到双 AIV 可见的 UB，供融合 Epilogue 消费。

### SwiGLU 双源拼接

- `LayoutB_` 可直接传入 B 布局，也可传入 `tuple<LayoutB, LayoutScaleB>`。当 B 为 NZ/ZN 时，
  `LayoutScaleB` 仍分别使用 `scaleb_nd_layout_ptn` / `scaleb_dn_layout_ptn`。
- 融合路径把完整 N 维平均拆成左右两个半区。`gmBLeft/gmBRight` 与
  `gmScaleBLeft/gmScaleBRight` 必须分别指向两个半区的同一 M/K 窗口。
- 普通 NZ/ZN Tensor 使用双源 overload；该 overload 通过编译期校验拒绝其他 B 布局。
  ND/DN 可使用单源 overload，由 `CopyConcatGM2L1` 按完整 N 的 stride 取得右半区。
- 提供自定义 `CopyConcatGM2L1` 搬运重载的 Tensor view（例如 MegaMoe 的
  `Gmm1NzWeightView`）继续使用单源 overload；编译期 `HasDualSource` 门控保留
  其自定义搬运，不将该 view 转交普通 NZ/ZN 双源 Tensor Copy。
- `singleShape` 的 N 是单个半区宽度；拼接后的 L1 B、ScaleB 和 UB C 的 N 宽度均为其 2 倍，
  即 UB 输出逻辑形状为 `[blockM, 2 * blockN]`。
- 普通 NZ/ZN 双源 Tensor 在按需补零后，通过 tensor_api NZ2NZ/ZN2ZN 分别搬运左右半区。
  单源 view 始终使用 `CopyConcatGM2L1` 通用拼接。

## 特殊静态常量

| 常量 | 说明 |
|------|------|
| `TRANS_A` | A 是否转置 |
| `TRANS_B` | B 是否转置 |
| `C0_SIZE` | 数据 C0 对齐大小，FP4 为 64，FP8 为 32 |
| `SCALE_C0` | Scale 布局的 C0 对齐大小，固定为 2 |
| `SCALE_BUFFER_NUM` | Scale 双缓冲数量 |
| `HALF_L0_SIZE` | L0A/L0B 半缓冲大小 |
| `HALF_L0C_SIZE` | L0C 半缓冲大小 |

## 特殊类型别名

| 别名 | 含义 |
|------|------|
| `ProblemShape` | 当前 group 的问题规模 |
| `BlockShape` | 单核 block 形状 |
| `MxL0AType` | A 在 L0 中的数据类型 |
| `MxL0BType` | B 在 L0 中的数据类型 |
| `LayoutB` | `LayoutB_` 的 B 数据布局；tuple 形式取第 0 项 |
| `LayoutScaleB` | ScaleB 的 GM 布局；tuple 形式取第 1 项，否则按 B 是否转置推导 |

## 特殊数据结构

### Params
```cpp
struct Params {
    GM_ADDR aGmAddr;
    GM_ADDR bGmAddr;
    GM_ADDR cGmAddr;
    GM_ADDR biasGmAddr;
    GM_ADDR scaleAGmAddr;
    GM_ADDR scaleBGmAddr;
};
```

参数说明：

| 参数 | 说明 |
|------|------|
| `aGmAddr` | A 的 GM 地址 |
| `bGmAddr` | B 的 GM 地址 |
| `cGmAddr` | C 的 GM 地址 |
| `biasGmAddr` | bias 的 GM 地址，可为空 |
| `scaleAGmAddr` | A 对应的 per-token scale 地址 |
| `scaleBGmAddr` | B 对应的 per-group scale 地址 |

### L1Params
```cpp
struct L1Params {
    uint64_t kAL1;
    uint64_t kBL1;
    uint64_t scaleKL1;
};
```

参数说明：

| 参数 | 说明 |
|------|------|
| `kAL1` | A 的 L1 K 轴切分 |
| `kBL1` | B 的 L1 K 轴切分 |
| `scaleKL1` | ScaleA/ScaleB 共享的 L1 K 轴切分 |

参数约束：

- `kAL1`、`kBL1`、`scaleKL1` 均需大于 0。
- `kAL1`、`kBL1` 建议按 `MXFP_DIVISOR_SIZE`（64）对齐；末尾 K tail 由实现内部 padding 处理。
- 当 `kAL1 >= kBL1` 时，外层按 `kAL1` 复用 A，要求 `kAL1` 是 `kBL1` 的整数倍。
- 当 `kBL1 > kAL1` 时，外层按 `kBL1` 复用 B，要求 `kBL1` 是 `kAL1` 的整数倍。
- `scaleKL1` 必须不小于外层 L1 K 窗口 `max(kAL1, kBL1)`，且应为该外层窗口的整数倍；Scale 只在 `scaleKL1` 边界搬运一次，并在窗口内复用。

### MmadParams
```cpp
struct MmadParams {
    BlockShape tileShapeL0;
    L1Params l1Params;
    bool isBias;
    bool enableL0cPingPong;
    uint8_t l1BufferStage{DOUBLE_BUFFER_COUNT};
};
```

参数说明：

| 参数 | 说明 |
|------|------|
| `tileShapeL0` | L0 tile 形状 |
| `l1Params` | A、B 和 Scale 的 L1 K 轴切分参数 |
| `isBias` | 是否计算 bias |
| `enableL0cPingPong` | 是否启用 L0C ping-pong |
| `l1BufferStage` | A/B 的 L1 缓冲级数，默认值为 `DOUBLE_BUFFER_COUNT` |

`l1BufferStage` 取值说明：

- `DOUBLE_BUFFER_COUNT`：使用双缓冲，L1 占用较小，也是默认配置。
- `TRIPLE_BUFFER_COUNT`：使用三缓冲，增加 A/B 搬运与计算的流水重叠，以提升性能。
- 仅值为 `TRIPLE_BUFFER_COUNT` 时启用三缓冲；其他值均按双缓冲处理。
- 该字段只控制 A/B 的 L1 缓冲级数，ScaleA/ScaleB 仍使用 `SCALE_BUFFER_NUM` 指定的双缓冲。
- 启用三缓冲前，调用方应通过 tiling 确认 L1 空间可以容纳对应布局；空间不足时应设置为 `DOUBLE_BUFFER_COUNT`。

## 特殊成员方法

### Init 函数
```cpp
__aicore__ inline void Init(
    const ProblemShape& problemShape,
    const MmadParams& params)
```

功能：
- 初始化当前 group 的 `m/n/k`
- 设置 L0 tile 和 L1 切分参数
- 根据 `l1BufferStage` 初始化并重置组内双缓冲或三缓冲状态

### UpdateParamsForNextProblem 函数
```cpp
__aicore__ inline void UpdateParamsForNextProblem(const ProblemShape& problemShape)
```

功能：
- 在 grouped matmul 切换到下一个 group 时刷新 `m/n/k`
- 保持同一个 block 组件在不同 group 间复用

### operator() 函数
```cpp
template <typename TensorA, typename TensorB, typename TensorScaleA,
          typename TensorScaleB, typename TensorBias, typename TensorC>
__aicore__ inline void operator()(
    const TensorA& gmA,
    const TensorB& gmB,
    const TensorScaleA& gmScaleA,
    const TensorScaleB& gmScaleB,
    const TensorBias& gmBias,
    const TensorC& gmC,
    const BlockShape& blockShape)
```

功能：
- 处理一个单核 block 的 MX grouped matmul 计算
- 输入 tensor 由 kernel 层完成 slice 后传入

### SwiGLU 拼接 operator() 函数

```cpp
template <typename TensorA, typename TensorB, typename TensorScaleA,
          typename TensorScaleB, typename TensorUbC>
__aicore__ inline void operator()(
    TensorA gmA, TensorB gmBLeft, TensorScaleA gmScaleA,
    TensorScaleB gmScaleBLeft, TensorUbC& ubC,
    const BlockShape& singleShape);

template <typename TensorA, typename TensorB, typename TensorScaleA,
          typename TensorScaleB, typename TensorUbC>
__aicore__ inline void operator()(
    TensorA gmA, TensorB gmBLeft, TensorB gmBRight,
    TensorScaleA gmScaleA, TensorScaleB gmScaleBLeft,
    TensorScaleB gmScaleBRight, TensorUbC& ubC,
    const BlockShape& singleShape);
```

功能：

- 第一个 overload 供 ND/DN 单源拼接路径及提供自定义搬运重载的 Tensor view 使用。
- 第二个 overload 供 NZ/ZN 双源路径使用，左右 B 和 ScaleB slice 均由 Kernel/Scheduler 显式传入。
- 按拼接后的 N 宽度执行 MMAD，结果按 `[left, right]` 顺序写入同一 UB，供 SwiGLU Epilogue 使用。

## 调用示例

### 组件组装
```cpp
using AType = fp8_e4m3fn_t;
using BType = fp8_e4m3fn_t;
using CType = float;
using BiasType = float;

using LayoutA = asc::te::nd_ext_layout_ptn;
using LayoutB = asc::te::nz_layout_ptn;
using LayoutC = asc::te::nd_ext_layout_ptn;
using LayoutBias = asc::te::nd_ext_layout_ptn;

using DispatchPolicy = Blaze::Gemm::GroupedMatmulWithScaleMx<0>;
using BlockMmad = Blaze::Gemm::Block::BlockMmad<
    DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC, BiasType, LayoutBias>;
```

### 组件初始化
```cpp
BlockMmad blockMmad;

BlockMmad::ProblemShape problemShape{m, n, k, 0};
BlockMmad::BlockShape tileShapeL0{baseM, baseN, baseK, 0};
BlockMmad::L1Params l1Params{kAL1, kBL1, scaleKL1};
uint8_t l1BufferStage = TRIPLE_BUFFER_COUNT;
BlockMmad::MmadParams params{tileShapeL0, l1Params, isBias, enableL0cPingPong, l1BufferStage};

blockMmad.Init(problemShape, params);
```

### 组件执行
```cpp
auto gmBlockA = gmA.slice(...);
auto gmBlockB = gmB.slice(...);
auto gmBlockScaleA = gmScaleA.slice(...);
auto gmBlockScaleB = gmScaleB.slice(...);
auto gmBlockBias = gmBias.slice(...);
auto gmBlockC = gmC.slice(...);

const int64_t blockK = problemK;
BlockMmad::BlockShape blockShape{blockM, blockN, blockK, 0};
blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockBias, gmBlockC, blockShape);
```

## 数据流
```text
GM(A/B) + GM(ScaleA/ScaleB) + GM(Bias)
    -> L1
    -> L0A/L0B + Scale Buffer
    -> MmadTraitMX
    -> L0C
    -> GM(C) 或 UB(C，MIX Epilogue)

SwiGLU NZ/ZN：
GM(A) + GM(B-left/B-right) + GM(ScaleA/ScaleB-left/ScaleB-right)
    -> 拼接 L1 B/ScaleB
    -> 按拼接后的 N 宽度执行 Mmad
    -> UB([blockM, 2 * blockN])
```

## 适用场景
- MX 量化 grouped matmul 的 tensor_api kernel
- group 间 `m/n/k` 动态变化的场景
- 同时处理输入 scale 与权重 scale 的 grouped matmul 场景
