# Kernel Matmul Mix Weight Prologue
> [代码位置](../../../../include/blaze/gemm/kernel/kernel_matmul_mix_weight_prologue.h)

## 功能说明

本页介绍 `GemmUniversal` 的 arch35 Weight Quant MX 特化。该特化在 AIV（Vector）侧使用
`KernelMatmulMixWeightPrologue`，在 MMAD 前将 packed FP4 权重转换为 FP8，并把转换后的权重和
可选 bias 写入共享 L1；AIC（Cube）消费相同 Scheduler 生成的 tile，执行 MX MMAD 和 Fixpipe 输出。

该路径的 bias 和权重转换属于 MMAD 前处理，不使用 `kernel_qbmm_mix` 的 AIV 后处理协议。

**继承自**：[Kernel 基础框架](./kernel.md)

**配套组件**：[BlockMmad Weight Prologue MX](../block/block_mmad_weight_prologue_mx.md)、
[Matmul SWAT Scheduler](../block/block_scheduler_matmul_swat_with_tail_split.md) 和
[Weight Quant MX 前处理 Tile](../tile/tile_weight_quant_mx_preprocess.md)

## 特殊约束

### 架构支持

该特化仅在 `__NPU_ARCH__ == 3510`（arch35）下定义，其他架构不能实例化此实现。

### 调度策略

`BlockMmad::DispatchPolicy::ScheduleType` 必须为 `KernelMixWithWeightPrologue`，对应
`MatmulWithWeightQuantMx`。`BlockEpilogue` 固定为 `void`；B/bias 由 AIV prologue 提供，C 由
`BlockMmad` 内的 Fixpipe 直接写回。

### 数据类型和布局

| 数据 | 类型 | Layout | 说明 |
| :--- | :--- | :--- | :--- |
| A | `fp8_e4m3fn_t` | `nd_ext_layout_ptn` | 激活矩阵 |
| B | `fp4x2_e2m1_t` | `zn_layout_ptn` 或 `dn_ext_layout_ptn` | packed FP4 转置权重 |
| ScaleA/ScaleB | `fp8_e8m0_t` | 由对应 scale layout 表达 | MX scale |
| C | `half` 或 `bfloat16_t` | `nd_ext_layout_ptn` | 输出矩阵 |
| Bias | 与 C 一致 | `nd_ext_layout_ptn` | 可选，首个 K window 参与计算 |

`LayoutB` 是 Weight NZ/ND 的唯一来源：NZ 使用 `zn_layout_ptn`，ND 使用 `dn_ext_layout_ptn`。
输入 B 必须是转置布局。各输入的 layout pattern 由 `BlockMmad` 类型提供，GM Tensor shape 由
`ProblemShape` 派生。

### 缓冲和同步

- `l1BufferNum` 由 host tiling 保证为 2 或 4；B、bias 的 L1 半区/槽位地址必须与 AIC 的
  `WeightL1Storage` 保持一致。
- AIV 是 B/bias L1 的生产者，发送 ready 标志；AIC 等待 ready 标志，完成 MMAD 后发送
  free 标志。
- AIV/AIC 必须使用相同的 `ProblemShape`、Scheduler 参数和 tile 序列。
- 输入合法性（包括 K 为 8 的倍数）由 host tiling/API 校验，device 侧不重复校验。

## 模板参数

```cpp
template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
class GemmUniversal;
```

调用方使用以上四个模板参数组装 Kernel；主模板的第五个 `Enable` 参数由 SFINAE 特化匹配自动确定。

| 参数 | 要求 |
| :--- | :--- |
| `ProblemShape` | `asc::te::shape<int64_t, int64_t, int64_t>`，维序为 `(M, N, K)` |
| `BlockMmad` | `MatmulWithWeightQuantMx` 特化的 `BlockMmad` |
| `BlockEpilogue` | 必须为 `void` |
| `BlockScheduler` | 通常为 `BlockSchedulerMatmulSwatWithTailSplit<ProblemShape>` |

## 特殊数据结构

### `PrologueParams`

```cpp
struct PrologueParams {
    GM_ADDR bGmAddr;
    GM_ADDR biasGmAddr;
    uint64_t kBubSize;
    uint64_t nBubSize;
};
```

| 字段 | 说明 |
| :--- | :--- |
| `bGmAddr` | packed FP4 权重 GM 地址 |
| `biasGmAddr` | bias GM 地址；无 bias 时可为空 |
| `kBubSize` | AIV 单个 UB 权重分片的 K 方向容量 |
| `nBubSize` | AIV 单个 UB 权重分片的 N 方向容量 |

### `Params`

```cpp
struct Params {
    ProblemShape problemShape;
    BlockMmad::Params mmadParams;
    PrologueParams prologueParams;
    BlockScheduler::Params schedulerParams;
};
```

`mmadParams.l1TileShape` 的维序为 `(baseM, baseN, kL1, scaleKL1)`，
`mmadParams.l0TileShape` 的维序为 `(baseM, baseN, baseK)`。Scheduler 的八个参数及尾块 split
语义见 [Matmul SWAT Scheduler](../block/block_scheduler_matmul_swat_with_tail_split.md#params)。

## 特殊成员方法

### 构造函数和析构函数

```cpp
__aicore__ inline GemmUniversal();
__aicore__ inline ~GemmUniversal();
```

Kernel 使用默认构造；每次调用由 `operator()` 将参数传给 AIV/AIC 路径。

### `operator()` 函数

```cpp
__aicore__ inline void operator()(const Params& params);
```

执行 AIV/AIC 分流、Scheduler 构造和 tile 循环。

## 执行流程

1. AIV 和 AIC 使用相同 Scheduler 计算 tile 数、坐标和 shape。
2. AIV 按 tile 的 K/N 分片从 GM 搬运 packed FP4，执行 FP4→FP8 转换，并将 B/bias 写入 L1。
3. AIV 为每个 L1 weight window 发送 ready 标志；没有有效数据的 AIV sub-block 也必须完成对应通知。
4. AIC 先将当前窗口的 ScaleA、ScaleB 和 A 从 GM 搬入 L1，并在等待 ready 标志前清理
   `[Align32(K), Align64(K))` 的 B 物理 K 尾部。
5. AIC 等待 AIV ready 标志，再将 A、B、ScaleA、ScaleB 搬入 L0，执行 MX MMAD 和 Fixpipe；完成
   MMAD 后释放 weight buffer，允许 AIV 复用。

## 格式和尾块

- **Weight NZ（`zn_layout_ptn`）**：AIV 沿 K 方向分片，转换后 UB 使用标准 ZN layout 搭配显式 stride。
- **Weight ND（`dn_ext_layout_ptn`）**：Tensor 坐标为 `(K, N)`，GM 物理数据按 `(N, K)` 行主序存放。
  FP8 UB 中每个 K32 slab 的 N pitch 为 `(Align16(N) + 1) * 32B`，UB→L1 时剥离额外 gap。
- **K 尾块**：AIV 写入物理 K32 block；AIC 在 MMAD 前清理 `[Align32(K), Align64(K))`。
- **Bias**：仅首个 K window 处理，按 MX MMAD 要求乘以 `1/64`；UB backing storage 至少按
  `CeilAlign(N, 256 / sizeof(BiasType))` 个元素分配。

## 调用示例

以下类型组合与 `examples/acl/weight_quant_batch_matmul_mx/weight_quant_batch_matmul_mx_swat` 一致。`m`、`n`、`k` 为
`int64_t`，tiling 尺寸为 `uint64_t`：

```cpp
using AType = fp8_e4m3fn_t;
using BType = fp4x2_e2m1_t;
using ScaleType = AscendC::fp8_e8m0_t;
using CType = half;
using BiasType = half;
using LayoutA = asc::te::nd_ext_layout_ptn;
using LayoutB = asc::te::zn_layout_ptn;  // ND 权重时改为 dn_ext_layout_ptn
using LayoutC = asc::te::nd_ext_layout_ptn;
using LayoutScaleA = asc::te::scalea_nd_layout_ptn;
using LayoutScaleB = asc::te::scaleb_dn_layout_ptn;
using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t>;
using DispatchPolicy = Blaze::Gemm::MatmulWithWeightQuantMx;
using BlockMmad = Blaze::Gemm::Block::BlockMmad<
    DispatchPolicy, AscendC::Std::tuple<AType, ScaleType>, AscendC::Std::tuple<LayoutA, LayoutScaleA>,
    AscendC::Std::tuple<BType, ScaleType>, AscendC::Std::tuple<LayoutB, LayoutScaleB>, CType, LayoutC, BiasType,
    LayoutC>;
using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulSwatWithTailSplit<ProblemShape>;
using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, void, BlockScheduler>;

Kernel::Params params{
    asc::te::make_shape(m, n, k),
    {aGm, scaleAGm, scaleBGm, cGm,
     asc::te::make_shape(
         static_cast<int64_t>(baseM), static_cast<int64_t>(baseN), static_cast<int64_t>(tileShapeKL1),
         static_cast<int64_t>(tileShapeScaleKL1)),
     asc::te::make_shape(
         static_cast<int64_t>(baseM), static_cast<int64_t>(baseN), static_cast<int64_t>(baseK)),
     l1BufferNum, hasBias},
    {bGm, biasGm, kBubSize, nBubSize},
    {baseM, baseN, 1U, 1U, 1U, 1U, 0U, 0U}};
Kernel kernel;
kernel(params);
```

## 适用场景

- arch35 的 MXA8W4（A=FP8、B=packed FP4、ScaleA/ScaleB=E8M0）矩阵乘。
- Weight NZ 或 Weight ND 输入，输出 `half`/`bfloat16_t`，可选 bias。
- 当前 QuantBatchMatmulV4 集成中的单 Batch `(M, N, K)` 推理路径。
