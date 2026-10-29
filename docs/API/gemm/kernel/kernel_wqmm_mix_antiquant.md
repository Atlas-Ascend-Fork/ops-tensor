# Kernel WQMM Mix Antiquant

> [代码位置](../../../../include/blaze/gemm/kernel/kernel_wqmm_mix_antiquant.h)

## 功能说明

该组件将量化权重 `weight` 反量化为输入 `x` 的类型，再执行矩阵乘并写回输出 `y`。
AIV 负责权重搬运和反量化，AIC 负责矩阵乘、可选的 `bias` 加法及输出写回。

逻辑矩阵维度为 `x: (M,K)`、`weight: (K,N)`、`y: (M,N)`。

计算过程为：

$$
\begin{aligned}
\mathrm{weightDequant}
&= \left(\operatorname{cast}_{x}(\mathrm{weight})
   + \mathrm{antiquantOffset}\right)\odot\mathrm{antiquantScale}, \\
y &= x\,\mathrm{weightDequant} + \mathrm{bias}.
\end{aligned}
$$

其中，$\operatorname{cast}_{x}$ 表示转换为 `x` 的类型，$\odot$ 表示逐元素乘法，
反量化参数按下述量化模式广播；`bias` 为可选的 N 元素向量，沿 M 轴广播。
未启用 `antiquantOffset` 或 `bias` 时省略对应加法。转换、反量化加法和乘法分别按 `x` 的类型舍入，
矩阵乘使用 FP32 累加，结果转换为 `y` 的类型。

### per-tensor 量化模式

`antiquantScale` 和可选的 `antiquantOffset` 各包含一个元素，整张权重矩阵共用同一组反量化参数。

### per-channel 量化模式

`antiquantScale` 和可选的 `antiquantOffset` 各包含 N 个元素，每个输出通道使用一组反量化参数，
同一通道的参数沿 K 轴广播。

## 支持范围

| 项目 | 支持范围 |
| --- | --- |
| 架构 | Ascend 950 / DAV_3510 |
| 输入 `x`、输出 `y` | FP16 或 BF16，类型相同 |
| 权重 `weight` | INT8、FP8 E4M3、HiFloat8 |
| GM 权重存储 | 连续的 `(N,K)` 或 `(K,N)` 数组，见下表 |
| `antiquantScale`、`antiquantOffset` | 类型与 `x` 相同；per-tensor 或 per-channel；偏移可选 |
| `bias` | 形状 `(N,)`，类型为 `x` 类型或 FP32，可选 |
| AIV 配置 | 每个 AIC 对应两个 AIV，输入缓冲区数量为 2 或 4 |

该特化不支持输入 `x` 转置、INT4、FP4、MX 量化和 GM WeightNZ 格式。
`ProblemShape` 只描述 `(M,N,K)`，不包含 batch 维度。

### 权重的布局方向

数学计算中的 `weight` 始终为 `(K,N)`。代码接口 `LayoutB` 以 `(N,K)` 为坐标描述权重，
其布局名称需要与 GM 实际存储顺序区分：

| `LayoutB` | GM 存储顺序 | 内部 `TRANS_B` | 反量化后的 UB/L1 布局，按 `(K,N)` 描述 |
| --- | --- | --- | --- |
| `nd_ext_layout_ptn` | `(N,K)`，K 连续 | `true` | ZN |
| `dn_ext_layout_ptn` | `(K,N)`，N 连续 | `false` | NZ |

Kernel 通过 `!IsTrans<LayoutB>::value` 推导方向。N 轴分块起点为 `coordN` 时，
权重地址偏移分别为 `coordN*K` 和 `coordN` 个权重元素；per-channel 的反量化参数偏移为
`coordN` 个参数元素，per-tensor 参数不随 N 分块偏移。

## 模板与参数

### Kernel 组装

`GemmUniversal<ProblemShape_, BlockMmad_, BlockEpilogue_, BlockScheduler_>` 的该偏特化要求
`BlockEpilogue_ = void`，且 `BlockMmad_::DispatchPolicy::ScheduleType` 为
`KernelMmadAPrefetchBAntiquant`。

| 模板参数 | 含义 |
| --- | --- |
| `ProblemShape_` | `(M,N,K)` 的 Shape 类型，通常为 `asc::te::shape<int64_t, int64_t, int64_t>` |
| `BlockMmad_` | [WQMM BlockMmad](../block/block_mmad_wqmm_mix_weight_prologue.md)，负责 AIC 计算 |
| `BlockEpilogue_` | `void`；输出写回由 BlockMmad 完成 |
| `BlockScheduler_` | [WQMM Scheduler](../block/block_scheduler_wqmm.md)，提供 M/N 分块 |

调度策略为 `MatmulWithWeightAntiquant<AivNum_, UbMte2InnerSize_, UbMte2BufNum_, AntiquantType_, HasAntiquantOffset_>`：

| 参数 | 含义及取值 |
| --- | --- |
| `AivNum_` | 每个 AIC 对应的 AIV 数量，取 2 |
| `UbMte2InnerSize_` | 权重输入 UB 行距，单位为字节；默认 512 |
| `UbMte2BufNum_` | 输入缓冲区数量，取 2 或 4，默认 2 |
| `AntiquantType_` | `QuantMode::PERCHANNEL_MODE` 或 `QuantMode::PERTENSOR_MODE` |
| `HasAntiquantOffset_` | 是否使用 `antiquantOffset`，默认 `false` |

### Kernel Params

公开调用接口为 `__aicore__ inline void operator()(const Params& params) const`。

| `Params` 成员 | 类型 | 含义 |
| --- | --- | --- |
| `problemShape` | `ProblemShape` | `(M,N,K)`，各轴长度以元素计 |
| `mmadParams` | `BlockMmadParams` | `x`、`y`、`bias` 地址和 L1/L0 分块配置，详见 BlockMmad 文档 |
| `aPreloadSize` | `uint64_t` | 每核预取的 `x` 元素数，0 表示关闭 |
| `aElementCount` | `uint64_t` | `x` 的总元素数，用于限制预取范围 |
| `prologueParams` | `PrologueParams` | 权重及反量化参数地址，见下表 |
| `schedulerParams` | `BlockSchedulerParams` | M/N 分核及三段 N 区间配置 |

公开结构 `PrologueParams`：

| 成员 | 类型 | 含义 |
| --- | --- | --- |
| `bGmAddr` | `GM_ADDR` | 量化权重起始地址，共 `K*N` 个 1 字节元素 |
| `scaleGmAddr` | `GM_ADDR` | `antiquantScale`；per-tensor 为 1 个元素，per-channel 为 N 个元素 |
| `offsetGmAddr` | `GM_ADDR` | 可选 `antiquantOffset`，形状及类型与 `antiquantScale` 相同；未启用时可为 `nullptr` |
| `weightL2Cacheable` | `uint32_t` | 非零使用 `NORMAL_FIRST_VICTIM`，0 使用 `NOTALLOC_KEEP` 缓存模式 |

### AIV Prologue

`KernelWqmmMixAntiquantPrologue` 封装权重前处理。其模板参数指定输入与权重类型、UB 行距与缓冲区数量、
权重方向、反量化模式、同步协议和共享 L1 地址提供者。`AntiquantScaleType_` 必须与 `XType_` 相同。

其 `Params` 为公开结构：

| 成员 | 类型 | 含义 |
| --- | --- | --- |
| `kbL1Size` | `uint64_t` | L1 权重分块的 K 长度，单位为元素 |
| `weightStride` | `uint64_t` | GM 权重物理行距，单位为权重元素；两种方向分别为 K 或 N |
| `weightCacheMode` | `asc_load_l2_cache_mode` | GM 权重读取的缓存模式 |
| `scaleValue` | `XType_` | per-tensor 的 `antiquantScale` 标量 |
| `offsetValue` | `XType_` | per-tensor 且启用偏移时使用的 `antiquantOffset` 标量 |

```cpp
__aicore__ inline explicit KernelWqmmMixAntiquantPrologue(
    const Params& params, const SharedMemProvider_& sharedMemProvider);

__aicore__ inline void operator()(
    __gm__ WeightType_* blockB,
    __gm__ AntiquantScaleType_* blockScale,
    __gm__ AntiquantScaleType_* blockOffset,
    uint64_t tileN, uint64_t kSize);
```

`blockB` 指向当前 N 分块的量化权重，`blockScale` 和 `blockOffset` 指向对应通道参数。
per-tensor 使用构造参数中的标量；未启用偏移时不读取 `blockOffset`。
`tileN`、`kSize` 分别是当前权重分块的有效 N 长度和完整 K 长度，以元素计。
共享地址提供者返回两份 L1 权重缓冲区的地址。VF 函数和内部容量常量为私有实现。

## 调用示例

以下示例组装 INT8 权重、FP16 输入、GM `(N,K)` 权重存储、per-channel 且无偏移的 Kernel 类型：

```cpp
using Policy = Blaze::Gemm::MatmulWithWeightAntiquant<
    2, 512, 2, Blaze::Gemm::QuantMode::PERCHANNEL_MODE, false>;
using Layout = asc::te::nd_ext_layout_ptn;
using Shape = asc::te::shape<int64_t, int64_t, int64_t>;
using Block = Blaze::Gemm::Block::BlockMmad<
    Policy, half, Layout, AscendC::Std::tuple<int8_t, half>,
    AscendC::Std::tuple<Layout, Layout>, half, Layout, half, Layout>;
using Scheduler = Blaze::Gemm::Block::BlockSchedulerWqmmTailResplit<Shape>;
using Kernel = Blaze::Gemm::Kernel::GemmUniversal<Shape, Block, void, Scheduler>;
```

填充 `Kernel::Params` 后在混合核入口调用 `Kernel{}(params)`。
完整的参数组装和运行方式见
[权重反量化样例](../../../../examples/acl/weight_quant_batch_matmul/weight_quant_batch_matmul_antiquant/README.md)。
