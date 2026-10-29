# Kernel QBMM Per-tensor StreamK
> [代码位置](../../../../include/blaze/gemm/kernel/kernel_qbmm_pertensor_streamk.h)

## 功能说明

QBMM per-tensor StreamK 的独立 `GemmUniversal` 特化，负责组装和协调：

- `BlockSchedulerMatmulStreamK`：划分 M/N tile 和 K 分片；
- 复用 `block_mmad_a8w8_fixpipe_quant.h` 中的 QBMM BlockMmad：Data Parallel（DP）block 直接反量化
  并写入 C，StreamK（SK）block 将当前 K 分片的累加结果写入 workspace；
- `BlockEpilogueQbmmPertensorStreamK`：AIV 归约并反量化；
- AIC→AIV 跨核同步。

该 Kernel 与 MX StreamK 的 scale 数据流不同，因此使用独立文件和独立 schedule tag，不依赖 `kernel_qbmm_streamk.h`。

## 特化条件

```cpp
template <
    class ProblemShape,
    class BlockMmad,
    class BlockEpilogue,
    class BlockScheduler>
class GemmUniversal<
    ProblemShape,
    BlockMmad,
    BlockEpilogue,
    BlockScheduler,
    enable_if_t<
        is_same_v<
            KernelQbmmPertensorMultiBlockStreamK,
            typename BlockMmad::DispatchPolicy::ScheduleType> &&
        is_same_v<
            KernelQbmmPertensorMultiBlockStreamK,
            typename BlockEpilogue::DispatchPolicy::ScheduleType>>>;
```

BlockMmad 和 BlockEpilogue 的 schedule tag 必须同时为 `KernelQbmmPertensorMultiBlockStreamK`，防止与普通 Matmul StreamK 或 MX StreamK 错误组合。

## 模板组件

| 组件 | 要求 |
|------|------|
| `ProblemShape` | `(M, N, K, B)` 四维 Shape |
| `BlockMmad` | `MatmulWithScaleFixpipeQuant<FullLoadMode, false, KernelQbmmPertensorMultiBlockStreamK>` 特化 |
| `BlockEpilogue` | `BlockEpilogueQbmmPertensorStreamK` |
| `BlockScheduler` | `BlockSchedulerMatmulStreamK<ProblemShape>` |

## 编译期 dtype/format 约束

Kernel 在模板实例化入口通过 `static_assert` 校验：

- A/B 支持 `int8_t` 同型、`hifloat8_t` 同型，以及 `fp8_e4m3fn_t` / `fp8_e5m2_t` 的 FP8 组合；
- C 支持 `half`、`bfloat16_t` 和 `float`；
- BlockMmad 与 BlockEpilogue 的输出和 workspace 类型必须一致；
- `LayoutA` 支持 ND/DN，`LayoutB` 支持 ND/DN/NZ/ZN，`LayoutC` 仅支持 `nd_ext_layout_ptn`。

Kernel 不额外校验 bias 类型、`LayoutBias`、scale 类型或 bias/scale 配对关系。scale 的存储类型、
实际启用的量化模式及 MMAD/Epilogue 参数仍须遵循下文所述的数据流要求；通过模板校验不代表
任意 bias/scale 组合都具备运行时支持。

不满足上述范围的组合会在编译期返回包含具体组件和类型/格式要求的错误信息，不再进入深层模板实例化。

## Params

```cpp
struct Params {
    ProblemShape problemShape;
    BlockMmadParams blockMmadParams;
    BlockEpilogueParams epilogueParams;
    BlockSchedulerParams schParams;
};
```

### BlockMmadParams

保存 A/B/C、bias 和 scale 地址。Kernel 使用 scheduler 参数初始化共享 QBMM BlockMmad，并将 per-tensor scale 转换为 DP Fixpipe 所需的 scalar scale。

### BlockEpilogueParams

保存最终输出、workspace、X2 scale、可选 X1 per-tensor 标量和 AIV bias。

### BlockSchedulerParams

```cpp
struct Params {
    int64_t usedCoreNum;
    int64_t baseM;
    int64_t baseN;
    int64_t baseK;
    int64_t singleCoreK;
    int64_t kL1;
    uint8_t isHf32;
    uint32_t l2CacheMode;
};
```

## 核心接口

### operator()

```cpp
__aicore__ inline void operator()(Params const& params);
```

执行流程：

1. 检查 `usedCoreNum > 0` 且 batch 为 1；
2. `Init()` 解析问题规模和 GM 地址；
3. 构造 StreamK scheduler；
4. AIC 执行 `ProcessOnAic()`；
5. AIV 执行 `ProcessOnAiv()`。

### AIC 流程

`ProcessOnAic()`：

1. 初始化 BlockMmad；
2. 计算 DP block 数、tail StreamK block 数；
3. 使用 `GetActualBlockIdx()` 调整尾部 StreamK block 的执行顺序；
4. Slice 当前 A/B/C/bias tile；
5. 为 SK K 分片建立 workspace tensor；
6. 调用共享 BlockMmad：DP 经 Fixpipe 反量化写 C，SK 将当前 K 分片的累加结果写入 workspace；
7. 所有 AIC 完成后设置 AIC→AIV flag。

### AIV 流程

`ProcessOnAiv()`：

1. 根据 tail M/N tile 数和 `skBlockNums` 计算需要参与的 AIV；
2. 等待 AIC 完成 flag；
3. 初始化专用 epilogue；
4. 归约 workspace、应用 scale/bias、写最终输出。

## Workspace 布局

每个 StreamK block 预留一个固定的 `256 × 256` tile 区域：

```text
workspace block offset
    = streamKBlockIndex × 256 × 256
```

实际 tensor 形状为当前 `singleCoreShape`，N 轴 stride 对齐到：

```text
GetVecLen() / sizeof(WorkspaceType)
```

`WorkspaceType` 由 BlockMmad 与 BlockEpilogue 共同约定：

- int8 输入：int32；
- FP8/HiFloat8 输入：float。

## Bias 处理

Kernel wrapper 根据 `BlockMmad::BIAS_IN_MMAD` 将 bias 唯一分配到 MMAD 或 AIV epilogue：

- int8 输入的 int32 bias 进入 MMAD；
- int8 输入、单路 FP32/BF16 scale 配合同类型 bias 时，bias 进入 AIV epilogue；
- FP8/HiFloat8 输入、整数编码 scale 配合 FP32 bias 时，bias 进入 MMAD；
- FP8/HiFloat8 输入、双 FP32 per-tensor scale 配合 FP32 bias 时，bias 进入 AIV epilogue；
- 无 bias 时两侧均传入空地址。

MMAD bias 只在 StreamK 的第 0 个 K 分片累加一次，其余分片不重复累加。

int8 单路 FP32/BF16 scale 的同类型 bias，以及 FP8/HiFloat8 双 FP32 scale 的
FP32 bias，都在反量化后处理。全 SK（all-SK）调度下，每个 block 都先将未反量化的累加结果
写入 workspace，再由 AIV 统一执行：

```text
INT8: 归约(分块累加结果) × X2 scale + 同类型 bias → 类型转换并写回
FP8/HiFloat8: 归约(分块累加结果) × X2 FP32 scale × X1 FP32 scale + FP32 bias → 类型转换并写回
```

DP block 通过 Fixpipe 直接写 C，不进入 AIV epilogue，因此 Host 对该组合增加
all-SK 约束。候选调度一旦包含 DP block，就回退非 StreamK MIX，避免漏加 bias
或在输出 cast 之后才加 bias。

无反量化后 Bias（post-dequant bias）的双 FP32 scale 与纯 Cube 通路一致：DP 和 SK 都先合并 X2/X1 scale，
再对合并结果应用一次 Fixpipe 掩码。因此该场景允许 DP+SK 混合调度。

## 组件组装示例

完整可编译、可运行并带 golden 校验的示例见
[quant_batch_matmul_cube](../../../../examples/acl/quant_batch_matmul/quant_batch_matmul_cube/README.md)，
对应 CSV 场景为 `qbmm_pertensor_streamk`。

```cpp
using AType = int8_t;
using BType = int8_t;
using X2ScaleType = float;
using CType = half;
using BiasType = int32_t;
using ProblemShape =
    asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
using Layout = asc::te::nd_ext_layout_ptn;
using DispatchPolicy =
    Blaze::Gemm::MatmulWithScaleFixpipeQuant<
        0, false, Blaze::Gemm::KernelQbmmPertensorMultiBlockStreamK>;
using BlockScheduler =
    Blaze::Gemm::Block::BlockSchedulerMatmulStreamK<ProblemShape>;
using BlockMmad = Blaze::Gemm::Block::BlockMmad<
    DispatchPolicy,
    AType,
    Layout,
    AscendC::Std::tuple<BType, X2ScaleType>,
    Layout,
    CType,
    Layout,
    BiasType,
    Layout>;
using BlockEpilogue =
    Blaze::Epilogue::Block::BlockEpilogueQbmmPertensorStreamK<
        typename BlockMmad::WorkspaceType,
        CType,
        DispatchPolicy,
        X2ScaleType,
        float>;
using Kernel = Blaze::Gemm::Kernel::GemmUniversal<
    ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
```

### 参数准备与 Kernel 执行

```cpp
const bool hasBias = biasGM != nullptr;
const GM_ADDR biasMmadGM = hasBias && BlockMmad::BIAS_IN_MMAD ? biasGM : nullptr;
const GM_ADDR biasEpilogueGM = hasBias && !BlockMmad::BIAS_IN_MMAD ? biasGM : nullptr;

Kernel::Params params{};
params.problemShape = {m, n, k, 1};
params.blockMmadParams = {x1GM, x2GM, yGM, biasMmadGM, perTokenScaleGM, scaleGM};
params.epilogueParams = {
    yGM, workspaceGM, scaleGM, perTokenScaleGM, biasEpilogueGM,
    biasEpilogueGM != nullptr, biasDtype};
params.schParams.usedCoreNum = usedCoreNum;
params.schParams.baseM = baseM;
params.schParams.baseN = baseN;
params.schParams.baseK = baseK;
params.schParams.singleCoreK = singleCoreK;
params.schParams.kL1 = kL1;
params.schParams.isHf32 = isHf32;
params.schParams.l2CacheMode = l2CacheMode;

Kernel kernel;
kernel(params);
```

`perTokenScaleGM == nullptr` 表示仅使用 X2 per-tensor scale；非空时该地址表示可选的
X1 per-tensor 标量，而不是 shape 为 `{M}` 的 per-token scale。Bias 必须根据
`BlockMmad::BIAS_IN_MMAD` 只传给 MMAD 或 AIV Epilogue 中的一侧。

## 数据流

```text
AIC:
                         ┌─ DP → Fixpipe dequant → C GM
A/B GM → L1/L0 → Mmad ──┤
                         └─ SK → 未反量化的 L0C 结果 → workspace
                                              │
                                              └─ AIC→AIV flag
AIV:
workspace → split-K 归约 → X2 scale → X1 scale → 反量化后 Bias → 类型转换 → C GM
```

## 约束

- 当前通路为单 batch StreamK 组装，batch 合法性由上层 tiling 保证；
- 仅支持 `KernelQbmmPertensorMultiBlockStreamK` schedule；
- workspace 大小和 offset 必须与固定 `256 × 256` block 约定一致；
- post-dequant bias 组合只允许 all-SK 调度，不能进入 DP+SK；
- shape、dtype、layout 和 StreamK 划分必须经过算子 tiling 校验；
- 本 Kernel 不应与 MX StreamK BlockMmad/Epilogue 混用。
