# Block Mmad QBMM MX L0C PingPong
> [代码位置](../../../../include/blaze/gemm/block/block_mmad_qbmm_mx_l0c_pingpong.h)

## 功能说明
`block_mmad_qbmm_mx_l0c_pingpong` 是 QBMM MX 量化场景的 BlockMmad 性能模板，基于 Tensor API 实现 AIC 侧矩阵乘计算。

该模板复用普通 MX BlockMmad 的 MxFP4/MxFP8 量化输入、ScaleA/ScaleB 反量化、K 轴 Padding、L1/L0 缓冲能力。未传入显式子块计划时，沿用原 L0C ping-pong 逻辑：
- 当当前 L0C 输出块可放入半个 L0C 时，直接开启 L0C ping-pong。
- 当当前 L0C 输出块无法放入半个 L0C 时，在 KL1 尾轮按 N 方向拆分输出子块，以满足 L0C ping-pong 使用条件。

传入非零 `mL0/nL0` 时，由 Host 决定不切分、切 M 或切 N，所有 K-L1 段执行相同计划。

## 使用限制
- 仅支持 `__NPU_ARCH__ == 3510`。
- 仅支持 AIC 计算，不支持 AIV 计算。
- 仅支持 `MatmulWithScaleMxL0CPingpong` 调度策略。
- 输入量化类型支持 MxFP4/MxFP8，Scale 类型固定为 `fp8_e8m0_t`。
- 输出通过 Fixpipe 搬出到输出 Tensor，支持 GM 或 UB，不支持 workspace。
- 子块拆分由模板内部执行，调用侧传入的 `singleShape` 需要与 GM Tensor Slice 保持一致。
- SplitK 场景下，仅最后一个 SplitK 分片写回输出。

### ScaleB 独立布局

`LayoutB` 参数保留原有矩阵布局写法，也接受 `tuple<WeightLayout, ScaleLayout>`。
前者沿用由 B 转置决定的 ND scale 布局；后者分别指定 B 与 ScaleB 布局。
ScaleNZ 使用 `nn_layout_ptn`，不随 B 转置改变。模板形参和调用接口不变。

E8M0 ScaleNZ 的逻辑 shape 为 `[batch_B..., ceil(K/64), N, 2]`，storage shape 为
`[batch_B..., ceil(N/16), ceil(K/64), 16, 2]`。应在 format cast 前交换真实轴和数据；
不能以 reshape 代替换轴，也不能在打包后转置 ScaleNZ。

此布局约定独立于矩阵类型；QBMM 的接口集成仍遵守 WeightNZ 的既有类型和转置限制。
shape 校验不能恢复历史转置操作，`N == ceil(K/64)` 时也无法区分同形状的错误轴含义。

## 外部接口
### L1Params
```cpp
struct L1Params {
    uint64_t kL1;
    uint64_t scaleKL1;
    uint64_t l1BufNum;
};
```

`l1BufNum` 支持 2、3 或 4 缓冲；其中 3 缓冲用于在 L1 容量不足以放下 4 缓冲时保留三缓冲流水。

### Params
```cpp
struct Params {
    GM_ADDR aGmAddr{nullptr};
    GM_ADDR bGmAddr{nullptr};
    GM_ADDR cGmAddr{nullptr};
    GM_ADDR biasGmAddr{nullptr};
    GM_ADDR scaleAGmAddr{nullptr};
    GM_ADDR scaleBGmAddr{nullptr};
    // Optional compute M/N; zero keeps the legacy path.
    uint32_t mL0{0};
    uint32_t nL0{0};
    ProblemShape problemShape{};
    BlockShape l1TileShape{};
    L1Params l1Params{};
    bool isBias{false};
    bool dbL0C{false};
    uint64_t splitKNum{1};
    uint64_t kL0{0}; // A zero kL0 selects the legacy GemmUniversal entry.
};
```

### Init
```cpp
__aicore__ inline void Init(
    const ProblemShape& problemShape,
    const BlockShape& l0TileShape,
    const L1Params& l1Params,
    bool isBias,
    bool dbL0C,
    uint64_t splitKNum = 1);
```

用于初始化问题规模、L0 tile shape、L1 切分参数、Bias 状态、L0C ping-pong 开关和 SplitK 数量。

### operator()
```cpp
template <
    typename TensorA_, typename TensorB_, typename TensorScaleA_, typename TensorScaleB_, typename TensorBias_,
    typename TensorC_>
__aicore__ inline void operator()(
    TensorA_ const& gmA,
    TensorB_ const& gmB,
    TensorScaleA_ const& gmScaleA,
    TensorScaleB_ const& gmScaleB,
    TensorBias_ const& gmBias,
    TensorC_ const& gmC,
    BlockShape const& singleShape,
    uint64_t splitKIdx = 0);
```

执行当前 Block 的 MX 量化矩阵乘计算。


### 通过 Params 一次完成初始化

```cpp
__aicore__ inline void Init(const Params& params);
```

调用方在 `Params` 中填写问题规模、块形状、L1 缓冲和开关。NN 在选中 PingPong 模板的 Kernel 入口填好这些字段，Tensor 直接把 `mmadParams` 传入 Block。`mL0/nL0` 的单位为元素，两者非零时执行 Host 显式计划，任一为零时沿用旧策略。

```cpp
BlockMmad::Params params{};
params.problemShape = problemShape;
params.l1TileShape = {baseM, baseN, kL1, 0};
params.kL0 = kL0;
params.l1Params = l1Params;
params.isBias = isBias;
params.dbL0C = dbL0C;
params.mL0 = mL0;
params.nL0 = nL0;
params.splitKNum = splitKNum;
blockMmad.Init(params);
```

`l1TileShape` 的 M/N/K 是 GM 到 L1 的搬运块尺寸，K 取 `kL1`。`mL0/nL0/kL0` 是 MMAD 的计算子块尺寸，`kL0` 取 Host tiling 的 `baseK`。
`qbmmParams.baseM/baseN` 保留原搬运和调度块大小。`L1Params::kL1` 为旧接口保留，单参数 `Init` 从 `l1TileShape` 读取搬运 K。

调用方应检查 M/N 子块能够整分基本块，`kL0` 能够整分 `kL1`，每个 C 子块不超过半 L0C，
A/B 子块可放入各自半 L0，并满足布局对齐。支持不切分、切 N 两份、切 M 两份；
NN 在 DAV_3510 的原生 MXFP8/MXFP4 路径按上述容量及对齐约束选择子块。QBMM 的带 batch 和无 batch 入口均传递 mL0/nL0。带 bias 时 L1 容量计算包含 bias 缓冲。InplaceAdd 的输入类型以该算子的接口规格为准。

所有 K-L1 段按同一计划执行。切片使用独立紧凑 L0C 布局和固定地址，仅最后 K 子块搬出。
当 `kL1 <= 2 * kL0` 时，切 N 的后片复用 A/ScaleA，切 M 的后片复用 B/ScaleB；
否则正常重新搬入。该传参调整不改变 GM→L1 分块、L1 分区或 Fixpipe 输出路径。

### 旧调用与兼容范围

原多参数 `Init(problemShape, blockShape, l1Params, isBias, dbL0C, splitKNum)` 保留。内部仍从旧 `blockShape` 取得基本块 M/N 和计算 K，搬运 K 取 `l1Params.kL1`。
旧 `GemmUniversal` 调用中的 `mmadParams.kL0` 默认为零，因此继续使用原多参数 `Init`。省略 `mL0/nL0` 时沿用原有尾段切分规则。

新增字段带默认值，普通六地址聚合初始化仍可编译；但 `Params` 大小发生变化，
包含它的外层 `GemmUniversal::Params` 布局也随之变化，不保证新旧布局或二进制混用兼容。
依赖固定大小、序列化、结构化绑定的调用方需要适配。
`Init` 新增同名重载后，`auto p = &BlockMmad::Init` 需要显式指定原函数指针类型。
使用新头文件的调用方应重新编译；外部调用者的完整兼容性回归尚未完成。
