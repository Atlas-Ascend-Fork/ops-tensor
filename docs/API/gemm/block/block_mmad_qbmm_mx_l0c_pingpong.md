# Block Mmad QBMM MX L0C PingPong
> [代码位置](../../../../include/blaze/gemm/block/block_mmad_qbmm_mx_l0c_pingpong.h)

## 功能说明
`block_mmad_qbmm_mx_l0c_pingpong` 是 QBMM MX 量化场景的 BlockMmad 性能模板，基于 Tensor API 实现 AIC 侧矩阵乘计算。

该模板复用普通 MX BlockMmad 的 MxFP4/MxFP8 量化输入、ScaleA/ScaleB 反量化、K 轴 Padding、L1/L0 缓冲能力，并针对 L0C 输出引入 ping-pong 机制：
- 当当前 L0C 输出块可放入半个 L0C 时，直接开启 L0C ping-pong。
- 当当前 L0C 输出块无法放入半个 L0C 时，在 KL1 尾轮按 N 方向拆分输出子块，以满足 L0C ping-pong 使用条件。

## 使用限制
- 仅支持 `__NPU_ARCH__ == 3510`。
- 仅支持 AIC 计算，不支持 AIV 计算。
- 仅支持 `MatmulWithScaleMxL0CPingpong` 调度策略。
- 输入量化类型支持 MxFP4/MxFP8，Scale 类型固定为 `fp8_e8m0_t`。
- 输出通过 Fixpipe 搬出到输出 Tensor，支持 GM 或 UB，不支持 workspace。
- N 方向拆分由模板内部完成，调用侧传入的 `singleShape` 需要与 GM Tensor Slice 保持一致。
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
### Params
```cpp
struct Params {
    GM_ADDR aGmAddr{nullptr};
    GM_ADDR bGmAddr{nullptr};
    GM_ADDR cGmAddr{nullptr};
    GM_ADDR biasGmAddr{nullptr};
    GM_ADDR scaleAGmAddr{nullptr};
    GM_ADDR scaleBGmAddr{nullptr};
};
```

### L1Params
```cpp
struct L1Params {
    uint64_t kL1;
    uint64_t scaleKL1;
    uint64_t l1BufNum;
};
```

`l1BufNum` 支持 2、3 或 4 缓冲；其中 3 缓冲用于在 L1 容量不足以放下 4 缓冲时保留三缓冲流水。

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
