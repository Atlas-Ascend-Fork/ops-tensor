# Kernel 层开发指导

> Kernel 层是 Blaze 三层架构的最外层入口，负责多核调度、GM Tensor 创建、Tile 遍历，组合 BlockMmad + BlockScheduler + BlockEpilogue 完成完整算子。

## 1. 核心架构

### 1.1 SFINAE 分发

所有 Kernel 通过 `GemmUniversal` 主模板 + SFINAE 特化实现编译期分发：

```cpp
// 主模板（kernel_universal.h）— 故意 static_assert
template <class PS, class BM, class BE, class BS, class Enable = void>
class GemmUniversal {
    static_assert(always_false_v<BM>, "Not implemented");
};

// 特化：通过 ScheduleType 标签匹配
template <class PS, class BM, class BE, class BS>
class GemmUniversal<PS, BM, BE, BS,
    enable_if_t<is_same_v<KernelMmadMultiBlockBasic,
                          typename BM::DispatchPolicy::ScheduleType>>> {
    // 实现
};
```

Attention 算子使用对应的 `AttentionUniversal` 主模板。

### 1.2 文件组织与 Include 约定

主模板（`GemmUniversal` / `AttentionUniversal`）的声明独立放在 `kernel_universal_decl.h` 中；
`kernel_universal.h` 是聚合头，包含 `kernel_universal_decl.h` 和**全部** `kernel_xx.h` 特化片段。
Include 约定：

- **特化片段**（`kernel_xx.h`）：自身 include `kernel_universal_decl.h` 获得主模板声明，保持自包含；
  **不得** include `kernel_universal.h`（避免循环包含）。使用方推荐只 include
  `kernel_universal.h`，直接 include 单个片段亦可编译。新增 Kernel 时，将新文件追加到聚合列表中。
- 仅两个头文件不参与聚合：`kernel_wqgmm_mix_weight_prologue.h` 定义独立的
  `GmmWeightQuantMxKernel` 类、不是 `GemmUniversal` 特化；`kernel_gmm_fixpipe_quant.h` 是
  `kernel_qgmm_mix_fixpipe_quant.h` 的废弃改名兼容头。两者的使用方直接 include 对应头文件。

### 1.3 Params 聚合

每个 Kernel 定义嵌套 `Params`，聚合所有子组件参数：

```cpp
struct Params {
    ProblemShape problemShape;              // (m, n, k, batch)
    typename BlockMmad::Params mmadParams;   // GM 地址 + L1/L0 尺寸
    typename BlockEpilogue::Params epiParams; // workspace、scale 等
    typename BlockScheduler::Params schParams; // tiling 参数
};
```

## 2. 标准执行流程

### 2.1 纯 AIC 路径（无 Epilogue）

适用于 Basic、FullLoad、QBMM MX 等不需要 AIV 后处理的场景：

```cpp
__aicore__ inline void operator()(Params& params) {
    if ASCEND_IS_AIV { return; }  // AIC-only

    Init(params);
    BlockScheduler bs(problemShape, schParams);
    BlockMmad mmad; mmad.Init(mmadParams);

    for (int64_t i = curBlockIdx; i < bs.GetBlockNums(); i += coreNums) {
        auto shape = bs.GetBlockShape(i);
        auto coord = bs.GetBlockCoord(i);
        // Slice GM tensors at (coordM, coordN)
        mmad(gmBlockA, gmBlockB, gmBlockBias, gmBlockC, shape);
    }
}
```

### 2.2 AIC+AIV 双路径（有 Epilogue）

适用于 StreamK、Fixpipe、MX 量化+激活等场景：

```cpp
__aicore__ inline void operator()(Params& params) {
    Init(params);
    BlockScheduler bs(problemShape, schParams);

    if ASCEND_IS_AIC {
        BlockMmad mmad; mmad.Init(mmadParams);
        for (tiles) {
            mmad(gmBlockA, gmBlockB, ..., shape);
        }
        Gemm::Sync::NotifyVector(AIC_SYNC_AIV_FLAG);  // 通知 AIV
    }

    if ASCEND_IS_AIV {
        Gemm::Sync::WaitForCube(AIC_SYNC_AIV_FLAG);  // 等 AIC 完成
        BlockEpilogue epi; epi.Init(epiParams);
        epi();  // 后处理 + 写 GM
    }
}
```

## 3. GM Tensor 创建

```cpp
// 1. 定义 Layout
using LayoutA = FrameLayoutFormat<NDExtLayoutPtn, C0_ELEMENT<half>>;

// 2. 创建 Tensor
auto gmA = MakeTensor(MakeMemPtr<Location::GM>(aGmAddr), LayoutA{}(m, k));

// 3. Slice 到当前 tile
auto gmBlockA = gmA.Slice(MakeCoord(coordM, 0), MakeShape(curM, curK));
```

**Layout 选择规则**：
- A 矩阵（激活）：通常 `NDExtLayoutPtn`
- B 矩阵（权重）：通常 `NZLayoutPtn` 或 `NDExtLayoutPtn`
- C 矩阵（输出）：通常 `NDExtLayoutPtn`
- Scale 矩阵：`ScaleANDLayoutPtn` / `ScaleBNDLayoutPtn`

## 4. Kernel Entry 函数

每个算子的 `__global__` 入口遵循固定模式：

```cpp
template <class AType, class BType, class CType, class BiasType>
__global__ __aicore__ void matmul_kernel_entry(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR tilingGM)
{
    // 1. 读 tiling 数据
    auto tiling = *reinterpret_cast<const TilingData*>(tilingGM);

    // 2. 组装类型栈
    using DispatchPolicy = MatmulMultiBlockBasic<...>;
    using BlockMmad = BlockMmad<DispatchPolicy, AType, LayoutA, ...>;
    using BlockEpilogue = BlockEpilogueEmpty;
    using BlockScheduler = BlockSchedulerMatmulBasic<ProblemShape>;
    using Kernel = GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

    // 3. 填充 Params 并执行
    typename Kernel::Params params;
    params.problemShape = {tiling.m, tiling.n, tiling.k, tiling.batch};
    params.mmadParams.aGmAddr = aGM;
    // ...
    Kernel{}(params);
}
```

## 5. 特殊场景

### 5.1 StreamK（DP+SK 混合）

SK block 输出到 workspace，AIV 归约后写 C：

```cpp
// AIC: SK block 写 workspace
auto workspace = MakeTensor(MakeMemPtr<Location::GM>(workspaceAddr + offset), layout);
mmad(gmA, gmB, bias, gmC, workspace, shape, kCntIdx, isSkBlock);

// AIV: 归约 workspace → C
epilogue.Init(epiParams, blockShape, tileL1Shape, coord, usedCoreNum, isSkScene);
epilogue();
```

### 5.2 分组 Matmul（GMM）

每组独立 M/N/K，需动态更新 ProblemShape：

```cpp
for (int g = 0; g < groupNum; ++g) {
    SetMNK(groupList[g]);           // 更新 m/n/k
    scheduler.UpdateNextProblem();   // 刷新调度器
    ProcessSingleGroup();            // 遍历该组的 tile
}
```

### 5.3 Weight Prologue（AIV 预处理权重）

AIV 做 W4→W8 转换写入 L1，AIC 消费：

```cpp
if ASCEND_IS_AIC {
    mmad();  // 读 L1 中已转换的权重
}
if ASCEND_IS_AIV {
    blockPrologue();  // FP4→FP8 转换，写 L1
}
// 通过块内同步 Flag（Gemm::Sync）交替同步（ready/free 协议）
```

## 6. 常用辅助操作

| 操作 | API |
|------|-----|
| HF32 模式 | `SetHF32(isHf32)` / `UnsetHF32(isHf32)` |
| L2 Cache | `SetL2Cache(gmA, gmB, mode)` |
| 原子加 | `AscendC::SetAtomicAdd<float>()` / `SetAtomicNone()` |
| 跨核同步 | `Gemm::Sync::NotifyVector/WaitForVector`（AIC 侧）、`Gemm::Sync::NotifyCube/WaitForCube`（AIV 侧）；部分组件直接使用 `AscendC::CrossCoreSetFlag/WaitFlag` |

## 7. 检查清单

- [ ] `GemmUniversal` 特化的 SFINAE 条件匹配 `ScheduleType`
- [ ] `Params` 聚合 ProblemShape + BlockMmad::Params + BlockEpilogue::Params + BlockScheduler::Params
- [ ] AIC-only 路径加 `if ASCEND_IS_AIV { return; }`
- [ ] AIC+AIV 路径用 `Gemm::Sync` 块内同步 Flag
- [ ] GM Tensor 用正确的 LayoutPattern 创建
- [ ] Tile 循环用 `blockIdx += coreNums` 步进
- [ ] Kernel entry 函数签名：`__global__ __aicore__ void xxx_kernel_entry(GM_ADDR..., GM_ADDR tilingGM)`
