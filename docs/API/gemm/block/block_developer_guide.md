# Block 层开发指导

> Block 层编排 Tile 原语，完成单个 Tile 的完整计算流水线：GM→L1→L0→MMAD→输出。

## 1. 两类组件

| 组件 | 职责 |
|------|------|
| **BlockMmad** | 计算引擎：数据搬运 + MMAD 计算 |
| **BlockScheduler** | 调度器：将 ProblemShape 切分为 Tile 网格 |

两者通过 **DispatchPolicy** 绑定。DispatchPolicy 内部定义 `ScheduleType` 标签，供 Kernel 层 SFINAE 分发。

## 2. BlockMmad

### 2.1 模板特化

主模板故意 `static_assert` 阻止误用，每个实现文件提供偏特化：

```cpp
// 主模板（block_mmad_decl.h）
template <class DispatchPolicy_, class AType_, class LayoutA_, ...>
class BlockMmad {
    static_assert(always_false_v<DispatchPolicy_>, "Not implemented");
};

// 偏特化（block_mmad_matmul_basic.h）
template <uint64_t FullLoadMode_, uint64_t FusedOpType_, ...>
class BlockMmad<MatmulMultiBlockBasic<FullLoadMode_, FusedOpType_, ...>,
                AType_, LayoutA_, BType_, LayoutB_, CType_, LayoutC_, BiasType_, LayoutBias_> {
    // ...
};
```

### 2.2 文件组织与 Include 约定

主模板 `BlockMmad` 的声明独立放在 `block_mmad_decl.h` 中（gemm 与 attention 各一份）；
`block_mmad.h` 是聚合头，包含 `block_mmad_decl.h` 和**全部** `block_mmad_xx.h` 特化片段。
Include 约定（与 Kernel 层 `kernel_universal.h` 约定一致）：

- **特化片段**（`block_mmad_xx.h`）：自身 include `block_mmad_decl.h` 获得主模板声明，保持自包含；
  **不得** include `block_mmad.h`（避免循环包含）。使用方推荐只 include `block_mmad.h`，
  直接 include 单个片段亦可编译。新增 BlockMmad 时，将新文件追加到聚合列表中。

BlockScheduler / BlockEpilogue 各自定义独立类，没有公共主模板，因此不参与上述
decl/聚合头模式，使用方按需直接 include 对应头文件。

### 2.3 必须暴露的接口

```cpp
using AType = AType_;  using BType = BType_;  using CType = CType_;  // 类型别名
using DispatchPolicy = MatmulMultiBlockBasic<...>;                    // 分发键

struct Params { GM_ADDR aGmAddr, bGmAddr, cGmAddr, biasGmAddr; uint64_t mL1, nL1, kL1; ... };

void Init(const Params& params);
void operator()(TensorA& gmA, TensorB& gmB, TensorBias& gmBias, TensorC& gmC, TupleShape& blockShape);
```

### 2.4 内部流水线（K-L1/K-L0 双层循环）

```
Init(): 计算 buffer 偏移、L1 slot 布局

operator()():
  curM/curN/curK = Get<MNK_M/N/K>(blockShape)
  for kL1Iter:                          // K 轴 L1 分块
    CopyGM2L1: A/B GM→L1
    for kL0Iter:                        // K 轴 L0 分块
      CopyL12L0A/B: A/B L1→L0
      CopyL12BT: Bias L1→BIAS（仅首次）
      Mmad: L0C += AL0 × BL0
  CopyL0C2GM/UB: 输出
```

### 2.5 关键细节

**MMAD 累加控制**：
```cpp
uint8_t unitFlag = lastKIter ? FINAL_ACCUMULATION : NON_FINAL_ACCUMULATION;
bool initCmatrix = firstKIter && !isBias_;
```

**L1 Layout**（TRANS 决定 NZ/ZN）：
```cpp
using MakeLayoutAL1 = conditional_t<TRANS_A,
    FrameLayoutFormat<ZNLayoutPtn, LayoutTraitDefault<AType>>,
    FrameLayoutFormat<NZLayoutPtn, LayoutTraitDefault<AType>>>;
```

**Buffer 管理**（推荐 BufferManager，旧代码用裸数组 + SetFlag/WaitFlag）：
```cpp
BufferManager<4, 4, 2> bufMgr;  // 4 A-L1, 4 B-L1, 2 L0
```

**Ping-Pong**：`l1BufId = loopCnt & (stages - 1)`，`l0Offset = HALF_L0_SIZE * (pingPong & 1)`

**变体分支**（`if constexpr`）：
- `NON_CONTIGUOUS_TYPE == SLICE` → 用 `CopySliceGM2L1`
- `FUSED_OP_TYPE == OP_TYPE_RELU` → 用 `CopyL0C2GMTraitRelu`
- `IS_INT8_OUT` → 加载 scale，量化 Fixpipe

## 3. BlockScheduler

### 3.1 标准接口

```cpp
BlockScheduler(const ProblemShape& shape, const Params& params);
int64_t GetBlockNums() const;
TupleShape GetBlockShape(int64_t blockIdx) const;   // (blkM, blkN, k, batch)
TupleCoord GetBlockCoord(int64_t blockIdx) const;   // (mOff, nOff, kOff, batchIdx)
```

### 3.2 调度策略速查

| 调度器 | 特点 |
|--------|------|
| `MatmulBasic` | 简单网格 + Z-scan，支持 tail 二次切分 |
| `MatmulStreamK` | DP+SK 混合：DP 全 K→C，SK 切 K→workspace |
| `SwatWithTailSplit` | 紧凑 tail，过滤无效切分 |
| `IterBatchBroadcast` | A/B 独立广播，4D batch |
| `GroupedMatmul` | offset/length/sparse 三种 group list |

## 4. 与上下层的交互

```
Kernel: 创建 Scheduler + BlockMmad → 遍历 tile → Slice GM → 调用 mmad()
Block:  编排 Tile 原语 → CopyGM2L1 → CopyL12L0 → Mmad → CopyL0C2GM
```

## 5. 检查清单

- [ ] DispatchPolicy 定义 `ScheduleType`
- [ ] 暴露类型别名 + `Params` + `Init` + `operator()`
- [ ] K-L1/K-L0 双层循环，unitFlag/initCmatrix 正确
- [ ] Bias 仅首次 K 迭代加载
- [ ] `if ASCEND_IS_AIV { return; }` 守卫 MTE 操作
- [ ] `if constexpr (CURRENT_ARCH_VERSION == ...)` 守卫架构代码
