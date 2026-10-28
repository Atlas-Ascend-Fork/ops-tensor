# Block Mmad Weight Prologue MX
> [代码位置](../../../../include/blaze/gemm/block/block_mmad_weight_prologue_mx.h)

## 功能说明

这是 QBMM/GMM 共用的 arch35 BlockMmad，负责 AIC（Cube）侧的单 tile
MX MMAD。B 和 bias 由 Kernel 内的 AIV（Vector）前处理写入共享 L1，因此本 BlockMmad 不接收
B/Bias Tensor，也不执行 B GM→L1 搬运。

**模板接口**：[BlockMmad 基础框架](./block_mmad.md)

**配套组件**：[QBMM Kernel](../kernel/kernel_matmul_mix_weight_prologue.md)、
[GMM Kernel](../kernel/kernel_wqgmm_mix_weight_prologue.md) 和
[Weight Quant MX 前处理 Tile](../tile/tile_weight_quant_mx_preprocess.md)

## 特殊约束

### 架构和 policy

仅支持 `__NPU_ARCH__ == 3510`（arch35）。`DispatchPolicy` 指定 K 分块方式和 ScaleB 来源：

- `MatmulWithWeightQuantMx`：固定 K 分块，AIC 从 GM 搬运 ScaleB。
- `GroupedMatmulWithWeightQuantMx`：执行 GMM Kernel 根据当前 M/N 选择的 ka/kb/k0，A 在 ka 窗口内跨 B 块复用；
  ScaleB 由 AIV 转换并写入 L1，AIC 只消费其 L1 视图。

GMM 的构造参数包含 L1 分区、缓冲容量、ready/free 标志和初始 B 槽位写入许可，
由配套 Kernel 提供；ka/kb/k0 随每个 tile 的四个 Tensor 传入计算接口。
两条路径共用 L0 搬运、bias 累加、K64 补零、MMAD 和 Fixpipe 写回。

### 模板类型约束

- `ATypeTuple`、`LayoutATuple`、`BTypeTuple`、`LayoutBTuple` 必须都是二元 tuple。
- Block 的静态约束要求 A 类型占 1 字节；当前配套 prologue/算子组装固定为 `fp8_e4m3fn_t`。
- B 类型必须为 packed FP4，ScaleA/ScaleB 必须为 `fp8_e8m0_t`。
- 配套 GMM Kernel 的 A 输入使用 `nd_ext_layout_ptn` 布局。
- `LayoutB` 必须是转置布局：NZ 使用 `zn_layout_ptn`，ND 使用 `dn_ext_layout_ptn`。
- 当前算子组装中 `LayoutC`/`LayoutBias` 均为 `nd_ext_layout_ptn`；bias 可选，有 bias 时
  `BiasType` 与 `CType` 一致。

### 缓冲和同步

- QBMM 调用方必须传入 2 或 4 个 L1 buffer；GMM Kernel 固定使用 2 个。
  缓冲数量应与配套 host tiling 及 AIV prologue 的配置一致。
- AIV 是 B/bias L1 的生产者，AIC 等待 ready 标志；AIC 完成当前 K window 后发送 free 标志。
- A、ScaleA 的 GM→L1 和全部操作数的 L1→L0 搬运由本组件负责；QBMM 还搬运 ScaleB 到 L1。
  B/bias 的 L1 地址和槽位所有权来自
  `WeightL1Storage` 协作契约。

## 模板类型和别名

```cpp
template <
    class DispatchPolicy,
    class ATypeTuple, class LayoutATuple, class BTypeTuple, class LayoutBTuple,
    class CType, class LayoutC, class BiasType, class LayoutBias>
class BlockMmad;
// DispatchPolicy 为 MatmulWithWeightQuantMx 或 GroupedMatmulWithWeightQuantMx。
```

`dispatch_policy.h` 中的策略只声明调度类型和两类行为，不包含 K 分块算法：

- QBMM：`K_SCHEDULE=WeightPrologueMxKSchedule::FIXED`，`SCALE_B_SOURCE=WeightPrologueMxScaleBSource::AIC`。
- GMM：`K_SCHEDULE=WeightPrologueMxKSchedule::DYNAMIC`，`SCALE_B_SOURCE=WeightPrologueMxScaleBSource::AIV`。

公共实现按上述枚举选择入口和 ScaleB 搬运分支。动态 K 的选择属于
[GMM Kernel](../kernel/kernel_wqgmm_mix_weight_prologue.md#动态-k-分块选择)：
AIC、AIV 共用 Kernel 内的 `CalcDynamicKBlock`，Block 只按传入的 ka/kb/k0 执行。
GMM 的 L1 布局与握手配置由 Params 工厂提供。

| 模板参数 | 公开类型别名 |
| :--- | :--- |
| `ATypeTuple` | `AType`、`ScaleAType` |
| `LayoutATuple` | `LayoutA`、`LayoutScaleA` |
| `BTypeTuple` | `BType`、`ScaleBType` |
| `LayoutBTuple` | `LayoutB`、`LayoutScaleB` |
| 独立类型/layout 参数 | `CType`、`LayoutC`、`BiasType`、`LayoutBias` |

## 特殊数据结构

### `Params`

```cpp
// QBMM uses host tiling; the GMM kernel supplies storage geometry and synchronization parameters.
struct Params {
    GM_ADDR aGmAddr;
    GM_ADDR scaleAGmAddr;
    GM_ADDR scaleBGmAddr;
    GM_ADDR cGmAddr;
    L1TileShape l1TileShape;
    L0TileShape l0TileShape;
    uint64_t l1BufferNum;
    bool hasBias;
    uint64_t aBufferBytes;
    uint64_t bBufferBytes;
    GM_ADDR biasGmAddr;
    bool explicitL1Offsets;
    uint64_t weightOffsets[4];
    uint64_t biasOffsets[4];
    uint64_t aOffsets[4];
    uint64_t scaleAOffsets[2];
    uint64_t scaleBOffsets[2];
    uint16_t readyFlag;
    uint16_t freeFlag;
    bool seedWeightCredits;
};
```

| 字段 | 说明 |
| :--- | :--- |
| `aGmAddr` | A 矩阵 GM 地址 |
| `scaleAGmAddr` / `scaleBGmAddr` | ScaleA/ScaleB GM 地址 |
| `cGmAddr` | 输出 C GM 地址 |
| `biasGmAddr` | Bias GM 地址，由配套 Kernel 的 AIV 分支使用 |
| `l1TileShape` | `(baseM, baseN, kL1, scaleKL1)` |
| `l0TileShape` | `(baseM, baseN, baseK)` |
| `l1BufferNum` | QBMM 为 2 或 4；GMM 为 2 |
| `hasBias` | 是否消费 AIV 写入的 bias |
| `aBufferBytes` / `bBufferBytes` | 非零时覆盖 A/B 物理槽位容量，零时由 tile shape 推导 |
| `explicitL1Offsets` / `*Offsets` | 使用调用方指定的 L1 分区，保证与 AIV producer 的分区一致 |
| `readyFlag` / `freeFlag` | AIV ready / AIC free 的跨核标志编号 |
| `seedWeightCredits` | 是否由 AIC 构造时发出初始 weight buffer credit |

### GMM 参数配置

```cpp
__aicore__ inline explicit BlockMmad(const Params& params);
```

构造函数根据 `Params` 初始化缓冲区和同步资源，构造参数应包含有效的 tile 尺寸、缓冲容量、L1 偏移及同步配置。
GMM Kernel 的 `params.mmad` 提供 A、ScaleA、ScaleB、C 和 Bias 的 GM 地址，完整的 BlockMmad 构造参数由 Kernel 配置。
AIC 和 AIV prologue 使用相同的 L1 分区约定；AIV 只读引用的地址配置在其执行期间必须有效。
动态 K 的容量约束与 AIC 的 A 缓冲容量一致。
GMM 使用 2 个 B 槽位，ready/free 标志为 1/0，`seedWeightCredits=false`。
QBMM 按 host tiling 设置固定 K 尺寸，并使用默认的 ready/free 标志 6/8 和初始 credit。

### 协作接口

`SyncProtocol` 提供 QBMM 默认 ready/free 标志的模式和编号；GMM 使用 Params 中的编号。
跨核握手通过 `sync.h` 的 CV 同步接口执行，AIC 使用 `PIPE_MTE1`，AIV 使用 `PIPE_MTE3`。
`WeightL1Storage` 提供 `Init`、
`MakeWeightTensor`、`MakeBiasTensor`、`WeightBufferSize`、`BiasBufferSize` 和 `BuffersPerHalf`。
它们是 Kernel prologue 与 BlockMmad 之间的内部协作契约，调用方通常通过 Kernel 使用，不应自行创建
第二套 L1 地址或同步协议。

## 特殊成员方法

### 构造函数和析构函数

```cpp
__aicore__ inline explicit BlockMmad(const Params& params);
__aicore__ inline ~BlockMmad();
```

构造函数初始化 L1 存储规划和各级缓冲槽位；析构函数关闭 MM layout transform。

### `operator()` 函数

```cpp
template <typename TensorA, typename TensorScaleA, typename TensorScaleB, typename TensorC>
__aicore__ inline void operator()(
    const TensorA& tensorA, const TensorScaleA& tensorScaleA,
    const TensorScaleB& tensorScaleB, const TensorC& tensorC);

// GMM：Kernel 为当前 tile 显式提供 K 分块。
template <typename TensorA, typename TensorScaleA, typename TensorScaleB, typename TensorC>
__aicore__ inline void operator()(
    const TensorA& tensorA, const TensorScaleA& tensorScaleA,
    const TensorScaleB& tensorScaleB, const TensorC& tensorC,
    uint64_t kaL1Size, uint64_t kbL1Size, uint64_t l0KSize);
```

A、ScaleA、C Tensor 为当前 scheduler tile 的 GM slice。QBMM 的 ScaleB 也传当前 tile 的 GM slice；
GMM 的入口额外接收 ka/kb/k0；AIC 不解引用或切分传入的 ScaleB GM Tensor，直接消费 AIV 写入的 L1 ScaleB。
B/bias 不作为参数传入，而由 AIV 通过共享 L1 提供。
QBMM 的固定 K 来自构造时的 tiling 参数；GMM 的 ka/kb/k0 必须由 Kernel 随当前 tile 传入。
两个入口在编译期限制对应的 policy，避免 GMM 误用固定 K 入口。

### 动态 K 参数约束

GMM Kernel 的 `CalcDynamicKBlock` 根据当前 tile 的 M/N 尺寸生成 ka/kb/k0，
分别表示 A 的 L1 驻留窗口、B 的 L1 分块尺寸和 L0 分块尺寸。该组参数用于 Kernel 与
BlockMmad 之间的内部调用，其约束如下：

- ka、kb、k0 均非零；ka 为 kb 的整数倍，kb 为 k0 的整数倍，k0 按 64 元素对齐。
- kb 整除 Scale K 窗口大小。GMM 的 Scale K 窗口为 4096，ka 与该窗口无整除要求，
  例如 `ka=768、kb=256、k0=128`。
- 令 `M16=Align16(tileM)`、`N16=Align16(tileN)`，A、B 各自单个 L1 缓冲区的容量分别为 `ABytes`、`BBytes`，
  则 `M16*ka*sizeof(AType) <= ABytes`，`N16*kb*sizeof(ConvertedBType) <= BBytes`。
  B 的 L1 缓冲区容量按 AIV 转换后的元素类型计算。
- `M16*k0*sizeof(AType) <= TOTAL_L0A_SIZE/2`，
  `N16*k0*sizeof(ConvertedBType) <= TOTAL_L0B_SIZE/2`，满足 L0 双缓冲容量要求。

L1 分区、缓冲容量和 ready/free 标志编号由 GMM Kernel 配置，AIC 与 AIV 使用相同的地址及握手约定。

## 计算流程

1. Scale 窗口开始时，AIC 在对应 L1 槽位锁保护下搬运 ScaleA；QBMM 还由 AIC 搬运 ScaleB，GMM 的 ScaleB 由 AIV 转换后写入 L1。
2. A 窗口开始时，AIC 在对应 L1 槽位锁保护下搬运 A，随后清理当前 B 块的 `[Align32(K), Align64(K))` 物理尾部。
3. 等待 AIV 为当前 L1 weight buffer 设置 ready 标志。
4. QBMM 按固定 `baseK`、GMM 按动态 `k0` 将 A、B、ScaleA、ScaleB 搬入 L0，
   使用 `MmadTraitMX` 累加到 L0C。
5. 首个 K tile 消费 AIV 已按 `1/64` 缩放的 bias，Fixpipe 将结果写回 C。
6. 释放 weight buffer，允许 AIV 继续生产下一 tile。

GMM 的 A 槽位按 ka 窗口切换，Scale 槽位按独立的 4096 元素 K 窗口切换；两者均可跨多个 kb 块复用。
GM 窗口偏移及切片坐标使用 `int64_t`，缓冲容量和缓冲索引仍使用无符号整数。

## Kernel 内部调用

该 Block 不能脱离 AIV producer 独立调用。QBMM 构造时向 AIV 释放初始 weight buffer credit，GMM 使用配套 prologue 的初始协议；执行时会等待 AIV
按相同 tile 序列写入 B/bias 并发送 ready 标志。完整类型组装和参数构造见
[Kernel Matmul Mix Weight Prologue](../kernel/kernel_matmul_mix_weight_prologue.md#调用示例)。

```cpp
BlockMmad blockMmad(params.mmadParams);
// AIV prologue and AIC must consume the same scheduler tile before this call completes.
blockMmad(gmBlockA, gmBlockScaleA, gmBlockScaleB, gmBlockC);
```

[GMM Kernel](../kernel/kernel_wqgmm_mix_weight_prologue.md) 内部调用示意：

```cpp
// MMAD parameters and K blocking belong to the GMM kernel.
BlockMmad blockMmad(mmadParams_);
uint64_t kaL1Size = 0;
uint64_t kbL1Size = 0;
uint64_t l0KSize = 0;
CalcDynamicKBlock(mL1Size, nL1Size, kaL1Size, kbL1Size, l0KSize);
// ScaleB is provided by the paired AIV prologue in L1.
blockMmad(gmBlockA, gmBlockScaleA, tensorScaleBGm, gmBlockC, kaL1Size, kbL1Size, l0KSize);
```

两条路径均通过构造参数中的 `hasBias` 设置 Bias 开关。

## 适用场景

- `GemmUniversal` Weight Prologue 特化的 AIC 计算侧。
- 配套 Kernel 支持 arch35 MXA8W4 Weight ND/NZ 输入，并由 AIV 负责 packed FP4→FP8 和 bias 前处理。
- 不适合单独调用；没有对应 AIV producer 时，B/bias L1 数据和 ready 标志不存在。
