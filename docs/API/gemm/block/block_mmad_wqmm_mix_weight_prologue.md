# Block Mmad WQMM Mix Weight Prologue

> [代码位置](../../../../include/blaze/gemm/block/block_mmad_wqmm_mix_weight_prologue.h)

## 功能说明

该组件是 `BlockMmad<MatmulWithWeightAntiquant<...>, ...>` 的偏特化，负责 AIC 侧的权重反量化矩阵乘。
AIV Prologue 将量化权重 `weight` 转换为输入 `x` 的类型并写入共享 L1；BlockMmad 装载 `x` 和可选的
`bias`，执行矩阵乘并通过 Fixpipe 将输出 `y` 写回 GM。

每次调用处理一个 M/N 分块，内部完成完整 K 轴的 FP32 累加。计算接口不接收 weight Tensor，
而是等待 Prologue 提供对应的 L1 权重分块，并在读取完成后通知 AIV 可以复用缓冲区。

## 使用限制

- 使用 `MatmulWithWeightAntiquant` 策略。默认调度类型服务于 DAV_3510 的 WQMM；
  GroupedMatmul 在组装处直接指定 `KernelWqgmmMixAntiquant` 调度类型和单 AIV 配置，面向 DAV_9201。
- 矩阵乘计算由 AIC 执行；AIV 可初始化组件并查询共享权重缓冲区地址。
- 输入 `x` 和输出 `y` 的类型支持 FP16、BF16；量化权重支持 INT8、FP8 E4M3、HiFloat8。
  反量化参数类型必须与 `x` 相同。配套 Kernel 的类型组合及布局限制见
  [WQMM Kernel 支持范围](../kernel/kernel_wqmm_mix_antiquant.md#支持范围)。
  GroupedMatmul 另支持 FP8 E5M2，x/y 同类型且为不转置 ND，bias 为 x 的类型或 BF16 x 配 FP32 bias；
  完整组合见 [GroupedMatmul Kernel](../kernel/kernel_wqgmm_mix_antiquant.md#输入与输出)。
- 必须配合 AIV Prologue 使用，双方按相同的 M/N 分块顺序及 K 分块配置处理数据。
  缺少对应的权重就绪信号时，AIC 无法完成计算。
- 输出写回 GM。分块配置须满足 L1、L0 和偏置缓冲区的容量及对齐要求；尾块计算使用真实有效长度。

数学上的权重为 `(K,N)`，而模板参数 `LayoutB` 的坐标约定为 `(N,K)`。
`nd_ext_layout_ptn` 对应 GM `(N,K)` 存储及 L1 ZN 布局；
`dn_ext_layout_ptn` 对应 GM `(K,N)` 存储及 L1 NZ 布局。L1 布局均按 `(K,N)` 描述。

## 外部接口

### Params

```cpp
using L1TileShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
using L0TileShape = asc::te::shape<int64_t, int64_t, int64_t>;

struct Params {
    GM_ADDR aGmAddr{nullptr};
    GM_ADDR cGmAddr{nullptr};
    GM_ADDR biasGmAddr{nullptr};
    L1TileShape l1TileShape;
    L0TileShape l0TileShape;
    bool hasBias{false};
    uint64_t kSize{0};
};
```

| 成员 | 含义 |
| --- | --- |
| `aGmAddr` | 输入 `x` 的 GM 起始地址，由配套 Kernel 用于构造 `(M,K)` Tensor |
| `cGmAddr` | 输出 `y` 的 GM 起始地址，由配套 Kernel 用于构造 `(M,N)` Tensor |
| `biasGmAddr` | 可选 `bias` 的 GM 起始地址，由配套 Kernel 用于构造 `(1,N)` Tensor；未启用时可为 `nullptr` |
| `l1TileShape` | `(baseM,baseN,KaL1,KbL1)`，后两项分别为输入和权重的 L1 K 分块配置，单位为元素 |
| `l0TileShape` | `(baseM,baseN,baseK)`，L0 分块配置，单位为元素 |
| `hasBias` | 是否读取并叠加 `bias`，默认 `false` |
| `kSize` | 完整 K 轴长度，单位为元素；调用前须设置为实际长度 |

BlockMmad 本身不读取上述三个 GM 地址字段，实际搬运地址来自 `operator()` 传入的 Tensor。
通过配套 Kernel 调用时，由 Kernel 读取这些字段并构造、切分 Tensor。
`KaL1` 与 `KbL1` 的大小关系决定输入装载方式，见[输入装载方式](#输入装载方式)。

### 构造函数和析构函数

```cpp
__aicore__ inline BlockMmad();
__aicore__ inline ~BlockMmad();
```

默认构造后，由 Kernel 调用 `Init` 完成初始化；析构函数在 AIC 上关闭矩阵布局转换。

### Init

```cpp
__aicore__ inline void Init(const Params& params);
```

根据 `params` 配置片上缓冲区，在 AIC 上启用矩阵布局转换。
Kernel 在 AIC/AIV 分支之前调用一次，完成后才能调用 `GetShareMemPtr` 或 `operator()`。

### GetShareMemPtr

```cpp
template <typename MemoryLocation, typename DType>
__aicore__ inline auto GetShareMemPtr(uint32_t bufferId) const;
```

返回指定共享权重缓冲区的类型化内存指针，供 AIV Prologue 构造 L1 Tensor。

| 参数 | 约束 |
| --- | --- |
| `MemoryLocation` | 必须为 `asc::te::location::l1` |
| `DType` | 必须与输入 `x` 的类型 `AType` 一致，即反量化后的权重类型 |
| `bufferId` | 取 0 或 1，指定两份共享权重缓冲区之一 |

该接口只提供地址，不执行搬运或核间同步。调用方通过配套 Prologue 管理权重写入及缓冲区复用。

### GetWeightL1Tensor

```cpp
__aicore__ inline auto GetWeightL1Tensor(uint32_t bufferId, int64_t k, int64_t n) const;
```

返回指定权重槽中逻辑形状为 `(k,n)` 的 L1 Tensor。GroupedMatmul 的 AIV 前处理通过此接口写入反量化权重；
`bufferId` 为 0 或 1。布局由 `LayoutB` 决定，与 AIC 读取同一段权重时的布局一致。

### operator()

```cpp
template <class TensorA, class TensorBias, class TensorC, class Shape>
__aicore__ inline void operator()(
    const TensorA& tensorA,
    const TensorBias& tensorBias,
    TensorC& tensorC,
    const Shape& actualShape);
```

在 AIC 上执行当前 M/N 分块的矩阵乘，遍历完整 K 轴后写回输出。

| 参数 | 含义 |
| --- | --- |
| `tensorA` | 当前输入 `x` 的 GM 切片，形状为 `(tileM,K)`，包含完整 K 轴 |
| `tensorBias` | 当前 N 分块的 `bias` GM 切片，形状为 `(1,tileN)`；`hasBias=false` 时不读取 |
| `tensorC` | 当前输出 `y` 的 GM 切片，形状为 `(tileM,tileN)` |
| `actualShape` | `(tileM,tileN)`，表示当前分块的真实有效长度，单位为元素 |

Tensor 的切片范围必须与 `actualShape` 和初始化参数中的 `kSize` 一致。
未启用偏置时仍需传入符合接口类型要求的 `tensorBias`。

## 调用流程

1. 配置 `Params`，在 AIC/AIV 分支之前默认构造 BlockMmad 并调用 `Init`。
2. AIV 通过 `GetShareMemPtr` 获取共享 L1 地址，由 Prologue 搬运并反量化当前分块的权重。
3. AIC 根据 Scheduler 的结果准备 `x`、`bias`、`y` 的 GM 切片，调用 `operator()`。
4. BlockMmad 装载输入和偏置，逐个 K 分块等待权重就绪，将数据搬入 L0 并执行 MMAD。
   对应权重读取完成后，通知 AIV 可以复用缓冲区。
5. 完成当前 M/N 分块的 K 轴累加后，通过 Fixpipe 写回 `y`。

AIC 和 AIV 并行执行，通过核间信号协调上述数据读写。输入预取由 Kernel 负责。
以下为 Kernel 内的调用片段，`params`、GM Tensor 和分块坐标均由 Kernel 准备。

初始化组件并构造共享地址提供者：

```cpp
BlockMmad blockMmad;
blockMmad.Init(params.mmadParams);
auto getSharedWeightMem = [&blockMmad](uint32_t bufferId) __aicore__ {
    return blockMmad.template GetShareMemPtr<asc::te::location::l1, AType>(bufferId);
};
```

Kernel 将 `getSharedWeightMem` 传给 AIV Prologue。在 AIC 侧，每个 M/N 分块执行：

```cpp
auto blockX = gmX.slice(
    asc::te::make_coord(coordM, 0UL), asc::te::make_shape(tileM, kSize));
auto blockBias = gmBias.slice(
    asc::te::make_coord(0UL, params.mmadParams.hasBias ? coordN : 0UL),
    asc::te::make_shape(1UL, tileN));
auto blockY = gmY.slice(
    asc::te::make_coord(coordM, coordN), asc::te::make_shape(tileM, tileN));
blockMmad(blockX, blockBias, blockY, asc::te::make_shape(tileM, tileN));
```

以上片段依赖配套 AIV Prologue，完整组件组装和参数准备见
[Kernel 调用示例](../kernel/kernel_wqmm_mix_antiquant.md#调用示例)。

## L1 空间划分

缓冲区是为分块数据预留的片上空间。以下地址和容量均以**字节**计：

- `l1Bytes`：L1 总容量。
- `weightBytes = baseN*KbL1*sizeof(AType)`：每份反量化权重容量，`AType` 为输入 `x` 的类型。
- `biasBytes = hasBias ? 4096 : 0`：每份 `bias` 预留容量。

| 区域 | 地址范围，左闭右开 |
| --- | --- |
| 权重缓冲区 0 | `[0, weightBytes)` |
| `bias` 缓冲区 0 | `[weightBytes, weightBytes+biasBytes)` |
| 输入 `x` 缓冲区 0 | `[weightBytes+biasBytes, l1Bytes/2)` |
| 输入 `x` 缓冲区 1 | `[l1Bytes/2, l1Bytes-weightBytes-biasBytes)` |
| `bias` 缓冲区 1 | `[l1Bytes-weightBytes-biasBytes, l1Bytes-weightBytes)` |
| 权重缓冲区 1 | `[l1Bytes-weightBytes, l1Bytes)` |

每份 `x` 缓冲区容量为 `xHalfBytes = l1Bytes/2-weightBytes-biasBytes`，
两份相邻缓冲区的总容量为 `2*xHalfBytes`。调用方须保证容量为正且足以容纳所选布局。
未启用 `bias` 时，其预留区域为空；启用时实际搬入 `tileN*sizeof(BiasType)` 字节，不能超过 4096 字节。

权重缓冲区基址由配置的 `baseN` 和 `KbL1` 决定，尾块不能改变其基址。
布局填充指为满足对齐而占用的额外元素；检查容量时需包含这些元素，不能只比较逻辑有效长度。

## 输入装载方式

| 方式 | 选择条件与存储内容 |
| --- | --- |
| 普通双缓冲 | `KaL1 <= KbL1`。逐 K 分块装载 `x`，交替使用两份缓冲区。 |
| 奇偶分散全载 | `KaL1 > KbL1`，输入不转置，`K % KbL1 == 0`，且较多的一半 K 分块能放入一份 `x` 缓冲区。当前 M 分块的全部输入按 K 分块交替存入两份缓冲区。 |
| 连续全载 | `KaL1 > KbL1` 且不满足奇偶分散条件。将完整 `(tileM,K)` 输入以 NZ 布局存入相邻的两份 `x` 缓冲区，再按 K 切片读取。 |

奇偶分散方式的容量条件为
`ceil((K/KbL1)/2)*KbL1*AlignUp(tileM,16)*sizeof(AType) <= xHalfBytes`。

连续全载要求完整 NZ 布局的物理大小不超过 `2*xHalfBytes`。
每次处理 M/N 分块时按真实 `tileM` 构造布局；尾块的搬运包含对齐填充，MMAD 使用真实有效 K 长度。

## 缓冲与同步

`BufferSlot` 管理片上缓冲区地址及本地流水同步。普通双缓冲按 K 分块保护输入的读写；
全载方式同时保护两份输入缓冲区，直到当前 M/N 分块的全部 L0 读取结束。
L0、偏置转换缓冲区和累加缓冲区使用各自的同步作用域。

AIC/AIV 通过 `MatmulWithWeightAntiquant::SyncProtocol` 协调共享权重缓冲区：

1. 两份权重缓冲区初始空闲，AIV 写入后发送就绪信号。
2. AIC 等待对应信号，将权重搬入 L0 后通知 AIV 该缓冲区可以复用。
3. AIV 复用缓冲区前等待空闲信号，结束时等待最后写入的权重被读取。

AIC 通过 `CrossCoreWaitFlag/CrossCoreSetFlag` 显式选择 mode 4、MTE1。`AivNum_=2` 时依次等待 READY=25、9，
依次发布 FREE=24、8；`AivNum_=1` 时仅处理 READY=9、FREE=8。两个 BL1 槽按段序号模 2 轮转，
共用同一组 mode 4 计数信号。输入 `x` 的装载方式不改变每个 K 分块所需的权重同步次数。

## GroupedMatmul 组装与 bias

`MatmulWithWeightAntiquant` 的第六个模板参数 `KernelSchedule_` 默认为原 WQMM 调度类型；
原有五参数组装、Params 字段及 `operator()` 四参数调用保持不变。
GroupedMatmul 在使用处声明 `MatmulWithWeightAntiquant<1,256,2,QuantMode::PERCHANNEL_MODE,HasOffset,KernelWqgmmMixAntiquant>`。
专家选择由 Kernel 通过 Tensor slice 完成，BlockMmad 不接收专家索引或 N 起点。
GroupedMatmul 设置 `KaL1=KbL1`，使用普通双缓冲装载 x。

每个输出块只在首 K 段搬入一次 bias，首 K 片由 L1 搬入 BT 并初始化累加结果。
GM/L1 bias 视图为 `[1,tileN]`，BT 视图为 `[1,Align16(tileN)]`。L1→BT 接口按 16 元素粒度向上取整搬运，实际读取范围包含补齐列并位于预留的 bias 槽内。
补齐列不清零，MMAD 和 Fixpipe 使用有效 N，补齐列不参与有效列的计算。
bias 搬运与读取分别受对应 AL1 槽的 MTE2/MTE1 锁保护。

GroupedMatmul 的主块配置和 UB 前处理见 [Kernel 的分块与 UB 布局](../kernel/kernel_wqgmm_mix_antiquant.md#分块与-ub-布局)。
