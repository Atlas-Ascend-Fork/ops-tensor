# Kernel Matmul StreamK
> [代码位置](../../../../include/blaze/gemm/kernel/kernel_matmul_streamk.h)

## 功能说明
StreamK 矩阵乘 Kernel，支持 AIC + AIV 双核协同计算。支持 workspace（用于 K 轴切分中间结果）、DP+SK 混合策略、AIC-AIV 跨核同步，适用于大矩阵、高并行度场景。

**继承自**：[Kernel Matmul 基础框架](./kernel.md)

## 特殊约束

### BlockEpilogue 限制
仅支持 `BlockEpilogueMatmulStreamK` 系列组件，不支持 `BlockEpilogueEmpty`。

### 计算模式
支持 AIC + AIV 双核协同：
- **AIC 核**：执行矩阵乘计算（BlockMmadStreamK）
- **AIV 核**：执行后处理（BlockEpilogueMatmulStreamK）

### Workspace 必需
必须提供 workspace 用于存储 K 轴切分的中间结果：
- AIC 计算结果输出到 workspace
- AIV 从 workspace 读取并汇聚（Add）
- 最终结果输出到 GM

### BlockScheduler 限制
仅支持 `BlockSchedulerStreamK` 调度器，支持 DP+SK 混合策略。

### HF32 模式
可选 HF32 计算模式，通过 scheduler 参数控制。

### AIC-AIV 同步
使用 `Gemm::Sync::NotifyVector`（AIC 侧）和 `Gemm::Sync::WaitForCube`（AIV 侧）进行跨核同步：
- AIC 完成后设置同步标志
- AIV 等待同步标志后执行后处理

### L0C2Out 模式
支持两种 Fixpipe 输出模式：
- **ON_THE_FLY**：实时输出模式
- **ND_FIXPIPE_1_2**：ND 1v2 优化模式（stride 对齐到 32B）

### L2 Cache 配置
可选禁用 A/B 矩阵的 L2 Cache，避免大矩阵场景下的缓存污染：
```
SetL2Cache(gmA, gmB, params.schParams.l2CacheMode);
```

## 特殊静态常量

| 常量 | 说明 |
|------|------|
| AIC_SYNC_AIV_MODE_4 | 同步模式（MODE_4） |
| AIC_SYNC_AIV_FLAG | AIC 同步 AIV 标志 ID（8） |
| FLAG_ID_MAX | 标志 ID 最大值（16） |
| BLOCK_BASE_M | Block 基础 M 维度（256） |
| BLOCK_BASE_N | Block 基础 N 维度（256） |
| BLOCK_BYTE_SIZE | Block 字节对齐大小（32） |


## 特殊数据结构

### Params
```
struct Params {
    ProblemShape problemShape;          // 问题 shape (m, n, k, batch)
    BlockMmadParams mmadParams;         // mmad 参数（包含 workspace 地址）
    BlockEpilogueParams epilogueParams; // epilogue 参数
    BlockSchedulerParams schParams;     // scheduler 参数
};
```

### BlockMmadParams（StreamK 特有）
```
struct Params {  // BlockMmad::Params
    GM_ADDR aGmAddr;         // A 矩阵 GM 地址
    GM_ADDR bGmAddr;         // B 矩阵 GM 地址
    GM_ADDR cGmAddr;         // C 矩阵 GM 地址（可选，DP 模式）
    GM_ADDR biasGmAddr;      // Bias GM 地址（可选）
    GM_ADDR workspaceGmAddr; // Workspace GM 地址（必需）
    ....
};
```

说明：workspace 用于存储 SK 模式下的中间计算结果。

## 特殊成员方法

### 构造函数
```
__aicore__ inline GemmUniversal()
```
功能：构造 GemmUniversal（KernelMatmulStreamK）对象。

### 析构函数
```
__aicore__ inline ~GemmUniversal()
```
功能：析构 GemmUniversal（KernelMatmulStreamK）对象。

### Init函数
```
__aicore__ inline void Init(Params const& params)
```
功能：初始化 Kernel，提取问题规模、GM 地址、workspace 地址。
执行流程：
1. 提取 BlockMmad 参数（包含 workspace 地址）
2. 设置 A、B、C、workspace 的 GM 地址
3. 判断 bias 地址是否为 nullptr

### SetL2Cache函数
```
template <typename TensorA, typename TensorB>
__aicore__ inline void SetL2Cache(TensorA& gmA, TensorB& gmB, uint32_t l2CacheMode)
```
功能：根据 L2CacheMode 配置 A/B 矩阵的 L2 Cache。
参数说明：
| 参数 | 类型 | 说明 |
|------|------|------|
| gmA | TensorA | A 矩阵 GM Tensor |
| gmB | TensorB | B 矩阵 GM Tensor |
| l2CacheMode | uint32_t | L2 Cache 配置模式 |

支持的 L2CacheMode：
- `L2_CACHE_DEFAULT`：L2 Cache 使能（默认）
- `A_L2_CACHE_DISABLE`：禁用 A 矩阵 L2 Cache
- `B_L2_CACHE_DISABLE`：禁用 B 矩阵 L2 Cache
- `ALL_L2_CACHE_DISABLE`：禁用所有 L2 Cache

### operator函数
```
__aicore__ inline void operator()(Params const& params)
```
功能：执行 StreamK 矩阵乘 Kernel 计算。
执行流程：
**公共部分**：
1. 调用 `Init(params)` 设置参数
2. 创建 BlockScheduler 实例
3. 获取 L1/L0 tile 形状、tile 数量

**AIC 核执行流程**：
1. Block 索引检查：超出实际数量则设置同步标志并返回
2. HF32 模式设置
3. BlockMmadStreamK 初始化
4. Layout 构建：A、B、C、Bias
5. GM Tensor 创建
6. L2 Cache 配置（通过 SetL2Cache）
7. Tile 循环处理：
   - **DP 模式**：结果输出到 GM
   - **SK 模式**：结果输出到 workspace
   - **Preload**：SK 模式下预加载下一轮 tile
8. AIC-AIV 同步：设置 `AIC_SYNC_AIV_FLAG`
9. 清理：关闭 HF32 模式

**AIV 核执行流程**：
1. Block 索引检查：超出处理范围则等待同步并返回
2. 等待 AIC 同步标志 `AIC_SYNC_AIV_FLAG`
3. 调用 `SyncAll` 全核同步
4. BlockEpilogueMatmulStreamK 初始化
5. 执行后处理：
   - 从 workspace 读取中间结果
   - 执行 Add 汇聚（K 轴切分累加）
   - 执行类型转换（float → half/bf16）
   - 执行可选 ReLU
   - 输出到 GM

### DP+SK 混合策略
```
// DP 模式：前 (mTileNum * nTileNum - tailMNTileNum) 个 tile
// SK 模式：后 tailMNTileNum * skKTileNum 个 tile
```

说明：
- **DP（Data Parallel）模式**：每个 tile 完整计算，结果直接输出到 GM
- **SK（StreamK）模式**：K 轴切分，结果输出到 workspace，AIV 汇聚

### Preload 优化
```
if (!bs.CheckIsSkScene(0)) { // SK Preload in DP+SK
    if (tileIdx % usedCoreNum < tailSKTotalTileNum &&
        (CeilDiv(tileIdx + 1, usedCoreNum) == (CeilDiv(tileNum, usedCoreNum) - 1))) {
        tmpTileIdx = tileIdx + usedCoreNum;  // Preload 下一轮 SK tile
    }
}
```

说明：在最后一轮 DP tile 时，预加载 SK tile 数据到 L1，减少 SK 阶段加载延迟。

## 调用示例

### Kernel 组装与调用

```cpp
// ============== 1. 类型定义 ==============
using AType = half;                                    // A 矩阵数据类型
using BType = half;                                    // B 矩阵数据类型
using CType = float;                                   // C 矩阵计算类型（L0C 累加）
using OutType = half;                                  // C 矩阵输出类型
using BiasType = float;                                // Bias 数据类型（可选）
using LayoutA = asc::te::nd_ext_layout_ptn;         // A 矩阵布局（NDExt 支持 stride）
using LayoutB = asc::te::nz_layout_ptn;             // B 矩阵布局（NZ 格式，权重优化）
using LayoutC = asc::te::nd_ext_layout_ptn;          // C 矩阵布局（NDExt 支持 stride）
using LayoutBias = LayoutC;                            // Bias 布局

// ============== 2. ProblemShape 定义 ==============
// 形状：(m, n, k, batch)，batch=0 或 1 表示单 batch
using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

// ============== 3. BlockScheduler 组装 ==============
// StreamK 使用专门的 BlockSchedulerStreamK
using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerStreamK<ProblemShape>;

// ============== 4. BlockMmad 组装 ==============
// DispatchPolicy: 调度策略，支持 L0C2Out 模式选择
// ON_THE_FLY: 实时输出模式
// ND_FIXPIPE_1_2: ND 1v2 优化模式（stride 对齐到 32B）
using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockWithStreamK<Blaze::Gemm::MatMulL0C2Out::ON_THE_FLY>;
using BlockMmad = Blaze::Gemm::Block::BlockMmad<
    DispatchPolicy, AType, LayoutA,
    BType, LayoutB, CType, LayoutC,
    BiasType, LayoutBias>;

// ============== 5. BlockEpilogue 组装 ==============
// StreamK 必须使用 BlockEpilogueMatmulStreamK，不支持 Empty
// 参数：计算类型，输出类型，调度策略
using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueMatmulStreamK<CType, OutType, DispatchPolicy>;

// ============== 6. Kernel 组装 ==============
using StreamKKernel = Blaze::Gemm::Kernel::GemmUniversal<
    ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

// ============== 7. Params 构造 ==============
using Params = typename StreamKKernel::Params;
Params params;

// --- ProblemShape 参数 ---
params.problemShape = {m, n, k, batch};              // (M, N, K, Batch)

// --- BlockMmad 参数 ---
params.mmadParams.aGmAddr = aGM;                     // A 矩阵 GM 地址
params.mmadParams.bGmAddr = bGM;                     // B 矩阵 GM 地址
params.mmadParams.cGmAddr = cGM;                     // C 矩阵 GM 地址
params.mmadParams.biasGmAddr = biasGM;               // Bias GM 地址（可选，nullptr 表示无 bias）
params.mmadParams.workspaceGmAddr = workspaceGM;    // Workspace GM 地址（必需，SK 模式使用）
params.mmadParams.mL1 = 256;                         // L1 M 维度尺寸
params.mmadParams.nL1 = 256;                         // L1 N 维度尺寸
params.mmadParams.kL1 = 128;                         // L1 K 维度尺寸
params.mmadParams.mL0 = 128;                         // L0 M 维度尺寸
params.mmadParams.nL0 = 128;                         // L0 N 维度尺寸
params.mmadParams.kL0 = 64;                          // L0 K 维度尺寸
params.mmadParams.l1Stages = 2;                      // L1 缓冲数量（双缓冲）
params.mmadParams.l0cStages = 1;                     // L0C 缓冲数量（单缓冲）

// --- BlockScheduler 参数 ---
params.schParams.usedCoreNum = 8;                    // 使用的核心数量
params.schParams.baseM = 128;                        // M 轴 L0 base 尺寸
params.schParams.baseN = 128;                        // N 轴 L0 base 尺寸
params.schParams.baseK = 64;                         // K 轴 L0 base 尺寸
params.schParams.singleCoreK = 256;                  // 单核 K 维度大小（用于 SK 模式 K轴切分）
params.schParams.kL1 = 128;                          // K 轴 L1 tile 尺寸
params.schParams.isHf32 = 0;                         // HF32 模式标志（0=关闭）
params.schParams.l2CacheMode = Blaze::Gemm::L2_CACHE_DEFAULT;  // L2Cache 使能

// --- BlockEpilogue 参数 ---
params.epilogueParams.cGmAddr = cGM;                 // C 矩阵 GM 地址（最终输出）
params.epilogueParams.workspaceGmAddr = workspaceGM; // Workspace GM 地址（中间结果读取）

// ============== 8. Kernel 调用 ==============
StreamKKernel streamk;
streamk(params);                                      // 执行 StreamK 矩阵乘计算
```

### 常用配置示例

**DP+SK 混合策略场景**：
```cpp
// DP 模式处理大部分 tile，SK 模式处理尾块
params.schParams.usedCoreNum = 8;                    // 使用 8 个核心
params.schParams.baseM = 128;
params.schParams.baseN = 128;
params.schParams.baseK = 64;
params.schParams.singleCoreK = k / 4;                // SK 模式 单核 K 维度
params.schParams.kL1 = 128;                          // L1 K 维度
params.mmadParams.l1Stages = 2;                      // 双缓冲提升性能
params.mmadParams.l0cStages = 1;
params.schParams.l2CacheMode = Blaze::Gemm::ALL_L2_CACHE_DISABLE;  // 禁用 L2 Cache
```

**大矩阵场景（ND_FIXPIPE_1_2 优化）**：
```cpp
// 使用 ND_FIXPIPE_1_2 模式，stride 对齐到 32B
using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockWithStreamK<Blaze::Gemm::MatMulL0C2Out::ND_FIXPIPE_1_2>;
params.schParams.baseM = 256;                        // 大 Block 尺寸
params.schParams.baseN = 256;
params.schParams.baseK = 128;
params.schParams.usedCoreNum = 16;                   // 使用更多核心
params.mmadParams.l1Stages = 2;                      // 双缓冲
params.mmadParams.l0cStages = 1;
params.schParams.l2CacheMode = Blaze::Gemm::ALL_L2_CACHE_DISABLE;  // 避免缓存污染
```

**Workspace 大小配置**：
```cpp
// Workspace 大小计算：tailMNTileNum × skKTileNum × BLOCK_BASE_M × BLOCK_BASE_N × sizeof(float)
// 示例：假设尾块 tile 数量为 4，SK 切分 K 轴数量为 2
constexpr int64_t BLOCK_BASE_M = 128;
constexpr int64_t BLOCK_BASE_N = 128;
size_t workspaceSize = 4 * 2 * BLOCK_BASE_M * BLOCK_BASE_N * sizeof(float);
GM_ADDR workspaceGM = AllocWorkspace(workspaceSize);
```

**Preload 优化配置**：
```cpp
// Preload 优化在最后一轮 DP tile 时预加载 SK tile 数据
// 无需额外参数配置，Kernel 内部自动判断 Preload 时机
params.mmadParams.l1Stages = 2;                      // 双缓冲支持 Preload
params.schParams.usedCoreNum = 8;
```

**HF32 计算模式**：
```cpp
// 启用 HF32 模式提升 FP16 计算精度
params.schParams.isHf32 = 1;                         // 启用 HF32 模式
// 注意：HF32 模式会增加 L0C 存储需求
```

## 数据流

### 存储层次
```
GM (A/B/Bias) → BlockScheduler (DP+SK 混合调度) → L1 → L0 → L0C
                                                          ↓
                                          DP: → GM (C)
                                          SK: → Workspace → AIV → GM (C)
```

### DP 模式流程
```
AIC: GM → L1 → L0 → Mmad → L0C → GM (C)
```

### SK 模式流程
```
AIC: GM → L1 → L0 → Mmad → L0C → Workspace (K 轴切分)
AIV: Workspace → Add 汇聚 → Cast → ReLU → GM (C)
```

### Kernel 执行流程
```
BlockScheduler 初始化（DP+SK 策略）
    ↓
DP Tile 循环（完整 tile）
    ↓
AIC 计算 → 输出到 GM
    ↓
SK Tile 循环（K 轴切分）
    ↓
AIC 计算 → 输出到 Workspace
    ↓
AIC-AIV 同步（`Sync::NotifyVector`）
    ↓
AIV 后处理 → 输出到 GM
```

### Workspace 布局
```
workspace 布局：
offsetWorkspace = ((tileIdx % usedCoreNum) / skKTileNum) * skKTileNum + kCntIndex) * BLOCK_BASE_M * BLOCK_BASE_N
```

说明：每个 (m, n) tile 的 K 轴切分结果按顺序存储在 workspace 中。

## 性能优化建议

### DP+SK 混合策略配置
- **策略选择**：
  - DP 模式：前 (mTileNum × nTileNum - tailMNTileNum) 个 tile 完整计算，结果直接输出到 GM
  - SK 模式：后 tailMNTileNum 个 tile 进行 K 轴切分，结果输出到 workspace，AIV 汇聚
- **参数调优**：
  - `usedCoreNum`：建议设置为物理核心数量的 1/2 到 2/3，预留资源给 AIV
  - `singleCoreK`：SK 模式下单核处理的 K 维度大小，建议 `k / 4`
  - `skKTileNum`：SK 模式下 K 轴切分数量，建议 2-4
- **负载均衡**：通过调整 `tailMNTileNum` 和 `skKTileNum`，确保 DP 和 SK 阶段负载均衡

### L0C2Out 模式选择
- **ON_THE_FLY 模式**：
  - 实时输出模式，适用于小到中等矩阵
  - 无需 stride 对齐要求
  - 较低的内存占用
- **ND_FIXPIPE_1_2 模式**：
  - ND 1v2 优化模式，适用于大矩阵
  - 要求 stride 对齐到 32B
  - 更高的输出效率
  - 配置示例：
    ```cpp
    using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockWithStreamK<Blaze::Gemm::MatMulL0C2Out::ND_FIXPIPE_1_2>;
    params.schParams.baseM = 256;  // 使用更大的 base 尺寸
    params.schParams.baseN = 256;
    params.schParams.baseK = 128;
    ```

### L2 Cache 配置
- **大矩阵场景**：
  - 建议禁用 L2 Cache 避免缓存污染
  - 使用 `ALL_L2_CACHE_DISABLE` 禁用所有 L2 Cache
  - 配置示例：`params.schParams.l2CacheMode = Blaze::Gemm::ALL_L2_CACHE_DISABLE;`
- **小矩阵场景**：
  - 可保留 L2 Cache 提升数据复用
  - 使用 `L2_CACHE_DEFAULT` 保留默认配置
- **选择性禁用**：
  - 禁用 A 矩阵 L2 Cache：`A_L2_CACHE_DISABLE`
  - 禁用 B 矩阵 L2 Cache：`B_L2_CACHE_DISABLE`

### L1/L0 缓冲配置
- **小矩阵场景**：
  - L1 单缓冲：`l1Stages = 1`
  - L0C 单缓冲：`l0cStages = 1`
  - 减少缓冲开销
- **中等矩阵场景**：
  - L1 双缓冲：`l1Stages = 2`
  - L0C 单缓冲：`l0cStages = 1`
  - 平衡性能和资源
- **大矩阵场景**：
  - L1 双缓冲或四缓冲：`l1Stages = 2` 或 `4`
  - L0C 单缓冲或双缓冲：`l0cStages = 1` 或 `2`
  - 提升数据搬运效率

### Workspace 配置
- **大小计算**：
  - 基本公式：`tailMNTileNum × skKTileNum × BLOCK_BASE_M × BLOCK_BASE_N × sizeof(float)`
  - 示例：尾块 4 个，SK 切分 2 次，Block 128×128
    ```cpp
    size_t workspaceSize = 4 * 2 * 128 * 128 * sizeof(float);  // 512KB
    ```
- **内存规划**：
  - 提前分配足够大的 workspace
  - workspace 地址对齐到 32B 边界
  - 避免 workspace 与其他 GM 数据冲突
- **性能影响**：
  - workspace 大小影响 SK 模式的存储效率
  - 过小的 workspace 会导致多次 K 轴切分，增加开销

### Preload 优化
- **原理**：在最后一轮 DP tile 时，预加载 SK tile 数据到 L1
- **配置要求**：
  - L1 双缓冲：`l1Stages = 2`
  - 无需额外参数，Kernel 自动判断 Preload 时机
- **性能提升**：
  - 减少 SK 阶段数据加载延迟
  - 平滑 DP 到 SK 的过渡
- **适用场景**：
  - DP+SK 混合策略
  - 大矩阵场景，数据加载延迟明显

### AIC-AIV 同步优化
- **同步机制**：
  - AIC 完成计算后设置同步标志：`Sync::NotifyVector`
  - AIV 等待同步标志后执行后处理：`Sync::WaitForCube`
  - 全核同步：`SyncAll`
- **优化建议**：
  - 确保 AIC 和 AIV 任务分配均衡
  - 避免过长的同步等待时间
  - 合理设置 `usedCoreNum`，预留 AIV 资源

### HF32 计算模式
- **精度提升**：启用 HF32 模式提升 FP16 计算精度
- **配置方式**：`params.schParams.isHf32 = 1;`
- **注意事项**：
  - HF32 模式会增加 L0C 存储需求
  - 适用于对精度要求较高的场景
  - 可能影响性能，需权衡精度和性能

### NZ 格式优化
- **权重矩阵（B）**：优先使用 NZ 格式，提升 L1/L0 搬运效率
- **激活矩阵（A）**：使用 NDExt 格式，支持灵活 stride 配置
- **输出矩阵（C）**：使用 NDExt 格式，便于后续处理

### 适用场景
- **大矩阵场景**：(m × n × k) 较大，需要多核并行
- **K 轴切分场景**：K 维度远大于 M/N，需要 K 轴切分优化
- **高并行度场景**：需要充分利用 AIC 和 AIV 双核
- **高精度场景**：需要 HF32 模式提升计算精度
