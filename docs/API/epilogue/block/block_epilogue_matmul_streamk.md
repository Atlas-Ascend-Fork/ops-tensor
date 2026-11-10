# Block Epilogue StreamK
> [代码位置](../../../../include/blaze/epilogue/block/block_epilogue_matmul_streamk.h)

## 功能说明
StreamK 矩阵乘后处理 Block，运行在 AIV 核。从 workspace 读取 AIC 计算的中间结果，执行 K 轴切分累加（Add）、类型转换（Cast）、ReLU 激活，并输出到 GM。

**继承自**：[Block Epilogue 基础框架](./block_epilogue.md)

## 特殊约束

### AIV 核专用
仅运行在 AIV（Vector）核，与 AIC（Cube）核协同工作：
- **AIC 核**：执行矩阵乘计算，结果输出到 workspace
- **AIV 核**：从 workspace 读取中间结果，执行后处理

### Workspace 必需
必须提供 workspace 用于存储 AIC 计算的中间结果：
- workspace 数据类型：`WorkspaceType`（通常为 float）
- 大小：`tailMNTileNum × skKTileNum × BLOCK_BASE_M × CeilAlign(BLOCK_BASE_N, 32) × sizeof(WorkspaceType)`（N 方向按 32 元素向上对齐）

### 输出类型
支持两种输出类型：
- **float → float**：无需 Cast，直接输出
- **float → half/bfloat16**：执行 Cast 转换后输出

### ReLU 支持
- **float 输出**：支持 ReLU（在 FP32 累加结果上执行）
- **half 输出**：支持 ReLU（Cast 后在 half 上执行）
- **bfloat16_t 输出**：支持 ReLU（硬件 Relu 指令不支持 bfloat16_t，先在 FP32 累加结果上执行 ReLU 再 Cast 为 bfloat16_t）

### L0C2Out 模式
支持两种 Fixpipe 输出模式：
- **ON_THE_FLY**：Compact 模式，紧凑布局
- **ND_FIXPIPE_1_2**：Normal 模式，stride 对齐到 32B

### AIV 并行度
- **SK 模式**：`aivMte2Num_ = GetTaskRation()`（多个 AIV 并行处理一个 AIC）
- **DP 模式**：`aivMte2Num_ = BLOCK_CUBE`（单个 AIV 处理）

### 数据对齐
- `burstLen`：对齐到 32B（`BLOCK_SIZE`）
- `srcGap`：workspace 中 K 轴切分之间的间隔
- `dstGap`：GM 输出中 N 轴的间隔（`n_ - curNL1InAiv`）

## 特殊静态常量

| 常量 | 说明 |
|------|------|
| BLOCK_BASE_M | Block 基础 M 维度（256） |
| BLOCK_BASE_N | Block 基础 N 维度（256） |
| NUM_TWO | 数值常量（2） |
| MAIN_WINDOW | 主窗口大小（4） |
| UB2GM_SRCGAP_UNIT | UB2GM srcGap 对齐单位（32） |

## 特殊类型别名

| 类型 | 说明 |
|------|------|
| WorkspaceType | Workspace 数据类型（模板参数） |
| OutType | 输出数据类型（模板参数） |
| DispatchPolicy | 调度策略（模板参数） |

## 特殊数据结构

### Params
```
struct Params {
    GM_ADDR cGmAddr{nullptr};         // C 矩阵 GM 地址
    GM_ADDR workspaceGmAddr{nullptr}; // Workspace GM 地址
};
```

### AivParams
```
struct AivParams {
    uint64_t indexParams = 0;      // AIV 处理的 tile 索引
    uint64_t mCntIndex = 0;       // M 轴 tile 索引
    uint64_t nCntIndex = 0;       // N 轴 tile 索引
    uint64_t kCntIndex = 0;       // K 轴切分索引
    uint64_t curML1InAiv = 0;     // 当前 tile M 维度
    uint64_t curNL1InAiv = 0;     // 当前 tile N 维度
    uint64_t curAlignedNInAiv = 0; // 对齐后的 N 维度（ND_FIXPIPE_1_2 模式）
};
```

### CopyGm2UbParams
```
struct CopyGm2UbParams {
    uint64_t offsetWorkspaceGM = 0; // workspace 起始偏移
    uint64_t kCnt = 0;              // K 轴切分数量
    uint64_t mBurstOri = 0;         // 原始 M burst 数量
    uint64_t mBurst = 0;            // 当前 AIV 处理的 M burst 数量
    uint64_t burstLen = 0;          // 每次 burst 的数据长度
    uint64_t srcGap = 0;            // workspace 中 K 切分之间的间隔
};
```

### CopyUb2GmParams
```
struct CopyUb2GmParams {
    uint64_t offsetCGm = 0;   // GM 输出偏移
    uint64_t mLength = 0;     // M 维度长度
    uint64_t burstLen = 0;    // burst 长度（N 维度）
    uint64_t dstGap = 0;      // GM 输出 N 轴间隔
    uint64_t srcGap = 0;      // UB 中 N 轴间隔
};
```

## 特殊成员方法

### 构造函数
```
__aicore__ inline BlockEpilogueMatmulStreamK()
```
功能：构造 BlockEpilogueMatmulStreamK 对象。

### 析构函数
```
__aicore__ inline ~BlockEpilogueMatmulStreamK()
```
功能：析构 BlockEpilogueMatmulStreamK 对象。

### Init函数
```
__aicore__ inline void Init(
    Params const& params,         // 参数（cGmAddr, workspaceGmAddr）
    BlockShape blockShapeInAiv,   // 问题规模 (m, n, k)
    BlockShape tileL1ShapeInAiv,   // L1 tile 形状 (mL1, nL1, kL1)
    BlockCoord coordInAiv,        // tile 坐标 (mCnt, nCnt, kCnt)
    uint64_t usedCoreNum,         // 使用的核数
    bool checkIsSkScene)          // 是否为 SK 模式
```
功能：初始化 BlockEpilogueMatmulStreamK 组件。
参数说明：
| 参数 | 类型 | 说明 |
|------|------|------|
| params | Params | 包含 cGmAddr 和 workspaceGmAddr |
| blockShapeInAiv | BlockShape | 问题规模 `(m, n, k)` |
| tileL1ShapeInAiv | BlockShape | L1 tile 形状 `(mL1, nL1, kL1)` |
| coordInAiv | BlockCoord | tile 坐标 `(mCnt, nCnt, kCnt)` |
| usedCoreNum | uint64_t | 使用的核数 |
| checkIsSkScene | bool | true = SK 模式，false = DP 模式 |

执行流程：
1. 设置问题规模：`m_`, `n_`, `mL1_`, `nL1_`
2. 设置 tile 坐标：`mCnt_`, `nCnt_`, `kCnt_`
3. 设置 AIV 并行度：`aivMte2Num_`
4. 设置 GM buffer：`cGlobal_`, `workspaceGlobal_`
5. ICache 预加载（2 个 cache line）
6. DP 模式同步：设置并等待 `MTE3_MTE2` 标志

### Run函数
```
__aicore__ inline void Run()
```
功能：执行 StreamK 后处理。
执行流程：
1. **更新索引**：`UpdateAivBasicIndex()` 更新 AIV tile 索引
2. **更新块参数**：`UpdateAivBasicBlock()` 更新当前 tile 形状
3. **AIV 并行循环**：按 `aivMte2Num_` 循环
   - 更新参数：`UpdateAivParams(index)`
   - 搬运 workspace → UB：`DataCopyPad<float>`
   - 等待 MTE2 完成：`WaitFlag<MTE2_V>`
   - **K 轴切分累加**：循环执行 `Add`
   - **ReLU 激活**（可选）：`Relu`（float/bfloat16 输出在 FP32 域执行；half 输出在 Cast 后执行）
   - **类型转换**（可选）：`Cast<float → half/bfloat16>`
   - 等待 V 完成：`WaitFlag<V_MTE3>`
   - 搬运 UB → GM：`DataCopyPad<OutType>`

### operator函数
```
__aicore__ inline void operator()()
```
功能：执行后处理，调用 `Run()`。

### UpdateAivBasicIndex函数
```
__aicore__ inline void UpdateAivBasicIndex()
```
功能：更新 AIV tile 索引（mCntIndex, nCntIndex, kCntIndex）。
算法说明：
1. 计算 `newBlockIdx`：`GetBlockIdx() / (GetTaskRation() * kCnt_)`
2. 计算 `kCntIndex`：`GetBlockIdx() % (GetTaskRation() * kCnt_)`
3. 计算 `cGmIndex`：基于 DP+SK 混合策略的索引
4. 扫描窗口计算：`mainWindow = min(4, mCnt_)`
5. Z 型扫描：偶数行正向，奇数行反向

### UpdateAivBasicBlock函数
```
__aicore__ inline void UpdateAivBasicBlock()
```
功能：更新当前 tile 形状（curML1InAiv, curNL1InAiv, curAlignedNInAiv）。
算法说明：
- `curML1InAiv`：尾块使用剩余尺寸，否则使用 `mL1_`
- `curNL1InAiv`：尾块使用剩余尺寸，否则使用 `nL1_`
- `curAlignedNInAiv`：ND_FIXPIPE_1_2 模式对齐到 32B

### UpdateAivParams函数
```
__aicore__ inline void UpdateAivParams(uint64_t index)
```
功能：更新 AIV 数据搬运参数（copyGm2UbParams, copyUb2GmParams）。
算法说明：
1. **mBurstBase**：`CeilAlign(curML1InAiv / (kCnt_ * GetTaskRation()), 32 / curAlignedNInAiv)`
2. **mBurstCnt**：`CeilDiv(curML1InAiv, mBurstBase)`
3. **mBurstTail**：`curML1InAiv - (mBurstCnt - 1) * mBurstBase`
4. **offsetWorkspaceGM**：`(indexParams * kCnt_ * BLOCK_BASE_M * BLOCK_BASE_N) + (kCntIndex * mBurstBase + mBurst * index) * curAlignedNInAiv`
5. **offsetCGm**：`nCntIndex * nL1_ + mCntIndex * mL1_ * n_ + (kCntIndex * mBurstBase + mBurst * index) * n_`
6. **burstLen**：`CeilAlign(mBurst * curAlignedNInAiv, 32)`
7. **srcGap**：`BLOCK_BASE_M * BLOCK_BASE_N - burstLen`
8. **dstGap**：`n_ - curNL1InAiv`

## 事件同步

| 事件 | 用途 |
|------|------|
| MTE3_MTE2 | DP 模式下 AIC 与 AIV 同步 |
| MTE2_V | workspace → UB 完成同步 |
| V_MTE3 | UB → GM 完成同步 |

说明：DP 模式下需要等待 AIC 完成写入 GM，SK 模式下直接从 workspace 读取。

## 调用示例

### 组件组装
```
using WorkspaceType = float;
using OutType = half;
using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockWithStreamK<Blaze::Gemm::MatMulL0C2Out::ON_THE_FLY>;

using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueMatmulStreamK<
    WorkspaceType, OutType, DispatchPolicy>;
```

### 参数准备
```
using Params = typename BlockEpilogue::Params;
Params params = {
    cGmAddr,        // C 矩阵 GM 地址
    workspaceGmAddr // Workspace GM 地址
};
```

### 组件初始化
```
BlockEpilogue blockEpilogue;
BlockShape blockShape{m, n, k};
BlockShape tileL1Shape{mL1, nL1, kL1};
BlockCoord coord{mCnt, nCnt, kCnt};
uint64_t usedCoreNum = 8;
bool checkIsSkScene = true;  // SK 模式
blockEpilogue.Init(params, blockShape, tileL1Shape, coord, usedCoreNum, checkIsSkScene);
```

### 组件执行
```
blockEpilogue();  // 或 blockEpilogue.Run();
```

## 数据流

### 存储层次
```
Workspace (float) → UB → Add 汇聚 → ReLU → Cast → GM (OutType)
（float/bfloat16 输出：ReLU 在 FP32 域、Cast 前；half 输出：先 Cast 后 ReLU）
```

### 执行流程
```
UpdateAivBasicIndex (更新 tile 索引)
    ↓
UpdateAivBasicBlock (更新 tile 形状)
    ↓
AIV 并行循环（aivMte2Num_ 次）
    ↓
UpdateAivParams (更新搬运参数)
    ↓
DataCopyPad: Workspace → UB (float)
    ↓
WaitFlag<MTE2_V>
    ↓
Add 循环：K 轴切分累加
    ↓
ReLU（可选，float/bfloat16 输出在 FP32 域执行）
    ↓
Cast（可选）：float → half/bfloat16（half 输出时 ReLU 在 Cast 后执行）
    ↓
WaitFlag<V_MTE3>
    ↓
DataCopyPad: UB → GM (OutType)
```

### Workspace 布局
```
offsetWorkspaceGM = indexParams * kCnt_ * BLOCK_BASE_M * BLOCK_BASE_N
                  + (kCntIndex * mBurstBase + mBurst * index) * curAlignedNInAiv
```

说明：每个 tile 的 K 轴切分结果按顺序存储在 workspace 中，间隔为 `BLOCK_BASE_M * BLOCK_BASE_N`。

### GM 输出布局
```
offsetCGm = nCntIndex * nL1_ + mCntIndex * mL1_ * n_ + (kCntIndex * mBurstBase + mBurst * index) * n_
```

说明：按 Z 型扫描顺序输出到 GM。

## 性能优化建议

### AIV 并行度配置
- **SK 模式**：`aivMte2Num_ = GetTaskRation()`，多 AIV 并行处理一个 AIC
- **DP 模式**：`aivMte2Num_ = BLOCK_CUBE`，单 AIV 处理
- 建议：SK 模式下增加 AIV 数量提升并行度

### K 轴切分累加
- 累加循环：`for (i = 1; i < kCnt; ++i) { Add(ub, ub, ub[i * burstLen], burstLen); }`
- 首次 K 切分数据在 `ub[0]`，后续切分在 `ub[burstLen]` 开始

### 类型转换优化
- float → half：使用 `Cast(ubDst, ubSrc, RoundMode::CAST_RINT, burstLen)`
- ReLU 位置：float/bfloat16_t 输出在 Cast 前（FP32 域），half 输出在 Cast 后

### L0C2Out 模式选择
- **ON_THE_FLY**：Compact 模式，`srcGap` 计算：`alignedN - actualN`
- **ND_FIXPIPE_1_2**：Normal 模式，`curAlignedNInAiv` 对齐到 32B

### 数据对齐
- `burstLen` 对齐到 32B
- `srcGap` 确保 K 轴切分数据正确间隔
- `dstGap` 确保 GM 输出 N 轴正确间隔

### 适用场景
- **StreamK Kernel**：AIC + AIV 双核协同
- **K 轴切分汇聚**：多个 K 切分结果累加
- **类型转换**：float → half/bfloat16
- **ReLU 激活**：可选 ReLU 后处理
