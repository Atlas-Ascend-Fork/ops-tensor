# Gemm/Block 类模板概述

## 开发指导

新组件开发请参阅：[Block 层编写指导](./block_developer_guide.md)

## API 清单

### BlockMmad（矩阵乘计算）
| 组件名 | 描述 |
| :----------------------------------------------------------- | :------: |
| [block_mmad_matmul_basic](./block_mmad_matmul_basic.md) | 基础矩阵乘 Block，基于 Tensor API，支持 L1/L0 双缓冲 |
| [block_mmad_matmul_bl1_full_load](./block_mmad_matmul_bl1_full_load.md) | B L1 全载矩阵乘 Block，B 常驻 L1，支持 ON_THE_FLY 直销和 Fixpipe（L0C→UB）两种输出 |
| [block_mmad_matmul_fixpipe_opti](./block_mmad_matmul_fixpipe_opti.md) | Fixpipe 非全载矩阵乘 Block，A/B 均流水，输出固定 L0C→UB，用于 Fixpipe1v1/fixpipe1v2 场景 |
| [block_mmad_a8w8_fixpipe_quant](./block_mmad_a8w8_fixpipe_quant.md) | 量化矩阵乘 Block，支持 int8/HiFloat8/FP8 输入，并通过 Fixpipe 按需反量化 |
| [block_mmad_qbmm_mx](./block_mmad_qbmm_mx.md) | MX 量化矩阵乘 Block，支持 Scale 因子、MxFP4/MxFP8 量化 |
| [block_mmad_a8w8_mix](./block_mmad_a8w8_mix.md) | MIX 模板 A8W8 量化矩阵乘 Block，int32 累加 + L0C→UB（fixpipe NoQuant），不做 scale/bias |
| [block_mmad_qgmm_mx](./block_mmad_qgmm_mx.md) | MX 量化 Grouped Matmul Block，支持 ScaleA/ScaleB 及 SwiGLU 的左右 N 半区拼接、NZ/ZN 双源输入 |
| [block_mmad_qbmm_mx_l0c_pingpong](./block_mmad_qbmm_mx_l0c_pingpong.md) | MX 量化矩阵乘 L0C PingPong Block，支持 N 方向拆分、Scale 复用和 SplitK 写回控制 |
| [block_mmad_matmul_streamk](./block_mmad_matmul_streamk.md) | StreamK 矩阵乘 Block，支持 workspace 输出、K 轴切分 |
| [block_mmad_weight_prologue_mx](./block_mmad_weight_prologue_mx.md) | QBMM/GMM 共用的 MX AIC BlockMmad，策略区分固定/动态 K 及 ScaleB 来源，Kernel 确定 K 分块 |
| [block_mmad_wqmm_mix_prologue_fixpipe_quant](./block_mmad_wqmm_mix_prologue_fixpipe_quant.md) | AIV 反量化 B 写入 L1 后的 T-CG per-group A8W4 AIC BlockMmad，Fixpipe 乘 yScale 输出 |
| [block_mmad_wqmm_mix_weight_prologue](./block_mmad_wqmm_mix_weight_prologue.md) | 权重反量化矩阵乘的 AIC 计算、L1 空间划分及缓冲同步 |
| [block_mmad_matmul_iterbatch](./block_mmad_matmul_iterbatch.md) | IterBatch 矩阵乘 Block，多 batch 驻留 L1、跨 tile 预取流水，ON_THE_FLY / ND_FIXPIPE_1_2 两种输出 |

### BlockScheduler（任务调度）
| 组件名 | 描述 |
| :----------------------------------------------------------- | :------: |
| [block_scheduler](./block_scheduler.md) | 公共框架：tile 切分、Z 型扫描、尾块处理 |
| [block_scheduler_matmul_basic](./block_scheduler_matmul_basic.md) | Basic 调度器：尾块切分、SplitK 切分、L2Cache 配置 |
| [block_scheduler_matmul_streamk](./block_scheduler_matmul_streamk.md) | StreamK 调度器：DP+SK 混合策略、K 轴切分 |
| [block_scheduler_wqmm](./block_scheduler_wqmm.md) | WQMM 的 M/N 分核与三段 N 区间调度，提供分块坐标和有效长度 |
| [block_scheduler_qbmm_mx](./block_scheduler_qbmm_mx.md) | QBMM 调度器：Batch 维度切分、量化对齐 |
| [block_scheduler_gmm_swat_with_tail_split](./block_scheduler_gmm_swat_with_tail_split.md) | QGMM 调度器：group 间连续分核、SWAT 扫描、末组 tail split，以及 SwiGLU 左右半区和输出 offset |
| [block_scheduler_grouped_matmul](./block_scheduler_grouped_matmul.md) | 非量化 GMM 调度器：组间偏移管理（含 sparse 直接定位）、SWAT 扫描、跨组延续起始核、tail split |
| [block_scheduler_matmul_swat_with_tail_split](./block_scheduler_matmul_swat_with_tail_split.md) | 通用 M/N SWAT 扫描、尾块合并和 compact tail split |
| [block_scheduler_wqmm_block_split](./block_scheduler_wqmm_block_split.md) | 固定核分核调度器：每核一个 M/N 责任矩形，核内 ORDER_M/ORDER_N swizzle 遍历 |
| [block_scheduler_matmul_iterbatch](./block_scheduler_matmul_iterbatch.md) | IterBatch 调度器：batch 分组 tile 映射、尾组余数分配（跨核步进由 kernel 承担） |

## 公共框架

### BlockMmad 公共框架
所有 BlockMmad 组件基于 [block_mmad.md](./block_mmad.md) 公共框架实现，包含统一的：
- 模板参数
- 数据结构（Params）
- 核心方法（Init、operator）

详见：[block_mmad.md](./block_mmad.md)

### BlockScheduler 公共框架
所有 BlockScheduler 组件基于 [block_scheduler.md](./block_scheduler.md) 公共框架实现，包含统一的：
- 模板参数（ProblemShape）
- 类型别名（BlockShape、BlockCoord）
- Z 型扫描策略
- 尾块处理

详见：[block_scheduler.md](./block_scheduler.md)

## 核心组件关系

```
BlockMmad
    ├── DispatchPolicy (调度策略)
    │       ├── MatmulMultiBlockBasic (Basic)
    │       ├── MatmulMultiBlockBFullLoad (B L1 全载)
    │       ├── MatmulMultiBlockFixpipeOpti (Fixpipe 非全载)
    │       ├── MatmulMultiBlockWithStreamK (StreamK)
    │       ├── MatmulWithScaleFixpipeQuant (Fixpipe 反量化输出)
    │       ├── MatmulWithScaleMx (QBMM MX 量化)
    │       ├── MatmulWithScaleMix (QBMM MIX A8W8 量化)
    │       ├── MatmulWithWeightQuantMx (QBMM MXA8W4 固定 K、AIC ScaleB)
    │       ├── GroupedMatmulWithWeightQuantMx (GMM MXA8W4 动态 K、AIV ScaleB)
    │       ├── MatmulWithScaleMxL0CPingpong (QBMM MX L0C PingPong)
    │       └── GroupedMatmulWithScaleMx (QGMM MX 量化)
    ├── 数据类型 (AType, BType, CType, BiasType)
    ├── 布局类型 (LayoutA, LayoutB, LayoutC, LayoutBias)
    └── 计算流程
            ├── GM → L1 → L0 数据搬运
            ├── Mmad 计算
            └── L0C → GM/workspace 结果搬出
```

## 实现差异对比

| Block 类型 | 调度策略 | 输出目标 | 量化支持 | Scale 支持 | L1 双缓冲 | L0C 双缓冲 | Bias 支持 | AIC-AIV 同步 | 适用场景 |
|-----------|---------|---------|---------|-----------|---------|-----------|---------|-------------|---------|
| BlockMmadBasic | MatmulMultiBlockBasic | GM | 不支持 | 不支持 | 可配置 (1 或 2) | 可配置 | 支持 | 无 | Basic Kernel |
| BlockMmadBL1FullLoad | MatmulMultiBlockBFullLoad | GM 或 UB | 不支持 | 不支持 | A 可配置，B 固定 1 | 可配置 | 支持 | 有（Fixpipe 模式） | B 全载，ON_THE_FLY / Fixpipe |
| BlockMmadFixpipeOpti | MatmulMultiBlockFixpipeOpti | UB | 不支持 | 不支持 | A/B 均 l1Stages | 可配置 | 支持 | 有 | 非全载 Fixpipe，小 K 场景 |
| BlockMmadStreamK | MatmulMultiBlockWithStreamK | GM 或 workspace | 不支持 | 不支持 | 固定双缓冲 | 固定单缓冲 | 支持 | 无（Kernel 层处理） | StreamK Kernel |
| BlockMmadA8W8FixpipeQuant（StreamK 调度） | MatmulWithScaleFixpipeQuant + KernelQbmmPertensorMultiBlockStreamK | C GM（DP）或 workspace 分块累加结果（SK） | int8/FP8/HiFloat8 | DP 使用 Fixpipe，SK 由 AIV epilogue 处理 | 2 或 4 | 固定单缓冲 | 支持反量化前 bias | 有（Kernel 层处理） | QBMM per-tensor StreamK |
| BlockMmadA8W8FixpipeQuant | MatmulWithScaleFixpipeQuant | GM | int8/HiFloat8/FP8 | X2 scale + Fixpipe | 可配置 (2、3 或 4) | 可配置 | 支持 | 无 | QBMM Cube Kernel / GMM Cube Kernel |
| BlockMmadMx | MatmulWithScaleMx | GM | MxFP4/MxFP8 | ScaleA + ScaleB | 可配置 (2、3 或 4) | 可配置 | 支持 | 无 | QBMM MX Kernel |
| BlockMmadA8W8Mix | MatmulWithScaleMix | UB (L0C→UB) | int8 (A8W8) | 不在本层（由 epilogue 处理） | 可配置 (2 或 4) | 可配置 | 不在本层 | 无（Kernel 层处理） | QBMM MIX Kernel |
| BlockMmadQGmmMx | GroupedMatmulWithScaleMx | GM 或拼接 UB | MxFP4/MxFP8；SwiGLU 路径为 MXFP8 | ScaleA + ScaleB | 可配置双/三缓冲 | 可配置 | 支持；SwiGLU 路径无 Bias | Kernel 层同步 | QGMM MX / SwiGLU MX Kernel |
| BlockMmadMxL0CPingpong | MatmulWithScaleMxL0CPingpong | GM | MxFP4/MxFP8 | ScaleA + ScaleB | 可配置 (2、3 或 4) | 固定双缓冲 | 支持 | 无 | QBMM MX L0C PingPong Kernel |
| BlockMmad（MX 权重前处理） | MatmulWithWeightQuantMx / GroupedMatmulWithWeightQuantMx | GM | FP8 + packed FP4 | ScaleA + ScaleB | 2 或 4 | 固定单缓冲 | AIV 提供 | 有（Kernel 层 ready/free 标志） | QBMM Weight ND/NZ；GMM Weight NZ、A 跨 B 块复用 |
| BlockMmadMatmulIterbatch | MatmulIterBatch | GM 或 UB | 不支持 | 不支持 | 整组双缓冲 | 可配置 | 支持 | 有（Fixpipe 模式） | IterBatch Matmul |

## 使用流程

1. **查看公共框架**：了解模板参数和核心接口 → [block_mmad.md](./block_mmad.md)
2. **选择具体实现**：根据 Kernel 类型选择 Basic、StreamK、FixpipeQuant、MX、Weight Prologue 或 L0C PingPong MX
3. **定义调度策略**：选择 DispatchPolicy（TensorApi、StreamK、FixpipeQuant、ScaleMx、WeightQuantMx 或 ScaleMxL0CPingpong）
4. **组装组件**：定义数据类型、布局类型
5. **初始化**：调用 Init 设置 tile 形状、缓冲策略
6. **执行计算**：调用 operator 执行矩阵乘
