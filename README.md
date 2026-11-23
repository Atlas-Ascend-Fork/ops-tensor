# ops-tensor

## 🔥Latest News

- [2026/10] 支持 `TransposeQuantBatchMatmul`（TQBMM，新增 HIFP8 数据类型与 FP8 per-token 反量化 MIX 路径）、`RotateQuant`（旋转量化，支持分组量化）算子，新增 `FlatQuant` 样例；`FusedMatMul`（FMM）支持 ReLU 激活，`QuantBatchMatmul`（QBMM）MX 支持 FP8/FP4 ScaleNZ 布局，新增 SiLU + MXFP8、A4W4 BF16 WeightNZ SwiGLU 量化 epilogue，GM→L1 搬运使能 PreAllocate，examples 新增 torch 调用路径。
- [2026/09] 支持 `GemmSyrk`（对称秩K更新 SYRK）、`GroupedMatmulFinalizeRouting`（GMMFR，MXFP4/MXFP8 + WeightNZ）算子；`GroupedMatmul`（GMM）非量化支持稀疏与 GELU 后处理融合，`QuantGroupedMatmul`（QGMM）SwiGLU MX 支持 weight NZ/ZN 布局，`WeightQuantBatchMatmul`（WQBMM）支持 A16W8 量化，`QuantBatchMatmul`（QBMM）新增伪量化 A8W4 PerGroup 量化，`BatchMatMul` 新增 IterBatch 模板，SwiGLU MX epilogue 支持 BLAS scale。
- [2026/08] 支持 `FusedMatMul`（MatMul + Scale + Add 融合矩阵乘）、`WeightQuantBatchMatmul`（Weight-only 量化）、`QuantGroupedMatmul`（QGMM）Swiglu 激活量化融合（GMMSGQ），`GroupedMatmul`（GMM）支持 S8S4/S4S4 量化。
- [2026/07] 支持 `TransposeBatchMatMul`（TBMM）、`QuantGroupedMatmul`（QGMM）算子。
- [2026/06] 支持 `GroupedMatmul`（GMM）算子。
- [2026/05] 支持 `MatMul`、`BatchMatMul` 算子。
- [2026/04] `Blaze` 为算子开发提供基础线性代数加速能力，支持 `QuantBatchMatMul`。

## 🚀概述

ops-tensor 内置了 **Blaze**（Basic Linear Algebra optimiZed Engine）基础线性代数加速引擎，构建 NPU 高性能 CUBE 底座，为算子开发提供 CUBE 公共能力。

## 🔍目录结构

ops-tensor 代码目录结构如下：

```
ops-tensor/
├── cmake/                               # 项目工程编译目录
├── CMakeLists.txt                        # 编译配置文件
├── docs/                                 # 项目文档介绍
│   └ API/                                # API文档
├── examples/						  	  # Samples 测试
├── include/                              # 项目公共头文件
│   ├── blaze/                            # Blaze 高性能线性代数引擎
│   └ tensor_api/                         # Tensor 抽象（Layout/Shape/Coord）
├── tests/                                # UT/ST 测试工程目录
│   └ ut/op_kernel/                       # Kernel UT 测试
├── build.sh                              # 编译脚本
├── README.md                              # 项目说明
├── QUICKSTART.md                          # 快速入门
├── QUICK_OP_INVOCATION.md                 # 算子开发指南
```

## 🔧支持的算子

| 算子名称 | 描述 | 实现变体 | 状态 |
|---------|------|---------|------|
| MatMul | 矩阵乘运算 | Basic / A 全载 / B 全载 / Fixpipe / StreamK | ✅ 已实现 |
| BatchMatMul | 批量矩阵乘运算 | Broadcast / IterBatch Broadcast | ✅ 已实现 |
| TransposeBatchMatMul（TBMM） | 转置批量矩阵乘运算 | Basic | ✅ 已实现 |
| QuantBatchMatMul（QBMM） | 量化批量矩阵乘运算 | Cube（per-tensor/per-channel）/ MX / MIX（A8W8）/ StreamK | ✅ 已实现 |
| WeightQuantBatchMatmulMX | 权重量化批量矩阵乘（weight-only，介于全量化与非量化矩阵乘之间） | Weight Prologue（SWAT） | ✅ 已实现 |
| GroupedMatmul（GMM） | 分组矩阵乘运算 | Fixpipe Quant | ✅ 已实现 |
| QuantGroupedMatmul（QGMM） | 量化分组矩阵乘运算 | MX / Swiglu MX | ✅ 已实现 |

各算子可运行的完整示例见 [examples/](./examples/) 目录。

## 💻SoC 支持

Ascend 950PR / Ascend 950DT

## ⚡️快速入门

若您希望快速体验项目，请访问[快速入门](./QUICKSTART.md)获取简易教程，包括环境搭建、编译执行、本地验证等操作。

- [环境准备](./QUICKSTART.md#环境准备)：安装软件包之前，需要完成搭建基础环境，包括第三方依赖等；基础环境搭建后需要完成社区版CANN软件包安装、环境变量配置等。
- [源码下载](./QUICKSTART.md#源码下载)：本项目源码下载。
- [编译执行](./QUICKSTART.md#编译执行)：环境准备好后，可对源码修改编译生成可执行的文件。
- [Kernel UT 测试](./QUICKSTART.md#Kernel-UT测试)：基于项目根目录的 build.sh 脚本，可执行 Kernel UT 用例，快速验证编译能力。
- [Samples 测试](./QUICKSTART.md#Samples测试)：基于项目根目录的 build.sh 脚本，可执行 Samples 用例，快速验证功能。

## 📖文档介绍

| 文档 | 说明 |
|------|------|
|[快速入门](./QUICKSTART.md)|快速体验项目的简易教程。|
|[算子开发指南](./QUICK_OP_INVOCATION.md)|使用 ops-tensor 实现算子开发的教程。|
|[Blaze 模块](./include/blaze/README.md)|Blaze 高性能线性代数引擎介绍。|
|[Blaze 接口文档](./docs/API/README.md)|Blaze 分层架构与各组件 API 详细说明。|

## 📝相关信息

- [贡献指南](CONTRIBUTING.md)
- [编程规范](CODING_CONVENTIONS.md)
- [安全声明](SECURITY.md)
- [许可证](LICENSE)
