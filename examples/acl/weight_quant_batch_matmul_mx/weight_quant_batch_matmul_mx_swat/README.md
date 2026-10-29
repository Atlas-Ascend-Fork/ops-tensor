# weight_quant_batch_matmul_mx_swat Example

## 概述

本示例演示基于 Blaze 框架的 Weight Quant Batch MatMul MX 算子在昇腾 NPU 上的实现。该算子使用
FP8 E4M3 激活、打包的 FP4 E2M1 权重和 E8M0 MX Scale 完成矩阵乘，并输出 FP16 结果。

- **算子**: weight_quant_batch_matmul_mx
- **场景**: weight_quant_batch_matmul_mx_swat
- **算法特点**: 支持 Weight ND 和 Weight NZ 布局，以及可选的 FP16 Bias
- **参考实现**: 基于 Blaze 框架 `blaze/gemm/kernel/kernel_matmul_mix_weight_prologue.h`

## 支持架构

| 架构 | SoC | 支持状态 |
|------|-----|----------|
| dav-3510 | Ascend950 | 支持 |

## 使用约束

- 输入 A shape: `[M, K]`，数据类型为 FP8 E4M3
- 输入 B 逻辑 shape: `[K, N]`，数据类型为打包的 FP4 E2M1
- 输出 C shape: `[M, N]`，数据类型为 FP16
- Scale A 逻辑 shape: `[M, align(K, 64) / 32]`，数据类型为 E8M0
- Scale B 逻辑 shape: `[align(K, 64) / 32, N]`，数据类型为 E8M0
- M、N 必须为正数，K 必须为 8 的倍数
- Weight NZ 场景的 N 必须为 8 的倍数
- bias 大小必须为 N 或 0（无 bias）
- 数据生成和 Golden 校验依赖 `numpy` 和 `ml_dtypes`

## 数据布局说明

Weight ND 场景按照逻辑 `[N, K]` 排列权重，并沿 K 方向将相邻两个 FP4 编码打包为一个字节。

Weight NZ 场景先将 K、N 分别补齐到 32、16 的倍数，再转换为 NZ 分形布局并打包 FP4 编码：

```text
Weight ND: [N, K / 2]
Weight NZ: [align(K, 32) / 32, align(N, 16) / 16, 16, 16]
```

Scale A 按 M 行存储；Scale B 的原始字节按 N 优先存储，对应 kernel 使用的 ScaleBDN 布局。计算结果 C
按照 `[M, N]` 的 ND 布局存储。

## CSV 驱动测试

### 执行方式

安装 Python 依赖并初始化 CANN 环境后，通过统一入口驱动，自动完成编译、数据生成、kernel 执行和精度验证：

```bash
source /path/to/cann/set_env.sh
bash examples/acl/common/run.sh --ops=weight_quant_batch_matmul_mx --target=weight_quant_batch_matmul_mx_swat
```

仅编译或复用已有构建结果：

```bash
bash examples/acl/common/run.sh --ops=weight_quant_batch_matmul_mx --target=weight_quant_batch_matmul_mx_swat --build-only
bash examples/acl/common/run.sh --ops=weight_quant_batch_matmul_mx --target=weight_quant_batch_matmul_mx_swat --skip-build
```

### 测试用例定义

本例使用统一 BlockMmad 的 `MatmulWithWeightQuantMx` 策略：按 host tiling 固定 K 分块，ScaleB 由 AIC
从 GM 搬运。GMM 使用同一实现的动态 K 策略与 AIV ScaleB producer，见
[统一 BlockMmad 文档](../../../../docs/API/gemm/block/block_mmad_weight_prologue_mx.md)。

QBMM 从 tiling 填写 `l1TileShape`、`l0TileShape`、`l1BufferNum` 和 `hasBias`，
以 `BlockMmad(params.mmadParams)` 创建实例，再调用固定 K 的四 Tensor 接口
`blockMmad(tensorA, tensorScaleA, tensorScaleB, tensorC)`。
AIV 转换 B 和可选 bias 后写入共享 L1，AIC 搬运 A、ScaleA、ScaleB 并执行 MMAD。
固定 K 路径中 A 与 B 按同一个 L1 K 块推进；GMM 由 Kernel 传入 ka/kb/k0，用独立 ka/kb 窗口复用 A。

测试用例定义在 `weight_quant_batch_matmul_mx_swat.csv` 中，格式如下：

```csv
casename,m,k,n,bias,layout,base_m,base_n,base_k,tile_k_l1,scale_k_l1,k_bub,n_bub,l1_buffers,block_num
ND_multik_tail_n,32,128,40,0,ND,32,32,64,64,64,64,32,2,1
NZ_multik_tail_n_bias,32,128,40,40,NZ,32,32,64,64,64,64,32,2,1
```

**列说明**：

| 列 | 说明 |
|----|------|
| casename | 用例名称 |
| m, k, n | 矩阵维度 |
| bias | bias 元素数量，必须为 n 或 0 |
| layout | 权重布局：ND / NZ |
| base_m, base_n, base_k | Cube 基础分块大小 |
| tile_k_l1 | L1 中 K 方向的 Tile 大小 |
| scale_k_l1 | L1 中 Scale K 方向的 Tile 大小 |
| k_bub, n_bub | BUB 中 K、N 方向的分块大小 |
| l1_buffers | L1 Buffer 数量 |
| block_num | 启动的 NPU Block 数量 |

### 结果输出

执行完成后结果写入 `weight_quant_batch_matmul_mx_swat_result.csv`。

## 数据与校验

### 输入数据

由 `examples/acl/weight_quant_batch_matmul_mx/scripts/gen_data.py` 在 `data/<casename>/` 下生成：

- `input_a.bin`: FP8 E4M3 激活矩阵
- `input_b.bin`: 打包的 FP4 E2M1 权重矩阵
- `scale_a.bin`: E8M0 激活 Scale
- `scale_b.bin`: E8M0 权重 Scale
- `bias.bin`: FP16 bias，禁用 bias 时内容为 0
- `initial_c.bin`: 初始化的 FP16 输出矩阵
- `golden_c.bin`: NumPy 计算得到的 FP16 Golden 结果

### 输出数据

- `npu_out.bin`: NPU 计算得到的 FP16 输出

### 验证标准

由 `examples/acl/weight_quant_batch_matmul_mx/scripts/verify_result.py` 执行校验。FP16 逐点绝对误差超过 `ratio_tol` 时记为误差点，误差点占比
不超过 `ratio_tol` 时校验通过：

| dtype | ratio_tol |
|-------|-----------|
| float16 | 1e-3 |

## 代码结构

```text
weight_quant_batch_matmul_mx_swat/
├── weight_quant_batch_matmul_mx_swat.cpp      # Kernel 和 Host 侧执行代码
├── weight_quant_batch_matmul_mx_swat.conf     # 参数路由配置
├── weight_quant_batch_matmul_mx_swat.csv      # CSV 测试用例
└── README.md                                   # 本文档
```

构建配置在 op 层 `examples/acl/weight_quant_batch_matmul_mx/CMakeLists.txt` 中统一管理；运行通过 `examples/acl/common/run.sh` 统一调度，数据生成和精度校验由 `examples/acl/weight_quant_batch_matmul_mx/scripts/` 下的 `gen_data.py` 和 `verify_result.py` 执行。

## Blaze 组件

本场景使用以下 Blaze 组件：

| 组件 | 头文件 | 职责 |
|------|--------|------|
| Kernel | `blaze/gemm/kernel/kernel_matmul_mix_weight_prologue.h` | 完整 kernel 入口和 AIV 权重预处理 |
| Block MMAD | `blaze/gemm/block/block_mmad_weight_prologue_mx.h` | Block 级 MXA8W4 矩阵乘 |
| Block Scheduler | `blaze/gemm/block/block_scheduler_matmul_swat_with_tail_split.h` | M/N SWAT 分核与尾块调度 |
| Tile | `blaze/gemm/tile/tile_weight_quant_mx_preprocess.h` | FP4 权重和 MX Scale 预处理 |
| Dispatch Policy | `blaze/gemm/policy/dispatch_policy.h` | 派发策略 `MatmulWithWeightQuantMx` |
