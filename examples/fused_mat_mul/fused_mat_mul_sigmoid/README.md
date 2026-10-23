# fused_mat_mul_sigmoid Example

## 概述

本示例演示基于 Blaze 框架的 FusedMatMul sigmoid 融合计算：

```text
y = sigmoid(op(A) @ op(B) + bias)
```

`op(A)`、`op(B)` 的逻辑形状分别为 `[M,K]`、`[K,N]`，输出形状为 `[M,N]`。
不传 bias 时省略加法；传入时按 N 轴广播到所有 M 行。
AIC 执行矩阵乘和 bias 累加，两个 AIV 分担 M 方向的 sigmoid 计算与输出搬运。
矩阵乘累加与 sigmoid 均使用 FP32，最后转换为输入类型。

## 支持架构

| 架构 | SoC |
| --- | --- |
| dav-3510 | Ascend950 |

## 使用约束

- 输入和输出为相同的 float16 或 bfloat16 类型，采用 ND 格式，二维、无 batch。
- 支持 NN、NT、TN、TT；转置输入在文件中按转置后的物理形状存储。
- bias 可省略，或为 N 个元素，类型为输入类型或 float32。
- M、K、N 均为正，使用设备 0。
- 示例使用固定的基本块 `baseM=64`、`baseN=64`、`baseK=32`，`mL1=nL1=64`、`kL1=64`。
  L1 和 L0C 使用双缓冲，UB 累加器使用单缓冲，开启 `splitM`，采用 MIX 1 AIC : 2 AIV。
  该分块用于演示组件组装，不代表各 shape 的最优性能配置。

## CSV 驱动测试

在仓库根目录配置 CANN 环境后执行：

```bash
bash build.sh --examples --ops=fused_mat_mul
```

也可使用公共样例入口：

```bash
bash examples/common/run.sh --ops=fused_mat_mul --target=fused_mat_mul_sigmoid
```

仅编译时添加 `--build-only`；复用已编译程序时添加 `--skip-build`。
公共入口按 CSV 逐条生成输入与 CPU golden、运行 kernel、校验结果。

用例见 [fused_mat_mul_sigmoid.csv](./fused_mat_mul_sigmoid.csv)，参数映射见
[fused_mat_mul_sigmoid.conf](./fused_mat_mul_sigmoid.conf)。

| 字段 | 说明 |
| --- | --- |
| casename | 用例名称 |
| m,k,n | M、K、N 逻辑长度 |
| transA,transB | `true` 表示转置，`false` 表示不转置 |
| dtype | `float16` 或 `bfloat16` |
| bias_dtype | `none`：无 bias；`same`：与输入同类型；`float32`：FP32 bias |

6 条用例覆盖两种输入类型、四种转置组合、无 bias/同类型 bias/FP32 bias，以及 M/N/K 尾块、
多轮 K 累加和 N=1 场景。

## 代码结构

```text
fused_mat_mul/
├── CMakeLists.txt
├── scripts/
│   ├── gen_data.py
│   └── verify_result.py
└── fused_mat_mul_sigmoid/
    ├── fused_mat_mul_sigmoid.cpp
    ├── fused_mat_mul_sigmoid.conf
    ├── fused_mat_mul_sigmoid.csv
    └── README.md
```

示例直接组装 `GemmUniversal`、`BlockMmad`、`BlockSchedulerMatmulBasic` 和
`BlockEpilogueMatmulWithSigmoid`，复用现有组件。组件接口见
[Kernel 文档](../../../docs/API/gemm/kernel/kernel_matmul_with_epilogue.md)。

## 输入输出文件

| 文件 | 内容 |
| --- | --- |
| `input/input_a.bin` | A，未转置时为 `[M,K]`，转置时为 `[K,M]` |
| `input/input_b.bin` | B，未转置时为 `[K,N]`，转置时为 `[N,K]` |
| `input/bias.bin` | 可选 bias，N 个元素 |
| `output/cpu_output.bin` | CPU golden，`[M,N]`，与输入同类型 |
| `output/npu_out.bin` | NPU 输出，`[M,N]`，与输入同类型 |
| `output/verify_metrics.json` | 精度校验结果，由公共入口汇总 |

数据生成固定随机种子 42，先将输入与 bias 转为指定类型，再执行 FP32 矩阵乘、bias 加法与 sigmoid，
最后转换输出类型。BF16 以原始 uint16 位表示保存，转换采用 round-to-nearest-even。

每个输出元素须满足 `abs(actual - golden) <= atol + rtol * abs(golden)`。
FP16 的 `atol`、`rtol` 均为 0.001；BF16 均为 0.008。非有限值或输出长度错误均判为失败。
