# fused_mat_mul_elemwise Example

## 概述

本示例演示基于 Blaze 框架的 FusedMatMul elementwise add/mul 融合计算：

```text
y = op(A) @ op(B) + x3    (add)
y = op(A) @ op(B) * x3    (mul)
```

`op(A)`、`op(B)` 的逻辑形状分别为 `[M,K]`、`[K,N]`，x3 与输出形状均为 `[M,N]`。
AIC 执行矩阵乘，两个 AIV 分担 elementwise 融合计算与输出搬运。

支持两种精度模式（`inner_precise`）：
- **高精度**（`inner_precise=0`）：累加器保持 FP32，融合在 FP32 域执行，结果 CAST_RINT 转回输入类型。此时 AIC 使用 DUAL_DST_SPLIT_M 将 FP32 累加器分给两个 AIV。
- **高性能**（`inner_precise=1`）：累加器先转为输入类型（B16），融合在元素域直接执行，无中间 cast。

支持两种执行模式（`mode`）：
- **basic**（默认）：`KernelMmadMultiBlockFixpipeOpti` 调度，固定基本块 `baseM=64`、`baseN=64`。
- **streamk**：`KernelMultiBlockStreamK` 调度，K 切分跨核累加（`mL1=32`、`kL1=256`，`singleCoreK=max(256, K/16)`），适用于小 M 大 K 场景。当前演示 tiling 的验证范围为 `M ≤ 64`（M 更大时 M 向 tile 数超过 2，超出本示例 tiling 的适用边界）。

## 支持架构

| 架构 | SoC |
| --- | --- |
| dav-3510 | Ascend950 |

## 使用约束

- 输入和输出为相同的 float16 或 bfloat16 类型，采用 ND 格式，二维、无 batch。
- x3 为 `[M,N]`，与输出同形状同类型。
- 支持 NN、NT、TN、TT；转置输入在文件中按转置后的物理形状存储。
- M、K、N 均为正，使用设备 0。
- 示例使用固定的基本块 `baseM=64`、`baseN=64`、`baseK=32`，`mL1=nL1=64`、`kL1=64`，1V2 fixpipe 模式。
  该分块用于演示组件组装，不代表各 shape 的最优性能配置。

## CSV 驱动测试

在仓库根目录配置 CANN 环境后执行：

```bash
bash build.sh --examples --ops=fused_mat_mul
```

也可使用公共样例入口：

```bash
bash examples/acl/common/run.sh --ops=fused_mat_mul --target=fused_mat_mul_elemwise
```

仅编译时添加 `--build-only`；复用已编译程序时添加 `--skip-build`。

用例见 [fused_mat_mul_elemwise.csv](./fused_mat_mul_elemwise.csv)，参数映射见
[fused_mat_mul_elemwise.conf](./fused_mat_mul_elemwise.conf)。

| 字段 | 说明 |
| --- | --- |
| casename | 用例名称 |
| m,k,n | M、K、N 逻辑长度 |
| transA,transB | `true` 表示转置，`false` 表示不转置 |
| dtype | `float16` 或 `bfloat16` |
| op_type | `add` 或 `mul` |
| inner_precise | `0` 高精度（FP32域融合）/ `1` 高性能（元素域融合） |
| mode | `basic`（fixpipe_opti 调度）/ `streamk`（K 切分调度） |

12 条用例覆盖两种输入类型、两种融合算子（add/mul）、两种精度模式、多种转置组合，
以及 M/N/K 尾块和 N=1 场景。

## 代码结构

```text
fused_mat_mul/
├── CMakeLists.txt
├── scripts/
│   ├── gen_data.py
│   └── verify_result.py
└── fused_mat_mul_elemwise/
    ├── fused_mat_mul_elemwise.cpp
    ├── fused_mat_mul_elemwise.conf
    ├── fused_mat_mul_elemwise.csv
    └── README.md
```

示例直接组装 `GemmUniversal`、`BlockMmad`、`BlockSchedulerMatmulBasic` 和
`BlockEpilogueMatmulElemwise`，复用现有组件。

## 输入输出文件

| 文件 | 内容 |
| --- | --- |
| `input/input_a.bin` | A，未转置时为 `[M,K]`，转置时为 `[K,M]` |
| `input/input_b.bin` | B，未转置时为 `[K,N]`，转置时为 `[N,K]` |
| `input/input_x3.bin` | x3，`[M,N]`，与输入同类型 |
| `output/cpu_output.bin` | CPU golden，`[M,N]`，与输入同类型 |
| `output/npu_out.bin` | NPU 输出，`[M,N]`，与输入同类型 |
| `output/verify_metrics.json` | 精度校验结果，由公共入口汇总 |

数据生成固定随机种子 42。先将输入与 x3 转为指定类型，再执行 FP32 矩阵乘，
然后与 x3 做 add/mul 融合（FP32 域），最后转换输出类型。
BF16 以原始 uint16 位表示保存，转换采用 round-to-nearest-even。

每个输出元素须满足 `abs(actual - golden) <= atol + rtol * abs(golden)`。
FP16 的 `atol`、`rtol` 均为 0.001；BF16 均为 0.008。非有限值或输出长度错误均判为失败。
