# RotateDualLevelQuant Example

本样例在 Ascend 950 上通过 <<< >>> 启动 Blaze `RotateDualLevelQuant`。AIC 计算 FP16/BF16 输入的旋转矩阵乘法，两个 AIV 按原始行执行可选 clamp、512 元素一级归一化和 32 元素二级 MX 量化。组件契约见 [RotateDualLevelQuant Kernel](../../../../docs/API/gemm/kernel/kernel_rotate_dual_level_quant.md)。

## 样例配置

`M=35`、`N=544`、`K=32`。Cube 每块处理 16 行、32 列。Scheduler 按 512 列分配任务，因此每行有 2 个一级尺度、17 个有效二级尺度和 1 个 E8M0 填充字节。输出 FP4 时每行占 272 字节，输出 FP8 时每行占 544 字节。

| 用例 | 分组 | 旋转布局 | 输入 → 输出 | Clamp |
| --- | --- | --- | --- | --- |
| `ordinary_shared_fp4` | 普通 | 共享 | BF16 → FP4 | 关闭 |
| `grouped_count_fp4_clamp` | count `[17,0,17]` | 组内共享 | FP16 → FP4 | 0.5 |
| `grouped_cumsum_e4m3_clamp` | cumsum `[17,17,34]` | 组内按列块 | FP16 → E4M3FN | 0.5 |
| `grouped_count_e5m2` | count `[17,0,17]` | 组内按列块 | BF16 → E5M2 | 关闭 |
| `all_groups_empty` | count `[0,0,0]` | 组内共享 | BF16 → FP4 | 关闭 |

样例的最终舍入模式为 rint，FP4 使用 `scaleAlg=0`，FP8 使用 `scaleAlg=1`。平台资源满足条件时启用两个 Fixpipe 接收槽，否则使用单槽。

## 编译与运行

在装有 Ascend 950 和 CANN Toolkit 的 Linux 环境中运行。统一执行器的环境配置及 Tensor API 子模块准备步骤见 [Examples 总览](../../../README.md)。样例的数据生成及校验仅使用 Python 标准库。

在仓库根目录执行以下命令。

```bash
./build.sh --examples --ops=rotate_dual_level_quant --target=rotate_dual_level_quant
```

也可使用公共执行器。编译后，用 `--ti` 选择从 0 开始的用例编号。

```bash
bash examples/acl/common/run.sh --ops=rotate_dual_level_quant --target=rotate_dual_level_quant --build-only
bash examples/acl/common/run.sh --ops=rotate_dual_level_quant --target=rotate_dual_level_quant --skip-build --ti=2
```

## 数据与校验

`gen_data.py` 生成符号随组、列块及元素变化的对角旋转矩阵，以及有限且可精确表示的 FP16/BF16 数据。每个 32 元素块的最大绝对值经一级归一化后等于目标类型最大有限值，因此对应的二级尺度为 1。使用这些条件直接生成 FP4/FP8 编码和两级尺度的预期结果。

| 文件 | 内容 |
| --- | --- |
| `input_x.bin`、`input_rotation.bin` | 连续 ND 输入和旋转矩阵 |
| `input_group_list.bin` | INT64 组表，普通模式为空文件 |
| `golden_y.bin`、`golden_level0_scale.bin`、`golden_level1_scale.bin` | 三路预期输出 |
| `npu_y.bin`、`npu_level0_scale.bin`、`npu_level1_scale.bin` | 执行后回读的设备输出 |
| `verify_metrics.json` | 公共执行器读取的校验结果 |

验证器逐字节比对三个输出。有效行末的 E8M0 填充字节须为 0。Kernel 启动前，Host 将三路输出填为 `0xa5`，据此检查只覆盖前 34 行、以及全部组为空时的输出保留行为。
