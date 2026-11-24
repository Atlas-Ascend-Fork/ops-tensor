# 二级量化分组矩阵乘 Kernel UT

本目录通过仓内 `AddOpTestCase` 注册 Ascend 950（`ascend950pr_9599`）测试，使用 `KERNEL_RUN_KF` 执行真实的 Blaze `GemmUniversal`、`BlockMmad` 和 `BlockEpilogueDualLevelScale` 组合。

## 用例范围

共 16 个参数化用例，覆盖以下场景：

- FP16、BF16 输出与 NZ、ZN 权重布局的四种组合。
- 可选 bias，以及前缀和、长度两种分组列表。
- M、N、K 尾块，单行输入，首尾空专家及中间空专家。
- 多个专家之间的 partial 槽位连续复用。
- K=4098 的一级 scale 窗口切换、256 元素的 level0 K 分段及 K=16386 的二级 scale 窗口切换。
- 多核执行及超出基本块的 M、N 切分。
- type=2 稀疏列表中按专家编号递增的非零行组、单个及多个末尾空专家，以及两种权重布局。

权重按对应转置场景的 FRACTAL_NZ 物理尺寸分配，输入采用非零 FP4 数据及非零 scale。无 bias 场景向 Kernel 传入空指针。GM 缓冲区采用 RAII 管理，并检查申请结果；输出有效区前后各保留 64 字节哨兵。

## 执行方法

在已安装 CANN Toolkit 并加载环境变量的 Linux 环境中执行：

```bash
bash build.sh --opkernel -u --ops=dual_level_grouped_matmul --soc=ascend950
```

构建后可单独运行：

```bash
./build/kernel_ut/ops_tensor_kernel_ut_ascend950 --gtest_filter='DualLevelMx/DualLevelGroupedMatmulKernelTest.*'
```

## 验证边界

这些用例检查模板实例化、CPU 调试执行状态和输出区前后哨兵，不将 CPU 指令桩的输出作为精度验证结果。CPU 调试头文件及仓内兼容层对部分 Cube、Fixpipe、VF 指令提供桩实现；完整计算正确性、跨核同步及硬件 DMA 边界仍需通过上板精度用例验证。

测试复用既有 QGMM CPU 调试兼容头，其宏只在包含相关设备头期间生效，随后立即撤销，不修改产品代码的计算路径。
