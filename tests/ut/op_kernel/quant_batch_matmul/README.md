# Quant Batch Matmul Kernel UT

## MX Mix 内部行为测试

`arch35/test_qbmm_mx_mix.cpp` 检查 `kernel_qbmm_mx_mix.h` 的调度和地址计算，
使用 `BlockSchedulerQuantBatchMatmulV3` 和 Tensor API 布局、切片逻辑。

MMAD 和 Epilogue 使用记录调用参数的测试组件。测试读取各 AIC/AIV 的 GM 记录，检查：

- 每个输出元素恰好被分配一次，跨核、尾块和 batch 无遗漏或重叠。
- 两个 AIV 收到的 tile 顺序、M/N 坐标和尺寸与对应 AIC 一致。
- A/B、两路 scale、输出和 bias 的地址符合独立的四维广播坐标推导。
- FP4 元素偏移正确转换为打包字节偏移，NZ/ZN 的 batch 步长包含物理 padding。
- B 的 L2 策略覆盖完整 M、N 对齐、转置、强制保留及同核 N 尾块恢复；atomic 输出禁用 L2。
- 双轴尾块重切分、跨 batch 保持尾块重切分、空 split 和空 batch 的行为正确。

共 12 个测试，包含 22 次 kernel 启动。覆盖 FP8/FP4 地址规则、ND/DN/NZ/ZN 输入、
ND/DN 输出、共享/逐 batch bias、空 bias，以及 `dbL0C=1/2` 参数传递。
测试不执行真实 MMAD/VF 计算；设备精度和硬件同步时序须在设备侧验证。

在已配置 CANN / tikicpulib 的 Linux 环境运行：

```bash
GTEST_FILTER='QbmmMxMix.*' bash build.sh --opkernel -u --ops=quant_batch_matmul --cov
```

`--cov` 生成本次执行的覆盖率报告。

## MX 显式 L0 子块

`QBMMV3Test.ExplicitMx*` 从 `kernel(params)` 下发非零 `mL0/nL0`，覆盖全程切 N 复用 A/ScaleA、
MXFP4 切 M 复用 B/ScaleB、切 M 重新搬入和 FP32 AtomicAdd。K=320、baseK=64，最后一个 L1 窗口只含一个计算子块。
AtomicAdd 的输出初值为 7。另有整块不切分和零 M/N 沿用旧策略的用例，覆盖一次初始化的显式计划和零参数回退。用例使用非恒定矩阵及 E8M0 scale。新增显式计划覆盖双边 batch、A/B 分别广播、共享 bias、逐 batch bias，以及 without-batch bias；输入及 scale 随 batch 变化，golden 逐 batch 计算。

默认 CPU-debug 用例验证内核执行。使用支持 Cube/Fixpipe 写回 GM 的仿真后端时，设置
`QBMM_MX_UT_VERIFY_OUTPUT=1` 启用逐元素 golden 校验。仅记录指令的后端会保留输出初值。

## HiF4 冒烟测试（Ascend960）

`arch92/test_qbmm_mx_hif4.cpp` 覆盖 batched 与 without_batch 的 HiF4 scale 寻址，
并静态校验 `MxScaleKLen`、`GetScaleC0`、`GetScaleGmPos` 的 HiF4 语义。
该用例仅在 `--soc=ascend960` 时编入：

```bash
bash build.sh --opkernel -u --soc=ascend960 --ops=quant_batch_matmul
```
