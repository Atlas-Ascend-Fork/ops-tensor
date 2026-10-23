# Quant Batch Matmul Kernel UT

## MX Mix 内部行为测试

`test_qbmm_mx_mix.cpp` 检查 `kernel_qbmm_mx_mix.h` 的调度和地址计算，
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
