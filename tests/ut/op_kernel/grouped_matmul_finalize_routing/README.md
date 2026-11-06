# GroupedMatmulFinalizeRouting Kernel UT

## 覆盖范围

本测试目录使用 Google Test、tikicpulib 和 KERNEL_RUN_KF 检查 arch35 GMMFR 的模板组装及 CPU 仿真执行是否完成。

| 测试文件 | 覆盖内容 |
| --- | --- |
| test_grouped_matmul_finalize_routing.cpp | 9 条 MX WeightNZ/ND/DN/ZN 用例，含 FP32/BF16 输出及 mixed FP8 |
| test_grouped_matmul_finalize_routing_quant.cpp | 5 条固定小 shape 用例，覆盖 INT8/FP8/HIFLOAT8 pertoken、INT8 确定性和 A8W4 入口 |

UT 使用非零量化输入，直接组装公开 Kernel/BlockMmad/Epilogue，不依赖 example 或 ops-transformer。仅检查模板编译和 KERNEL_RUN_KF 执行完成，不读取输出或比较 golden。尾块、bias/shared input 和精度场景由 example 覆盖。

CPU 调试使用 `grouped_matmul_finalize_routing_cpu_debug_compat.h`：复用 GMM A8W4 的设备类型、bias 常量及 UB 地址适配，将寄存器转换的 CastTrait 参数转交给 SDK 的枚举参数接口。转换仍由 SDK 实现执行；兼容宏仅在组件头文件包含期间生效，随后取消定义。

## 构建和运行

在已安装配套 CANN 与 tikicpulib 的环境，从 ops-tensor 根目录执行：

```bash
source /usr/local/Ascend/cann/set_env.sh
bash build.sh --opkernel -u --ops=grouped_matmul_finalize_routing
```

也可以独立配置：

```bash
cmake -S tests/ut/op_kernel -B build/kernel_ut \
  -DOP_KERNEL_OPS=grouped_matmul_finalize_routing -DKERNEL_UT_SOC=ascend950 \
  -DASCEND_HOME_PATH="$ASCEND_HOME_PATH"
cmake --build build/kernel_ut -j2
./build/kernel_ut/ops_tensor_kernel_ut_ascend950 \
  --gtest_filter='GroupedMatmulFinalizeRouting*'
```

## 检查边界

这些测试检查模板组装、调度和执行完成，不作为 NPU 数值精度证明。KERNEL_RUN_KF 传播子进程的崩溃、断言和失败状态。构建时应使用与当前 tensor_api 配套的 CANN 和 tikicpulib。

NPU 数值验证使用 [量化直调示例](../../../../examples/acl/grouped_matmul/grouped_matmul_finalize_routing/README.md) 的独立 CPU golden；同一目标也覆盖 MX WeightNZ/BF16 输出。
