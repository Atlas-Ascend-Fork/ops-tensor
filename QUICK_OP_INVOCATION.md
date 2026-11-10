# 算子开发快速指南

本指南介绍如何基于 `ops-tensor` 进行算子开发，包括编写 Kernel UT 测试、运行测试和调试。

## 一、开发流程概览

### 1. 算子开发步骤

```
1. 在 include/blaze/ 下实现算子 Kernel
2. 在 tests/ut/op_kernel/ 下编写 Kernel UT 测试
3. 运行 Kernel UT 验证正确性
4. 优化和迭代
```

### 2. 目录结构参考

```
tests/ut/op_kernel/
├── mat_mul/                      # MatMul 算子 Kernel UT
│   ├── mat_mul.cpp              # Kernel 入口（统一调度）
│   ├── mat_mul_basic.h          # Basic 实现的 Wrapper
│   ├── mat_mul_stream_k.h       # StreamK 实现的 Wrapper
│   ├── mat_mul_tiling_data.h    # Tiling 数据结构
│   ├── test_mat_mul.cpp         # GTest 测试用例
│   ├── CMakeLists.txt           # CMake 配置
│   └── matmul_data/             # 测试数据生成脚本
│       └── gen_data.py
│
├── quant_batch_matmul/           # QuantBatchMatmul 算子 Kernel UT
│   ├── qbmm_cpu_debug_stub.h   # CPU Debug 兼容定义
│   ├── qbmm_cube.h              # Cube / per-tensor StreamK Wrapper
│   ├── qbmm_mix.h               # MIX / MIX without-batch Wrapper
│   ├── qbmm_mx.h                # MX / L0C PingPong / MX StreamK Wrapper
│   ├── qbmm_tiling_data.h       # 公共 Tiling 数据结构
│   ├── qbmm_ut_fill_helpers.h   # 公共 Params 填充辅助函数
│   ├── test_quant_batch_matmul.cpp # GTest 测试用例
│   ├── CMakeLists.txt
│   └── qbmm_data/
│       └── gen_data.py          # smoke 用例输入数据生成脚本
│
└── kernel_ut_runner.h            # UT 运行器（捕获子进程失败）
```

## 二、编写 Kernel UT 测试

### 1. 创建测试目录

```bash
cd tests/ut/op_kernel
mkdir my_operator
cd my_operator
```

### 2. 编写 Kernel 入口（统一或按实现拆分）

可参考 `mat_mul.cpp` 创建统一入口文件，也可参考 `qbmm_cube.h`、`qbmm_mix.h`
和 `qbmm_mx.h` 按实现类型拆分 Wrapper 与 Kernel 入口：

```cpp
// my_operator.cpp
#pragma once
#include <cstring>
#include "my_operator_impl.h"

namespace MyOpUT {

enum OpApiType : int {
    OP_TYPE_BASIC = 0,
    OP_TYPE_ADVANCED = 1,
};

template <int OP_TYPE, class DTYPE_X, class DTYPE_Y>
__global__ __aicore__ void my_operator_kernel_entry(
    GM_ADDR xGM, GM_ADDR yGM, GM_ADDR tilingGM)
{
    MyOperatorTilingData tilingData;
    memcpy(&tilingData, tilingGM, sizeof(MyOperatorTilingData));

    if constexpr (OP_TYPE == OP_TYPE_BASIC) {
        MyOperatorBasicWrapper<DTYPE_X, DTYPE_Y>(xGM, yGM, tilingData);
    } else if constexpr (OP_TYPE == OP_TYPE_ADVANCED) {
        MyOperatorAdvancedWrapper<DTYPE_X, DTYPE_Y>(xGM, yGM, tilingData);
    } else {
        static_assert(sizeof(OP_TYPE) == 0, "Unsupported OP_TYPE");
    }
}

} // namespace MyOpUT
```

**关键点**：

- 使用 `namespace` 避免符号冲突
- 移除 `#pragma once`（测试文件直接 include）
- 统一入口通过 `OP_TYPE` 参数区分不同实现

### 3. 编写 Wrapper 实现

```cpp
// my_operator_impl.h
#pragma once
#include "blaze_kernel_stub.h"
#include "kernel_operator.h"
#include "tensor_api/tensor.h"

namespace MyOpUT {

struct MyOperatorTilingData {
    int64_t m;
    int64_t n;
    int64_t k;
};

template <typename XType, typename YType>
__aicore__ inline void MyOperatorBasicWrapper(
    GM_ADDR xGM, GM_ADDR yGM, const MyOperatorTilingData& tiling)
{
    // 调用 Blaze Kernel
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using LayoutX = asc::te::nd_ext_layout_ptn;

    auto gmX = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::gm>(xGM),
        LayoutX{}(tiling.m, tiling.k));

    // ... 调用 Kernel
}

} // namespace MyOpUT
```

### 4. 编写 GTest 测试用例

```cpp
// test_my_operator.cpp
#include "gtest/gtest.h"
#include "kernel_ut_runner.h"
#include "tikicpulib.h"
#include "my_operator.cpp"  // 直接 include 入口文件

using MyOpUT::OP_TYPE_BASIC;
using MyOpUT::my_operator_kernel_entry;

class MyOperatorTest : public testing::Test {
protected:
    GM_ADDR xGM = nullptr;
    GM_ADDR yGM = nullptr;
    GM_ADDR tilingGM = nullptr;

    void TearDown() override {
        if (xGM) AscendC::GmFree((void*)xGM);
        if (yGM) AscendC::GmFree((void*)yGM);
        if (tilingGM) AscendC::GmFree((void*)tilingGM);
    }
};

TEST_F(MyOperatorTest, TestBasicFP16) {
    const int64_t M = 16;
    const int64_t N = 16;
    const uint32_t blockNum = 1;

    xGM = (GM_ADDR)AscendC::GmAlloc(M * N * sizeof(half));
    yGM = (GM_ADDR)AscendC::GmAlloc(M * N * sizeof(half));
    tilingGM = (GM_ADDR)AscendC::GmAlloc(sizeof(MyOpUT::MyOperatorTilingData));

    auto* tiling = reinterpret_cast<MyOpUT::MyOperatorTilingData*>(tilingGM);
    tiling->m = M;
    tiling->n = N;

    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    auto kernelFunc = my_operator_kernel_entry<OP_TYPE_BASIC, half, half>;
    ASSERT_TRUE(KERNEL_RUN_KF(kernelFunc, blockNum, xGM, yGM, tilingGM))
        << "Kernel execution failed";
}
```

### 5. 配置 CMakeLists.txt

```cmake
# CMakeLists.txt
AddOpTestCase(my_operator "ascend950pr_9599" "")
```

## 三、运行 Kernel UT

### 1. 基本命令

```bash
# 构建并执行所有算子的 Kernel UT
./build.sh --opkernel -u

# 构建并执行指定算子
./build.sh --opkernel -u --ops=mat_mul

# 执行多个算子
./build.sh --opkernel -u --ops=mat_mul,quant_batch_matmul
```

### 2. 指定 SoC

```bash
# 为指定 SoC 执行 Kernel UT
./build.sh --opkernel -u --soc=ascend950
```

### 3. 测试超时设置

```bash
# 设置测试超时时间（默认 300 秒）
./build.sh --opkernel -u --test-timeout=600
```

### 4. 查看测试结果

成功输出：

```
[SUCCESS] Kernel UT all SoC tests passed
```

失败输出：

```
[ERROR] mat_mul failed (exit code: 1)
```

## 四、Kernel UT 运行机制

### 1. 测试框架架构

```
tests/ut/op_kernel/
├── kernel_ut_runner.h    # 捕获子进程失败
├── test_op_kernel_main.cpp  # GTest 主入口
└── k3_pvwrap.cpp         # CANN 包装器
```

### 2. KERNEL_RUN_KF 工作原理

`KERNEL_RUN_KF` 宏用于运行 Kernel 并捕获失败：

```cpp
// 捕获 stdout 输出
// 检测 [SUCCESS] / [ERROR] / [FAILED] 标记
// 返回 true 仅当有 [SUCCESS] 且无 [ERROR]/[FAILED]
ASSERT_TRUE(KERNEL_RUN_KF(kernelFunc, blockNum, args...));
```

**注意**：CANN SDK 的 `ICPU_RUN_KF` 会 fork 子进程，子进程失败时父进程无法感知。`KERNEL_RUN_KF` 通过捕获 stdout 解决此问题。

### 3. SoC 多版本测试

Kernel UT 会为每个支持的 SoC 生成独立可执行文件：

```
build/kernel_ut/
├── ops_tensor_kernel_ut_ascend950    # Ascend950 测试
└── ops_tensor_kernel_ut_ascend910b   # Ascend910B 测试（暂不支持）
```

## 五、调试技巧

### 1. 使用 VERBOSE 模式

```bash
./build.sh --opkernel -u --ops=mat_mul -v
```

### 2. 检查 Tiling 数据

在测试用例中打印 Tiling 数据：

```cpp
auto* tiling = reinterpret_cast<TilingData*>(tilingGM);
std::cout << "M=" << tiling->m << ", N=" << tiling->n << std::endl;
```

### 3. 使用 AscendC CPU Debug

```cpp
AscendC::SetKernelMode(KernelMode::MIX_MODE);
```

### 4. 查看临时文件

测试数据通常位于：

```
tests/ut/op_kernel/mat_mul/matmul_data/
├── input_a.bin
├── input_b.bin
└── golden_c.bin
```

## 六、常见问题

### Q1: OP_TYPE static_assert 错误

**原因**：多个测试文件 include 同一个入口文件，`enum` 定义冲突。

**解决**：使用 namespace 包裹 enum 定义：

```cpp
namespace MyOpUT {
enum OpType : int { ... };
}
```

### Q2: 符号重复定义

**原因**：入口文件使用 `#pragma once`。

**解决**：移除 `#pragma once`，让每个测试文件独立编译。

### Q3: 测试超时

**解决**：增加超时时间：

```bash
./build.sh --opkernel -u --test-timeout=600
```

### Q4: 子进程失败无法捕获

**解决**：使用 `KERNEL_RUN_KF` 替代 `ICPU_RUN_KF`：

```cpp
ASSERT_TRUE(KERNEL_RUN_KF(kernelFunc, blockNum, args...));
```

## 七、参考示例

### 1. MatMul Kernel UT

路径：`tests/ut/op_kernel/mat_mul/`

支持的测试类型：

- FP16 Basic MatMul
- FP16 StreamK MatMul
- FP32 MatMul

### 2. QuantBatchMatmul Kernel UT

路径：`tests/ut/op_kernel/quant_batch_matmul/`

支持的测试类型：

- INT8 A8W8 Cube
- INT8 A8W8 MIX
- MXFP8 / MXFP4 MatMul
- INT8 / FP8 StreamK

## 八、更多信息

- **完整构建文档**: [README.md](README.md)
- **环境安装指南**: [QUICKSTART.md](QUICKSTART.md)
- **API 文档**: [docs/API/](docs/API/)
- **Blaze 模块**: [include/blaze/README.md](include/blaze/README.md)
