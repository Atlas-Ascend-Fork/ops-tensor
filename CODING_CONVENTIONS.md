# 编程规范

## 一、命名约定

### 1.1 文件命名

文件名格式：`层级_算子_类型_策略`，各字段含义如下：

| 字段 | 说明 | 取值示例 |
|------|------|----------|
| 层级 | 代码所属层级 | `kernel` / `block_mmad` |
| 算子 | 算子功能 | `matmul` / `batchmatmul` / `qbmm` / `wqmm` / `qgmm` / `wqgmm` / `tbmm` |
| 类型(可选) | 任务类型，默认不填表示 cube | `mx`（mx量化的纯cube模板） / `mx_mix`（mx量化的mix模板） / `mix`(非mx量化的mix模板) |
| 策略 | 模板策略或应用场景（策略中如果要包括数据类型，数据类型放最前面），默认不填表示 basic；可由多个 `_` 连接的子策略组成 | `streamk` / `al1_full_load` / `basic_split_k` / `broadcast` / `iterbatch_broadcast` / `fixpipe_quant` / `fixpipe_opti` / `activation_quant` / `without_batch` / `l0c_pingpong` / ... |


```cpp
block_mmad.h                       // 基类文件

// 非量化 + 策略 —— 仓库已有合规文件
block_mmad_matmul_basic.h
kernel_matmul_streamk.h

// 类型（非mx量化的cube/mix模式）+ basic 策略 —— 仓库已有合规文件
kernel_qbmm_cube.h
kernel_qbmm_mix.h

// 类型（mx量化的纯cube模板）+ basic 策略 —— 仓库已有合规文件
kernel_qbmm_mx.h
block_mmad_qbmm_mx.h
block_mmad_qgmm_mx.h

// 类型（mx_mix = MX 量化 + Mix 模板）+ basic 策略 —— 仓库已有合规文件
kernel_qbmm_mx_mix.h
kernel_wqmm_mx_mix.h
block_mmad_wqmm_mx_mix.h
kernel_qgmm_mx_mix.h

// 类型 + 策略 —— 仓库已有合规文件
kernel_qbmm_pertensor_streamk.h    // pertensor + streamk
kernel_qbmm_mx_streamk.h           // mx + streamk
kernel_qgmm_mix_fixpipe_quant.h    // mix + fixpipe_quant

// 非量化 + 策略 —— 仓库已有合规文件
kernel_batchmatmul_broadcast.h             // broadcast
kernel_batchmatmul_iterbatch_broadcast.h   // iterbatch_broadcast

// 多子策略 —— 仓库已有合规文件
block_mmad_matmul_basic_split_k.h  // 非量化 + basic_split_k
block_mmad_matmul_streamk_split_k.h // 非量化 + streamk_split_k
```

### 1.2 类模板参数

带下划线后缀的大驼峰（UpperCamelCase + 后缀 `_`）。

```cpp
// 正确
template <class AType_, class LayoutA_, uint64_t FullLoadMode_>
class MyClass {};

// 错误
template <class a_type, class layout_a, uint64_t full_load_mode>
class MyClass {};
```

### 1.3 类型名称

命名空间、类（class）、结构体（struct）、联合体（union）、枚举类型（enum）、`typedef` / `using` 定义的类型别名，均使用大驼峰（UpperCamelCase）。

```cpp
// 正确
namespace Blaze {}
class BlockMmad {};
struct Params {};
enum class ErrorCode {};
using TupleShape = asc::te::shape<int64_t, int64_t>;
typedef int32_t IndexType;

// 错误
namespace BLAZE {}
class block_mmad {};
struct PARAMS {};
```

### 1.4 函数名称

全局函数、作用域内函数、成员函数均使用大驼峰（UpperCamelCase）。

```cpp
// 正确
void ComputeResult();
void InitParams();

// 错误
void compute_result();
void initParams();
```

### 1.5 常量、枚举值、宏

全大写，下划线分割（UPPER_SNAKE_CASE）。

```cpp
// 正确
static constexpr bool TRANS_A = true;
static constexpr uint64_t MAX_SIZE = 256;
#define BLAZE_GEMM_API
enum class ErrorCode { INVALID_PARAM, OUT_OF_MEMORY };

// 错误
static constexpr bool transA = true;
static constexpr uint64_t max_size = 256;
```

### 1.6 类成员变量

带下划线后缀的小驼峰（lowerCamelCase + 后缀 `_`）。

```cpp
// 正确
class MyClass {
private:
    int32_t dataSize_;
    GM_ADDR bufferAddr_;
};

// 错误
class MyClass {
private:
    int32_t m_dataSize;
    GM_ADDR data_size;
};
```

### 1.7 全局变量

带 `g_` 前缀的小驼峰。

```cpp
// 正确
int32_t g_maxCount;
GM_ADDR g_globalAddr;

// 错误
int32_t gMaxCount;
GM_ADDR global_addr;
```

### 1.8 局部变量、函数参数、宏参数、结构体/联合体成员变量

小驼峰（lowerCamelCase）。

```cpp
// 正确
void MyFunction(int32_t inputSize, GM_ADDR srcAddr) {
    int32_t localCount = 0;
    struct Params {
        GM_ADDR aGmAddr;
        uint64_t mL1;
    };
}

// 错误
void MyFunction(int32_t input_size, GM_ADDR src_addr) {
    int32_t local_count = 0;
}
```

### 1.9 命名约定汇总

| 类别 | 命名风格 | 示例 |
|------|----------|------|
| 文件名 | `层级_算子_类型_策略` | `kernel_matmul_basic.h`, `block_mmad_qbmm_mx.h`, `kernel_qbmm_mx_mix.h`, `block_mmad_matmul_basic_split_k.h` |
| 类模板参数 | 大驼峰 + `_` 后缀 | `AType_`, `FullLoadMode_` |
| 类型名（类/结构体/联合体/枚举/typedef/别名/命名空间） | 大驼峰 | `BlockMmad`, `ErrorCode` |
| 函数名（全局/成员/作用域内） | 大驼峰 | `ComputeResult()` |
| 常量、枚举值、宏 | 全大写 + 下划线 | `TRANS_A`, `MAX_SIZE` |
| 类成员变量 | 小驼峰 + `_` 后缀 | `dataSize_`, `bufferAddr_` |
| 全局变量 | `g_` + 小驼峰 | `g_maxCount` |
| 局部变量、函数参数、宏参数、结构体/联合体成员 | 小驼峰 | `inputSize`, `localCount` |

---

## 二、编码约束

### 2.1 禁止调用 AscendC 高阶 API

禁止直接调用 AscendC 高阶 API 接口（如 `AuxGetC0Size` 等），应使用 `tensor_api/utils` 下的接口（如 `c0_element`）。

```cpp
// 错误：直接调用 AscendC 高阶 API
constexpr uint32_t c0Size = AscendC::AuxGetC0Size<AType>()>;

// 正确：使用 tensor_api/utils 封装
constexpr uint32_t c0Size = asc::te::c0_element<AType>;
```

### 2.2 禁止全局作用域中引入大命名空间

禁止在全局作用域中使用 `using namespace` 引入较大的命名空间（如 `using namespace AscendC`）。

```cpp
// 错误：全局作用域引入命名空间
using namespace AscendC;       // 禁止
using namespace asc::te;   // 禁止

// 正确：函数作用域内局部引入，或使用全限定名
void MyFunc() {
    using asc::te::shape;  // 允许：局部引入单个名称
    shape<int64_t, int64_t> s;
}

void MyFunc() {
    AscendC::GlobalTensor<int64_t> tensor;  // 允许：全限定名
}
```

### 2.3 头文件引用路径约束

头文件 `#include` 必须使用从 `blaze` 根目录开始的绝对路径（如 `blaze/xx/xx.h`），不允许使用相对路径。

```cpp
// 错误：使用相对路径
#include "../include/block_mmad.h"
#include "./utils.h"

// 正确：使用从 blaze 根目录开始的绝对路径
#include "blaze/api/gemm/block/block_mmad.h"
#include "blaze/common/utils.h"
```

### 2.4 禁止引用特定目录

- 支持引用 `include` 目录下的 `tensor-api` 头文件。
- 禁止直接引用 `impl` 目录下的文件。

```cpp
// 错误：直接引用 impl 目录
#include "blaze/api/gemm/impl/block_mmad_internal.h"  // 禁止

// 正确：引用 include 目录下的公开头文件
#include "blaze/api/gemm/block/block_mmad.h"
```

### 2.5 禁止在类成员变量初始化表达式中使用被 `__NPU_ARCH__` 宏保护的函数或常量

在混合编译场景下，存在编译失败的风险。只能在类方法中使用，类方法存在 `__aicore__` 前缀，因此编译器会忽略函数内部实现。

```cpp
// 错误：成员变量声明处直接使用架构相关函数
template <class AType_, class LayoutA_>
class MyKernel {
private:
    // GetArchDepValue() 仅在 __NPU_ARCH__ 下可用，CPU 侧编译会失败
    uint32_t archValue_ = GetArchDepValue();
};

// 正确：在 __aicore__ 方法中初始化
template <class AType_, class LayoutA_>
class MyKernel {
private:
    uint32_t archValue_;

    __aicore__ void Init() {
        archValue_ = GetArchDepValue();
    }
};
```

### 2.6 控制对外接口可见性

严格控制对外接口，不对外开放的类方法和成员变量必须设置为 `private`。

对于 Kernel 来说，只有 `operator()` 接口对外开放，`Init` 和 `Run` 都应当设置为 `private`。

```cpp
// 错误：Init 和 Run 暴露为 public
template <class AType_, class LayoutA_>
class MyKernel {
public:
    void Init() { /* ... */ }
    void Run() { /* ... */ }
};

// 正确：仅 operator() 对外暴露
template <class AType_, class LayoutA_>
class MyKernel {
public:
    void operator()() {
        Init();
        Run();
    }

private:
    __aicore__ void Init() { /* ... */ }
    __aicore__ void Run() { /* ... */ }
};
```

### 2.7 Kernel 与 BlockMmad 职责划分

- **Kernel 层**：完成多核间任务分布调度，并确定单核需要处理的数据块大小。
- **BlockMmad**：完成单核内数据块的处理。

`ProblemShape` 总大小 → `BlockShape` 单核要处理的大小 → `TileShape` L0 的切分。

```cpp
// Kernel 层：确定单核要处理的 blockShape
template <class AType_, class LayoutA_>
class MyKernel {
public:
    void operator()() {
        // 根据 problemShape 和 核数 计算 blockShape
        auto blockShape = ComputeBlockShape(problemShape, coreNum);
        blockMmad_(blockShape, /* ... */);
    }

private:
    __aicore__ void ComputeBlockShape(/* ... */) { /* ... */ }
    BlockMmad<AType_, LayoutA_> blockMmad_;
};

// BlockMmad 层：处理单核内的数据块，不再关心多核切分
template <class AType_, class LayoutA_>
class BlockMmad {
public:
    void operator()(/* blockShape, ... */) {
        // 在 blockShape 内进一步做 L0 Tile 切分
    }
};
```

### 2.8 静态断言校验

各层接口须显式增加编译时 `static_assert` 校验，不支持的场景直接报错提示。比如输入的 dtype、shape、format 校验；不支持后处理的 kernel 需要增加后处理校验等。

```cpp
// 正确：编译期校验模板参数合法性
template <class AType_, class BType_, bool TransA, bool TransB>
class BlockMmad {
    // dtype 校验：仅支持 FP16 和 BF16
    static_assert(
        std::is_same_v<AType_, AscendC::half> || std::is_same_v<AType_, AscendC::bfloat16_t>,
        "AType_ must be half or bfloat16_t");
    static_assert(
        std::is_same_v<BType_, AscendC::half> || std::is_same_v<BType_, AscendC::bfloat16_t>,
        "BType_ must be half or bfloat16_t");

    // 仅支持 NT 排列
    static_assert(!TransA && TransB, "Only NT (TransA=false, TransB=true) is supported");
};

// 正确：Kernel 层校验不支持后处理场景
template <class AType_, bool HasEpilogue>
class MyKernel {
    static_assert(!HasEpilogue, "Epilogue is not supported in this kernel");
};
```

### 2.9 变量就近定义

变量应在使用前就近定义，避免过早声明。变量定义太早、使用的地方太晚，可能会触发 Load / Store，影响性能。编译器使用寄存器保存变量，当活跃变量数超过寄存器个数时，则会选一个寄存器通过 Store 指令保存到栈内存，等到再次使用此变量时，通过 Load 指令读回寄存器。

```cpp
// 错误：变量过早定义，中间大量代码可能溢出到栈
void Run() {
    int32_t result = 0;
    // ... 大量不涉及 result 的中间计算 ...
    // 编译器可能将 result 溢出到栈内存
    // ...
    Compute(result);  // 使用时再通过 Load 读回
}

// 正确：变量在首次使用时定义
void Run() {
    // ... 大量不涉及 result 的中间计算 ...
    int32_t result = ComputeIntermediate();
    UseResult(result);
}
```

### 2.10 优先使用局部变量

优先使用局部变量替代成员变量，以减少寄存器压力。

```cpp
// 错误：将临时中间状态设为成员变量，增加类内寄存器压力
template <class AType_>
class BlockMmad {
private:
    int32_t tempKIter_;    // 整个类生命周期占用寄存器
    int32_t tempBatchIdx_; // 整个类生命周期占用寄存器

    __aicore__ void Run() {
        for (int k = 0; k < kSize; ++k) {
            tempKIter_ = k;
            // ...
        }
    }
};

// 正确：中间变量定义为局部变量，生命周期结束后释放寄存器
template <class AType_>
class BlockMmad {
private:
    __aicore__ void Run() {
        for (int k = 0; k < kSize; ++k) {
            int32_t tempKIter = k;   // 仅在循环内占用寄存器
            int32_t tempBatchIdx = 0; // 仅在循环内占用寄存器
            // ...
        }
    }
};
```

### 2.11 UT 要求

- 新增功能必须补充对应 UT。
- 已有 UT 不能修改。

### 2.12 禁止修改核心类模板参数

禁止修改 `BlockMmad`、`GemmUniversal` 类定义的模板参数。这些模板参数是经过严格验证的公共接口，增删或重排会破坏向前兼容性。

```cpp
// 错误：在已有模板参数列表中插入新参数
template <class AType_, class BType_, class NewParam_, class CType_>
//                                      ↑ 插入到中间，破坏 API 兼容性
class BlockMmad {};

// 正确：新参数追加到模板参数列表末尾（如有必要）
template <class AType_, class BType_, class CType_, class NewParam_>
class BlockMmad {};
```

### 2.13 头文件自包含

头文件必须是自包含的（self-contained），即不依赖外部前置 `#include`。

```cpp
// 错误：my_header.h 依赖使用者先 include 其他头文件
// 使用者必须这样写：
//   #include <AscendC/core/tensor.h>  // 前置依赖，my_header.h 才能正常编译
//   #include "blaze/xx/my_header.h"

// 正确：my_header.h 内部包含所有依赖
#pragma once
#include <AscendC/core/tensor.h>   // 自身需要的依赖，自己引入
#include <type_traits>

namespace Blaze {
    // ...
}
```

### 2.14 禁止单独 Prelogue 目录

`blaze` 代码中不单独增加 `prelogue` 目录，前处理逻辑放在各自的 `kernel` 中。

```cpp
// 错误：创建独立的 prelogue 目录和文件
// blaze/api/gemm/prelogue/prelogue_qbmm.h

// 正确：前处理逻辑放入对应 kernel 文件内
// blaze/api/gemm/kernel/kernel_qbmm_cube.h（内部包含前处理逻辑）
```

### 2.15 Params 结构体约束

- `Params` 结构体中不要提供自定义构造函数，只能提供默认构造函数。
- 新增的参数必须放在结构体最后。

```cpp
// 错误：提供自定义构造函数、新字段插入中间
struct Params {
    GM_ADDR aGmAddr;
    uint32_t newField;    // 禁止：插入已有字段之前
    uint64_t mL1;
    uint64_t mHbm;

    Params(GM_ADDR a, uint64_t l1, uint64_t hbm)  // 禁止：自定义构造函数
        : aGmAddr(a), mL1(l1), mHbm(hbm) {}
};

// 正确：默认构造函数，新字段追加到末尾
struct Params {
    GM_ADDR aGmAddr;
    uint64_t mL1;
    uint64_t mHbm;
    uint32_t newField;    // 新字段放在最后
};
```

### 2.16 vf类接口需要使用__NPU_ARCH__包裹

兼容性要求，当前最早兼容版本中相关接口被__NPU_ARCH__包裹，不添加该宏会导致在对应版本的toolkit无法编译

```c++
#ifdef __NPU_ARCH__ // 使用宏定义包住AscendC::Reg
static constexpr AscendC::Reg::CastTrait CAST_FR_FP32_TO_BF16 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

static constexpr AscendC::Reg::CastTrait CAST_FR_BF16_TO_FP32 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};
#endif
```

### 2.17 不允许使用TPipe和TBuf接口

TPipe和TBuf用于管理同步，当前blaze已自行管理同步，性能更优

```c++
// 错误
AscendC::TPipe pipe_;
AscendC::TBuf<QuePosition::VECCALC> bufQueue_;
```

### 2.18 头文件中不允许使用匿名空间

匿名空间内定义内容仅对当前编译单元可见。头文件中使用匿名空间实际效果等于此头文件的源文件定义了一组私有成员，而头文件一般用于定义模块间的公共接口，容易引发混淆，不建议使用

```c++
// 错误
namespace {
    namespace BlockAttnResPrepareDetail {
        // xxxx
    }
}

// 正确
namespace BlockAttnResPrepareDetail {
    // xxxx
}
```
