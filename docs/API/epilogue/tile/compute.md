# Compute（Tile 架构派发入口）
> [代码位置](../../../../include/blaze/epilogue/tile/compute.h)

## 功能说明

`compute.h` 是 Block 层引入 Tile 组件的统一入口。
当前在定义 `__NPU_ARCH__` 时引入 `arch35/` 实现。

```cpp
#pragma once

#if defined(__NPU_ARCH__)
#include "blaze/epilogue/tile/arch35/dequant.h"
#include "blaze/epilogue/tile/arch35/gelu.h"
#include "blaze/epilogue/tile/arch35/initialize_empty_softmax.h"
#include "blaze/epilogue/tile/arch35/mx_quant.h"
#include "blaze/epilogue/tile/arch35/reduce_square.h"
#include "blaze/epilogue/tile/arch35/rms_softmax.h"
#include "blaze/epilogue/tile/arch35/sigmoid.h"
#include "blaze/epilogue/tile/arch35/swish.h"
#endif
```

## 特殊约束

### include 规范
Block Epilogue 通过 `blaze/epilogue/tile/compute.h` 引入组件，避免直接依赖 `tile/arch35/<name>.h`。

### 注册规范
新增 Tile 时，在 `defined(__NPU_ARCH__)` 守卫内添加 include，并保持字母序。

### 架构守卫
当前 `arch35` 实现面向 Ascend 950 / dav-3510，在定义 `__NPU_ARCH__` 时引入。
Block 在对应架构路径中通过本入口调用 Tile，调用处不再重复添加架构判断。

## 使用方式

```cpp
#include "blaze/epilogue/tile/compute.h"

// BlockEpilogue 实现内部
auto layout = Gemm::MakeNDExtLayout(mSize, nSize, nAligned);
auto srcTensor = asc::te::make_tensor(
    asc::te::make_mem_ptr<asc::te::location::ub, float>(0), layout);
auto dstTensor = asc::te::make_tensor(
    asc::te::make_mem_ptr<asc::te::location::ub, bfloat16_t>(dstOffset), layout);
Blaze::Epilogue::Tile::Gelu<bfloat16_t, float> gelu;
gelu.GeluTanh(srcTensor, dstTensor, mSize, nSize);
```

## 数据流

```
BlockEpilogue（管理 UB 偏移/同步/slot）
    ↓ #include
tile/compute.h（按 __NPU_ARCH__ 派发）
    ↓
tile/arch35/<name>.h（架构实现，VF 寄存器级计算）
```
