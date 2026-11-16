# Block Epilogue 编译期兼容适配

头文件：`blaze/epilogue/block/block_epilogue_compat.h`

该头文件供 `9.2.0` 已有 Kernel 兼容不同版本的自定义 Epilogue，不要求新增 Epilogue 提供旧式初始化接口。`operator()` 保持各 Epilogue 原有入参形式，不引入 `ExecParams`。

## 接口检测

| 类型 | 用途 |
|------|------|
| `HasParamsOnlyInit<BlockEpilogue, Params>` | 检测是否支持 `Init(const Params&)` |
| `HasGetAccumulatorTensor<BlockEpilogue>` | 检测是否提供 `GetAccumulatorTensor` 及对应参数类型 |

检测结果为静态成员 `value`，用于 `if constexpr` 分支。初始化与 UB Tensor 获取按接口能力适配，执行阶段始终沿用原有 `operator()` 调用。

## 初始化适配

新接口先将 `problemShape` 写入 `Params`，再调用单参数 `Init`；旧接口调用 `Init(params, problemShape)`，不访问旧 `Params` 中不存在的字段。调用方在局部 `Params` 对象上补齐字段时，必须保证该对象的生命周期覆盖 Epilogue 的后续使用。
