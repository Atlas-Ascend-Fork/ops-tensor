# Block Epilogue Per-Token Scale

> [代码位置](../../../../include/blaze/epilogue/block/block_epilogue_per_token_scale.h)

## 功能说明

`BlockEpiloguePerTokenScale` 是 GMM Fixpipe 量化 Kernel 的 AIV 后处理组件。它读取 Fixpipe 结果，应用 per-token scale，并在非对称量化场景加入 row-sum 与 offset 修正，最后转换为目标输出类型。

对输出元素 `y[m,n]`，计算规则为：

```text
offset 为空： y[m,n] = fixpipe[m,n] * perTokenScale[m]

offset 非空： y[m,n] =
    (fixpipe[m,n] + xRowSum[m] * offset[n]) * perTokenScale[m]
```

其中 Fixpipe 结果已经包含权重 scale；本组件不负责 INT4 数据展开、K 轴矩阵乘或 row-sum 归约。

## 模板参数

| 参数 | 说明 |
|------|------|
| `OutType` | 最终输出类型，支持 `float`、`half`、`bfloat16_t` |
| `FixpipeType` | Fixpipe 中间结果类型，支持 `float`、`half`；默认为 `half` |

## Params

| 字段 | 说明 |
|------|------|
| `workspaceGmAddr` | Fixpipe 中间结果 workspace 地址；`operator()` 的 `workspaceOffset` 以 `FixpipeType` 元素数（非字节）为单位在此地址上偏移 |
| `perTokenScaleGmAddr` | 每个 token 的 scale 地址 |
| `offsetGmAddr` | 非对称量化 offset 地址 |
| `xRowSumGmAddr` | 激活行归约结果地址 |
| `outGmAddr` | 最终输出地址 |
| `n` | 完整输出 N |
| `baseM/baseN` | 基本块大小 |
| `withOffset` | 是否执行 row-sum 与 offset 修正 |

## 执行流程

```text
读取 Fixpipe 结果和 per-token scale
    ↓
按需读取 xRowSum 和 offset
    ↓
以 FP32 完成修正和缩放
    ↓
转换为 OutType 并写回输出
```

组件按每个 Cube 核对应的两个 Vector 子核拆分 M 轴任务，每个 Vector 子核只处理自己负责的行。
