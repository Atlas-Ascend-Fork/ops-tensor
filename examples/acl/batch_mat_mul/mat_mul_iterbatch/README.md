# mat_mul_iterbatch Example

## 概述

本示例演示基于 Blaze 框架的 IterBatch MatMul 载入模版在昇腾 NPU 上的实现。该模版将多个 batch 同时驻留 L1 并在 L0 流水线中流水处理：每核每主循环处理 `iterBatchL1` 个 batch，kernel 将下一组 GM→L1 预取与当前组 MMAD 重叠执行；输出支持 ON_THE_FLY（L0C→GM 直接 fixpipe）与 ND_FIXPIPE_1_2（L0C→UB 多帧 fixpipe + AIV 后处理 ND 写回）两种输出方式。适用于 A/B/C batch 维度完全一致的批量矩阵乘场景。

- **算子**: mat_mul
- **场景**: mat_mul_iterbatch
- **算法特点**: iterbatch L1/L0 流水线 + 跨 tile L1 预取 + bias 广播；支持 FP16/BF16/FP32
- **参考实现**: 基于 Blaze 框架 `blaze/gemm/kernel/kernel_matmul_iterbatch.h`

## 支持架构

| 架构     | SoC       | 支持状态 |
| -------- | --------- | -------- |
| dav-3510 | Ascend950 | ✅       |

## 使用约束

- 输入 A shape: `[batch, M, K]`（transA=false）或 `[batch, K, M]`（transA=true）
- 输入 B shape: `[batch, K, N]`（transB=false）或 `[batch, N, K]`（transB=true）
- 输出 C shape: `[batch, M, N]`
- 数据类型: float16, bfloat16, float32
- batch 约束: A/B/C 的 batch 维度必须完全一致（无广播；广播语义请使用 `mat_mul_iterbatch_broadcast`）
- bias: 1D 向量，大小必须等于 n 或 0（无 bias）
- L1 容量约束: `iterBatchL1` 组的 A/B/bias 总量不超过 L1 单缓冲区（host tiling 自动推导）

## 输出模式

| mode | 名称           | 任务类型   | 数据通路                                                     |
| ---- | -------------- | ---------- | ------------------------------------------------------------ |
| 0    | ON_THE_FLY     | AIC_ONLY   | MMAD 后逐 tile 多帧 fixpipe L0C→GM（Tensor API 批量拷贝）     |
| 2    | ND_FIXPIPE_1_2 | MIX_AIC_1_2 | AIC 多帧 fixpipe L0C→UB，AIV 经 CrossCore 握手消费 UB 并 ND 写回 GM |

## IterBatch 流水线说明

```
标准 BMM:   逐 batch 串行处理，每个 batch 独立完成 L1 加载 → L0 计算
IterBatch:  多个 batch 同时驻留 L1，在 L0 流水线中交错计算
Preload:    组 i 计算期间（MTE1/M/FIX），组 i+1 的 GM→L1（MTE2）预取
            至另一 L1 半区，隐藏 GM→L1 延迟
```

**iterbatch 参数**：

| 参数        | 层级      | 含义                                          |
| ----------- | --------- | --------------------------------------------- |
| iterBatchL1 | L1 缓存   | 同时驻留 L1 的 batch 数，控制 L1 数据复用深度 |
| iterBatchL0 | L0 流水线 | L0 流水线并行 batch 数，控制计算重叠深度      |

**batch 调度**（与源 iterbatch 模版一致）：

| 循环   | 每 core batch 数                                                  |
| ------ | ----------------------------------------------------------------- |
| 主循环 | iterBatchL1                                                       |
| 尾循环 | 每个核从自身 blockIdx 起以 blockNum 为步长处理 tile，尾组 batch 数为余数 |

## CSV 驱动测试

### 执行方式

通过统一入口驱动，自动完成编译、数据生成、kernel 执行和精度验证：

```bash
bash examples/acl/common/run.sh --ops=batch_mat_mul --target=mat_mul_iterbatch
```

### 测试用例定义

测试用例定义在 `mat_mul_iterbatch.csv` 中，格式如下：

```csv
casename,m,k,n,batch,batchA,batchB,mode,bias,dtype,transA,transB,hf32
iterbatch_fp16_otf,64,64,160,64,64,64,0,0,float16,false,false,false
iterbatch_fp16_mix,64,64,160,64,64,64,2,0,float16,false,false,false
...
```

## 支持范围

| 维度         | 支持范围                                                     |
| ------------ | ------------------------------------------------------------ |
| 数据类型     | float16, bfloat16, float32                                   |
| 转置         | transA/transB 任意组合                                       |
| 输出方式     | ON_THE_FLY（mode=0）、ND_FIXPIPE_1_2（mode=2）                |
| batch        | A/B/C batch 维度一致，batch ≥ 1                              |
| bias         | bias 向量大小，必须等于 n 或 0（无 bias）                     |
| 尾块         | M/K/N 非 16 对齐由 fixpipe/搬运边界处理                       |
| 非连续 innerBatch | 暂不支持（B 非连续场景待扩展）                           |

## 组件构成

| 组件            | 路径                                                        | 说明                                  |
| --------------- | ----------------------------------------------------------- | ------------------------------------- |
| Kernel          | `blaze/gemm/kernel/kernel_matmul_iterbatch.h`   | 完整 kernel 入口（AIC 流水线 + AIV 后处理调度） |
| Block MMAD      | `blaze/gemm/block/block_mmad_matmul_iterbatch.h`             | Block 级矩阵乘（标志位轮次预取流水） |
| Block Scheduler | `blaze/gemm/block/block_scheduler_matmul_iterbatch.h`        | batch 分组调度器              |
| Epilogue        | `blaze/epilogue/block/block_epilogue_iterbatch.h`     | AIV 侧 UB 消费与 ND 写回（MIX 模式）   |
| Dispatch Policy | `blaze/gemm/policy/dispatch_policy.h`                       | 派发策略（`MatmulIterBatch<FixpOpt>`） |
| Example         | `mat_mul_iterbatch.cpp`                               | host 侧运行器（读 bin/写 bin）         |
| 数据生成        | `../scripts/gen_data.py`                                   | 输入与 CPU golden（与 broadcast 共用）|

## 文件结构

```
mat_mul_iterbatch/
├── mat_mul_iterbatch.cpp         # host 侧运行器（读 bin/写 bin）
├── mat_mul_iterbatch.conf        # 参数路由配置
├── mat_mul_iterbatch.csv         # CSV 测试用例
└── README.md                     # 本文档

../scripts/
└── gen_data.py                    # 数据生成与 CPU golden（与 broadcast 共用）
```
