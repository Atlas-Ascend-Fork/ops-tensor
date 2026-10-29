# Kernel Matmul IterBatch
> [代码位置](../../../../include/blaze/gemm/kernel/kernel_matmul_iterbatch.h)

## 功能说明
IterBatch MatMul Kernel，支持 ON_THE_FLY（仅 AIC）和 ND_FIXPIPE_1_2（AIC+AIV）两种输出模式。集成 BlockSchedulerMatmulIterBatch 调度、BlockMmad 矩阵乘和 BlockEpilogueEmpty / BlockEpilogueIterbatch 后处理组件，适用于批量矩阵乘场景。

**继承自**：GemmUniversal 基础模板（按 `MatmulIterBatch<FixpOpt>` 调度策略特化）

## 关联组件

| 组件 | 路径 |
|------|------|
| BlockMmad | [block_mmad_matmul_iterbatch.h](../../../../include/blaze/gemm/block/block_mmad_matmul_iterbatch.h)（[文档](../block/block_mmad_matmul_iterbatch.md)） |
| BlockScheduler | [block_scheduler_matmul_iterbatch.h](../../../../include/blaze/gemm/block/block_scheduler_matmul_iterbatch.h)（[文档](../block/block_scheduler_matmul_iterbatch.md)） |
| BlockEpilogue | [block_epilogue_iterbatch.h](../../../../include/blaze/epilogue/block/block_epilogue_iterbatch.h)（[文档](../../epilogue/block/block_epilogue_iterbatch.md)，仅 ND_FIXPIPE_1_2） |
| 示例 | `examples/acl/batch_mat_mul/mat_mul_iterbatch/` |

## Params 参数结构

| 字段 | 类型 | 说明 |
|------|------|------|
| problemShape | ProblemShape | 问题规模 `(m, n, k, batch)`，A/B/C 共用 batch 维 |
| mmadParams | BlockMmadParams | GM 地址（a/b/c/bias，bias 为 nullptr 表示无 bias）与 m/n/k、baseM/baseN/baseK、iterBatchL1/iterBatchL0 |
| epilogueParams | BlockEpilogueParams | 仅 ND_FIXPIPE_1_2 需要设置 `cGmAddr`（ON_THE_FLY 使用 BlockEpilogueEmpty，无需设置） |
| schedulerParams | BlockSchedulerParams | 见 [BlockSchedulerMatmulIterBatch Params](../block/block_scheduler_matmul_iterbatch.md#params-参数结构)，含 isHf32/l2CacheDisable |

## 支持范围

- 数据类型：float16, bfloat16, float32（FP32 可选 HF32 模式，`schedulerParams.isHf32`）
- A/B/C batch 维一致；bias 广播至所有 batch
- 输出模式：ON_THE_FLY（AIC_ONLY）/ ND_FIXPIPE_1_2（MIX_AIC_1_2）
- 多核调度：`tileNum = CeilDiv(batch, iterBatchL1)`，每个核从自身 `blockIdx` 起以 `blockNum` 为步长处理 tile，尾组 batch 为余数
- B 非连续 innerBatch 暂不支持
