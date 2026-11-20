# BlockMmad Matmul IterBatch
> [代码位置](../../../../include/blaze/gemm/block/block_mmad_matmul_iterbatch.h)

## 功能说明
IterBatch 矩阵乘 Block，多个 batch 同时驻留 L1 并在 L0 流水线中流水处理。支持 GM→L1 跨组预取流水（下一组搬运与当前组 MMAD 重叠）、L1→L0 分块 MMAD、L0C→GM/UB 多帧 fixpipe，以及 bias 广播至所有 batch。适用于批量矩阵乘场景。

**特化自**：[block_mmad.md](./block_mmad.md) 公共模板，按 `MatmulIterBatch<FixpOpt>` 调度策略特化。

## 调度策略

```cpp
template <MatMulL0C2Out FixpOpt_ = MatMulL0C2Out::ON_THE_FLY, uint64_t FusedOpType_ = 0,
          uint64_t NonContiguousType_ = 0>
struct MatmulIterBatch {
    using ScheduleType = KernelIterBatch;
    static constexpr MatMulL0C2Out FIXP_OPT = FixpOpt_;
    static constexpr uint64_t FUSED_OP_TYPE = FusedOpType_;          // 预留，暂无消费者
    static constexpr uint64_t NON_CONTIGUOUS_TYPE = NonContiguousType_;  // kernel 构造 GM 非连续布局
};
```

### FixpOpt 输出模式
| 值 | 含义 | 输出行为 |
|----|------|---------|
| `ON_THE_FLY` (0) | 直销 GM | L0C→GM 多帧 fixpipe（单条 `asc::te::copy_l0c_to_gm`，多帧 tensor） |
| `ND_FIXPIPE_1_2` (2) | Fixpipe | L0C→UB 多帧 fixpipe（`set_register` 多帧 + 单条 `copy_l0c_to_ub`，`sub_block_id` 按 tile 奇偶路由）+ AIC/AIV `Sync` MODE_4 握手，AIV 侧经 [BlockEpilogueIterbatch](../../epilogue/block/block_epilogue_iterbatch.md) ND 写回 |

## 数据流

```
GM(A/B, iterBatchL1 组) ──(MTE2, 组间预取重叠)──→ L1 双半区 [bias|A|B]
L1(A: NZ/ZN, B: NZ/ZN) ──(单条多帧 load_data, MTE1)──→ L0A/L0B ──→ Mmad ──→ L0C（batch 维多帧）
L0C ──(ON_THE_FLY: 多帧 fixpipe→GM / ND_FIXPIPE_1_2: 多帧 fixpipe→UB)──→ 输出
```

## 特殊说明

- **L1→L0 批量拷贝**：batch 轴折叠为单条多帧 load_data（`copy_l1_to_l0a/l0b`，`batchCnt == 1` 退化为单帧），无逐 batch 回退分支。折叠拷贝要求源数据在 L1 中连续：多 batch 进 L0（`iterBatchL0 > 1`）时 base 块须覆盖全对齐 m/n/k；M/N/K 需要分块时 `iterBatchL0` 必须为 1。此约束由 host tiling 保证。
- **L0C 多帧**：`iterBatchL0` 个 batch 的结果以 NZ(batch, m, n) 帧布局驻留 L0C，一次 fixpipe 全部搬出。
- **非连续布局**：MBK A、NBK/KBN B 由 kernel 描述 GM stride，block 通过现有多 batch copy 搬入连续的 L1 帧。
