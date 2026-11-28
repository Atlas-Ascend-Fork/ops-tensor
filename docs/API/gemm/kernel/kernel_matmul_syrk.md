# Kernel Matmul Syrk
> [代码位置](../../../../include/blaze/gemm/kernel/kernel_matmul_syrk.h)

## 功能说明
对称秩 k 更新 kernel：`C = alpha * (A @ A^T) + beta * C`，配 [BlockMmadSyrk](../block/block_mmad_matmul_syrk.md) 与 `BlockEpilogueFmmWithScaleAdd` 使用。

遍历完整 M×N block 网格但只处理上三角（`coordN >= coordM`），下三角槽位即刻跳过。每个被处理槽位：

1. 驱动一次 `BlockMmadSyrk` 调用：两次共享的 nd2nz GM→L1 搬运（对角块一次）产出 C[i, j]，**每个 pair 单条 Mmad 链**（cube 量 ≈ 1/2）。
2. AIV 跑 scale-add epilogue 写出 (i, j) tile。
3. 非对角槽位调用 `WriteTransposedTile` 将刚写出的结果**转置写出**到镜像位置 C[j, i] = C[i, j]^T（依赖输入 C 对称，syrk 需求已约定）。

### WriteTransposedTile

- 内部 16×16 块：tensor_api 分块跨步 GM→UB 回读（L2 热）→ `asc_transpose`（3510 `vtranspose`，b16 位级 16×16 转置，half/bf16 通用）→ 分块跨步 UB→GM 写出。
- 边角块与 CPU 仿真回退**标量路径**（`ASCENDC_CPU_DEBUG == 1` 时整条走标量）。
- **同步**：`TRANSPOSE_EVENT = 3`（epilogue 占用 0/1/2）。每块链路 `MTE3_MTE2 drain → gather → MTE2_V → vtranspose → V_MTE3 → scatter`；首部的 MTE3→MTE2 等待经 epilogue 的 V→MTE3 同步传递性排空此前全部向量工作。
- **scratch 布局**：两块 512B（16×16 b16）位于 **UB 顶端（TOTAL_UB_SIZE - 1024）**——避开 AIC 并发 fixpipe（下一 tile 累加器受 tiling 契约约束 ≤ L0C/2 < UB 顶）与下一 epilogue staging（AIV 程序序，ready-flag 握手之后才开始）。
- **splitM**：开启时两个 AIV 子块按 rowBlocks 对分转置写出工作。

## Host tiling 契约

见 [BlockMmadSyrk](../block/block_mmad_matmul_syrk.md)：`mL1/nL1 <= min(baseM, baseN)`，`mTailCnt == nTailCnt == 1`，`baseM * baseN * sizeof(float) <= L0C_SIZE / 2`，baseN 16 对齐。

## 数据流

```
GM(A) ─→ [BlockMmadSyrk: nd2nz×2 → L1镜像 → L0A/L0B → Mmad 单链] ─→ L0C ─→ UB(0)
                                                                        │ AIC/AIV 握手
GM(C) ◄─ [epilogue: alpha*acc + beta*C] (i,j)                          ▼
GM(C) ◄─ [WriteTransposedTile: GM→UB → vtranspose → UB→GM] (j,i) = (i,j)^T
```
