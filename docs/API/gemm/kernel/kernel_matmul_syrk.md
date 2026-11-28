# Kernel Matmul Syrk
> [代码位置](../../../../include/blaze/gemm/kernel/kernel_matmul_syrk.h)

## 功能说明
对称秩 k 更新 kernel：`C = alpha * (A @ A^T) + beta * C`，配 [BlockMmadSyrk](../block/block_mmad_matmul_syrk.md) 与 `BlockEpilogueFmmWithScaleAdd` 使用。

遍历**紧凑上三角槽位空间**（`BlockSchedulerSyrkTriangular`，stride 轮询，每核 slots/核数 ±1；下三角槽位不占索引）。每个被处理槽位：

1. 驱动一次 `BlockMmadSyrk` 调用：单次 nd2nz GM→L1 搬运产出 L1 双视图（对角块一次搬运同时喂 L0A/L0B），单条 Mmad 链产出 (i, j) 累加器。
2. fixpipe 双发（`BlockMmadSyrk::CopyOutFromL0C2UB`）：
   - `nz2nd` → AIV sub0 的 (i, j) ND 图像；
   - `nz2dn`（硬件转置）→ AIV sub1 的 (j, i) DN 图像。
   两路写回均无 GM 回读、无软件转置。
3. 两个 AIV 并行跑 scale-add epilogue：sub0 写 C[i, j]、sub1 写 C[j, i]。

## fill_mode（DispatchPolicy 模板参数，编译期）

| fill_mode | 行为 |
|---|---|
| `full` | 双图像（nz2nd + nz2dn），完整对称矩阵输出 |
| `up` | 仅 (i, j) ND 图像，`DUAL_DST_SPLIT_M` fixpipe 把行对半分给两个 AIV；对角块按元素粒度三角掩码（`triMask=1`：行 <= 列保留计算值，其余恢复原 C） |
| `low` | kernel 层交换 A/B tile 先算出 (j, i) 转置 tile，再走与 up 相同的 split-M 输出；对角块 `triMask=2`（行 >= 列保留） |

`up`/`low` 固定走 split-M 双 AIV 输出：`SPLIT_M` 由 `FILL_MODE` 编译期派生，不经 Params 传递。

## 三角掩码（triMask）

- `0`：整 tile 写出（非对角块，整块都在目标三角内）。
- `1`/`2`：对角块逐元素比较全局行号与列号（`rowBase` 校准 sub 半幅的行偏移，`vcmps_ge`/`vcmps_le` 生成掩码，`vsel` 选择），掩码外元素写回 staged 原 C，保证 bit-exact。
- `beta == 0` 且 `triMask != 0` 时仍强制加载 x3 staging（掩码恢复需要原值），计算主体用 `skipX3` 的 VF 实例（不读、不乘 C，NaN/Inf 不传播）。

## staging 同步

x3 staging 与输出复用同一 UB 区。每块写前等两个方向（`MTE3→MTE2` 供 x3 load、`MTE3→V` 供向量输出），store 后两个方向各 set 一次；每块每方向严格 set/wait 配对。对角块（load）与 `beta == 0` 非对角块（skip）混合轮转时不产生跨类型窗口。

## Host tiling 契约

见 [BlockMmadSyrk](../block/block_mmad_matmul_syrk.md)：`mL1 == nL1 == baseM == baseN == baseBlock`（对称方块）、每轴单尾块、16 对齐、L1 双缓冲；`baseBlock² × sizeof(float)` 受 L0C 预算约束——`full` 取 L0C/2 上限，`up`/`low` 单图像取全量 L0C 预算。

## 数据流

```
GM(A) ─→ [BlockMmadSyrk: nd2nz(对角1次/非对角2次) → L1镜像 → L0A/L0B → Mmad 单链] ─→ L0C
                                                                                    │ fixpipe ×2
                                          ┌─────────────────────────────────────────┴──────────────┐
                                    nz2nd → UB(sub0) (i,j) ND                        nz2dn → UB(sub1) (j,i) DN
                                          │ AIC/AIV 握手                                          │
                                    [epilogue] 写 C[i,j]                              [epilogue] 写 C[j,i]
```
