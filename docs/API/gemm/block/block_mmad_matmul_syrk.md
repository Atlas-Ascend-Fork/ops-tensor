# BlockMmad Syrk
> [代码位置](../../../../include/blaze/gemm/block/block_mmad_matmul_syrk.h)

## 功能说明
对称秩 k 更新（syrk）的 BlockMmad 特化：`C[i, j] = A[i-rows] @ A[j-rows]^T`，仅处理上三角槽位。

利用分形对偶性实现单次搬运：**NZ(X)(m, k) 与 ZN(X^T)(k, m) 字节级相等**。每个 A 行块每 k-chunk 仅一次 `nd2nz CopyGM2L1`，同一份 L1 镜像同时喂两个 cube 输入：

- L0A ← 行块 I 的 NZ 视图（`CopyL12L0A`）
- L0B ← 行块 J 的 ZN 视图（`CopyL12L0B`）
- 对角块（i == j）一次搬运同时喂 L0A 与 L0B

镜像 tile (j, i) **不在本 Block 内计算**：它等于 (i, j)^T，由 kernel 对已算出的 (i, j) 结果做转置写出（见 [kernel_matmul_syrk.md](../kernel/kernel_matmul_syrk.md)），因此每个槽位对只需**单条 Mmad 链**（cube 计算量减半，nB(nB+1)/2）。

使用 `BufferManager` 统一管理 L1、L0、L0C 缓冲和事件同步。基于 Tensor API 实现 L0/L1 数据搬运和 Mmad 计算。

**特化自**：[block_mmad.md](./block_mmad.md) 公共模板，按 `MatmulSyrk` 调度策略特化。

## 调度策略

```cpp
using DispatchPolicy = MatmulSyrk;
```

`ScheduleType = KernelMmadSyrk`，由 `GemmUniversal` 的 syrk 专化（上三角遍历 + 转置写出）装配。

## Kernel 侧使用契约（host tiling 强制）

- `mL1 <= min(baseM, baseN)` 且 `nL1 <= min(baseM, baseN)`：镜像 tile 交换 M/N 量纲，两者须同时满足 epilogue 的单 N-chunk 规则与 baseM 行 clamp。
- `baseM * baseN * sizeof(float) <= L0C_SIZE / 2`：单累加器位于一个 L0C 槽（槽粒度即 L0C/2）。
- `mTailCnt == nTailCnt == 1`：不允许对 pair tile 做按核尾块切分。
- `ubDB == 1`（kernel 强制）：累加器 fixpipe 落在 UB 偏移 0，由 AIC/AIV ready/free 握手串行化。

## 数据流

```
                     AIC
GM(A) ──(nd2nz, 每行块每k-chunk一次)──→ L1 镜像 ──NZ视图──→ L0A ──┐
                                          │                      Mmad ──→ L0C ──→ UB(偏移0)
                                          └──ZN视图──→ L0B ──────┘        (fixpipe)
                                                                AIC/AIV 握手 4/6 (+16 splitM)
```

### L1 缓冲区布局

```
L1 slot[i]  ├─ A L1[i]（行块 I 的 NZ 镜像） ├─ B L1[i]（行块 J 的 NZ 镜像，对角块复用 A L1[i]）
```

对角槽位时 `CopyL1FromGM` 只执行一次，I/J 两个 L0 视图取自同一 slot；非对角槽位两次 nd2nz 搬运，GM→L1 总量 = nB²（下界，相比通用装配减半）。

## 与通用装配对比

| 特性 | BlockMmadSyrk | 通用装配（FixpipeOpti + ScaleAdd） |
|------|---------------|-----------------------------------|
| B 来源 | A 自身存储的 ZN 视图（分形对偶） | DNExt 列主序 A^T 视图（dn2zn） |
| GM→L1 次数 | 每行块每 k-chunk 1 次 | 每行块每 k-chunk 2 次（A、B 各一） |
| cube 计算量 | nB(nB+1)/2（单链 + 转置写出） | nB² |
| 输出 | 上三角槽位 + 镜像转置写出 | 全网格 |
