# BlockMmad Syrk
> [代码位置](../../../../include/blaze/gemm/block/block_mmad_matmul_syrk.h)

## 功能说明
对称秩 k 更新（syrk）的 BlockMmad 特化：`C[i, j] = A[i-rows] @ A[j-rows]^T`，仅处理上三角槽位。

利用分形对偶性实现单次搬运：**NZ(X)(m, k) 与 ZN(X^T)(k, m) 字节级相等**。每个 A 行块每 k-chunk 仅一次 `nd2nz CopyGM2L1`，同一份 L1 镜像同时喂两个 cube 输入：

- L0A ← 行块 I 的 NZ 视图（`CopyL12L0A`）
- L0B ← 行块 J 的 ZN 视图（`CopyL12L0B`）
- 对角块（i == j）一次搬运同时喂 L0A 与 L0B

镜像 tile (j, i) 不在本 Block 内计算：`CopyOutFromL0C2UB` 把单个 L0C 累加器 fixpipe 两次——`nz2nd` 出 (i, j) 的 ND 图像、`nz2dn`（硬件转置）出 (j, i) 的 DN 图像——每个槽位对只消耗**单条 Mmad 链**（cube 计算量减半，nB(nB+1)/2）。

## 调度策略

```cpp
template <uint8_t FillMode_ = SYRK_FILL_MODE_FULL>
struct MatmulSyrk {
    using ScheduleType = KernelMmadSyrk;
    static constexpr uint8_t FILL_MODE = FillMode_;
};
```

`FillMode_` 取 `SYRK_FILL_MODE_FULL`/`UP`/`LOW`（`SyrkFillMode` 枚举）；非法值由特化内 `static_assert` 拒绝（否则会混用 split-M fixpipe 与 full 镜像 epilogue，静默写错）。

**特化自**：[block_mmad.md](./block_mmad.md) 公共模板，按 `MatmulSyrk` 调度策略特化。

## Params

| 字段 | 说明 |
|---|---|
| `aGmAddr` / `workspaceGmAddr` | 唯一 GM 数据源（B 经 ZN L1 视图复用 A 的搬运）；workspace |
| `oriK` | K 维长度 |
| `block` | 对称方块尺寸（`mL1 == nL1 == baseM == baseN`） |
| `kL1` / `kL0` | L1 k-chunk 与 L0 k-step |
| `l1Stages` | L1 缓冲级数（kernel 固定 2） |

`splitM` 不再是 Params 字段：由 `FILL_MODE` 在类内编译期派生（`FILL_MODE != FULL`）。

## 输出路径（CopyOutFromL0C2UB）

- `FILL_MODE == FULL`：`nz2nd` + `nz2dn` 双图像（完整对称输出）。
- `FILL_MODE != FULL`（up/low）：`DUAL_DST_SPLIT_M` 单图像，行对半分给两个 AIV。硬件要求 `m_size` 为偶数，奇数尾块在源（L0C NZ）与目的（UB ND）两侧 layout 各补一行 padding（AIC err 169 的修复），epilogue 按 ceil/floor 分行只读有效行。

## Kernel 侧使用契约（host tiling 强制）

- 块几何坍缩为一个对称方块尺寸：`mL1 == nL1 == baseM == baseN == baseBlock`。
- 每轴单尾块（`mTailCnt == nTailCnt == 1`，不做核间尾块切分）。
- L0C 单累加器槽（无 ping-pong）；UB 容纳 fp32 累加器镜像与两个 AIV 的 staging。
- L1 双缓冲（`SYRK_L1_STAGES = 2`），`kL1` 使一个 stage 的 A 图 + B 图各占 L1 一半。

## 数据流

```
                     AIC
GM(A) ──(nd2nz, 每行块每k-chunk一次)──→ L1 镜像 ──NZ视图──→ L0A ──┐
                                          │                      Mmad ──→ L0C
                                          └──ZN视图──→ L0B ──────┘
                                                          │ fixpipe ×2
                        ┌─────────────────────────────────┴──────────────────┐
                  nz2nd → UB(sub0) (i,j) ND                    nz2dn → UB(sub1) (j,i) DN
                        │ AIC/AIV 握手                                        │
                  [epilogue] 写 C[i,j]                        [epilogue] 写 C[j,i]
```

### L1 缓冲区布局

```
L1 stage[i] ├─ A L1[i]（行块 I 的 NZ 镜像） ├─ B L1[i]（行块 J 的 NZ 镜像，对角块复用 A L1[i]）
```

对角槽位只执行一次 `CopyGM2L1`；非对角槽位两次，GM→L1 总量 = nB²（下界，相比通用装配减半）。

## 与通用装配对比

| 特性 | BlockMmadSyrk | 通用装配（FixpipeOpti + ScaleAdd） |
|------|---------------|-----------------------------------|
| B 来源 | A 自身存储的 ZN 视图（分形对偶） | DNExt 列主序 A^T 视图 |
| GM→L1 次数 | 每行块每 k-chunk 1 次 | 每行块每 k-chunk 2 次（A、B 各一） |
| cube 计算量 | nB(nB+1)/2（单链 + nz2dn 镜像） | nB² |
| 输出 | 上三角槽位 + nz2dn 镜像；`up`/`low` 单图像 split-M | 全网格 |
