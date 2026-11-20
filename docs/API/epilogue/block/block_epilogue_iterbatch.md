# BlockEpilogue Iterbatch
> [代码位置](../../../../include/blaze/epilogue/block/block_epilogue_iterbatch.h)

## 功能说明
IterBatch ND_FIXPIPE_1_2 模式的 AIV 侧后处理 Block。从 AIC 经多帧 fixpipe 写入 UB 的批量矩阵乘结果中，按奇偶门控消费路由到自身 UB 窗口的 tile，按ND格式输出到GM（自动跳过行尾padding）。BMN保持整tile批量搬出；MBN按batch帧搬出，复用相同同步协议。可选`innerBatch`指定MBN输出组内的batch数（未拆分时为B，拆分时为B/S），默认1保持原BMN行为。

## 同步协议

MODE_4 flag id 空间按 sub-block 划分（`FLAG_ID_MAX = 16`，框架保留 12/13）。`slot = l0CEventId & 0x1` 为 tile 奇偶（即 fixpipe `sub_block_id` 路由目标）：

| 方向 | 机制 | 发起侧 flag id | 对端表现 |
|------|------|---------------|---------|
| AIC→AIV（UB 数据就绪） | `Sync::NotifyVector<MODE_4, PIPE_FIX>` | `slot * FLAG_ID_MAX`（0/16） | 目标 AIV sub-block 本地 flag 0 |
| AIV→AIC（槽位释放） | `Sync::NotifyCube<MODE_4, PIPE_MTE3>` | `SYNC_OFFSET`（2） | AIC 侧 `SYNC_OFFSET + slot * FLAG_ID_MAX`（2/18） |

- AIC 在覆写同奇偶 sub-block 的 UB 窗口前 `Sync::WaitForVector` 该 sub-block 的释放 flag（首个 tile 免等，`l0cEventId > 1` 守卫）；fixpipe 完成后 `NotifyVector` 就绪 flag。
- AIV 在拷出后仅当 `l0CEventId + SYNC_OFFSET < totalTiles` 才 `Set` 释放 flag（每个 sub-block 全 launch 的最后一个 tile 抑制释放）；两个 sub-block 的 `l0CEventId` 计数器保持锁步，set/wait flag 次数精确配对（`totalTiles` 由 kernel 以相同的 tile 遍历规则预计算）。

## 约束

- 双 sub-block 奇偶门控：每个 AIV sub-block 仅消费路由到自身 UB 窗口的 tile（`l0CEventId & 0x1 == subBlockIdx`），窗口单 tile 复用（fixpipe 目的偏移 0，整窗一次持有一个 tile）。
- 槽内布局：帧距 `curM × alignN` 元素（紧凑帧——`alignM` 行帧的 padding 行被下一帧覆盖，各帧有效 `curM` 行密集排布），行宽 `alignN`（含 16 对齐填充列，搬出时自动跳过填充列，有效列宽为 `n`）。
- 仅由 AIC 侧 `BlockMmadIterbatch`（ND_FIXPIPE_1_2）驱动，不独立使用。
