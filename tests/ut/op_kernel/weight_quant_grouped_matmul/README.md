# Grouped Matmul 权重反量化 UT

## 测试内容

本目录使用 gtest 和仓库的 Kernel 测试框架，检查分组矩阵乘的任务分配与 Tensor 布局。目标为 Ascend 960（`ascend960DT_968B7`），通过 [CMakeLists.txt](./CMakeLists.txt) 的 `AddOpTestCase` 注册。

M 表示输入/输出的行数，N 表示输出通道数，K 表示矩阵乘累加维度，E 表示专家数。每个矩阵计算核 AIC 配对一个向量计算核 AIV，两者需要取得相同的输出块序列。

权重方向使用 NN/NT：NN 表示 x、weight 均不转置，权重连续存储为 `[E,K,N]`；NT 表示 x 不转置、weight 转置，权重连续存储为 `[E,N,K]`。两者的逻辑计算维度均为 `[M,K]×[K,N]`。

## 用例组织

### 调度测试

[test_a16w8_scheduler.cpp](./test_a16w8_scheduler.cpp) 调用实际的 `BlockSchedulerGmmSwatWithTailSplit`，使用 8 对 AIC/AIV，分别检查有 bias 和无 bias 的基本块配置。

| 配置 | 输入 |
| --- | --- |
| 每专家行数 | `[0,1,129,0,255,1025]`，含空专家和非均匀分组 |
| N、K | N=273，K=1041 |
| M 基本块 | 无 bias 为 256，有 bias 为 240 |
| N 基本块 | 256 |

测试 Kernel 将每个核、每个专家的任务记录为 `(行起点,列起点,行数,列数)`。Host 读取记录后检查：

- 每个专家、每个物理核的任务数量符合独立预期，配对的 AIC/AIV 任务数量、顺序及块尺寸一致。
- 空专家没有任务；有效块全部位于该专家的输出范围内。
- 每个有效输出元素恰好被一个 AIC 任务覆盖。
- N 方向的块起点按 16 元素对齐，跨专家继续分配任务。

每核、每专家的记录容量按两组配置的最大原始块数推导：`ceil(1025/240) × ceil(273/256) = 10`，基础轮数上界为 `ceil(10/8) = 2`。每组先将尾块拆分系数复位为 `(1,1)`，再执行一次自动拆分；自动拆分最多为每核增加一轮，因此记录容量为 `2+1=3`。

各组的精确任务数如下，数组按物理核 0–7 排列，忽略行数或列数为零的拆分块：

| 专家编号 | 行数 | 无 bias | 有 bias |
| --- | --- | --- | --- |
| 0 | 0 | [0,0,0,0,0,0,0,0] | [0,0,0,0,0,0,0,0] |
| 1 | 1 | [1,1,1,1,0,0,0,0] | [1,1,1,1,0,0,0,0] |
| 2 | 129 | [0,0,0,0,1,1,1,1] | [0,0,0,0,1,1,1,1] |
| 3 | 0 | [0,0,0,0,0,0,0,0] | [0,0,0,0,0,0,0,0] |
| 4 | 255 | [1,1,1,1,1,1,1,1] | [1,1,1,1,1,1,1,1] |
| 5 | 1025 | [2,2,2,2,1,1,1,1] | [2,2,2,2,2,2,2,2] |

测试 Kernel 只在容量内写记录，但计数始终累加；Host 在读取记录前用致命断言检查容量及配对核的数量一致性。`SCOPED_TRACE` 提供 M 基本块大小、专家编号和物理核编号，便于定位数量或覆盖范围不一致的位置。

### Tensor 视图测试

[test_a16w8_tensor_views.cpp](./test_a16w8_tensor_views.cpp) 构造实际 Tensor layout 和 slice，使用独立的地址公式检查坐标映射。

| 用例 | 检查内容 |
| --- | --- |
| BatchedNdNnAndNtTails | E=4、K=529、N=273 时，专家维和 K/N 尾块的地址映射；两种方向的 L1 布局 |
| PaddedVfPitchDoesNotShrinkOnTail | Vector 输出尾块保持原有分组跨度，避免尾块改变物理行距 |
| TwoOrientationSpecificVfTilesCoverBl1WithoutOverlap | 两次 Vector 计算的结果拼入一份 L1 权重块，覆盖完整有效区域且不重叠 |
| InputStageSlicesRetainFullRowPitch | 从 256×256 输入缓冲区切出第二个子块后，仍使用父 Tensor 的连续维度跨度 |

Vector 输出以 16 个 FP16/BF16 元素为一个基本组，称为 C0。NN 按 N 的 C0 分组，NT 按 K 的 C0 分组；每组物理跨度为 `129×16` 个元素。测试同时检查有效数据范围与搬运所需的 C0 补齐范围。

`TwoOrientationSpecificVfTilesCoverBl1WithoutOverlap` 使用分层 `SCOPED_TRACE` 记录权重方向、`kL1/n` 和当前子块的 `kStart/nStart/tileK/tileN`。断言失败时，gtest 将这些上下文与失败位置一起输出，可据此定位两次 Vector 计算中的具体子块。

UB 是 Vector 使用的片上缓冲区，BL1 是 L1 中保存转换后权重的缓冲区。一个完整 256×256 BL1 块由两次 Vector 函数计算填充：NN 沿 K 分成两个 128×256 子块，NT 沿 N 分成两个 256×128 子块。尾块测试覆盖 1、15、16、17、63、64、65、127、128、129、255、256 等相关边界。

### 前处理配置与参数视图测试

[test_weight_antiquant_config.cpp](./test_weight_antiquant_config.cpp) 通过 `kernel_universal.h` 聚合入口实例化生产代码的 `GroupedWeightAntiquantConfig`，检查两种输入类型、四种 weight 类型、NN/NT 和合法 offset 组合，共 20 组前处理配置。bias 不参与配置选择。

用例检查逐通道 Policy 的单 AIV 约束与核间同步配置、输入/输出槽字节数、GM 权重坐标映射、输出尾块的 C0 补齐和固定 pitch，以及跨专家、跨 K 段的参数子视图。UB 总预算由 UT 按 `Config::OUTPUT_BASE + Policy::UB_MTE2_BUFFER_NUM * Config::OUTPUT_SLOT_BYTES` 计算，先核对 265,216 B（259 KiB）的布局预算，再通过 `AscendC::GetUBSizeInBytes()` 取得 UT 编译目标的 SDK 可用 UB 容量，检查容量非零且总占用不超过该容量。该接口在 CPU UT 中采用目标平台定义，不查询已连接的设备；容量不可用时测试失败，不回退到手写常量。生产配置保留各区域大小和布局约束。参数视图保持 `[E,N]` 的逐通道含义。

## 构建与运行

准备目标 SoC 的 CANN 和 CPU-debug 运行库后，在仓库根目录执行：

```bash
bash build.sh --opkernel -u \
  --soc=ascend960 --ops=weight_quant_grouped_matmul
```

`--ops` 对应本目录名称。调度用例使用 `blaze_kernel_stub.h` 和 `KERNEL_RUN_KF` 启动测试 Kernel；视图用例直接比较 Tensor 的形状和坐标映射。构建入口及目标 SoC 映射见 [Kernel UT CMake](../CMakeLists.txt)。

## 配套数值测试

完整权重反量化和矩阵乘的输入生成、设备调用及输出比较由 [grouped_matmul_weight_antiquant 示例](../../../../examples/acl/grouped_matmul/grouped_matmul_weight_antiquant/README.md) 提供。该示例有 50 组类型、bias、offset 和权重方向配置，并附带 Python 数据生成器与校验器的单元测试。

接口、计算公式和缓冲区说明见 [Kernel 文档](../../../../docs/API/gemm/kernel/kernel_wqgmm_mix_antiquant.md)；AIC 的矩阵乘流程和 bias 处理见 [BlockMmad 文档](../../../../docs/API/gemm/block/block_mmad_wqmm_mix_weight_prologue.md)。
