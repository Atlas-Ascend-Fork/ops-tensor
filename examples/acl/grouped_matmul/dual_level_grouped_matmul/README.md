# DualLevelGroupedMatmul 组件示例

Ascend 950 上组合原 GMM MX Policy、BlockMmad、Epilogue 和 Scheduler。AIC 保留完整 K 流水，AIV 消费 FP32 partial 并执行 level0 缩放及归约；ScaleTile 保留在 epilogue 内部。

```bash
source /usr/local/Ascend/cann/set_env.sh
cmake -S examples -B build_examples -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" \
  -DCMAKE_PREFIX_PATH="$ASCEND_HOME_PATH/x86_64-linux/asc/lib/cmake"
cmake --build build_examples --target dual_level_grouped_matmul -j2
```

运行 `dual_level_grouped_matmul CASE_DIR`。目录包含 x/a0/a1/weight/b0/b1/groups 的二进制文件、可选 bias.bin 以及 112-byte tiling.bin，输出 output.bin。协议由本例源码的 Tiling 结构定义；请求核数须在实际设备可用范围内。

使用稀疏分组列表时，非零行组在前，零行组在后，两类分组内部均按专家编号递增排列。

组件接口见 [Epilogue 文档](../../../../docs/API/epilogue/block/block_epilogue_dual_level_scale.md)。算子精度统一使用 ops-transformer 的 ACLNN/E2E ST 和随算子交付的 CPU reference。
