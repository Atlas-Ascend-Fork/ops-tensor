# ConfidenceHeadFused Kernel UT

## 运行方式

在仓库根目录加载 CANN 环境后执行：

```bash
bash build.sh --opkernel -u --ops=confidence_head_fused
```

使用 Google Test 与 tikicpulib，注册目标为 ascend950pr_9599。

## 覆盖范围

| 用例组 | 用例数 | 覆盖内容 |
| --- | --- | --- |
| 非空 K | 5 | FP16/BF16、N/K 尾块、双缓冲、多轮 K 加 bias、markov-only、batch bias。 |
| 零 K | 2 | bias/零 logits、温度零值回退和单侧截断。 |
| 空输出 | 1 | N=0 时直接返回，不进入调度和跨核同步。 |
| Sigmoid | 1 | FP32/FP16/BF16 输出实例化。 |

共 9 条用例。Kernel 用例检查启动与同步，保留 batch bias、N 尾块和空输出覆盖。CPU 模拟器不完整模拟 MMAD/Reg 数值，设备精度由 [端到端示例](../../../../examples/acl/confidence_head_fused/confidence_head_fused_mix/README.md) 和 NN ST 校验。
