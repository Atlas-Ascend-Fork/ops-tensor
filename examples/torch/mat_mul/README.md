# torch/mat_mul — 通过 PyTorch 接口直接调用 Blaze MatMul kernel

本示例在纯 C++（`examples/acl/`）样例之外提供一条 **torch 路径**：将 Blaze MatMul kernel
注册为 torch 自定义算子，在 Python 侧直接调用并跑到 NPU 上的 Blaze kernel。

覆盖 `examples/acl/mat_mul/` 下的四个 C++ 变体：

| torch 算子 | 对应 C++ 样例 | 说明 |
| --- | --- | --- |
| `blaze_mat_mul` | `acl/mat_mul/mat_mul_basic` | 基础 multi-block tiling |
| `blaze_mat_mul_a_fullload` | `acl/mat_mul/mat_mul_a_fullload` | A 全量加载进 L1，适合小 M（m ≤ 256） |
| `blaze_mat_mul_b_fullload` | `acl/mat_mul/mat_mul_b_fullload` | B 全量加载进 L1，适合小 N（n ≤ 256） |
| `blaze_mat_mul_streamk` | `acl/mat_mul/mat_mul_streamk` | StreamK：M*N tile 打不满核时沿 K 切分 |

## 结构

每个变体一个子目录（与 `acl/mat_mul/` 组织方式一致），各含一对源文件与自身的
CMakeLists（通过 `target_sources` 挂到顶层模块）：

```
torch/mat_mul/
├── CMakeLists.txt                        # 独立构建（torch 为可选依赖，未挂入 examples/ 总构建），
│                                         # 公共配置 + add_subdirectory 四个变体
├── test_mat_mul.py                       # 端到端测试：对比 torch.matmul 参考结果
├── mat_mul_torch_common.h                # 四个 binding 共享的校验、输出分配、取流与 dtype 分发
├── README.md
├── mat_mul_basic/                        # 基础 multi-block tiling
│   ├── CMakeLists.txt
│   ├── mat_mul_basic_torch_kernel.cpp    # Blaze kernel + extern "C" launcher（ASC 编译）
│   └── mat_mul_basic_torch_binding.cpp   # torch 算子注册（g++ 编译）
├── mat_mul_a_fullload/                   # A 全量加载进 L1（m ≤ 256）
│   ├── CMakeLists.txt
│   ├── mat_mul_a_fullload_torch_kernel.cpp
│   └── mat_mul_a_fullload_torch_binding.cpp
├── mat_mul_b_fullload/                   # B 全量加载进 L1（n ≤ 256）
│   ├── CMakeLists.txt
│   ├── mat_mul_b_fullload_torch_kernel.cpp
│   └── mat_mul_b_fullload_torch_binding.cpp
└── mat_mul_streamk/                      # StreamK 沿 K 切分（kernel 需要 GM workspace，
    ├── CMakeLists.txt                    #   workspace 在绑定侧按最大 L0 tile 分配）
    ├── mat_mul_streamk_torch_kernel.cpp
    └── mat_mul_streamk_torch_binding.cpp
```

每个变体一对源文件：`*_torch_kernel.cpp` 由 ASC（bisheng/ccec）编译，含设备 kernel 与
`<<<blockNum, nullptr, stream>>>` 启动 stub；`*_torch_binding.cpp` 由 g++ 编译，用
`TORCH_LIBRARY_FRAGMENT` / `TORCH_LIBRARY_IMPL(PrivateUse1)` 把算子注册到
`ops_tensor_examples` 命名空间。四个 binding 共享 `mat_mul_torch_common.h` 中的输入校验、
输出分配、取流与 dtype 分发，各自只保留变体特有的 shape 约束（fullload 上限）与资源分配
（StreamK workspace）。所有变体最终链接成同一个 `ops_tensor_examples.so`。

调用链路（以 basic 为例）：

```
python: torch.ops.ops_tensor_examples.blaze_mat_mul(a, b)
        └─ TORCH_LIBRARY_IMPL (PrivateUse1)   [mat_mul_basic/mat_mul_basic_torch_binding.cpp, g++]
           └─ extern "C" launch_mat_mul_*()    [mat_mul_basic/mat_mul_basic_torch_kernel.cpp, ccec host pass]
              └─ <<<blockNum, nullptr, stream>>>  Blaze GemmUniversal kernel
```

## 算子定义

四个算子签名一致：

```
blaze_mat_mul[_a_fullload|_b_fullload|_streamk](Tensor a, Tensor b) -> Tensor
```

- `a`: [M, K]，`b`: [K, N]，输出 [M, N]
- dtype：float16 / bfloat16 / float32（两个输入须一致）
- 布局：ND x ND -> ND（不支持转置与 bias；转置/NZ 布局可参考 `acl/mat_mul/` 各 C++ 样例扩展）
- 形状约束：a_fullload 要求 m ≤ 256（A 常驻 L1）；b_fullload 要求 n ≤ 256（B 常驻 L1）
- 零尺寸输入不下发 kernel：m=0 或 n=0 直接返回空输出；k=0 为空累加，返回全零输出

## 构建与运行

```bash
source <cann>/set_env.sh          # 需要 CANN 环境与已安装的 torch、torch_npu
cd examples/torch/mat_mul
cmake -S . -B build
cmake --build build -j
python3 test_mat_mul.py                               # 跑全部四个变体（默认形状）
python3 test_mat_mul.py blaze_mat_mul_streamk         # 只跑 streamk 默认形状
python3 test_mat_mul.py blaze_mat_mul 512 512 512     # 指定变体与形状
```

构建产物 `build/ops_tensor_examples.so`，python 侧通过 `torch.ops.load_library`
加载（见 `test_mat_mul.py`）。

## 说明

- 各变体的 kernel 与 tiling 策略分别沿用 `acl/mat_mul/` 下对应 C++ 样例；`blockNum` 由
  `PlatformAscendCManager::GetCoreNumAic()` 查询。
- basic 变体 fp32 的 L0 tile 取 128x128x64（而非 mat_mul_basic 的 128x256x64）：baseN=256 时
  L0B 双缓冲需要 baseK*baseN*4B*2=128KB，超出 L0B 容量会在 N>128 的 shape 上崩溃，
  这是 mat_mul_basic fp32 配置的已知限制，本示例主动规避。
- StreamK 变体需要 GM workspace 做跨核部分和累加，绑定侧按每核最大 L0 tile
  （256x256 fp32）+ 20MB RPC padding 以 torch byte tensor 分配。
- StreamK 是 MIX（AIC+AIV 1:2）kernel：n % 16 != 0 且满足规模条件时走
  ND_FIXPIPE_1_2 fixpipe 搬出路径，否则走 ON_THE_FLY。
- 流使用输入张量所在卡的 torch_npu 当前 stream（`c10_npu::OptionalNPUGuard` 切换设备 +
  `getCurrentNPUStream(device_index)`，多卡安全），与 torch 侧的 NPU 任务在同一流上保序。
- 混合编译：kernel 源文件由 ASC 编译器编译（设备代码 + host 启动 stub），绑定
  源文件由 g++ 编译（torch 头文件），最终由 CXX 链接器产出单个 `.so`。
