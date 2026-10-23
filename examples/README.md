# ops-tensor Examples

## 1. 功能说明

examples 目录提供基于 Blaze header-only GEMM 框架的算子样例程序，用于：

- 验证算子在 NPU 上的编译、执行和精度正确性
- 演示如何使用 Blaze 框架编写 StreamK、WeightNz 等 kernel
- 提供端到端的测试数据生成、执行、精度验证工作流
- 支持多种数据类型（float16/bfloat16/float32）的测试和验证

每个样例包含完整的 host 侧代码（数据读取、ACL 内存管理、kernel 调度）和 device 侧 kernel（Blaze 模板实例化），通过 CSV 测试用例表驱动执行。

## 2. 环境依赖

examples 的完整编译与运行涉及三类依赖：**系统工具链**、**CANN / 昇腾环境**、**Python 依赖**。`common/run.sh` 的 `preflight()` 函数会在执行前自动检查关键依赖（`ASCEND_HOME_PATH`、`bisheng`、`g++`、`python3`、`cmake`），缺失时报错退出。

### 2.1 系统工具链

| 依赖    | 最低版本          | 用途                                                                                | 检查方式              |
| ------- | ----------------- | ----------------------------------------------------------------------------------- | --------------------- |
| cmake   | 3.16              | 构建系统（`examples/CMakeLists.txt` 中 `cmake_minimum_required(VERSION 3.16)`） | `cmake --version`   |
| g++     | 默认标准 ≥ C++14 | Host 侧 C++ 编译器（`project(... LANGUAGES CXX)`）                                | `g++ --version`     |
| bisheng | —                | ASC 语言编译器，编译 device 侧 kernel（`find_package(ASC REQUIRED)`）             | `bisheng --version` |
| make    | —                | 由 cmake 调用的底层构建工具                                                         | `make --version`    |
| python3 | 3.7+              | 驱动 CSV 解析、数据生成、精度验证脚本                                               | `python3 --version` |

> `bisheng` 编译器随 CANN Toolkit 安装，`source set_env.sh` 后自动加入 PATH。

### 2.2 CANN / 昇腾环境

| 依赖         | 要求      | 说明                                    |
| ------------ | --------- | --------------------------------------- |
| CANN Toolkit | ≥ 9.1.0  | 提供 ACL 运行时、bisheng 编译器、头文件 |
| NPU 硬件     | Ascend950 | 当前 examples 仅支持 Ascend950 芯片     |

### 2.3 Python 依赖

数据生成和精度验证脚本依赖以下 Python 包：

| 包        | 用途                                                                | 安装命令                  |
| --------- | ------------------------------------------------------------------- | ------------------------- |
| numpy     | 二进制数据读写、数组操作                                            | `pip install numpy`     |
| ml_dtypes | `float8_e4m3fn` golden 计算                                       | `pip install ml_dtypes` |
| en_dtypes | `hifloat8` golden 计算（quant_batch_matmul_cube 样例）            | `pip install en-dtypes` |
| torch     | CPU golden 参考计算（`torch.matmul`/`torch.addmm`）、dtype 映射 | `pip install torch`     |

### 2.4 依赖检查清单

执行前可运行以下命令快速验证环境：

```bash
# 1. CANN 环境
echo $ASCEND_HOME_PATH && [ -d "$ASCEND_HOME_PATH" ] && echo "OK" || echo "MISSING"

# 2. 编译器
cmake --version | head -1
g++ --version | head -1
bisheng --version 2>&1 | head -1

# 3. Python 依赖
python3 -c "import numpy; print('numpy', numpy.__version__)"
python3 -c "import ml_dtypes; print('ml_dtypes', ml_dtypes.__version__)"
python3 -c "import torch; print('torch', torch.__version__)"

# 4. NPU 设备
npu-smi info | head -20
```

> `common/run.sh` 执行时会自动运行 preflight 检查，缺失关键依赖会报错退出并给出修复提示。

## 3. 目录结构

```
examples/
├── CMakeLists.txt              # 全局 CMake 配置（编译器、链接库、宏定义、add_subdirectory）
├── README.md                   # 本文件
├── common/                     # 公共基础设施
│   ├── run.sh                  #   统一执行入口（编译 + 运行 + 验证）
│   ├── run_case.py             #   通用批跑引擎（CSV 解析 → gen_data → kernel → verify）
│   ├── CONF_FORMAT.md          #   .conf 文件格式规范
│   └── data_utils.h            #   ACL_CHECK 宏、文件读写工具
│
└── {op}/                       # 算子级目录，如 mat_mul、batch_mat_mul、grouped_matmul
    ├── CMakeLists.txt          #   注册本算子下所有样例可执行文件
    ├── scripts/                #   算子级脚本（数据生成 + 精度验证）
    │   ├── gen_data.py         #     输入数据生成 + CPU golden 计算
    │   └── verify_result.py    #     NPU 输出 vs CPU golden 精度比对
    │
    └── {example}/              #   样例级目录，如 mat_mul_basic
        ├── {example}.cpp       #   样例源码
        ├── {example}.conf      #   执行参数配置（INI 格式）
        ├── {example}.csv       #   CSV 测试用例表
        ├── README.md           #   样例说明文档
        ├── input/              #   运行时生成：输入数据 + CPU golden（每条用例覆盖，样例执行后自动清理）
        └── output/             #   运行时生成：NPU 输出 + verify_metrics.json（每条用例覆盖，样例执行后自动清理）
```

**层级关系**：

| 层级 | 目录               | CMakeLists.txt 职责                                                                                  |
| ---- | ------------------ | ---------------------------------------------------------------------------------------------------- |
| L1   | `examples/`      | 编译器发现、链接库配置、`ops_example_add_executable` 宏定义、`add_subdirectory({op})` 加载各算子 |
| L2   | `examples/{op}/` | 通过`ops_example_add_executable(name subdir/source.cpp)` 直接注册本算子下所有样例可执行文件        |

**当前算子目录**：

| 算子目录                           | 样例                                                                                         | 说明                                     |
| ---------------------------------- | -------------------------------------------------------------------------------------------- | ---------------------------------------- |
| `mat_mul/`                       | mat_mul_basic, mat_mul_streamk, mat_mul_a_fullload, mat_mul_b_fullload, mat_mul_fixpipe_opti | 矩阵乘法样例                             |
| `batch_mat_mul/`                 | mat_mul_bmm_broadcast, mat_mul_iterbatch_broadcast, mat_mul_iterbatch                     | 批量矩阵乘法样例                         |
| `fused_mat_mul/` | [fused_mat_mul_sigmoid](fused_mat_mul/fused_mat_mul_sigmoid/README.md) | 矩阵乘与 sigmoid 融合，支持 FP16/BF16、转置及可选 bias |
| `transpose_batch_mat_mul/`       | transpose_batch_mat_mul_basic                                                                | 转置批量矩阵乘法样例                     |
| `quant_batch_matmul/`            | quant_batch_matmul_cube, quant_batch_matmul_mix, quant_batch_matmul_mx | 量化批量矩阵乘法样例 |
| `block_attn_res_prepare/`        | block_attn_res_prepare                                                                        | Attention residual prepare 样例           |
| `weight_quant_batch_matmul_mx/`  | weight_quant_batch_matmul_mx_swat                                                            | 权重量化批量矩阵乘法样例                 |
| `weight_quant_batch_matmul/` | [weight_quant_batch_matmul_antiquant](weight_quant_batch_matmul/weight_quant_batch_matmul_antiquant/README.md) | INT8 权重反量化矩阵乘，FP16/BF16 输入、两种权重存储方向及尾块 |
| `grouped_matmul/`               | quant_grouped_matmul_mx, quant_grouped_matmul_activation_quant_mx, [quant_grouped_matmul_swiglu_quant_mx](grouped_matmul/quant_grouped_matmul_swiglu_quant_mx/README.md), [quant_grouped_matmul_cubeonly](grouped_matmul/quant_grouped_matmul_cubeonly/README.md), [grouped_matmul_mx_a8w4](grouped_matmul/grouped_matmul_mx_a8w4/README.md), [grouped_matmul_no_quant](grouped_matmul/grouped_matmul_no_quant/README.md), [grouped_matmul_finalize_routing_mx](grouped_matmul/grouped_matmul_finalize_routing_mx/README.md) | Grouped MatMul 与量化、激活量化、SwiGLU MX、非量化及 FinalizeRouting MX 样例 |
| `weight_quant_matmul/` | [quant_batch_matmul_pergroup_a8w4](weight_quant_matmul/quant_batch_matmul_pergroup_a8w4/README.md) | T-CG per-group 权重反量化 A8W4 矩阵乘样例 |
| `quant_matmul_activation_quant/` | quant_matmul_activation_quant                                                                | 量化矩阵乘激活融合样例                   |
| `transpose_quant_batch_matmul/`  | tqbmm_mx                                                                                     | 转置量化批量矩阵乘法样例                 |
| `gemm_syrk/`                      | [gemm_syrk_basic](gemm_syrk/gemm_syrk_basic/README.md)                                        | 对称秩k更新（syrk）样例，单次搬运 + 上三角调度，C 原地更新 |

## 4. 执行方法

### 4.1 通过 build.sh 执行

从仓库根目录执行，`build.sh --examples` 内部调用 `common/run.sh`：

```bash
# 运行所有算子的所有样例
./build.sh --examples

# 运行指定算子下的所有样例
./build.sh --examples --ops=mat_mul

# 运行指定样例
./build.sh --examples --ops=mat_mul --target=mat_mul_basic

# 运行多个算子
./build.sh --examples --ops=mat_mul,grouped_matmul

# 运行算子下多个样例
./build.sh --examples --ops=mat_mul --target=mat_mul_basic,mat_mul_streamk
```

### 4.2 通过 common/run.sh 执行

`common/run.sh` 是统一执行入口：

```bash
bash examples/common/run.sh --ops=<names> [--target=<names>] [--case=<path>] [--ti=<N|N-M>] [-j<N>|--jobs=<N>] [--skip-build] [--build-only]
```

**参数说明**：

| 参数                 | 说明                                                                                                              |
| -------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `--ops=<names>`    | 算子目录名，支持逗号分隔多个（如`mat_mul` 或 `mat_mul,grouped_matmul`）                                       |
| `--target=<names>` | 样例名，支持逗号分隔多个（如`mat_mul_basic` 或 `mat_mul_basic,mat_mul_streamk`）。多值时 `--ops` 必须为单个 |
| `--case=<path>`    | CSV 测试用例文件路径。仅支持`--ops` 和 `--target` 均为单个值时使用                                            |
| `--ti=<N>`         | 仅运行第 N 条用例（0-based 索引）。仅支持`--ops` 和 `--target` 均为单个值时使用                               |
| `--ti=<N-M>`       | 运行第 N 到第 M 条用例（含两端）。仅支持`--ops` 和 `--target` 均为单个值时使用                                |
| `--skip-build`     | 跳过 CMake 编译阶段                                                                                               |
| `--build-only`     | 仅编译，不运行和验证                                                                                               |
| `-j<N>` / `--jobs=<N>` | 编译与样例运行阶段的并行作业数（两阶段各自自适应）。编译默认 min(核数, 目标数, 内存GB/16)；运行默认 4×NPU 卡数（无法探测卡数时为 8），不超过样例数。显式指定时不超过核数。单样例（或 `-j1`）时运行阶段串行、输出直接打印到控制台；并行运行时各样例完整输出写入 `examples/build/logs/run/{op}_{example}.log`，同时控制台实时回显带 `[样例名]` 前缀的用例级结果（PASS/FAIL + 精度信息） |

**多值约束**：

| 场景         | --ops | --target | --case | --ti |
| ------------ | ----- | -------- | ------ | ---- |
| 单算子单样例 | 单个  | 单个     | 允许   | 允许 |
| 单算子多样例 | 单个  | 多个     | 禁止   | 禁止 |
| 多算子       | 多个  | 禁止     | 禁止   | 禁止 |
| 全部样例     | 不传  | 不传     | 禁止   | 禁止 |

**用法示例**：

```bash
# 运行 mat_mul 下所有样例
bash examples/common/run.sh --ops=mat_mul

# 运行单个样例
bash examples/common/run.sh --ops=mat_mul --target=mat_mul_basic

# 运行多个算子的所有样例
bash examples/common/run.sh --ops=mat_mul,grouped_matmul

# 运行单算子下多个样例
bash examples/common/run.sh --ops=mat_mul --target=mat_mul_basic,mat_mul_streamk

# 指定 CSV 文件（需单 ops + 单 target）
bash examples/common/run.sh --ops=mat_mul --target=mat_mul_basic --case=path/to/cases.csv

# 仅运行第 3 条用例（需单 ops + 单 target）
bash examples/common/run.sh --ops=mat_mul --target=mat_mul_basic --ti=3

# 运行第 0 到第 5 条用例（需单 ops + 单 target）
bash examples/common/run.sh --ops=mat_mul --target=mat_mul_basic --ti=0-5

# 仅编译不运行
bash examples/common/run.sh --ops=mat_mul --target=mat_mul_basic --build-only

# 跳过编译直接运行
bash examples/common/run.sh --ops=mat_mul --target=mat_mul_basic --skip-build
```

**执行流程**：

```
bash examples/common/run.sh --ops=mat_mul --target=mat_mul_basic
    │
    ├─→ preflight（检查 ASCEND_HOME_PATH / bisheng / g++ / python3 / cmake）
    ├─→ cmake + cmake --build             # 编译（并行）
    ├─→ 并行运行各样例（worker 池；多样例时各样例完整输出写入
    │   examples/build/logs/run/{op}_{example}.log，控制台实时回显
    │   带 [样例名] 前缀的用例级结果）
    │   每个样例读取 {example}.csv，逐条执行:
    │       ├─ gen_data.py（按 .conf [gen_data] 参数生成数据）
    │       ├─ ./mat_mul_basic（按 .conf [kernel] 参数执行 kernel）
    │       └─ verify_result.py（按 .conf [verify] 参数验证精度）
    └─→ 结果汇总
```

### 4.3 CSV 测试用例格式

每个样例目录下有一个与样例同名的 CSV 文件（如 `mat_mul_basic.csv`），定义所有测试用例：

```csv
casename,m,k,n,bias,dtype,transA,transB,hf32,layoutA,layoutB
mat_mul_basic_fp16,128,512,128,128,float16,false,false,false,ND,ND
mat_mul_basic_bf16,128,512,128,128,bfloat16,false,false,false,ND,ND
mat_mul_basic_fp32,128,512,128,128,float32,false,false,false,ND,ND
mat_mul_basic_hf32,128,512,128,128,float32,false,false,true,ND,ND
mat_mul_basic_weightNz,128,512,128,0,float16,false,false,false,ND,NZ
```

> 具体列定义参见各样例的 CSV 文件和 README。

### 4.4 .conf 配置文件格式

每个样例目录下有一个 `{example}.conf` 文件（INI 格式），定义数据生成、kernel 执行和精度验证的参数路由。详细格式参见 [`examples/common/CONF_FORMAT.md`](common/CONF_FORMAT.md)。

基本结构：

```ini
[scripts]
gen_data=gen_data.py           # 可选，默认 gen_data.py
verify=verify_result.py        # 可选，默认 verify_result.py

[gen_data]
params=...                     # 逗号分隔的参数条目
bool_flags=...                 # 可选，布尔 flag

[kernel]
params=...
bool_flags=...                 # 可选

[verify]
params=...
bool_flags=...                 # 可选
```

| Section        | 用途                                                                                   |
| -------------- | -------------------------------------------------------------------------------------- |
| `[scripts]`  | 指定 gen_data 和 verify 脚本文件名（可选，默认`gen_data.py` / `verify_result.py`） |
| `[gen_data]` | 传递给数据生成脚本的参数                                                               |
| `[kernel]`   | 传递给 kernel 执行的参数                                                               |
| `[verify]`   | 传递给精度验证脚本的参数                                                               |

### 4.5 两种执行方式的关联

- `build.sh --examples` 是顶层入口，内部调用 `common/run.sh`
- `common/run.sh` 是统一执行器，负责完整的 preflight → 编译 → CSV 解析 → 逐条执行 → 结果汇总流程

## 5. 新增样例

### 5.1 新增算子

```bash
# 1. 创建算子目录
mkdir -p examples/new_op/scripts

# 2. 创建数据生成和验证脚本
# examples/new_op/scripts/gen_data.py
# examples/new_op/scripts/verify_result.py

# 3. 创建算子级 CMakeLists.txt，注册样例
cat > examples/new_op/CMakeLists.txt << 'EOF'
ops_example_add_executable(new_op_basic new_op_basic/new_op_basic.cpp)
EOF

# 4. 在 examples/CMakeLists.txt 中注册算子
# 末尾追加: add_subdirectory(new_op)
```

### 5.2 新增样例（已有算子下）

```bash
# 1. 创建样例目录
mkdir -p examples/new_op/new_op_basic

# 2. 创建样例源码 new_op_basic.cpp

# 3. 创建 .conf 配置文件（参考 common/CONF_FORMAT.md）
cat > examples/new_op/new_op_basic/new_op_basic.conf << 'EOF'
[gen_data]
params=...
[kernel]
params=...
[verify]
params=...
EOF

# 4. 创建 CSV 测试用例表 new_op_basic.csv

# 5. 在算子级 CMakeLists.txt 中注册
# 追加: ops_example_add_executable(new_op_basic new_op_basic/new_op_basic.cpp)
```

注册后即可通过以下方式运行：

```bash
# 通过 build.sh
./build.sh --examples --ops=new_op --target=new_op_basic

# 通过 common/run.sh
bash examples/common/run.sh --ops=new_op --target=new_op_basic
```
