# .conf 文件格式规范

本文档定义了 `run_case.py` 使用的 INI 格式 `.conf` 文件格式，用于配置样例执行参数。

## 文件命名与位置

- **命名**：`{example_name}.conf`
- **位置**：样例目录内（如 `examples/acl/mat_mul/mat_mul_basic/mat_mul_basic.conf`）

## INI 结构

每个 `.conf` 文件最多包含四个 section：

```ini
[scripts]
gen_data=gen_data.py       # 可选，默认 gen_data.py
verify=verify_result.py    # 可选，默认 verify_result.py

[gen_data]
params=...
bool_flags=...  # 可选

[kernel]
params=...
bool_flags=...  # 可选

[verify]
params=...
bool_flags=...  # 可选
```

| Section | 用途 |
|---------|------|
| `[scripts]` | 指定 gen_data 和 verify 脚本文件名（可选，默认 `gen_data.py` / `verify_result.py`） |
| `[gen_data]` | 传递给数据生成脚本的参数 |
| `[kernel]` | 传递给 kernel 执行文件的参数 |
| `[verify]` | 传递给验证脚本的参数 |

### [scripts] section

| 字段 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `gen_data` | 否 | `gen_data.py` | gen_data 脚本文件名，必须位于算子的 `scripts/` 目录下 |
| `verify` | 否 | `verify_result.py` | verify 脚本文件名，必须位于算子的 `scripts/` 目录下 |

### 字段

| 字段 | 必填 | 说明 |
|------|------|------|
| `params` | 是 | 逗号分隔的参数条目列表（位置参数、flag 参数、运行时令牌）。位置参数的顺序很重要。 |
| `bool_flags` | 否 | 逗号分隔的布尔 flag 条目列表（条件性包含）。 |

## params 条目语法

params 字段支持以下 5 种条目类型：

### 1. 位置参数 (`column_name`)

将 CSV 值作为位置参数传递。

```ini
params=m,k,n
```

对于 CSV 行 `m=1024,k=512,n=256`，生成：`1024 512 256`

### 2. Flag 参数 (`--flag-name:column_name`)

将 CSV 值作为命名 flag 参数传递。

```ini
params=--m:m,--k:k,--dtype:dtype
```

对于 CSV 行 `m=1024,k=512,dtype=float16`，生成：`--m 1024 --k 512 --dtype float16`

### 3. 运行时令牌 (`$TOKEN_NAME`)

注入由 `run_case.py` 提供的运行时值，不从 CSV 查找。

可用令牌：
- `$OUTPUT_DIR` — 解析为 `examples/{op}/{example}/output/`
- `$INPUT_DIR` — 解析为 `examples/{op}/{example}/input/`

```ini
params=--output-dir:$OUTPUT_DIR
```

### 4. 运行时令牌拼接路径 (`$TOKEN/path`)

解析运行时令牌后，追加路径后缀。

```ini
params=$OUTPUT_DIR/golden_c.bin,$OUTPUT_DIR/npu_out.bin
```

对于 `$OUTPUT_DIR/golden_c.bin`，结果为 `examples/{op}/{example}/output/golden_c.bin`。

### 5. 字面量字符串 (`="value"`)

传递硬编码的字面量字符串，不来自 CSV 或运行时。

```ini
params==golden.bin
```

## bool_flags 条目类型

bool_flags 字段支持以下 1 种条目类型：

### 1. 布尔 flag (`++flag:column`)

仅当 CSV 值为 `true`（不区分大小写）时包含 `--flag`。

```ini
bool_flags=++trans-a:transA,++hf32:hf32
```

对于 CSV 行 `transA=true,hf32=false`，生成：`--trans-a`（不包含 `--hf32`）

## 约束

1. **列存在性**：`params` 和 `bool_flags` 中引用的所有列名必须存在于 CSV 表头中，运行时令牌（`$OUTPUT_DIR`、`$INPUT_DIR` 等）和字面量字符串（`="value"`）除外。
2. **顺序敏感**：`params` 中的位置参数按出现顺序传递。
3. **必填字段**：每个 section 必须有 `params` 字段。`bool_flags` 为可选。
4. **可用运行时令牌**：`$OUTPUT_DIR`（{example}/output/）、`$INPUT_DIR`（{example}/input/）。

## 完整示例

### 示例 1：位置参数风格（mat_mul）

```ini
[gen_data]
params=m,k,n,transA,transB,dtype,bias,format

[kernel]
params=m,k,n,transA,transB,dtype,hf32,bias,format

[verify]
params=m,n,dtype
bool_flags=++hf32:hf32
```

### 示例 2：Flag 风格含路径拼接（quant_grouped_matmul_mx）

```ini
[gen_data]
params=--group-num:groupNum,--m:m,--n:n,--k:k,--dtype:dtype,--layout-b:layoutB,--group-list-type:groupListType,--group-type:groupType,--single-w:singleW,--is-bias:isBias,--group-list:groupList,--output-dir:$OUTPUT_DIR

[kernel]
params=groupNum,m,n,k,baseM,baseN,baseK,kAL1,kBL1,scaleKAL1,scaleKBL1,isBias,dbL0C,l1BufferStage,groupType,groupListType,singleW,dtype,layoutA,layoutB,aFullLoad,$OUTPUT_DIR,$OUTPUT_DIR/npu_out.bin

[verify]
params=$OUTPUT_DIR/golden_c.bin,$OUTPUT_DIR/npu_out.bin,--groups:groupNum,--m:m,--n:n
```

使用了：flag 参数（`--group-num:groupNum`）、位置参数（`groupNum,m,n,k,...`）、布尔 flag（无）、运行时令牌（`$OUTPUT_DIR`）、路径拼接（`$OUTPUT_DIR/npu_out.bin`）。

### 示例 3：混合位置参数和 flag 含路径拼接（quant_batch_matmul_cube verify）

```ini
[verify]
params=$OUTPUT_DIR/golden_c.bin,$OUTPUT_DIR/npu_out.bin,--batch:batch,--m:M,--n:N,--dtype:CType
```

混合了位置路径参数（`$OUTPUT_DIR/golden_c.bin`、`$OUTPUT_DIR/npu_out.bin`）和 flag 参数（`--batch:batch`、`--m:M` 等）。
