# 自定义算子包构建与安装步骤（MoE Init Routing Custom）

> `vllm_fl/_cann_ops_custom/` 是**构建/安装产生的运行包目录**（含 `op_api/lib/*.so`、
> `op_impl/ai_core/tbe/kernel/**/*.o`、`bin/set_env.bash` 等），体积大且与具体 CANN 版本绑定，
> 仓库**不提交**其中任何生成文件。需要时按下述步骤在容器内重新生成。

## 1. 前置条件

- Ascend 910B3 容器，CANN 9.0.0（`source /usr/local/Ascend/ascend-toolkit/set_env.sh`）
- Python 3.11 + torch_npu，vLLM 0.20.2 环境
- 插件仓库（本分支）

## 2. 插件构建（editable）

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
VLLM_VENDOR=ascend pip install -e . --no-build-isolation
```

## 3. 构建 aclnn 自定义算子包

```bash
cd <repo_root>
bash build_aclnn.sh
```

## 4. 安装到插件目录（隔离安装，不污染系统 CANN）

```bash
bash <生成的算子包>.run --install-path=<repo_root>/vllm_fl/_cann_ops_custom
```

安装后目录结构（由构建产生，不入库）：

```text
vllm_fl/_cann_ops_custom/vendors/custom_transformer/
├── bin/set_env.bash              # 环境脚本（见第 5 节）
├── op_api/include/aclnnop/       # aclnn 头文件
├── op_api/lib/libcust_opapi.so   # 算子库
├── op_impl/ai_core/tbe/kernel/   # kernel 描述与 .o
└── version.info
```

## 5. 环境脚本（按脚本所在目录定位）

原脚本把路径写死为 `/workspace/Main/vllm-plugin-FL/...`，换机器或换目录即失效。
`vendors/custom_transformer/bin/set_env.bash` 建议改为：

```bash
#!/bin/bash
# 按脚本自身位置定位算子包根目录，避免写死绝对路径
CUSTOM_OPP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ASCEND_CUSTOM_OPP_PATH="${CUSTOM_OPP_ROOT}:${ASCEND_CUSTOM_OPP_PATH}"
export LD_LIBRARY_PATH="${CUSTOM_OPP_ROOT}/op_api/lib/:${LD_LIBRARY_PATH}"
```

使用：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source <repo_root>/vllm_fl/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash
```

## 6. 验证算子安装成功

```bash
python3 tests/custom_ops_tests/test_moe_init_routing_custom.py
```

## 7. 边界说明

- 本次提交只包含测试、基准脚本与文档，**未修改模型调用链**；仓库当前模型侧 MoE 路由仍为
  `torch_npu.npu_moe_init_routing_v2`（`routing_v2`）。
- 上述目录及其中的 `.so` / `.o` 等生成物不入库。
