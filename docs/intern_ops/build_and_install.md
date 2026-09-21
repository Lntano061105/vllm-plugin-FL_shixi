# 自定义算子包构建与安装步骤（MoE Init Routing Custom）

> `vllm_fl/_cann_ops_custom/` 是**构建/安装产生的运行包目录**（含 `op_api/lib/*.so`、
> `op_impl/ai_core/tbe/kernel/**/*.o`、`bin/set_env.bash` 等），体积大且与具体 CANN 版本绑定，
> 仓库**不提交**其中任何生成文件。需要时按下述步骤在容器内重新生成。

## 1. 前置条件

- Ascend 910B3 容器，CANN 9.0.0（`source /usr/local/Ascend/ascend-toolkit/set_env.sh`）
- Python 3.11 + torch_npu，vLLM 0.20.2 环境（本容器为 `/workspace/venvs/vllm0202`）
- 插件仓库（本分支）

## 2. 插件构建（editable）

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
VLLM_VENDOR=ascend pip install -e . --no-build-isolation
```

## 3. 构建 aclnn 自定义算子包

```bash
cd <repo_root>
bash csrc/ascend/build_aclnn.sh            # 可追加 SOC 版本参数，省略时默认 ascend910b
```

- 脚本位于 `csrc/ascend/`，仓库根目录下**没有** `build_aclnn.sh`，须按上述相对路径调用。
- 可选参数：`ascend910b`（910B/A2，默认）或 `ascend910_93`（910C/A3）；其它取值脚本直接报错退出，
  `ascend310*` 会提示无自定义算子并正常返回。
- 需先 `source` CANN 环境（见第 2 节）。脚本内部按 `${BASH_SOURCE[0]}` 推导仓库根目录，
  因此可在任意工作目录调用。
- 脚本会先清理 `csrc/ascend/{build,output,build_out}`，再执行
  `csrc/ascend/build.sh --pkg --ops=<自定义算子列表> --soc=<SOC>` 生成 `.run` 包。
- 依赖 catlass 子模块（`csrc/ascend/third_party/catlass`）：缺失时脚本自动执行
  `git submodule update --init --recursive`，离线环境需提前准备。

## 4. 安装到插件目录（隔离安装，不污染系统 CANN）

`build_aclnn.sh` 末尾已自动完成安装（等价命令如下），通常无需手动执行：

```bash
bash csrc/ascend/build/cann-ops-transformer*.run --install-path=<repo_root>/vllm_fl/_cann_ops_custom
```

`.run` 包输出在 `csrc/ascend/build/cann-ops-transformer*.run`，兜底目录为 `csrc/ascend/build_out/`。

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

安装器生成的 `vendors/custom_transformer/bin/set_env.bash` 会把打包环境下的绝对路径写死
（本容器实测为 `/workspace/Main/vllm-plugin-FL/...`），换机器或换目录即失效。
安装后应将其内容替换为：

```bash
#!/bin/bash
# 按脚本自身位置定位算子包根目录，避免写死绝对路径
CUSTOM_OPP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ASCEND_CUSTOM_OPP_PATH="${CUSTOM_OPP_ROOT}:${ASCEND_CUSTOM_OPP_PATH}"
export LD_LIBRARY_PATH="${CUSTOM_OPP_ROOT}/op_api/lib/:${LD_LIBRARY_PATH}"
```

核对方式（确认无外部机器路径残留）：

```bash
grep -n 'custom_transformer:' <repo_root>/vllm_fl/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash
```

使用：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source <repo_root>/vllm_fl/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash
```

## 6. 验证算子安装成功

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source <repo_root>/vllm_fl/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash
cd <repo_root>
/workspace/venvs/vllm0202/bin/python tests/custom_ops_tests/test_moe_init_routing_custom.py
```

预期输出：`moe_init_routing_custom test: 9/9 passed`（2026-09-20 于 Ascend 910B3 / CANN 9.0.0 实测）。

解释器说明：算子运行须先 `source` CANN 环境再 `import torch_npu`；容器默认 `python3` 为
Python 3.10.12（无 torch_npu / vLLM），直接执行会失败，须使用装有 torch_npu 2.11.0 与
vLLM 0.20.2 的解释器（本环境为 `/workspace/venvs/vllm0202/bin/python`）。

## 7. 边界说明

- 本次提交只包含测试、基准脚本与文档，**未修改模型调用链**；仓库当前模型侧 MoE 路由仍为
  `torch_npu.npu_moe_init_routing_v2`（`routing_v2`）。
- 上述目录及其中的 `.so` / `.o` 等生成物不入库。
