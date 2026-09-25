# week1~3 操作步骤与指令详解（Fused GDN Gating 实习汇总）

> 人员：龚昊磊（容器用户名：ghl）
> 合并自 `week1_save.md` / `week2_save.md` / `week3_save.md`（2026-08-25），原文件已删除；周报正文见 `week_report.md`。
> 注：各周内部对原文件名的交叉引用（如 `week3_save.md §2.2`）仍沿用原名，对应本文档同名周次章节。
> 路径说明：本文档位于仓库 `ghl/` 子目录；文中仓库相对路径（`csrc/`、`vllm_fl/`、`tests/`、`docs/intern_ops/` 等）相对仓库根 `/workspace/vllm-plugin-FL/`（即加 `../` 前缀）。

---

## 使用规范（run_rule.md 速记，2026-08-25 记录）

> 来源：`run_rule.md`（与本文档同目录，容器环境使用规范）。后续所有周操作以此为准；违反安全约定导致环境损坏或影响他人的，首次书面警告并修复环境，多次暂停容器使用权限。

### 代码修改权限
- ✅ 允许：`vllm-plugin-fl`（FL 插件源码）、`flaggems`（FlagGems 源码）、其他已 `pip install -e .` 的可编辑 Python 包
- ❌ 禁止：系统目录（`/usr/`、`/etc/`、`/opt/` 等）、Ascend 驱动与固件（`/usr/local/Ascend/`）、其他用户目录、共享模型目录 `/models`（只读挂载）

### 测试结果持久化
- 所有测试输出必须保存在 `/workspace/results/`（挂载自宿主机 `/data2/shixisheng/results/`）
- 目录组织：`/workspace/results/ghl（用户名）/YYYYMMDD_测试描述/`
- 必存环境信息（便于复测）：测试时间/人员、vLLM / FL插件 / FlagGems 版本（commit ID）、CANN 版本 / 驱动版本 / NPU 信息（`npu-smi info`）、完整运行命令及关键环境变量、模型名称及路径

### 文件删除安全
- ❌ 严禁：`rm -rf /*`、`rm -rf *`、`rm -rf ./*`、任何未指定明确路径的通配符删除
- ✅ 必须指定至少一层明确路径（推荐绝对路径），删除前先用 `ls` 确认目标内容

### 容器使用基本规范
- 固定使用分配的容器（端口 3222~3228），不得切换他人容器
- 模型文件 `/models` 为只读挂载，不可修改
- 个人脚本放 `/workspace/scripts/ghl（用户名）/`
- 禁止随意 `apt install` 或修改系统配置；使用 `exit` 正常退出容器

### 违规处理
- 首次：书面警告并修复环境；多次：暂停容器使用权限

### 第 1 周合规自查（2026-08-25 复核）
- ✅ 修改范围：所有改动均在仓库内（vllm-plugin-FL / tests / benchmarks）；`/usr/local/Ascend/` 仅 `ls` 读取检查，`/models` 仅读取模型路径
- ✅ 测试结果：统一脚本 run 落在 `/workspace/results/atp_*`，环境信息由脚本写入 `run_info.env` / `run_summary.txt` / `run_command.sh`（已核验实际 run 目录）
- ⚠️ 结果目录组织：`/workspace/results/atp_*` 平铺结构（统一脚本约定），未按 run_rule.md 的"用户名/YYYYMMDD_描述"组织——沿用项目组统一脚本约定，非个人违规，记录待项目组确认
- ⚠️ §5.5 手动 profiling 文档建议结果目录 `/workspace/new_results/qwen0.6b_basic`，不符合"所有测试输出必须保存在 `/workspace/results/`"（该流程截至 2026-08-25 未执行，且 `/workspace/new_results` 不存在）；后续手动分步冒烟时结果目录应改到 `/workspace/results/` 下
- ✅ 未发现违规删除 / 越权修改 / 切换他人容器 / 随意 apt install 等操作

### 第 2 周合规自查（2026-08-25 复核）
- ✅ 修改范围：所有改动均在仓库内；`/usr/local/Ascend/` 仅检查使用，`/models` 仅读取权重路径
- ✅ 删除安全：§1.1 构建脚本内 `rm -rf build output build_out` —— 明确命名路径、无通配符，符合删除规范（相对路径可接受，规范"推荐绝对路径"）
- ✅ 测试结果：模型级 run 结果在 `/workspace/results/atp_*`（统一脚本），环境信息已由脚本记录（`run_info.env` 等）
- ⚠️ §4.6 手动分步冒烟仍引用 `/workspace/new_results/qwen0.6b_basic` 结果目录（截至 2026-08-25 未执行；若执行需改到 `/workspace/results/` 下）
- ✅ 未发现其他越权修改 / 违规删除 / 切换他人容器操作

### 第 3 周合规自查（2026-08-25 复核）
- ✅ 修改范围：4 个模型级 run 均在 `/workspace/results/atp_*`，`package_op_statistic.sh` 二次打包为 `*_op_statistic.tar.gz` 同目录存放（与统一脚本约定一致）；`/models` 仅读取
- ✅ 微基准/日志存档位置（2026-08-25 已整改）：`fused_gdn_gating_microbench_20260820.txt` 与 `model_*.log` 已从仓库 `benchmark_results/` 移入 `/workspace/results/ghl/20260820_算子级微基准/` 与 `20260820_模型级基准_自研脚本/`；仓库 `benchmark_results/` 仅剩 7/18 他人数据（未动）
- ✅ §1.4 计数钩子验证文件（`gdn_count_test.txt` 等）已随外层日志一并移入 `/workspace/results/ghl/20260824_20260825_模型级1K1K验证外层日志/`
- ✅ 踩坑 #4"GB 级导出文件用完即删"未见具体 rm 命令，无通配符删除迹象
- ✅ 未发现越权修改 / 违规删除 / 切换他人容器操作

---

## 第 1 周（原 week1_save.md）

> 原对应周报：`week1.md`（已并入 `week_report.md`）
> 人员：龚昊磊（容器用户名：ghl）
> 目标：定位算子在 Qwen3.6 推理计算图中的位置，理解输入输出数据类型与数据布局，读懂参考实现并确认环境与构建链路；同时完成项目组 Profiling 统一脚本的就绪检查，为模型级验证统一接入服务化压测 + Profiler 流程铺路。

## 0. 前置环境

- 硬件：Ascend 910B3 × 8（本机）
- 软件：Python 3.11、torch 2.8.0 + torch_npu 2.8.0.post2、vllm 0.13.0、CANN 8.5/9.0
- 关键产物位置：
  - torch 绑定库：`vllm_fl/_C_ascend.cpython-311-aarch64-linux-gnu.so`
  - 框架算子库：`vllm_fl/libvllm_fl_kernels.so`
  - CANN 自定义算子包：`vllm_fl/_cann_ops_custom/vendors/custom_transformer/`
  - 源码目录：`csrc/ascend/attention/fused_gdn_gating/`
  - Profiling 统一脚本：`/workspace/scripts/`（run_vllm_ascend_profile_unified.sh / run_vllm_fl_profile_unified.sh / package_op_statistic.sh）
  - 手动分步 profiling 流程说明：`手动profiling测试.md`（统一脚本内部流程的手动展开，Qwen3-0.6B 示例）
  - 结果目录：`/workspace/results/`（`atp_*` 命名的 run 目录，`latest_run_dir.txt` 指向最新 run）

---

## 1. 调用链梳理（定位算子在计算图中的位置）

自顶向下的调用链为：

```
Qwen3NextGatedDeltaNet.forward
  → torch.ops.vllm.gdn_attention_core          (vllm_fl/models/qwen3_next.py:477 → 1285)
    → Qwen3NextGatedDeltaNet._forward_core      (qwen3_next.py:497，Triton 基线在第 604 行调 fused_gdn_gating)
      → torch.ops._C_ascend.npu_fused_gdn_gating (AscendC 路径，patch 后)
        → fused_gdn_gating_torch_adpt.h          (shape/dtype 校验 + 调 aclnn)
          → aclnnFusedGdnGating                  (op_host/op_api/，GetWorkspaceSize + Execute)
            → AscendC kernel                     (op_kernel/fused_gdn_gating.cpp，AI Core 上执行)
```

复现/核验用指令：

```bash
# 1) 确认模型层入口与 Triton 基线调用位置
grep -n "gdn_attention_core\|fused_gdn_gating\|_forward_core" vllm_fl/models/qwen3_next.py
#    477: torch.ops.vllm.gdn_attention_core(...)
#    604: g, beta = fused_gdn_gating(self.A_log, a, b, self.dt_bias)   ← Triton 基线
#   1285: def gdn_attention_core(...)  → 分派到 self._forward_core(...)

# 2) 确认 torch schema 注册位置（周报所指 torch_binding.cpp:2688）
sed -n '2687,2696p' csrc/ascend/torch_binding.cpp
#    ops.def("npu_fused_gdn_gating(Tensor A_log, Tensor a, Tensor b, Tensor dt_bias,
#            float beta=1.0, float threshold=20.0) -> (Tensor g, Tensor beta_output)");

# 3) 确认 AscendC 接入点（patch 后的 _forward_core 里调用）
grep -n "npu_fused_gdn_gating" vllm_fl/dispatch/backends/vendor/ascend/patches/patch_qwen3_6_gdn.py
#   386: torch.ops._C_ascend.npu_fused_gdn_gating(self.A_log, a, b, self.dt_bias.to(self.A_log.dtype))

# 4) 确认算子目录结构（op_host=Tiling/InferShape，op_kernel=AI Core 实现）
find csrc/ascend/attention/fused_gdn_gating -type f
```

**定位结论**：该算子位于 `conv1d` 之后、recurrent/chunk gated delta rule 之前——即 GDN 层的门控计算（`g`、`beta`），上游是 causal conv 输出，下游是 linear attention 状态递推。

---

## 2. 算子接口确认（输入输出 dtype 与布局）

直接读三处源码即可确认全部约束：

```bash
# schema（默认属性 beta=1.0, threshold=20.0）
sed -n '2688,2695p' csrc/ascend/torch_binding.cpp

# Torch Adapter：shape/dtype 校验 + 输出张量创建
cat csrc/ascend/attention/fused_gdn_gating/fused_gdn_gating_torch_adpt.h

# op_api：C 层 dtype 支持列表与参数校验
head -90 csrc/ascend/attention/fused_gdn_gating/op_host/op_api/aclnn_fused_gdn_gating.cpp
```

接口汇总（与 vLLM Triton 基线 `qwen3_next.fused_gdn_gating` 语义一致）：

| 参数 | Shape | dtype | 说明 |
|---|---|---|---|
| `A_log` | `(num_heads,)` | fp32 | 每头对数衰减，模型参数 |
| `a` | `(batch, num_heads)` | bf16/fp16 | 时间步投影，需与 `b` 同 dtype |
| `b` | `(batch, num_heads)` | bf16/fp16 | 门控投影 |
| `dt_bias` | `(num_heads,)` | fp32 | 时间步偏置，需与 `A_log` 同 dtype |
| `g`（输出） | `(1, batch, num_heads)` | fp32 | `g = -exp(A_log) * softplus(a + dt_bias)` |
| `beta_output`（输出） | `(1, batch, num_heads)` | 与 `b` 同 | `beta = sigmoid(b)` |

kernel 内按 tiling key 分派 6 种模板实例（bf16/fp16 输入 × fp32/bf16/fp16 输出），AIV 核执行：

```bash
grep -n "TILING_KEY_IS\|KERNEL_TYPE_AIV" csrc/ascend/attention/fused_gdn_gating/op_kernel/fused_gdn_gating.cpp
```

---

## 3. 环境确认

```bash
# 硬件健康检查
npu-smi info            # 期望 8 张 910B3，Health 正常

# 软件版本确认
python -c "import torch, torch_npu, vllm; print(torch.__version__, torch_npu.__version__, vllm.__version__)"
ls /usr/local/Ascend/   # CANN 安装目录（8.5/9.0 两套）

# 构建产物就位检查
ls -la vllm_fl/_C_ascend*.so vllm_fl/libvllm_fl_kernels.so
ls vllm_fl/_cann_ops_custom/vendors/custom_transformer/{op_api,op_impl,bin}
```

---

## 4. 验证操作（三项，对应周报验证结果）

### 4.1 连通性测试

```bash
# 必须先在当前 Shell source 环境脚本（与 LD_LIBRARY_PATH/ASCEND_CUSTOM_OPP_PATH 相关）
source /workspace/vllm-plugin-FL/vllm_fl/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash

# 直接以脚本方式运行（脚本内置 _check_custom_op_env 检查）
python tests/custom_ops_tests/test_fused_gdn_gating.py
# 期望输出：npu_fused_gdn_gating test passed
```

> 注意：`set_env.bash` 只导出 `ASCEND_CUSTOM_OPP_PATH` 与 `LD_LIBRARY_PATH` 两个变量；若未 source，脚本会直接报错提示。

### 4.2 算子级固定测试（17 项）

```bash
source /workspace/vllm-plugin-FL/vllm_fl/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash

pytest tests/ops/ascend/test_fused_gdn_gating.py -v
# 或仅跑 gpu 标记（测试文件内已 pytestmark = pytest.mark.gpu）
pytest tests/ops/ascend/test_fused_gdn_gating.py -m gpu
```

- 用例构成：固定 `NUM_HEADS=32`、token 数 `[1, 4, 16, 64]`、dtype `bf16/fp16`，分别对比 PyTorch 参考实现与 vLLM Triton 基线（`rtol=atol=1e-2`），外加 softplus 越阈退化路径（`a=50` 时 `g == -exp(A_log)*x`）与 shape/dtype 校验用例。
- 文件顶部自带 `os.environ.setdefault("VLLM_PLUGINS", "fl")`，无需手工设插件。
- 测试代码直接调用 `torch.ops._C_ascend.npu_fused_gdn_gating(...)`，不经过模型层。

### 4.3 回退开关验证

```bash
# 默认：走 AscendC 路径（patch_qwen3_6_gdn() 返回 True）
python -c "import vllm_fl.dispatch.backends.vendor.ascend.patches.patch_qwen3_6_gdn as m; print(m.patch_qwen3_6_gdn())"
# 日志可见：Patched Qwen3NextGatedDeltaNet ... (AscendC causal_conv1d / fused_gdn_gating / ...)

# 打开开关：强制保持 Triton 路径（返回 False）
VLLM_FL_DISABLE_ASCENDC_GDN=1 python -c "import vllm_fl.dispatch.backends.vendor.ascend.patches.patch_qwen3_6_gdn as m; print(m.patch_qwen3_6_gdn())"
# 日志可见：VLLM_FL_DISABLE_ASCENDC_GDN=1, keep Triton GDN path
```

开关逻辑在 `patch_qwen3_6_gdn.py` 的 `_ascendc_ops_available()`（第 146 行）：开关为 1、`_C_ascend` 不可导入、或 5 个必需 op（`npu_causal_conv1d_custom`/`npu_fused_gdn_gating`/`npu_recurrent_gated_delta_rule`/`npu_gemma_rms_norm`/`npu_add_rms_norm_bias`）缺失时返回 False 回退 Triton。默认情况下 patch 会自动 bootstrap CANN 算子环境，无需手动 source。

---

## 5. Profiling 统一脚本就绪检查（本周新增）

项目组统一脚本位于 `/workspace/scripts/`，共 3 个（用途说明见 `unify_sh.md`）：

| 脚本 | 用途 |
|---|---|
| `run_vllm_ascend_profile_unified.sh` | `VLLM_PLUGINS=ascend`，原生 vllm-ascend 起服务 + 采集 torch/NPU profiler |
| `run_vllm_fl_profile_unified.sh` | `VLLM_PLUGINS=fl`（vllm-plugin-FL），起服务 + 采集 profiler（本任务用这个） |
| `package_op_statistic.sh` | 对已完成的 run 目录二次打包，提取算子统计/API 统计/benchmark 摘要 |

### 5.1 参数核对（--help）

```bash
/workspace/scripts/run_vllm_fl_profile_unified.sh --help
```

关键参数（与 `unify_sh.md` 一致部分略）：`--cases "I,O,NP;..."`（正式测试，格式=输入长度,输出长度,请求数）、`--warmup "I,O,C,NP"`（预热，4 段）、`--mode graph|eager`、`--chunked true|false`、`--bench-profile true|false`（是否给 `vllm bench serve` 传 `--profile`）、`--tp N`（默认 4）、`--gmem FLOAT`（默认 0.6）、`--devices IDS`（默认 0,1,2,3）、`--port`、`--run-label`、`--package tar.gz|tar|none`（默认 tar.gz）、`--skip-analyse`、`--mtp N`（MTP 快捷参数，不能与 `--speculative-config` 同用）、`--flaggems-ops`（FlagGems 白名单，默认 `unquantized_fused_moe_method,topk_softmax`）。

### 5.2 执行流程与产物结构（已核对脚本实现）

```bash
# 执行流程（run_vllm_fl_profile_unified.sh）
# 1) 解析/校验参数 → 创建 RUN_DIR（atp_{model_tag}_fl_{mode}_{chunk}_{label}_tp{tp}_gmem{gmem}_{ts}）
# 2) 记录 run_command.sh / run_info.env / run_summary.txt
# 3) 内置 source CANN 环境（cann-9.0.0、nnal/atb），设置 VLLM_PLUGINS=fl 等全部环境变量
# 4) setsid 启动 vllm serve，等待 /v1/models 可访问（超时 1200s）
# 5) warmup（默认 128,128,2,4）→ 逐个 --cases 跑 vllm bench serve（--bench-profile true 时带 --profile）
# 6) 每个 case 前写 .current_case_dir，profiler 产物归档到对应 case 目录
# 7) 汇总 Serving Benchmark Result 到 request_benchmark_results.txt
# 8) 优雅停服 → torch_npu.profiler.profiler.analyse（SKIP_ANALYSE 或 bench-profile=false 时跳过）
# 9) 修复权限 → 按 --package 打包（默认 tar.gz）

# 产物：RUN_DIR 下含 server.log / client.log / warmup_terminal.log /
#       request_benchmark_results.txt / analysis_terminal.log /
#       torch_profile/i32_o32_np1_c1/.../ASCEND_PROFILER_OUTPUT/
#         {analysis.db, op_statistic.csv, api_statistic.csv, kernel_details.csv,
#          operator_details.csv, communication*.json, trace_view.json}
```

### 5.3 脚本实现与 unify_sh.md 文档的差异（核对记录）

| 点 | unify_sh.md 描述 | 脚本实际实现（以代码为准） |
|---|---|---|
| FL 脚本默认端口 | usage 文本 8113 | 代码变量 `PORT="8080"`（--help 显示 8113，属 usage 文本笔误） |
| analyse 跳过条件 | FL 脚本仅在 `--skip-analyse` 时跳过 | 实际 `--skip-analyse` 或 `--bench-profile false` 都跳过（L599） |
| server 就绪超时 | 360 秒 | `SERVER_WAIT_TIMEOUT=1200` |
| 环境变量 | 未提 | 脚本内置 `--flaggems-ops`（FlagGems 白名单），并透传外部 export 的环境变量给子进程 |

> 结论：外部 `export VLLM_FL_DISABLE_ASCENDC_GDN=1`（Triton 对照）与 `VLLM_FL_GDN_COUNT_FILE=<path>`（调用计数钩子）会在 `vllm serve` 子进程中生效，AscendC/Triton 双路径可直接复用统一脚本。

### 5.4 TP 约束与多卡适配

- 27B/35B 的 GDN 层 conv 状态维度 `conv_dim = head_k_dim*num_k_heads*2 + head_v_dim*num_v_heads = 8192`，vLLM 要求其能被 TP 整除（`gated_delta_net_state_shape` 内 `ensure_divisibility`）。
- `8192 % 3 = 2` → **TP3 结构性不可行**（35B 历史跑批用 TP4，卡数不足时取 TP2）。
- 适配命令：`--tp 1`（27B）/ `--tp 2`（35B）+ 对应 `--devices`；TP2 下 8192 可整除，无需调整 `max_num_batched_tokens`。
- 35B 需注意 `--gmem`（默认 0.6，若 prefill 激活 OOM 再降到 0.55）。

### 5.5 手动分步 profiling 流程核验（手动profiling测试.md，本周新增）

`手动profiling测试.md` 把 `run_vllm_fl_profile_unified.sh` 的核心执行流程拆成最基础的手动命令（以 Qwen3-0.6B 为例），用于逐步验证与故障定位。经对照核验，**手动流程 = 统一脚本内部步骤的手动展开**：

| 统一脚本内部步骤 | 手动分步（手动profiling测试.md） |
|---|---|
| 创建 RUN_DIR / PROFILE_ROOT，写 `.current_case_dir` | `mkdir -p .../torch_profile/i32_o32_np1_c1` + `echo <case_dir> > .current_case_dir` |
| 设置环境变量 + `setsid vllm serve ...` | 终端 1 手动 export + 完整 serve 参数（含 `--profiler-config` JSON，与脚本 PROFILER_CONFIG 一致） |
| 等待 `/v1/models` 可访问 | 终端 2 `curl http://127.0.0.1:8080/v1/models` |
| `warmup_once`（默认 128,128,2,4） | 终端 2 手动 `vllm bench serve`（random-input-len 128 / output 128 / 并发 2 / 4 请求） |
| `run_case` 逐个跑 `vllm bench serve --profile` | 终端 2 手动 `vllm bench serve ... --profile`（case 1/2/3） |
| 优雅停服（stop_server） | 终端 1 `Ctrl+C` |
| `analyse_profile`（torch_npu.profiler.profiler.analyse） | 步骤 8 可选：手动 `python - <<'PY' ... analyse(d)` |

**与统一脚本的差异点（记录）**：

| 点 | 手动profiling测试.md | run_vllm_fl_profile_unified.sh |
|---|---|---|
| serve 附加参数 | 额外 `--no-async-scheduling` | SERVE_ARGS 无此项 |
| 结果目录 | `/workspace/new_results/qwen0.6b_basic` | `/workspace/results/atp_*` |
| 可见设备 | `ASCEND_RT_VISIBLE_DEVICES=0` 先导出，随后覆盖为 `4,5,6,7`（最终生效 TP4，前一行冗余） | 由 `--devices` 单点控制 |
| 示例模型 | Qwen3-0.6B（小模型快速冒烟） | 27B/35B |
| analyse | 手动可选 | 自动（`--skip-analyse` 或 `--bench-profile false` 时跳过） |

**结论**：统一脚本一键失败时，可按 `手动profiling测试.md` 手动分步定位故障环节（serve 启动 / warmup / 具体 case / analyse）；0.6B 小模型冒烟是权重未就位或快速复验时的首选路径。

---

## 6. 下周计划

### 6.1 完整构建部署闭环（干净状态复现）

这是从零复现的命令序列，建议按顺序执行：

```bash
# 1) 编译 torch 扩展（编译出 vllm_fl/_C_ascend*.so）
VLLM_VENDOR=ascend python setup.py build_ext --inplace

# 2) 编译并打包 CANN framework 算子（自动：拉取 catlass 子模块 → build.sh --pkg → 安装到 vllm_fl/_cann_ops_custom）
cd csrc/ascend && bash build_aclnn.sh ascend910b
cd /workspace/vllm-plugin-FL
#    build_aclnn.sh 内部等价于：
#      bash build.sh --pkg --ops="...fused_gdn_gating..." --soc="ascend910b"
#      找到 build/ 或 build_out/ 下 cann-ops-transformer*.run
#      bash <pkg>.run --install-path=<ROOT>/vllm_fl/_cann_ops_custom

# 3) source 部署生成的环境脚本
source /workspace/vllm-plugin-FL/vllm_fl/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash

# 4) 重新打开一个 Shell（模拟"重开终端后仍可加载"的部署要求）
#    新 Shell 内直接验证：
python tests/custom_ops_tests/test_fused_gdn_gating.py
pytest tests/ops/ascend/test_fused_gdn_gating.py -m gpu
```

其中步骤 3/4 的核心是 `set_env.bash` 导出的两个变量：

```bash
export ASCEND_CUSTOM_OPP_PATH=/workspace/vllm-plugin-FL/vllm_fl/_cann_ops_custom/vendors/custom_transformer:${ASCEND_CUSTOM_OPP_PATH}
export LD_LIBRARY_PATH=/workspace/vllm-plugin-FL/vllm_fl/_cann_ops_custom/vendors/custom_transformer/op_api/lib/:${LD_LIBRARY_PATH}
```

> 注意：`LD_LIBRARY_PATH` 必须在 Python 进程启动前设置（glibc 只缓存启动时的库搜索路径），所以"重新打开 Shell 后再 source 再运行"正是验证部署闭环的关键。

### 6.2 统一脚本接入模型级 1K/1K 验证（前置）

模型权重就位后，用统一脚本跑 1K/1K（替代/互补自研脚本）：

```bash
# AscendC 路径
/workspace/scripts/run_vllm_fl_profile_unified.sh \
  --model-path /models/Qwen3.6-27B --model-tag qwen3.6-27b \
  --mode eager --chunked true --devices 0,1,2 --tp 1 \
  --gmem 0.6 --cases "1024,1024,1" --bench-profile true \
  --run-label ascendc_r1

# Triton 对照（外层 export 开关即可，子进程继承）
export VLLM_FL_DISABLE_ASCENDC_GDN=1
... 同参数，--run-label triton_r1
```

> 执行结果（2026-08-25）：27B/35B 双路径 4 个 run 全部完成，TTFT/TPOT/吞吐与 27B AscendC Profiler 证据（`op_statistic.csv`）见 week3_save.md §2.2/§2.3。

---

## 7. 关键文件速查表

| 用途 | 路径 |
|---|---|
| 模型层调用 | `vllm_fl/models/qwen3_next.py:477/1285` |
| schema 注册 | `csrc/ascend/torch_binding.cpp:2688` |
| Torch 适配层 | `csrc/ascend/attention/fused_gdn_gating/fused_gdn_gating_torch_adpt.h` |
| op_api 入口 | `csrc/ascend/attention/fused_gdn_gating/op_host/op_api/aclnn_fused_gdn_gating.cpp` |
| Host 侧 Tiling/InferShape | `csrc/ascend/attention/fused_gdn_gating/op_host/` |
| AscendC kernel | `csrc/ascend/attention/fused_gdn_gating/op_kernel/fused_gdn_gating.cpp` |
| 框架接入 patch | `vllm_fl/dispatch/backends/vendor/ascend/patches/patch_qwen3_6_gdn.py` |
| 连通性测试 | `tests/custom_ops_tests/test_fused_gdn_gating.py` |
| 算子级固定测试 | `tests/ops/ascend/test_fused_gdn_gating.py` |
| 构建脚本 | `csrc/ascend/build_aclnn.sh`（CANN 算子）、`setup.py build_ext --inplace`（torch 扩展） |
| Profiling 统一脚本 | `/workspace/scripts/run_vllm_ascend_profile_unified.sh`、`/workspace/scripts/run_vllm_fl_profile_unified.sh`、`/workspace/scripts/package_op_statistic.sh` |
| 手动分步 profiling 说明 | `手动profiling测试.md`（统一脚本流程的手动展开，0.6B 冒烟/故障排查用） |
| 脚本说明文档 | `unify_sh.md` |
| 结果目录 | `/workspace/results/`（`latest_run_dir.txt` 指向最新 run） |

---

## 第 2 周（原 week2_save.md）

> 原对应周报：`week2.md`（已并入 `week_report.md`）
> 人员：龚昊磊（容器用户名：ghl）
> 目标：完成干净环境下的构建部署闭环复现、参考分支差异核对、异常用例补充，并完成模型级 1K/1K 验证的统一脚本接入（dry-run 与双路径开关确认，脚本就绪待权重）。

## 0. 前置环境

- 硬件：Ascend 910B3 × 8（本机）
- 软件：Python 3.11、torch 2.8.0 + torch_npu 2.8.0.post2、vllm 0.13.0、CANN 8.5/9.0
- 关键产物位置：
  - torch 绑定库：`vllm_fl/_C_ascend.cpython-311-aarch64-linux-gnu.so`
  - 框架算子库：`vllm_fl/libvllm_fl_kernels.so`
  - CANN 自定义算子包：`vllm_fl/_cann_ops_custom/vendors/custom_transformer/`
  - 构建产物（自解压包）：`csrc/ascend/build/cann-ops-transformer-custom_linux-aarch64.run`（约 21MB）
  - 源码目录：`csrc/ascend/attention/fused_gdn_gating/`
  - Profiling 统一脚本：`/workspace/scripts/run_vllm_fl_profile_unified.sh`（vllm-plugin-FL 服务化压测 + Profiler）
  - 结果目录：`/workspace/results/`（`latest_run_dir.txt` 指向最新 run）

---

## 1. 构建部署闭环复现（干净状态全量重建 + 安装 + 新 Shell 验证）

### 1.1 执行构建

```bash
cd csrc/ascend && bash build_aclnn.sh ascend910b
```

`build_aclnn.sh` 内部等价步骤（自顶向下）：

```bash
# 1) SOC 分派（默认 ascend910b；ascend310 系列直接跳过；ascend910_93 走另一份算子列表）
SOC_VERSION=${1:-ascend910b}

# 2) 依赖 catlass：检查 csrc/ascend/third_party/catlass/include，
#    缺失则 git submodule update --init --recursive 拉取，并导出 CPATH
if [[ ! -d "${CATLASS_PATH}" ]]; then
    git submodule update --init --recursive
fi
export CPATH=${ABSOLUTE_CATLASS_PATH}:${CPATH}

# 3) 算子选择：CUSTOM_OPS 分号分隔列表（37 项，含 fused_gdn_gating）
# 4) 干净状态编译：先删 build/ output/ build_out/ 再打包
rm -rf build output build_out
bash build.sh --pkg --ops="$CUSTOM_OPS" --soc="ascend910b"

# 5) 找到生成的 .run 包并安装到 vllm_fl/_cann_ops_custom（与系统 CANN 隔离）
RUN_PACKAGE=$(ls build/cann-ops-transformer*.run 2>/dev/null | head -n1)  # 或 build_out/
bash "${RUN_PACKAGE}" --install-path="${ROOT_DIR}/vllm_fl/_cann_ops_custom"
```

> 注意：`build_aclnn.sh` 的 CUSTOM_OPS 列表实际为 **37 项**（`echo $CUSTOM_OPS | tr ';' '\n' | grep -v '^$' | wc -l`），差异以实际列表为准。

### 1.2 产物确认

```bash
# 自解压包（约 21MB）
ls -la csrc/ascend/build/cann-ops-transformer-custom_linux-aarch64.run

# 安装目录结构（op_api=aclnn 头/库，op_impl=AI Core 算子，op_proto=算子原型）
find vllm_fl/_cann_ops_custom/vendors/custom_transformer -maxdepth 2 -type d | sort
#    .../bin        → set_env.bash
#    .../op_api     → include/ + lib/（LD_LIBRARY_PATH 指向 op_api/lib）
#    .../op_impl    → ai_core/（算子二进制）
#    .../op_proto   → inc/ + lib/
#    .../scripts
```

### 1.3 部署闭环验证（核心：重新打开 Shell 再 source 再运行）

```bash
# 重新打开一个 Shell（模拟"重开终端后仍可加载"的部署要求）
source /workspace/vllm-plugin-FL/vllm_fl/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash

python tests/custom_ops_tests/test_fused_gdn_gating.py
# 期望输出：npu_fused_gdn_gating test passed
```

> 要点：
> 1. `set_env.bash` 只导出两个变量——`ASCEND_CUSTOM_OPP_PATH` 与 `LD_LIBRARY_PATH`（`op_api/lib`）。
> 2. `LD_LIBRARY_PATH` 必须在 Python 进程启动前设置（glibc 只缓存启动时的库搜索路径），所以"重开 Shell → source → 运行"正是闭环验证的关键。
> 3. 部署为自包含：算子已安装进 `vllm_fl/_cann_ops_custom`，**不依赖手工复制 .so**，与系统 CANN 隔离。
> 4. 若本机已有 `ASCEND_CUSTOM_OPP_PATH`，`set_env.bash` 采用前置追加方式导出。

---

## 2. 参考分支核对（appleinsky/qwen36_dense_moe）

### 2.1 算子源码逐文件对比

```bash
# 本仓库 fused_gdn_gating 源码清单
find csrc/ascend/attention/fused_gdn_gating -type f | sort
# 共 14 个文件：
#   fused_gdn_gating_torch_adpt.h
#   op_host/CMakeLists.txt
#   op_host/fused_gdn_gating_def.cpp / _infershape.cpp / _tiling.cpp / _tiling.h / _tiling_utils.h
#   op_host/op_api/aclnn_fused_gdn_gating.{cpp,h} / fused_gdn_gating.{cpp,h}
#   op_kernel/fused_gdn_gating.{cpp,h} / fused_gdn_gating_tiling_data.h

# 与参考分支 appleinsky/qwen36_dense_moe（R4/R8）逐文件 diff 核对
git diff <reference-commit> --stat -- csrc/ascend/attention/fused_gdn_gating/
```

**对比结论**：`csrc/ascend/attention/fused_gdn_gating/` 下算子源码文件与参考分支**完全一致**（来源 R4，报告中注明）。

### 2.2 框架补丁差异（本地裁剪版）

```bash
# 必需 op 清单（5 个 AscendC 核心 op）
grep -n "_REQUIRED_OPS" -A 8 vllm_fl/dispatch/backends/vendor/ascend/patches/patch_qwen3_6_gdn.py
#    "npu_causal_conv1d_custom",
#    "npu_fused_gdn_gating",
#    "npu_recurrent_gated_delta_rule",
#    "npu_gemma_rms_norm",
#    "npu_add_rms_norm_bias",
```

| 内容 | 参考分支 | 本地 patch |
|---|---|---|
| 5 个 AscendC 核心 op 接入（causal_conv1d / fused_gdn_gating / recurrent_gated_delta_rule / gemma_rms_norm） | ✅ | ✅ 保留 |
| PTO megakernel | 实验性路径 | 删除 |
| fused Triton decode kernel | 实验性路径 | 删除 |
| conv1d 权重缓存 | 实验性路径 | 删除 |
| `tests/ops/ascend/test_fused_gdn_gating.py` | — | 本地新增 |

**设计结论**：patch 仅保留 AscendC 核心接入，符合"单算子、最小改动、可独立回退"原则（`VLLM_FL_DISABLE_ASCENDC_GDN=1` 可整体回退 Triton 路径）。

---

## 3. 异常用例补充（TestFusedGdnGatingValidation，8 项）

覆盖 Torch Adapter（`fused_gdn_gating_torch_adpt.h`）全部 `TORCH_CHECK` 校验分支：

```bash
grep -n "TORCH_CHECK" csrc/ascend/attention/fused_gdn_gating/fused_gdn_gating_torch_adpt.h
```

| 行号 | 校验内容 | 错误文案 |
|---|---|---|
| 22 | `A_log` 1-D | `A_log should be 1-D [num_heads]` |
| 23 | `dt_bias` 1-D | `dt_bias should be 1-D [num_heads]` |
| 24 | `a` 2-D | `a should be 2-D [batch, num_heads]` |
| 25 | `b` 2-D | `b should be 2-D [batch, num_heads]` |
| 26-27 | `a`/`b` shape 一致 | `a and b must have the same shape` |
| 28-30 | `a`/`b` dtype 一致 | `a and b must have the same dtype` |
| 31-33 | `A_log`/`dt_bias` dtype 一致 | `A_log and dt_bias must have the same dtype` |
| 34-36 | num_heads 匹配 | `a second dim (num_heads) must equal A_log first dim` |

运行：

```bash
source /workspace/vllm-plugin-FL/vllm_fl/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash

# 全量 25/25 通过
pytest tests/ops/ascend/test_fused_gdn_gating.py -v

# 仅新增校验类
pytest tests/ops/ascend/test_fused_gdn_gating.py -k Validation -v
```

用例构成（合计 25 项，固定 `NUM_HEADS=32`、token 数 `[1,4,16,64]`、dtype `bf16/fp16`，`rtol=atol=1e-2`）：

| 测试类 | 用例 | 数量 |
|---|---|---|
| `TestFusedGdnGatingOutputs` | g/beta 对比 PyTorch 参考（token×dtype 参数化） | 8 |
| `TestFusedGdnGatingOutputs` | softplus 越阈退化路径（`a=50` 时 `g == -exp(A_log)*x`） | 1 |
| `TestFusedGdnGatingVsTriton` | 对比 vLLM Triton 基线（同上参数化） | 8 |
| `TestFusedGdnGatingValidation` | 8 个 `TORCH_CHECK` 异常分支 | 8 |

---

## 4. 模型级验证统一脚本接入（本周调整）

自研服务化压测脚本不再作为模型级验证主载体，统一改用项目组 Profiling 脚本（`/workspace/scripts/run_vllm_fl_profile_unified.sh`），理由：任务书要求"用 Profiler 证明算子实际进入推理路径"与"模型侧报告 TTFT、TPOT 或吞吐"，统一脚本的产物（`ASCEND_PROFILER_OUTPUT/op_statistic.csv` + `request_benchmark_results.txt`）正好完整覆盖。

### 4.1 脚本能力与任务书对应

| 任务书要求 | 统一脚本对应 |
|---|---|
| 模型级 1K 输入 / 1K 输出验证 | `--cases "1024,1024,1"`（输入长度,输出长度,请求数） |
| 用 Profiler 证明实际调用 | `--bench-profile true` → `ASCEND_PROFILER_OUTPUT/op_statistic.csv` 中 `npu_fused_gdn_gating` 出现次数与耗时 |
| 模型侧报告 TTFT、TPOT 或吞吐 | `request_benchmark_results.txt`（vllm bench serve 的 Serving Benchmark Result 摘要） |
| AscendC/Triton 双路径 | AscendC 默认；Triton 在外层 `export VLLM_FL_DISABLE_ASCENDC_GDN=1`（子进程继承） |

### 4.2 已核对项（无需模型）

```bash
# 1) 参数解析（已执行）：--help 输出与 unify_sh.md 一致，含 --cases/--bench-profile/--tp/--gmem/--devices/--flaggems-ops
/workspace/scripts/run_vllm_fl_profile_unified.sh --help

# 2) 参数校验失败路径（可验证项，未执行）：
/workspace/scripts/run_vllm_fl_profile_unified.sh --mode invalid --cases "1,1,1" --package none
# → [ERROR] --mode must be graph or eager

# 3) 环境变量透传（脚本实现核对结论，未实测）：脚本内置 source CANN 环境并 export 一堆变量，
#    外部 export 的变量（VLLM_FL_DISABLE_ASCENDC_GDN / VLLM_FL_GDN_COUNT_FILE）
#    由 setsid 启动的 vllm serve 子进程继承 → AscendC/Triton 双路径可直接复用
```

### 4.3 TP 约束（本周确认的约束）

- GDN 层 conv 状态维度 `conv_dim = 8192`，vLLM 要求能被 TP 整除 → `8192 % 3 = 2`，**TP3 结构性不可行**。
- 27B 用 `--tp 1`；35B 用 `--tp 2`（TP2 下 8192 可整除）。
- 35B 建议 `--gmem 0.6` 起步（prefill 激活 OOM 时降到 0.55）。

### 4.4 正式用法（待 27B/35B 权重就位后执行）

```bash
# AscendC 路径（默认）
/workspace/scripts/run_vllm_fl_profile_unified.sh \
  --model-path /models/Qwen3.6-27B --model-tag qwen3.6-27b \
  --mode eager --chunked true --devices 0 --tp 1 \
  --gmem 0.6 --cases "1024,1024,1" --bench-profile true \
  --run-label ascendc_r1

# Triton 对照路径（外层 export 开关即可）
export VLLM_FL_DISABLE_ASCENDC_GDN=1
/workspace/scripts/run_vllm_fl_profile_unified.sh \
  --model-path /models/Qwen3.6-27B --model-tag qwen3.6-27b \
  --mode eager --chunked true --devices 0 --tp 1 \
  --gmem 0.6 --cases "1024,1024,1" --bench-profile true \
  --run-label triton_r1

# 35B（TP2）
/workspace/scripts/run_vllm_fl_profile_unified.sh \
  --model-path /models/Qwen3.6-35B-A3B --model-tag qwen3.6-35b-a3b \
  --mode eager --chunked true --devices 0,1 --tp 2 \
  --gmem 0.6 --cases "1024,1024,1" --bench-profile true \
  --run-label ascendc_r1
```

### 4.5 结果归档与二次打包

```bash
# 每次 run 自动写入最新目录指针
cat /workspace/results/latest_run_dir.txt

# 二次打包算子统计（op_statistic.csv / api_statistic.csv / step_trace_time.csv / request_benchmark_results.txt）
/workspace/scripts/package_op_statistic.sh "$(cat /workspace/results/latest_run_dir.txt)"
# → /workspace/results/{RUN_NAME}_op_statistic.tar.gz
```

### 4.6 0.6B 小模型手动分步冒烟（备用排查路径，截至 2026-08-25 未执行）

按 `手动profiling测试.md` 用 Qwen3-0.6B 手动分步验证（统一脚本内部流程的手动展开，见 week1_save.md §5.5）：

```bash
# 终端 1：手动启动 serve（含手动 export 环境变量 + --no-async-scheduling）
export PYTHONPATH=/workspace/vllm-plugin-FL:$PYTHONPATH
export VLLM_PLUGINS=fl VLLM_FL_PLATFORM=ascend GEMS_VENDOR=ascend
export VLLM_ATTENTION_BACKEND=TORCH_SDPA ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
export VLLM_FL_FLAGOS_WHITELIST=unquantized_fused_moe_method,topk_softmax
vllm serve /models/Qwen3-0.6B --served-model-name qwen0.6b --trust-remote-code \
  --tensor-parallel-size 4 --host 0.0.0.0 --port 8080 --gpu-memory-utilization 0.6 \
  --compilation-config '{"cudagraph_mode":"FULL",...}' --enable-chunked-prefill \
  --max-model-len 32768 --max-num-seqs 1 \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"/workspace/new_results/qwen0.6b_basic/torch_profile",...}' \
  --no-async-scheduling

# 终端 2：curl 检查 → warmup（128,128,2,4）→ 分步 case（32,32,1 / 32,128,1 / 4096,4096,1，各 --profile）
# 终端 1：Ctrl+C 停服 → 终端 2：手动 torch_npu.profiler.profiler.analyse
```

用途：① 权重未就位（或 27B/35B 加载成本高）时的快速冒烟；② 统一脚本一键失败时，按此流程逐步定位故障环节（serve 启动 / warmup / 具体 case / analyse）。

---

## 5. 下周计划

1. **单算子最小改动提交**：整理 `fused_gdn_gating` 提交（与项目组确认提交基线与 PR 方式）。
2. **0.6B 手动分步冒烟**：按 `手动profiling测试.md` 用 Qwen3-0.6B 跑通手动 serve → curl → warmup → 分步 case → analyse 全流程（故障排查路径就绪）。
3. **模型级 1K/1K 验证（统一脚本）**：27B/35B 权重已就位，`--cases "1024,1024,1" --bench-profile true` 双路径，从 `request_benchmark_results.txt` 取 TTFT/TPOT/吞吐，从 `op_statistic.csv` 取 `npu_fused_gdn_gating` 调用证据，`package_op_statistic.sh` 二次打包；失败时按 §4.6 手动分步定位。
4. **W3 交付物**：按 W3 目标推进算子级测试交付物（算子级 Microbenchmark）。

---

## 6. 关键文件速查表

| 用途 | 路径 |
|---|---|
| 算子源码（与 R4 一致） | `csrc/ascend/attention/fused_gdn_gating/` |
| 构建脚本 | `csrc/ascend/build_aclnn.sh`、`csrc/ascend/build/cann-ops-transformer-custom_linux-aarch64.run` |
| 框架接入 patch（本地裁剪版） | `vllm_fl/dispatch/backends/vendor/ascend/patches/patch_qwen3_6_gdn.py` |
| 算子级固定测试（25/25） | `tests/ops/ascend/test_fused_gdn_gating.py` |
| 连通性测试 | `tests/custom_ops_tests/test_fused_gdn_gating.py` |
| Profiling 统一脚本 | `/workspace/scripts/run_vllm_fl_profile_unified.sh`、`/workspace/scripts/package_op_statistic.sh` |
| 手动分步 profiling 说明 | `手动profiling测试.md`（统一脚本流程的手动展开，0.6B 冒烟/故障排查用） |
| 脚本说明文档 | `unify_sh.md` |
| 结果目录 | `/workspace/results/`（`latest_run_dir.txt` 指向最新 run） |
| 参考分支 | `appleinsky/qwen36_dense_moe`（R4/R8） |

---

## 第 3 周（原 week3_save.md）

> 原对应周报：`week3.md`（已并入 `week_report.md`）
> 人员：龚昊磊（容器用户名：ghl）
> 目标：整理 fused_gdn_gating 单算子最小改动提交；在 27B/35B 权重就位后完成模型级 1K 输入/1K 输出验证（统一脚本 + Profiler 证据，AscendC/Triton 双路径）；推进 W3 算子级测试交付物（算子级 Microbenchmark）。

## 0. 前置环境

- 硬件：Ascend 910B3 × 8（本机 3 卡可用时 TP3 不可用，见 §2.3）
- 软件：Python 3.11、torch 2.8.0 + torch_npu 2.8.0.post2、vllm 0.13.0、CANN 8.5/9.0、flag_gems 5.0.2
- 模型权重（本周就位，`/models/` 挂载盘）：
  - `/models/Qwen3.6-27B`（64 层，48 个 GDN 层，15 个 safetensors）
  - `/models/Qwen3.6-35B-A3B`（40 层，30 个 GDN 层，26 个 safetensors，67GB）
- 关键产物位置：
  - torch 绑定库：`vllm_fl/_C_ascend.cpython-311-aarch64-linux-gnu.so`
  - CANN 自定义算子包：`vllm_fl/_cann_ops_custom/vendors/custom_transformer/`
  - 算子源码：`csrc/ascend/attention/fused_gdn_gating/`（14 文件）
  - Profiling 统一脚本：`/workspace/scripts/run_vllm_fl_profile_unified.sh`、`/workspace/scripts/package_op_statistic.sh`
  - 结果目录：`/workspace/results/`（`atp_*` run 目录，`latest_run_dir.txt` 指向最新 run）
  - 算子级微基准（本周新增，已移入个人脚本目录）：`/workspace/scripts/ghl/benchmark_fused_gdn_gating.py`
  - 结果存档（2026-08-25 已归档）：`/workspace/results/ghl/20260820_算子级微基准/`、`/workspace/results/ghl/20260820_模型级基准_自研脚本/`

---

## 1. 单算子最小改动提交整理（本周任务 1）

### 1.1 提交基线确认

```bash
git merge-base HEAD origin/main        # 906fa07e93809c9d910468df41592f1205f6b3b7
git log --oneline -3 origin/main       # main 最新 344e42b
git ls-files csrc/ascend | wc -l       # 主干上 csrc/ascend/ 仅 1 个占位 CMakeLists.txt
```

**结论**：提交基线为项目组 `main` 分支。当前工作分支 `add-qwen3_6_ascendc_gdn_ops` 上，Ascend 后端源码整体未入库（untracked），需要按"单算子 + 必要公共改动"最小集合组织提交。

### 1.2 算子源码与参考分支 R4 一致性核对

```bash
for f in $(find csrc/ascend/attention/fused_gdn_gating -type f | sort); do
  git cat-file -e "appleinsky/qwen36_dense_moe:$f" 2>/dev/null \
    && diff -q <(git show "appleinsky/qwen36_dense_moe:$f") "$f" >/dev/null 2>&1 \
    && echo "IDENTICAL $f"
done
# 14 个文件全部 IDENTICAL（来源 R4，commit ab33933，2026-07-20）
```

### 1.3 最小改动文件清单

| 类别 | 文件 | 说明 |
|---|---|---|
| 算子本体 | `csrc/ascend/attention/fused_gdn_gating/`（14 文件） | 与 R4 完全一致（op_host/op_kernel/op_api/torch_adpt） |
| schema 注册 | `csrc/ascend/torch_binding.cpp:51`（include）+ `:2687-2696`（`ops.def`/`ops.impl`） | `npu_fused_gdn_gating(Tensor A_log, a, b, dt_bias, float beta=1.0, float threshold=20.0) -> (g, beta_output)` |
| 构建清单 | `csrc/ascend/build_aclnn.sh:31`（CUSTOM_OPS 含 `fused_gdn_gating`） | CANN framework 算子打包 |
| 构建公共 | `csrc/CMakeLists.txt`（+152）、`csrc/ascend/CMakeLists.txt`（+659） | 扩展分发器 + CANN 算子工程（与其他同学共用） |
| 环境自举 | `vllm_fl/__init__.py`（+46，`_bootstrap_cann_custom_op_env`） | 免 source 加载自定义算子包（共用） |
| 框架接入 | `vllm_fl/dispatch/backends/vendor/ascend/patch.py`（`patch_qwen3_6_gdn()`） | 注册 patch（共用） |
| 框架接入 | `vllm_fl/dispatch/backends/vendor/ascend/patches/patch_qwen3_6_gdn.py`（本地裁剪版，577 行 vs R8 907 行） | `_REQUIRED_OPS` 第 87 行 + `_forward_core` 第 386 行挂钩 `npu_fused_gdn_gating`；本周新增默认关闭的 `VLLM_FL_GDN_COUNT_FILE` 计数钩子（`_install_gdn_gating_counter`） |
| 文档 | `vllm_fl/dispatch/backends/vendor/ascend/patches/README.md`（+1 行） | patch 说明 |
| 固定测试 | `tests/ops/ascend/test_fused_gdn_gating.py`（25 用例，本地新增） | — |
| 连通性测试 | `tests/custom_ops_tests/test_fused_gdn_gating.py`（本地新增） | — |
| 模型级验证 | 统一脚本 `/workspace/scripts/run_vllm_fl_profile_unified.sh`（`--cases "1024,1024,1"`）+ `/workspace/scripts/ghl/benchmark_model_gdn_1k.py`（补充调用计数等） | 1K/1K 双路径验证 |
| 算子级微基准 | `/workspace/scripts/ghl/benchmark_fused_gdn_gating.py`（本周新增） | — |

> 提交方式：与项目组确认后以 PR 方式提交到 `main`；工作树中含其他同学的同名 patch 公共改动，提交时仅 stage 本人范围 + 必要公共改动。

### 1.4 调用计数钩子（本周新增，默认关闭）

```bash
# patch_qwen3_6_gdn.py 内 _install_gdn_gating_counter()：
# 设置 VLLM_FL_GDN_COUNT_FILE=<path> 后，每次真实调用 npu_fused_gdn_gating
# 都会计数，每 1024 次及进程退出时追加 "<pid> <count>" 行到文件。
# 不设置时为零开销 no-op，不改变任何现有行为。

# 验证（单进程 2050 次调用 → 文件中 1024 + 1026）：
VLLM_FL_GDN_COUNT_FILE=/tmp/gdn_count_test.txt VLLM_PLUGINS=fl python -c "
import torch, torch_npu, vllm_fl._C_ascend
from vllm_fl.dispatch.backends.vendor.ascend.patches.patch_qwen3_6_gdn import patch_qwen3_6_gdn
patch_qwen3_6_gdn()
A_log = torch.randn(32, dtype=torch.float32, device='npu:0')
a = torch.randn(4, 32, dtype=torch.bfloat16, device='npu:0')
b = torch.randn(4, 32, dtype=torch.bfloat16, device='npu:0')
dt_bias = torch.randn(32, dtype=torch.float32, device='npu:0')
for _ in range(2050):
    torch.ops._C_ascend.npu_fused_gdn_gating(A_log, a, b, dt_bias)
"
```

---

## 2. 模型级 1K 输入 / 1K 输出验证（本周任务 2，统一脚本）

### 2.1 运行方式（统一脚本，无需手工 source 环境）

> ✅ 本轮模型级验证**已完成**（2026-08-25）：27B/35B 双路径 4 个 run 数据齐备（§2.2/§2.3）。27B AscendC 提供 Profiler 算子调用证据（`op_statistic.csv`）；35B 因磁盘/时间约束以 `--bench-profile false` 跑性能对照（Profiler 证据以 27B 为准）。失败排查路径 §2.5 保留备用（0.6B 冒烟未执行）。

统一脚本 `run_vllm_fl_profile_unified.sh` 内置 source CANN 环境并设置全部 vLLM-FL 环境变量（含 FlagGems 白名单 `VLLM_FL_FLAGOS_WHITELIST=unquantized_fused_moe_method,topk_softmax`，规避 flag_gems 5.0.2 的 pow 算子 Triton kernel 编译失败）。只需两条前置保证：

```bash
# 1) AscendC 路径（默认，走 npu_fused_gdn_gating）
# 2) Triton 对照：外层 export 开关（子进程继承），不需要任何脚本改动
export VLLM_FL_DISABLE_ASCENDC_GDN=1
```

### 2.2 27B 结果（Qwen3.6-27B，64 层 / 48 GDN 层）

> 实际执行（2026-08-25，NPU4，`--gmem 0.9 --max-model-len 4096`；gmem 0.6 会因 27B bf16 权重 ~50GiB + KV 预算不足报 `No available memory for the cache blocks`，见 §3 踩坑记录）：

```bash
# AscendC 路径（Profiler 证据 run，--bench-profile true）
/workspace/scripts/run_vllm_fl_profile_unified.sh \
  --model-path /models/Qwen3.6-27B --model-tag qwen3.6-27b \
  --mode eager --chunked true --devices 4 --tp 1 \
  --gmem 0.9 --max-model-len 4096 --cases "1024,1024,1" \
  --bench-profile true --run-label ascendc_r1 --port 8113 --package none

# Triton 对照路径（外层置 VLLM_FL_DISABLE_ASCENDC_GDN=1；性能对照，--bench-profile false）
export VLLM_FL_DISABLE_ASCENDC_GDN=1
/workspace/scripts/run_vllm_fl_profile_unified.sh \
  --model-path /models/Qwen3.6-27B --model-tag qwen3.6-27b \
  --mode eager --chunked true --devices 4 --tp 1 \
  --gmem 0.9 --max-model-len 4096 --cases "1024,1024,1" \
  --bench-profile false --skip-analyse --run-label triton_r1 --port 8113 --package none
```

**结果（request_benchmark_results.txt，1 请求 1024 in / 1024 out，eager + chunked）**：

| 指标 | AscendC（profiler run） | Triton（对照） |
|---|---|---|
| Benchmark duration (s) | 247.96（tqdm） | 167.83 |
| 生成吞吐 Output tok/s | ~4.3（server.log 引擎日志稳态值） | 6.10 |
| Mean TTFT (ms) | 未测得* | 922.18 |
| Mean TPOT (ms) | 未测得* | 163.15 |
| Mean ITL (ms) | — | 162.99 |

> *AscendC profiler run 因 profiler 导出阶段脚本卡住（trace_view.json 达 15.5GB，CANN 解析 11m43s），`vllm bench serve` 未打印结果块即挂起——总耗时/吞吐从 `request_terminal.log`（tqdm）与 `server.log`（引擎日志）提取，TTFT/TPOT 无法补测（数据已在 run 目录 `request_benchmark_results.txt` 如实标注）。性能对照以 Triton（--bench-profile false）为基准。

**调用证据（op_statistic.csv + 计数钩子双证据）**：

```bash
# 1) Profiler 证据：ASCEND_PROFILER_OUTPUT/op_statistic.csv 中
#    npu_fused_gdn_gating 的出现次数 / device 侧耗时（--bench-profile true 时产出）
# 2) patch 计数钩子：run 时外层 export VLLM_FL_GDN_COUNT_FILE=/tmp/gdn_count_27b.txt，
#    汇总后与 profiler 次数、理论值（48 GDN 层 × 1024 decode 步 = 49152）交叉核对
```

### 2.3 35B 结果（Qwen3.6-35B-A3B，40 层 / 30 GDN 层，TP2）

**TP 选择约束**（重要）：GDN 层 conv 状态维度 `conv_dim = head_k_dim*num_k_heads*2 + head_v_dim*num_v_heads = 8192`，vLLM 要求其能被 TP 整除（`gated_delta_net_state_shape` 内 `divide(conv_dim, tp_size)`）。8192 % 3 = 2，故 **TP3 不可行**（历史 35B 跑批用 TP4，本机可用卡不足时取 TP2）。

```bash
# AscendC（性能对照；实际执行 devices 4,5 / gmem 0.7 / max-model-len 4096）
/workspace/scripts/run_vllm_fl_profile_unified.sh \
  --model-path /models/Qwen3.6-35B-A3B --model-tag qwen3.6-35b-a3b \
  --mode eager --chunked true --devices 4,5 --tp 2 \
  --gmem 0.7 --max-model-len 4096 --cases "1024,1024,1" \
  --bench-profile false --skip-analyse --run-label ascendc_r1 --port 8113 --package none

# Triton 对照（外层置 VLLM_FL_DISABLE_ASCENDC_GDN=1，同参数）
export VLLM_FL_DISABLE_ASCENDC_GDN=1
/workspace/scripts/run_vllm_fl_profile_unified.sh \
  --model-path /models/Qwen3.6-35B-A3B --model-tag qwen3.6-35b-a3b \
  --mode eager --chunked true --devices 4,5 --tp 2 \
  --gmem 0.7 --max-model-len 4096 --cases "1024,1024,1" \
  --bench-profile false --skip-analyse --run-label triton_r1 --port 8113 --package none
```

**结果（request_benchmark_results.txt，1 请求 1024 in / 1024 out，eager + chunked，TP2）**：

| 指标 | AscendC | Triton |
|---|---|---|
| Benchmark duration (s) | 180.18 | 193.43 |
| 生成吞吐 Output tok/s | 5.68 | 5.29 |
| Mean TTFT (ms) | 1058.51 | 1125.14 |
| Mean TPOT (ms) | 175.09 | 187.98 |

> 说明：35B 两个 run 均为 `--bench-profile false`（磁盘/时间约束，仅性能对照），无 op_statistic 证据——Profiler 调用证据以 27B AscendC run 为准（调用计数理论核对值：30 GDN 层 × 2 rank × 1024 decode 步 = 61440，未单独 profiler 核验）。与 8/20 自研脚本基线（35B AscendC 3.42 / Triton 4.60 tok/s）**方向相反**：本次 35B AscendC（5.68）快于 Triton（5.29）约 7%，可能与 TP2 + gmem0.7 配置及脚本差异有关，待多次重复数据确认。

**踩坑记录（2026-08-25 实测确认）**：
1. `8192 is not divisible by 3`：GDN conv 状态维度 8192 无法被 TP3 整除（`distributed/utils.py:55` 的 `ensure_divisibility`）——TP3 对 35B 结构性不可行（本次 35B 实测 TP2 成功）。
2. gmem 预算：27B bf16 权重 ~50GiB，`--gmem 0.6` 会报 `No available memory for the cache blocks`，实测 `--gmem 0.9 --max-model-len 4096` 成功；35B（TP2，权重每卡 ~33.5GB）实测 `--gmem 0.7` 成功。
3. flag_gems 5.0.2 pow 算子编译失败：统一脚本内置 `--flaggems-ops unquantized_fused_moe_method,topk_softmax` 白名单，加载日志确认 `Enable only the following ops: [...]`，无需手工设置（本次 4 个 run 均正常加载）。
4. `--bench-profile true` profiler 导出卡死（27B AscendC run）：trace_view.json 达 15.5GB、CANN 解析 11m43s，`vllm bench serve` 结果块未打印即挂起 → 性能对照一律 `--bench-profile false --skip-analyse`（快、无此问题）；任何 profiler 采集前先查磁盘余量，GB 级导出文件用完即删。

### 2.4 结果归档（package_op_statistic.sh）

```bash
# 每次 run 自动写入最新目录指针；二次打包算子统计
/workspace/scripts/package_op_statistic.sh "$(cat /workspace/results/latest_run_dir.txt)"
# → /workspace/results/{RUN_NAME}_op_statistic.tar.gz
#    内含 op_statistic.csv / api_statistic.csv / step_trace_time.csv / request_benchmark_results.txt
```

### 2.5 失败排查路径（手动分步，对应 `手动profiling测试.md`）

统一脚本一键跑失败（如 27B AscendC 首次运行 51s exit 1）时，按 `手动profiling测试.md` 手动分步复现定位故障环节（该文档 = 统一脚本内部流程的手动展开，见 week1_save.md §5.5）：

```bash
# 0) 先确认残留进程与结果目录：latest_run_dir.txt 指向的 run 目录、
#    server.log / client.log 是否已创建（失败环节的初步线索）

# 1) 小模型冒烟（排除流程问题）：Qwen3-0.6B 手动 serve + warmup + 分步 case + analyse
#    （/models/Qwen3-0.6B，结果目录 /workspace/new_results/qwen0.6b_basic）

# 2) 27B 复现：手动 serve（终端 1，含 --no-async-scheduling）→
#    curl /v1/models（终端 2，确认服务是否真正就绪）→
#    warmup → 分步 case（1024,1024,1 --profile）→ Ctrl+C → analyse
#    逐环节对比统一脚本日志，定位是 serve 启动 / warmup / case / analyse 哪一步失败
```

排查要点：统一脚本 exit 1 前 `server.log` 最后几行（vllm serve 是否启动成功/加载模型是否报错）、`client.log` 的 `[WAIT_SERVER]`/`[SERVER_READY]`/`[ERROR]` 标记、失败时是否已进入 warmup/case 阶段。

---

## 3. 算子级 Microbenchmark（本周任务 3，W3 交付物）

### 3.1 脚本设计

`/workspace/scripts/ghl/benchmark_fused_gdn_gating.py`（符合任务书：性能脚本含预热、同步、重复测试）：

- 三后端：`ascendc`（`npu_fused_gdn_gating`）/ `triton`（vLLM `fused_gdn_gating` 基线）/ `torch`（PyTorch 参考实现）
- 规模：token 数 1/4/16/64/256/1024 × bf16/fp16，固定 `NUM_HEADS=32`（27B/35B GDN 层，TP1）
- 每次调用前后 `torch.npu.synchronize(device)` 单独同步（只同步目标设备，避免带起其他卡粘性错误），warmup 100 次 + 测量 500 次
- 正确性校验先行：g/beta 与 PyTorch 参考对比（rtol/atol=1e-2，与固定测试一致），全部通过后才计时
- 输出：min/mean/P50/P90/P99/max（µs）+ AscendC/Triton mean 比值

```bash
# 运行（选空闲卡；--device 支持 npu:N）
source /workspace/vllm-plugin-FL/vllm_fl/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash
python /workspace/scripts/ghl/benchmark_fused_gdn_gating.py \
    --backends ascendc,triton,torch --tokens 1,4,16,64,256,1024 \
    --dtypes bf16,fp16 --reps 500 --device npu:0 \
    > /workspace/results/ghl/20260820_算子级微基准/fused_gdn_gating_microbench_20260820.txt
```

### 3.2 结果要点（36 项配置全部通过正确性校验）

| 后端 | tokens=1 | tokens=1024 | 备注 |
|---|---|---|---|
| AscendC | ~90µs | ~88µs | 不随 token 数增长（kernel 内 1 次 AIV 调用） |
| Triton 基线 | ~200µs | ~360µs | 随 token 数增长 |
| PyTorch 参考 | ~370µs | ~340µs | 逐元素算子，无融合 |

- **AscendC/Triton mean 比值 0.26x–0.44x**（AscendC 快 2.3–3.8 倍），tokens=1024 时最快（0.26x ≈ 3.8 倍）
- 退出码：0 正常 / 3 op 未注册 / 4 无 NPU

---

## 4. 模型级 vs 算子级性能差异（如实记录，下周用 Profiler 数据定位）

| 层 | 27B AscendC | 27B Triton | 35B AscendC | 35B Triton |
|---|---|---|---|---|
| 算子级 mean 时延 | ~85-90µs | ~200-360µs | —（同算子） | — |
| 模型级 1K/1K 生成吞吐 | ~4.3 tok/s* | 6.10 tok/s | 5.68 tok/s | 5.29 tok/s |
| 模型级 TPOT | 未测得* | 163.15ms | 175.09ms | 187.98ms |

> *27B AscendC 为 profiler run（`--bench-profile true`，有 profiler 开销；TTFT/TPOT 因导出卡死未打印），其余三个 run 均为 `--bench-profile false` 性能对照。

**实测结论（2026-08-25，如实记录）**：
- **27B**：模型级 AscendC（~4.3 tok/s，profiler run）慢于 Triton（6.10），与算子级方向相反（AscendC 快 2.3-3.8 倍）；与 8/20 自研脚本基线（AscendC 4.07/4.52 vs Triton 5.32）方向一致。严格对比需补无 profiler 的 27B AscendC 对照。
- **35B**：模型级 AscendC（5.68）略快于 Triton（5.29，约 +7%），方向与算子级一致；但与 8/20 基线（3.42 vs 4.60）相反——配置（TP2/gmem0.7）与脚本差异可能导致，待多次重复数据确认。
- **疑似开销因素（待定位）**：decode 路径 `npu_recurrent_gated_delta_rule` AscendC 算子或 mamba state 布局转换（`(Hv,Dv,Dk)` ↔ `(Hv,Dk,Dv)` 转置、dense KV 重组）引入额外开销；`op_statistic.csv` 显示 GDN 全路径（FusedGdnGating 0.448% + RecurrentGatedDeltaRule 2.005% + CausalConv1d 0.893% + chunk 系列 kernel）合计约 3.7% device 耗时，后续用 `api_statistic.csv` / `trace_view.json` 深挖。

---

## 5. 下周计划

1. 完成最终 PR 提交（按 §1.3 最小改动集合提交到 main 基线，与项目组确认方式）。
2. （2026-08-25 已完成 1K/1K 双路径验证，见 §2.2/§2.3；原"27B AscendC exit 1 排查"已定位——实为 `--gmem 0.6` 权重装不下，见 §2.3 踩坑 #2）后续：按需补跑多次重复稳定性数据（当前各路径各 1 次）；用统一脚本 profiler 产物定位 27B 模型级 AscendC 与 Triton 差异（§4）；0.6B 冒烟作答辩/流程复验素材（§2.5，可选）。
3. 完善个人技术报告（`../docs/intern_ops/`，3-5 页：调用链、编译、测试、性能、限制）与答辩材料。

---

## 6. 所遇问题

- **模型级 1K/1K 验证已完成（2026-08-25）**：27B/35B 双路径 4 个 run 全部完成（§2.2/§2.3），27B AscendC 提供 Profiler 调用证据（`op_statistic.csv`：FusedGdnGating 49152 次 / 0.448%）。遗留问题：27B AscendC profiler run 因 profiler 导出阶段脚本卡住（trace_view.json 15.5GB、CANN 解析 11m43s）未打印 vllm bench 结果块，TTFT/TPOT 未测得（总耗时/吞吐从 request_terminal.log 与 server.log 提取，已在 `request_benchmark_results.txt` 标注）；性能对照 run 全部 `--bench-profile false --skip-analyse`，未复现卡住。模型级 AscendC vs Triton 方向在 27B（AscendC 慢）与 35B（AscendC 快）上不一致，见 §4。
- **flag_gems 5.0.2 pow 算子编译失败**：模型加载时 `base ** arange`（RoPE 频率计算）经 flag_gems 的 pow Triton kernel 报 `TypeError: cannot convert None to tensor`；统一脚本内置 `--flaggems-ops` 白名单规避（加载日志可见 `Enable only the following ops: [...]`，与历史 atp 配置一致）。
- **TP3 对 35B 结构性不可行**：GDN conv 状态维度 8192 无法被 3 整除（vLLM `ensure_divisibility` 断言）。
- **并发跑批瞬态 aivec 错误**：模型加载/生成期间其他卡上的进程出现 `MTE DDR out of range` aivec 错误（错误上报为 Device:0，模型日志本身 0 错误，模型结果完整）；不可复现，疑似跨卡同步时的粘性错误上报。

---

## 7. 关键文件速查表

| 用途 | 路径 |
|---|---|
| 算子源码（与 R4 一致，14 文件） | `csrc/ascend/attention/fused_gdn_gating/` |
| schema 注册 | `csrc/ascend/torch_binding.cpp:2687-2696` |
| 框架接入 patch（本地裁剪版 + 计数钩子） | `vllm_fl/dispatch/backends/vendor/ascend/patches/patch_qwen3_6_gdn.py` |
| 固定测试（25 用例） | `tests/ops/ascend/test_fused_gdn_gating.py` |
| 连通性测试 | `tests/custom_ops_tests/test_fused_gdn_gating.py` |
| Profiling 统一脚本（模型级 1K/1K） | `/workspace/scripts/run_vllm_fl_profile_unified.sh` |
| 二次打包脚本 | `/workspace/scripts/package_op_statistic.sh` |
| 手动分步 profiling（排查路径） | `手动profiling测试.md`（0.6B 冒烟 / 统一脚本失败定位） |
| 算子级微基准（本周新增，个人脚本） | `/workspace/scripts/ghl/benchmark_fused_gdn_gating.py` |
| 模型级补充脚本（个人脚本） | `/workspace/scripts/ghl/benchmark_model_gdn_1k.py` |
| 微基准结果存档 | `/workspace/results/ghl/20260820_算子级微基准/fused_gdn_gating_microbench_20260820.txt` |
| 27B/35B 模型级日志 | `/workspace/results/ghl/20260820_模型级基准_自研脚本/model_27b_*.log`、`model_35b_*.log` |
| 模型权重 | `/models/Qwen3.6-27B`、`/models/Qwen3.6-35B-A3B` |
| 参考分支 | `appleinsky/qwen36_dense_moe`（R4 commit ab33933 / R8） |
