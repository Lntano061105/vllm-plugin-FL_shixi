# MoE Init Routing Custom 算子接入 —— 实习个人报告

- 姓名：邝珈慧（PerseusM34）
- 任务：vLLM-Plugin-FL 自定义算子接入（MoE Init Routing Custom）
- 周期：2026-07 ～ 2026-09（约 6 周）
- 验收：2026-09-18

---

## 1. 任务目标

在 vLLM-Plugin-FL（Ascend 910B3 后端）上完成 MoE Init Routing 自定义算子的接入与验证：

1. 将 CANN 框架算子 `MoeInitRoutingCustom` 接入插件运行链路，torch 侧接口为
   `torch.ops._C_ascend.npu_moe_init_routing_custom`；
2. 打通 AscendC 构建链，产出可在容器内直接运行的算子包；
3. 完成正确性对拍（CPU 参考实现 + 真实权重）与 serve 路径生效验证；
4. 输出 Qwen3.6-27B / 35B-A3B 在 graph / eager 两种模式下的吞吐 baseline，并定位性能瓶颈。

算子语义：输入 token 张量 `x [num_rows, hidden]` 与 `expert_idx [num_rows, top_k]`，输出
展开后的 token、行索引映射（gather map）、每专家 token 数/累加量，支持
Dropless / Drop-Pad 两种模式与 CUMSUM / COUNT / KEY_VALUE 三种计数类型。

## 2. 环境与构建

| 项 | 值 |
|---|---|
| 硬件 | Ascend 910B3 × 8（baseline 用 4 卡 TP=4） |
| CANN | 9.0.0 |
| Python / torch | 3.11.14 / 2.11.0 + torch_npu |
| vLLM | 0.20.2 |
| 工作仓 | `/workspace/vllm-plugin-FL.bak.20260817`，分支 `kjh/add-qwen3_6_ascendc_gdn_ops` |

构建链上定位并修复的阻塞点：

| # | 阻塞现象 | 处理 |
|---|---|---|
| 1 | 构建期依赖隔离报错 | 改用 `pip install -e . --no-build-isolation` 在已激活的 CANN 环境内构建 |
| 2 | `csrc/ascend` 的 aclnn 目标缺失 / 命令错误 | 修正 `CMakeLists.txt` 目标与 `build_aclnn.sh` 调用 |
| 3 | 头文件找不到（`acl_base_rt.h`） | 补齐 AscendC 头文件包含路径 |
| 4 | `exponential_` 在 int64 上的类型不匹配 | 修正 dtype 后重新构建 |
| 5 | 真机对拍阶段双重转置导致 59GB 显存申请失败 | 去掉冗余转置，改为按块拷贝 |
| 6 | 运行时版本不匹配（编译期 / 运行期 CANN 版本） | 统一算子包环境脚本，运行时 `source set_env.bash` |

## 3. 接入与生效验证

- 算子经 `torch_binding.cpp` 注册为 `_C_ascend::npu_moe_init_routing_custom`，
  与 `vllm_fl/ops/custom_ops.py::register_oot_ops()` 的 OOT 注册机制对接；
- 通过 `VLLM_FL_OOT_WHITELIST / VLLM_FL_OOT_BLACKLIST` 开关控制启用与回退；
- serve 路径验证：推理日志中出现 640 次 `[ASCENDC_IMPL]` 调用记录，说明请求实际走到了
  AscendC 实现而非参考实现；
- 输出一致性：serve 端输出与参考实现余弦相似度 ≈ 0.99999。

## 4. 正确性验证

| 验证项 | 方法 | 结果 |
|---|---|---|
| CPU 参考对拍 | 以纯 torch 参考实现（Dropless / Drop-Pad + unpermute）为 golden | 通过 |
| 真实权重对拍 | Qwen3.6 真实权重逐层比对 | MAX_ABS_DIFF = 4.88e-04 |
| 边界用例 | top-k 全零、部分专家为空、单专家全量路由 | 通过，无 NaN / 越界 |
| 端到端 | serve 输出与参考余弦相似度 | ≈ 0.99999 |

单测落库：`tests/custom_ops_tests/test_moe_init_routing_custom.py`，覆盖
token 展开 / 专家排序 / 索引恢复 / 计数四类校验点与 3 类边界用例。

## 5. 性能 Baseline

条件：Ascend 910B3 × 4（TP=4），cases `1024,1024,128`，concurrency 64，
max-num-seqs 64，max-model-len 8192，gmem 0.6，chunked prefill 开启，graph 模式为 PIECEWISE。

| 模型 | 模式 | Output tok/s | Total tok/s | Mean TTFT (ms) | Mean TPOT (ms) | 成功/失败 |
|---|---|---|---|---|---|---|
| Qwen3.6-27B | graph | **490.34** | 980.68 | 8441.34 | 121.33 | 128/0 |
| Qwen3.6-35B-A3B | graph | **326.11** | 652.22 | 13842.02 | 181.88 | 128/0 |
| Qwen3.6-27B | eager | **293.21** | 586.41 | 7821.11 | 209.24 | 128/0 |
| Qwen3.6-35B-A3B | eager | **268.38** | 536.76 | 14831.91 | 222.60 | 128/0 |

- graph 相对 eager：27B 吞吐 +67.2%、35B +21.5%，Mean TPOT 分别下降 42.0% / 18.3%。
- 四组均 128 请求成功、0 失败。

瓶颈定位（逐层插桩）：主要耗时不在本算子，而在 vLLM 原生 GDN triton kernel，
单层约 26–29 ms；本算子在整网耗时中占比很小，因此本次接入的收益体现为
"可运行 + 数值正确 + 支持回退"，而非吞吐提升。

## 6. 代码规范自检（导师 7 项）

| # | 规范项 | 落地情况 |
|---|---|---|
| 1 | 命名 | 新增文件与算子目录沿用既有 `moe_init_routing_custom` / `op_host` / `op_kernel` 命名，脚本与 csv 为 snake_case |
| 2 | 风格 | 新增 `.py` / `.sh` 与仓库既有风格对齐；Python 侧不做仓库未启用的强制格式化 |
| 3 | 注释文档 | 算子适配头、单测、脚本均带用途说明；本目录报告 + `benchmarks/ops/ascend/README.md` 说明参数与复现步骤 |
| 4 | 目录 | 单测入 `tests/custom_ops_tests/`，脚本入 `benchmarks/ops/ascend/`，报告入 `docs/intern_ops/` |
| 5 | 提交 | 提交信息用 `feat/test/bench/docs(ascend): ...` 前缀；单测、脚本、文档分目录提交，不混入生成物 |
| 6 | 质量 | 单测覆盖边界用例；脚本绝对路径参数化（`--fl-repo-path` / `--result-root`）；无调试残留 |
| 7 | 依赖 | 环境清单（vllm / torch / CANN / 构建命令）固化在本报告与 `benchmarks/ops/ascend/README.md` |

## 7. 贡献边界与遗留问题

**可主张**：算子整合适配与构建链打通排错、真实权重正确性对拍、serve 路径生效验证、
四项 Baseline 压测与汇总、瓶颈定位、单测与脚本补交。

**需谨慎表述**：`csrc/ascend` 算子源码与 `vllm_fl` 注册 / 开关机制在 main 中由团队
上游提交（`f4c54af`、`6be3b62`），本人分支在这两个目录无增量改动。

**遗留问题**：

1. 09-03 批次 graph 模式出现 63/128 请求失败，失败点不在 custom op 调用路径
   （日志无算子侧 Traceback）；09-15 同配置重跑 4 组均 128/0，未复现，作为遗留项披露；
2. 09-11 首次 27B graph 跑批在 KV cache 分配阶段 OOM（卡组 0-3），换卡组 4-7 后通过；
3. 性能提升的进一步空间在 GDN triton kernel，不属于本次算子接入范围。

**复现入口**：`benchmarks/ops/ascend/run_command.sh`（四条 baseline 命令）、
`benchmarks/ops/ascend/analysis.py`（结果汇总）。
