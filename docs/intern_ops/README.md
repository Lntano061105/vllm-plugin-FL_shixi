# docs/intern_ops — MoE Init Routing Custom 算子接入

本目录为 vLLM-Plugin-FL 自定义算子接入实习任务（MoE Init Routing Custom）
的提交物归档：个人报告、完整技术报告、验证附录与性能 baseline。

## 一页结论

| 维度 | 结论 |
|---|---|
| 接入范围 | `MoeInitRoutingCustom`（aclnn 框架算子）→ torch 接口 `_C_ascend::npu_moe_init_routing_custom` → vllm-plugin-FL OOT 注册链路 |
| 正确性 | 真实权重逐层对拍 MAX_ABS_DIFF = 4.88e-04；serve 输出与参考余弦相似度 ≈ 0.99999；边界用例通过 |
| 生效证据 | serve 日志 640 次 `[ASCENDC_IMPL]` 调用；`VLLM_FL_OOT_WHITELIST / BLACKLIST` 开关可回退 |
| 性能 baseline | 27B graph 490.34 / 35B graph 326.11 / 27B eager 293.21 / 35B eager 268.38（Output tok/s，128/0） |
| 瓶颈 | GDN triton kernel 单层 26–29 ms，非本算子；本次接入收益 = 可运行 + 数值正确 + 可回退 |
| 单测 | `tests/custom_ops_tests/test_moe_init_routing_custom.py`（7 用例） |

## 环境基线

| 项 | 值 |
|---|---|
| 硬件 | Ascend 910B3 × 8（baseline 使用 4 卡 TP=4） |
| CANN | 9.0.0 |
| Python / torch | 3.11.14 / 2.11.0 + torch_npu |
| vLLM | 0.20.2 |
| 构建 | `VLLM_VENDOR=ascend pip install -e . --no-build-isolation` |
| 运行前 | `source vllm_fl/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash` |

## 文件索引

| 文件 | 内容 |
|---|---|
| [moe_init_routing_custom_report.md](moe_init_routing_custom_report.md) | 个人报告（任务目标 / 环境与构建 / 接入验证 / 正确性 / 性能 / 规范自检 / 贡献边界） |
| [moe_init_routing_custom_full_report.md](moe_init_routing_custom_full_report.md) | 完整技术报告（7 章 + 附录，含逐层插桩与排错细节） |
| [appendix_verification.md](appendix_verification.md) | 验证附录（参考实现、对拍脚本、日志摘录） |
| [baseline_qwen36_4cases.md](baseline_qwen36_4cases.md) | 四项 Baseline 汇总（09-15 批次，含运行目录与补跑说明） |

## 复现入口

```bash
# 单测（需先 source set_env.bash）
python3 tests/custom_ops_tests/test_moe_init_routing_custom.py

# 四项 baseline
bash benchmarks/ops/ascend/run_command.sh
python3 benchmarks/ops/ascend/analysis.py --result-root /workspace/results
```
