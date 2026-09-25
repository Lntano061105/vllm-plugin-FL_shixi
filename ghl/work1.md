vLLM-Plugin-FL 自定义算子接入、编译部署与模型推理验证
项目周期	6周（自任务发布之日起）	适用对象	项目组实习生
目标模型	 Qwen3.6 27B / 35B	提交方式	技术报告 + 汇报

1. 背景
本项目面向 vLLM-Plugin-FL 的框架级推理优化。实习生需要复现项目组已有的自定义算子接入工作，理解算子从 csrc 源码、CANN/AscendC 工程、CMake 编译、Python 扩展注册到 vLLM 模型执行路径的完整链路，并在项目指定的模型上完成调用验证。重点不是重新设计复杂算法，而是掌握可复用、可测试、可部署的算子工程方法。
每名实习生独立负责一个算子。参考分支可用于理解实现，但不得直接整体复制或合并实验分支；最终提交应基于项目组指定基线，以单算子、最小改动和可独立回退为原则。
2. 学习目标
• 能够定位算子在 Qwen 推理计算图中的位置，并说明其输入、输出、数据类型和数据布局。
• 能够将已有 AscendC/CANN 算子工程接入 vLLM-Plugin-FL 的 csrc、构建系统和 Python 调用层。
• 能够在干净环境中完成编译、安装和动态库加载，并通过固定输入输出测试验证正确性。
• 能够通过开关控制自定义算子与基线实现，使用 Profiler 证明算子实际进入模型推理路径。
• 能够形成可审查的 Pull Request、测试脚本、性能数据和简洁技术报告。
3. 统一任务要求
模块	最低要求	说明
源码接入	算子进入 csrc	包含算子源码、Host/Tiling 或 Torch Adapter，以及必要的头文件和注册代码。
编译部署	可由项目构建流程安装	不得依赖手工复制 .so；重新打开 Shell 后仍能加载。
Python 调用	可通过 torch.ops 或项目封装调用	能输出算子注册名称、动态库路径和调用结果。
固定测试	至少完成一组项目规定的算子级固定 Shape 与输入输出测试，并完成模型级 1K 输入、1K 输出验证	算子级测试需明确各输入张量的 Shape、dtype 和数据布局，并与 基线比较数值结果；模型级测试统一采用输入长度 1024 tokens、最大输出长度 1024 tokens，验证算子调用、推理稳定性及性能。
框架接入	进入 Qwen 推理路径	保留基线回退开关，并用日志或 Profiler 证明实际调用。
性能验证	完成 Microbenchmark	报告平均时延及 P50/P90；模型侧报告与该算子相关的 TTFT、TPOT 或吞吐。
项目汇报	文档/PPT	具体时间与方式待定。
4. 提交物
提交物	建议位置	要求
算子实现	csrc/ascend/...	只包含本人负责算子及必要公共改动。
单元测试	tests/ops/ascend/	可独立运行，固定随机种子，包含参考实现。
性能脚本	benchmarks/ops/ascend/	包含预热、同步和重复测试。
框架补丁	vllm_fl/...	提供启用/禁用开关及回退路径。
个人报告	docs/intern_ops/	建议 3–5 页，说明调用链、编译、测试、性能和限制。
5. 个人任务分配
序号	姓名	负责算子	适用路径	固定测试重点	参考
1	龚昊磊	Fused GDN Gating	27B/35B GDN	固定 Token 数和 Head 数，对比 g 与 beta 输出。	R4、R8
2	邓永权	Add RMSNorm Bias / GemmaRMSNorm	27B/35B 公共路径	对比归一化输出、rstd 和 residual。	R5、R8
3	刘国新	Causal Conv1D	27B/35B GDN	分别完成 Prefill 与 Decode 固定用例，检查状态缓存。	R6、R8
4	赵小苇	Recurrent Gated Delta Rule	27B/35B Decode	比较输出及调用前后 State，检查未选中槽位。	R7、R8
5	李泽泓	Chunk Gated Delta Rule	27B/35B Prefill	变长序列、初始 State、最终 State 和实际长度输入。	R3、R9
6	刘鑫	MoE Gating Top-K	35B MoE Router	校验 Top-K 权重、Expert ID、归一化与重复专家。	R10
7	邝珈慧	MoE Init Routing Custom	35B MoE Routing	校验 Token 展开、专家排序、索引恢复和计数总和。	R11
若成员的 C++、CANN 或模型结构基础存在明显差异，可在第1周结束前调整人员与算子的对应关系，但原则上保持“一人一算子、独立分支、独立测试、独立答辩”。
6. 评价方式
评价项	权重	主要依据
工程接入与构建部署	30%	代码能够从项目构建流程编译、安装、加载和回退。
正确性与测试质量	20%	固定用例、异常用例、参考实现和状态验证完整。
框架集成与性能结果	20%	模型调用证据清晰，性能方法合理，结论不过度推断。
代码质量、文档与答辩	30%	能够独立解释调用链和问题定位。
7. 参考材料
R4 参考实现：Fused GDN Gating https://github.com/appleinsky/vllm-plugin-FL/tree/qwen36_dense_moe/csrc/ascend/attention/fused_gdn_gating
R8 参考实现：Qwen GDN Python 接入补丁 https://github.com/appleinsky/vllm-plugin-FL/blob/qwen36_dense_moe/vllm_fl/dispatch/backends/vendor/ascend/patches/patch_qwen3_6_gdn.py
8. 学术与工程规范
允许参考公开代码，但必须在个人报告中注明来源和改动范围。禁止把参考分支整体合并后作为个人成果，禁止伪造性能数据或仅展示一次运行的最优值。所有结果应能够由其他成员按照提交说明复现。遇到环境问题时，应先记录软件版本、硬件型号、完整命令和最小错误日志，再在项目组中提问。