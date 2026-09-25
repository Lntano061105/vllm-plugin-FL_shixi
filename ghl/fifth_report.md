type: weekly-report
week: 2026-W37
project: "[[P-2026-002-实习生Ascend c算子接入项目]]"
owner:
- 龚昊磊
report_date: 2026-09-11
health: 🟢 正常推进
progress: 90

项目周报｜2026-W37

一、本周完成
完成技术报告定稿：更新报告日期与结论、新增交付物清单附录。新建答辩材料 ghl_reply.md，含 12 页汇报提纲、关键数据速查与 14 条 Q&A 预案。续写第五周周报，并归档 9/9~9/11 共享基线重测与 27b_graph 复测记录。

二、本周未完成及原因
PR 提交未执行：提交基线与方式待项目组确认。27B 模型级 AscendC 慢于 Triton 的差异未定位，需补无 profiler 对照与重复数据。27B graph 基线 5 次 PIECEWISE 复测全部 OOM，本环境取不到数据，需确认可用配置。

三、当前进度
当前进度：工程接入、编译部署、算子级与模型级验证、性能与 Profiler 证据、技术报告与答辩材料均已完成，项目进入收尾。
项目状态：🔵 进行中（选项：⚪ 未开始 / 🔵 进行中 / ⏸️ 已暂停 / ✅ 已完成）

四、下周计划
按排期完成答辩。项目组确认提交基线与方式后，按最小改动集合执行 PR 提交。若确认 27B graph 可用配置与资源，补 27B 模型级差异定位及 AscendC 无 profiler 对照数据。

五、问题与需要协调的事项
1. PR 提交基线与方式待项目组确认（最小改动集合已就绪，merge-base 906fa07）。2. 27B graph 基线在本环境不可复现，需项目组提供可用配置。3. 27B 差异定位需无 profiler 对照的卡资源。

六、相关材料
本周附件：/workspace/vllm-plugin-FL/ghl/ghl_report.md（技术报告定稿）、/workspace/vllm-plugin-FL/ghl/ghl_reply.md（答辩材料）、ghl/week_report.md（周报 week1~5）。复测归档：/workspace/results/ghl/20260911_27b_graph重测/。
