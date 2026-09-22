# 附录

## 附录 A 修改文件与回退清单

| 类型 | 文件 | 说明 | 回退方式 |
|---|---|---|---|
| 构建 | CMakeLists.txt | 修复 csrc 构建排错项，新增 custom op 编译目标 | 恢复 CMakeLists.txt.bak.* |
| 构建 | build_aclnn.sh | 修复构建脚本，接入 aclnn 包构建链 | 恢复 build_aclnn.sh.bak.* |
| 构建 | _C_ascend 重编译 | 重新编译生成 _C_ascend.so（16.9MB） | 重跑构建链覆盖 |
| 插件 | vllm_plugin_fl custom op 接入（早期开发阶段，非本次提交内容） | MoE Init Routing 自定义算子注册与调用路径 | git 独立分支回退 |
| 插件 | serve 钩子（早期开发阶段，非本次提交内容） | 精度钩子（cos≈0.99999 验证）与 [ASCENDC_IMPL] 日志 | git 独立分支回退 |

所有修改均位于主仓库独立分支，可一键回退；构建产物均有 .bak 备份。

## 附录 B 复现步骤

```bash
# 1. 进入容器（Ascend 910B3）
ssh -p 3228 root@127.0.0.1
# 2. 激活独立 venv（vllm 0.20.2 / torch_npu 2.11.0 / CANN 9.0.0）
source /workspace/venvs/vllm0202/bin/activate
# 3. 进入插件仓库，安装 editable 模式（需已 source CANN 环境）
cd <repo_root> && VLLM_VENDOR=ascend pip install -e . --no-build-isolation
# 4. 构建 aclnn 包（脚本位于 csrc/ascend/，末尾自动安装到 vllm_fl/_cann_ops_custom）
bash csrc/ascend/build_aclnn.sh
# 5. 运行正确性验证（CPU 参考 / NPU 冒烟 / 真实权重）
#    脚本与预期输出见 /workspace/results/邝珈慧/ 对应日期目录
# 6. 启动 serve（当前仓库模型侧仍为 routing_v2 路径）
#    早期开发阶段曾在隔离环境以 serve 钩子观察 [ASCENDC_IMPL] 计数与精度输出，见 4.4/4.5 节（非本次提交内容）
```

## 附录 C 数据归档清单

- `/workspace/results/邝珈慧/` 下按日期分目录，包含：
  - 构建日志（pip / aclnn / _C_ascend）
  - CPU 参考与 NPU 冒烟输出
  - 真实权重对拍日志（MAX_ABS_DIFF、cos）
  - serve 压测与 profiling 日志（TPOT、GDN kernel 占比）
  - 阶段8 对拍与 profiling 汇总

## 附录 D 术语表

| 术语 | 含义 |
|---|---|
| GDN | Gated Delta Net，gated delta rule 线性注意力网络 |
| TPOT | Time Per Output Token，单 token 生成耗时 |
| DBO | Delta-Boost 相关状态（shared_experts 场景） |
| ASCENDC_IMPL | 自定义算子实现分支的运行时标记 |
| cos | 余弦相似度，精度对齐指标 |
| aclnn | Ascend CANN 算子库封装包 |
| _C_ascend | Ascend 侧 C 扩展编译产物（.so） |
