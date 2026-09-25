# 容器环境使用规范

# 代码修改权限

1. 允许修改的范围

    - `vllm-plugin-fl`（FL 插件源码）

    - `flaggems`（FlagGems 源码）

    - 其他已通过 `pip install -e .` 安装的可编辑 Python 包

2. 禁止修改的内容

    - 系统目录（`/usr/`、`/etc/`、`/opt/` 等）

    - Ascend 驱动与固件（`/usr/local/Ascend/`）

    - 其他用户目录及共享模型目录 `/models`

---

# 测试结果持久化规范

1. 结果存放路径：所有测试输出必须保存在 `/workspace/results/`（挂载自宿主机 `/data2/shixisheng/results/`）

2. 目录组织：`/workspace/results/你的用户名/YYYYMMDD_测试描述/`

3. 必存环境信息（便于复测）：

    - 测试时间、人员

    - vLLM / FL插件 / FlagGems 版本（commit ID）

    - CANN 版本、驱动版本、NPU 信息（`npu-smi info`）

    - 完整运行命令及关键环境变量

    - 模型名称及路径

---

# 文件删除安全规范

1. 严禁以下操作：

    - ❌ `rm -rf /*`

    - ❌ `rm -rf *`

    - ❌ `rm -rf ./*`

    - ❌ 任何未指定明确路径的通配符删除

2. 安全删除要求：

    - 必须指定至少一层明确路径，推荐使用绝对路径

    - 删除前先用 `ls` 确认目标内容

3. bash

正确rm \-rf /workspace/results/your\_name/temp\_cache/rm \-f /workspace/scripts/old\_test\.py\# 错误（禁止）rm \-rf \./temp/\*rm \-rf \*

---

# 容器使用基本规范

1. 每个用户使用分配的容器（端口 3222\~3228），不得切换他人容器

2. 模型文件 `/models` 为只读挂载，不可修改

3. 个人脚本放 `/workspace/scripts/你的用户名/`

4. 禁止随意 `apt install` 或修改系统配置

5. 使用 `exit` 正常退出容器

---

# 违规处理

违反安全约定导致环境损坏或影响他人的：

- 首次：书面警告并修复环境

- 多次：暂停容器使用权限

---

请每位使用者仔细阅读并遵守，共同维护稳定的研发环境。