项目名称：Minimal Coding Agent
Git 仓库：https://github.com/happy-299/CodingAgent

项目简介

Minimal Coding Agent 是一个简单而完整的编程智能体。项目使用 Python 标准库直连 DeepSeek 原生 tool calling，让模型在真实工作区自主完成编程任务。

如何运行

在 .env 中配置 OPENAI_API_KEY 后运行：

python3 coding_agent.py

指定项目：

python3 coding_agent.py --workspace /path/to/project

设计特色

1. Terminal-first：最简洁而通用的工具

大模型已能熟练组合 Shell 命令。项目以 terminal(command, timeout) 统一完成探索、读写、检索、运行和验证，以单一可组合接口兼顾简洁与通用。update_plan 是可选状态工具，只维护结构化待办与进度。

2. 由模型自主决定是否规划

规划不是固定步骤。模型按复杂度和未知信息自行判断：简单任务直接执行，复杂任务调用 update_plan 拆解目标并随证据更新。

3. 先探索理解，再采取行动

模型先以最小充分的侦察理解项目结构、现有实现、行为约定和测试入口，再确定修改范围。terminal 结果进入下一轮决策，形成“探索—行动—观察—调整”闭环。

4. TDD 驱动的工程闭环

Agent 从需求和既有测试提炼验收标准，先编写或复用能暴露问题的测试，再完成最小实现；随后根据测试证据迭代，结束前执行回归测试。测试持续指导设计、实现和完成判断。

5. 完整的 Agent 能力

Agent 具备规划、执行、观察、上下文管理和终止判断。它以工作记忆和历史压缩保留任务、计划与证据；terminal 结构化返回退出码、输出、截断和超时，使失败可被观察并恢复；完成审计依据验收标准和证据决定是否自然退出，轮次上限作为安全边界。SSE 实时展示思考、回答、计划和工具执行，并支持 API 重试、命令超时、输出截断和协议恢复。
