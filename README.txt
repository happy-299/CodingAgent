项目名称：Minimal Coding Agent
Git 仓库：https://github.com/happy-299/CodingAgent

项目简介

Minimal Coding Agent 是一个简单而完整的编程智能体，使用 Python 标准库直连 DeepSeek 原生 tool calling，在真实工作区自主完成编程任务。

如何运行

在 .env 中配置 OPENAI_API_KEY 后运行：

python3 coding_agent.py

指定项目：

python3 coding_agent.py --workspace /path/to/project

设计故事：把舞台还给模型

构建 Agent 时，人们容易不断增加工具、规则和提示，为每种任务铺设轨道。但强模型已经从训练中学会阅读代码、使用 Shell 和解决问题。我们因此反问：不是还能增加什么，而是最少需要给它什么？

答案是一把万能工具和一张可选地图。terminal(command, timeout) 是统一执行入口，模型在 Shell 中自由组合命令，完成探索、读写、运行与验证；单一接口让选择更简单、行动更连贯。update_plan 像一张可折叠地图，只维护待办与进度，需要时才由模型展开和调整。

所有引导都是能力选项，而非强制流程。除用户目标与安全边界外，探索深度、是否规划、使用哪些命令、是否新增测试、如何验证和何时结束，都由模型结合实际任务判断。提示词只提供原则，不枚举任务解法；Agent 鼓励先以最小充分的侦察理解项目，再采取行动。

测试是旅程的路标。Agent 以 TDD 思路提炼可失败的验收标准，由模型判断编写或复用测试，再完成最小实现并依据证据迭代。terminal 的结构化结果进入“探索—行动—观察—调整”循环；工作记忆和历史压缩保留任务、计划与证据，完成审计据此判断是否自然退出。

克制不等于简陋：SSE 流式界面、错误恢复和安全轮次边界保证过程清晰可靠。Minimal 不是能力更少，而是用更少的干预释放模型已有的能力。
