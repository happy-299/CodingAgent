项目名称：Minimal Coding Agent
Git 仓库地址：https://github.com/happy-299/CodingAgent

运行方法：需要 Python 3.10+。在项目根目录的 .env 中配置 OPENAI_API_KEY、OPENAI_BASE_URL=https://api.deepseek.com、OPENAI_MODEL=deepseek-v4-flash，然后执行 python3 coding_agent.py；也可执行 python3 coding_agent.py "任务描述" 运行一次性任务。运行测试：python3 -m unittest discover -v。

特色功能：不使用任何 Agent 框架或 SDK，直接调用模型原生 tool calling。智能体自行维护多轮对话、模型推理内容、工具调用和工具结果；只向模型提供一个通用终端，由模型通过标准命令完成目录浏览、代码搜索、读写文件、构建和测试，降低工具选择与提示词开销。模型可反复观察结果、修改代码并验证，直到给出最终答复。

可靠性设计：终端从工作区启动且不继承 Key、Token、Secret 等敏感环境变量；工具参数和 JSON 调用均校验；长输出自动截断；命令超时时终止子进程组；API 临时故障自动重试；最大循环数防止失控。上下文按完整用户回合裁剪，不破坏 assistant/tool 消息配对。命令执行并非操作系统沙箱，应只在可信项目中使用。
