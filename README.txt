项目名称：Minimal Coding Agent
Git 仓库：https://github.com/happy-299/CodingAgent

运行：需要 Python 3.10+，无需安装依赖。在 .env 中配置 DeepSeek API Key，执行 python3 coding_agent.py 进入交互模式；也可执行 python3 coding_agent.py --workspace /path/to/project "任务"。测试命令为 python3 -m unittest discover -v。

设计：早期模型需要 list/read/write/search 等窄工具帮助行动；随着大模型的代码、Shell 和长上下文能力增强，这些工具逐渐成为 ls、rg、sed、Python 的重复封装，并增加 schema 占用与选择成本。因此本项目只提供一个可组合的 terminal，让模型直接复用真实开发环境完成探索、编辑、构建和测试。只有终端无法表达、评测持续失败或需要独立授权边界时才新增工具。

实现：项目不使用 Agent 框架、SDK 或托管代码执行，只用标准库直连 DeepSeek 原生 tool calling。自行维护多轮历史、reasoning_content、工具调用与结果；按完整回合裁剪上下文。terminal 返回结构化退出码、输出长度和截断状态，启用 pipefail，不继承敏感环境变量，超时终止进程组。API 支持重试；输出达到 token 上限可保留上下文续写；最大轮数保证终止。

验证：15 项框架测试全部通过。真实 V4 Flash 完成多模块 Task API，覆盖线程安全、原子 JSON 持久化、严格 HTTP 校验及错误映射；最终可见测试 6/6、隐藏并发与协议测试 3/3 通过。评测先后推动了输出上限恢复、实时日志、结构化 terminal 结果和管道失败传播。
