# Minimal Coding Agent

一个约 400 行、只使用 Python 标准库的编程智能体。它直接调用 DeepSeek 的 OpenAI 兼容接口，没有使用 Agent 框架、Agent SDK、托管代码执行或文件 API；对话历史、原生 tool calling、本地执行、结果回传、循环终止和错误恢复均由项目自行实现。

GitHub：https://github.com/happy-299/CodingAgent

## 从“很多小工具”到一个通用终端

早期模型不擅长稳定组合命令，给它 `list_files`、`read_file`、`write_file`、`search_files` 等窄工具，能够降低行动难度。但随着模型的代码能力、Shell 能力和长上下文能力增强，这些工具越来越像对 `ls`、`rg`、`sed`、Python 脚本和重定向的薄封装：它们增加了工具描述占用、工具选择分支和实现维护量，却没有增加模型真正能做的事情。

因此本项目选择 **terminal-first**：只向模型暴露一个 `terminal(command, timeout)`。终端不是“功能少”，而是一种可组合的统一接口。模型可以用同一工具探索仓库、读取与修改代码、运行测试、查看 Git diff，并把前一步输出自然地组合到下一步。宿主机已有的编译器、测试框架和项目脚本也能直接复用，不需要为每一种生态重新设计工具协议。

这种选择不是越少越好。新增工具需要满足至少一个条件：终端无法表达该能力；命令行实现经过评测仍频繁失败；或能力需要独立的授权/隔离边界，例如浏览器交互、图片理解、人工审批。当前高难度评测没有出现这些证据，所以工具仍保持为一个，优化集中在 terminal 的执行语义和 agent 循环。

## 工作方式

```text
用户任务
  -> DeepSeekClient：发送完整上下文和 terminal schema
  -> 模型返回原生 tool_calls
  -> CodingAgent：校验参数并执行 TerminalTool
  -> 结构化回传 exit_code / output / output_chars / truncated
  -> 模型继续观察、修改、测试
  -> 无 tool_calls 时结束；达到边界时恢复或安全终止
```

核心模块都在 [`coding_agent.py`](coding_agent.py)：

- `DeepSeekClient`：用 `urllib` 调用 `/chat/completions`，处理认证、超时、限流及临时服务错误重试。
- `CodingAgent`：保存 user/assistant/tool 历史，保留 DeepSeek `reasoning_content`，解析原生工具调用并驱动观察-行动循环。
- `TerminalTool`：在工作区启动 Bash/PowerShell，合并输出，返回结构化结果；POSIX 管道启用 `pipefail`。
- `Config`：从环境变量或 `.env` 读取配置并校验范围；Key 不进入仓库。

## 与考核要求的对应关系

| 要求 | 实现 |
|---|---|
| 对话历史与上下文管理 | 保留多轮消息、工具调用及结果；按完整用户回合裁剪，不产生孤立 tool 消息 |
| 工具定义与本地执行 | 自行定义原生 function schema；`subprocess` 在本地工作区执行唯一 terminal 工具 |
| 模型输出解析 | 校验 tool 名称、JSON 参数、类型和异常，并按 `tool_call_id` 回传结构化结果 |
| 循环终止 | 模型返回最终文本时结束；最大轮数硬停止；输出达到 token 上限时保留上下文自动续写 |
| 错误处理 | API 重试、配置校验、命令超时、进程组终止、非零退出码、长输出截断、未知工具错误回传 |

终端子进程不会继承名称包含 Key、Token、Secret、Password、Credential 的敏感环境变量。命令超时时会终止整个子进程组，避免遗留后台进程。它仍不是操作系统级沙箱，因此应在可信仓库中使用，并在提交前检查 diff。

## 运行

需要 Python 3.10+，无需安装依赖。

```bash
cp .env.example .env
# 编辑 .env，填入 OPENAI_API_KEY
python3 coding_agent.py
```

一次性任务与指定工作区：

```bash
python3 coding_agent.py "阅读项目并修复测试"
python3 coding_agent.py --workspace /path/to/project "实现需求并运行测试"
```

使用 `--workspace` 时优先读取启动目录的 `.env`，凭据无需复制到目标仓库。交互模式支持 `/clear` 和 `/exit`。

配置示例：

```dotenv
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_API_KEY=your-deepseek-api-key
OPENAI_MODEL=deepseek-v4-flash
AGENT_API_TIMEOUT=180
AGENT_MAX_ROUNDS=20
AGENT_MAX_HISTORY_CHARS=300000
AGENT_MAX_OUTPUT_TOKENS=32768
```

## 测试与演进

```bash
python3 -m unittest discover -v
```

15 项单元测试覆盖 terminal、上下文、模型输出解析和 agent 循环。真实 DeepSeek V4 Flash 端到端评测使用了一个多模块持久化 Task API：包括线程安全、原子 JSON 持久化、HTTP CRUD、严格输入校验及隐藏并发测试。第一次运行暴露输出上限直接终止；修复自动续写与输出预算后，可见测试 4/4、隐藏测试 3/3 通过。随后增量实现持久化删除与 HTTP DELETE，可见测试 6/6、隐藏测试 3/3 通过。详细过程见 [`EVALUATION.md`](EVALUATION.md)。

## 已知边界

- terminal 能执行任意本地命令，不提供容器级隔离或逐命令人工审批。
- 当前 API 调用为非流式；工具事件实时显示，但模型生成中的 token 不逐字输出。
- 上下文按字符近似控制，不依赖特定模型 tokenizer；这保持零依赖，但不等同于精确 token 计数。
