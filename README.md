# Minimal Coding Agent

一个只使用 Python 标准库、保持实现紧凑的编程智能体。它直接调用 DeepSeek 的 OpenAI 兼容接口，没有使用 Agent 框架、Agent SDK、托管代码执行或文件 API；对话历史、原生 tool calling、本地执行、结果回传、循环终止和错误恢复均由项目自行实现。

GitHub：https://github.com/happy-299/CodingAgent

## 从“很多小工具”到一个通用终端

早期模型不擅长稳定组合命令，给它 `list_files`、`read_file`、`write_file`、`search_files` 等窄工具，能够降低行动难度。但随着模型的代码能力、Shell 能力和长上下文能力增强，这些工具越来越像对 `ls`、`rg`、`sed`、Python 脚本和重定向的薄封装：它们增加了工具描述占用、工具选择分支和实现维护量，却没有增加模型真正能做的事情。

因此本项目选择 **terminal-first**：主要执行工具是 `terminal(command, timeout)`，另外只提供一个不执行外部操作的 `update_plan` 状态工具。终端不是“功能少”，而是一种可组合的统一接口；规划状态则用于让 agent 显式维护任务进度。模型可以用同一终端探索仓库、读取与修改代码、运行测试、查看 Git diff，并把前一步输出自然地组合到下一步。

这种选择不是越少越好。新增工具需要满足至少一个条件：终端无法表达该能力；命令行实现经过评测仍频繁失败；或能力需要独立的授权/隔离边界，例如浏览器交互、图片理解、人工审批。当前高难度评测没有出现这些证据，所以执行工具仍保持为一个；优化集中在 terminal 的执行语义、规划状态、上下文管理和 agent 循环。

评测任务只用于暴露通用故障模式。进入运行时的优化必须由协议或执行信号触发，例如 `finish_reason`、退出码、超时、输出大小和轮数；任务名称、固定文件、特定命令及 verifier 规则不得进入系统提示词或控制分支。具体任务仅记录在评测文档中，不参与 agent 决策。

## 工作方式

```text
用户任务
  -> DeepSeekClient：发送完整上下文和 terminal schema
  -> 模型返回原生 tool_calls
  -> CodingAgent：校验参数；更新计划或执行 TerminalTool
  -> 结构化回传 exit_code / output / output_bytes / truncated
  -> 模型继续观察、修改、测试
  -> 计划完成或首次最终回答时触发一次基于证据的完成审计
  -> 计划完成后结束；达到边界时恢复或安全终止
```

核心模块都在 [`coding_agent.py`](coding_agent.py)：

- `DeepSeekClient`：用 `urllib` 调用 `/chat/completions`，处理认证、超时、限流及临时服务错误重试。
- `CodingAgent`：保存 user/assistant/tool 历史，保留 DeepSeek `reasoning_content`，维护有界工作记忆和计划，解析原生工具调用并驱动观察—行动循环；探索后会提醒模型自主判断是否规划，计划长期未更新时会触发进度检查点。
- `update_plan`：只更新宿主内的有限计划状态，不访问文件系统、不执行命令；它与 terminal 的职责严格分离。
- `TerminalTool`：在工作区启动 Bash/PowerShell，合并输出，返回结构化结果；POSIX 管道启用 `pipefail`。输出先写匿名临时文件，只读取有限头尾，内存不会随日志量线性增长。
- `TerminalUI`：零依赖渲染固定高度 TUI，底部固定输入/状态栏，活动区支持鼠标滚轮回看；每轮思考以时间线卡片呈现，推理时在卡片内自动滚动，正式输出开始后自动收起；工具名称、命令、结果和输出按卡片组织，并渲染常用 Markdown；非 TTY 自动使用纯文本模式。
- `Config`：从环境变量或 `.env` 读取配置并校验范围；Key 不进入仓库。

## 与考核要求的对应关系

| 要求 | 实现 |
|---|---|
| 对话历史与上下文管理 | 保留多轮消息、工具调用及结果；按完整协议单元压缩，保留当前任务、计划和最近证据，不产生孤立 tool 消息 |
| 工具定义与本地执行 | 自行定义原生 function schema；`subprocess` 在本地工作区执行唯一 terminal 工具，`update_plan` 仅维护状态 |
| 模型输出解析 | 校验 tool 名称、JSON 参数、类型和异常，并按 `tool_call_id` 回传结构化结果 |
| 规划与完成判断 | 模型先侦察再根据任务复杂度自主决定是否规划；计划停滞时重新对齐原始验收标准，计划完成后立即审计证据，计划项必须全部完成或明确阻塞 |
| 流式交互 | 以 SSE 增量读取 reasoning/content/tool calls；命令事件和模型增量均实时渲染，工具调用仍在完整组装后执行 |
| 循环终止 | 模型返回最终文本并通过完成审计时结束；输出达到 token 上限时保留上下文自动续写；默认 80 轮仅作为安全边界，达到上限后禁用工具并给予一次诚实收尾机会 |
| 错误处理 | API 重试、配置校验、命令超时、进程组终止、非零退出码、长输出截断、未知工具错误回传 |

终端子进程不会继承名称中含独立 Key、Token、Secret、Password、Credential 标识或常见凭据后缀的敏感环境变量，同时避免误伤普通单词中的相同字母片段。命令超时时会终止整个子进程组，避免遗留后台进程。它仍不是操作系统级沙箱，因此应在可信仓库中使用，并在提交前检查 diff。

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

交互模式默认启用适合 Demo 录制的固定布局终端界面，展示模型状态、工具调用卡片、完整命令、有界命令输出和 Markdown 回答。每一次模型推理都会在对应输出之前生成独立思考卡片：推理进行时，卡片保持固定高度并在内部自动滚动；正式输出开始后，卡片自动收起并保留在时间线原位。输入 `/thinking` 可展开或收起最近一轮思考，鼠标滚轮可浏览完整活动记录。脚本调用或需要稳定纯文本时可使用：

```bash
python3 coding_agent.py --plain "阅读项目并运行测试"
```

录制 Demo 时建议将终端宽度设为至少 120 列，然后运行 `python3 coding_agent.py`。选择一个能产生文件修改和测试验证的任务，即可展示完整的观察—行动闭环。

配置示例：

```dotenv
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_API_KEY=your-deepseek-api-key
OPENAI_MODEL=deepseek-v4-flash
AGENT_API_TIMEOUT=180
AGENT_MAX_ROUNDS=80
AGENT_MAX_HISTORY_CHARS=160000
AGENT_MAX_OUTPUT_TOKENS=32768
AGENT_REASONING_EFFORT=medium
```

## 测试与演进

```bash
python3 -m unittest discover -v
```

48 项单元测试覆盖 terminal、终端 UI、配置、上下文、模型输出解析、agent 循环、规划检查点和流式协议，其中包含固定布局、逐轮思考时间线、Markdown、中文边框对齐和滚动行为测试。真实 DeepSeek V4 Flash 除完成三个修复类任务外，还从六个全空目录执行了同一句 2048 高层任务；完整日志和独立行为验证暴露了范围膨胀、重复验证、“只看退出码”的假阳性、文本化工具协议和上下文成本问题，并推动了推理预算、规划节奏、证据检查、协议恢复和分层上下文的通用优化。详细过程见 [`EVALUATION.md`](EVALUATION.md)。

## 已知边界

- terminal 能执行任意本地命令，不提供容器级隔离或逐命令人工审批。
- API 使用 SSE 流式读取；思考、回答和工具调用增量会实时显示，同时在服务端事件结束后完整组装 tool call 再执行。
- 上下文按字符近似控制，不依赖特定模型 tokenizer；这保持零依赖，但不等同于精确 token 计数。
- 每次 terminal 调用是独立 shell；需要临时服务时，应在一条命令中完成启动、验证和清理。
