# Minimal Coding Agent

一个不依赖 Agent 框架或 SDK 的简洁编程智能体。它直接调用 DeepSeek 的 OpenAI 兼容 Chat Completions API，并自行实现对话历史、原生 tool calling、文件操作、命令执行、循环终止及错误处理。

## 运行

需要 Python 3.10+。项目只使用标准库，无需安装依赖。

```bash
cp .env.example .env  # 已有 .env 时跳过
# 在 .env 中填入 OPENAI_API_KEY
python3 coding_agent.py
```

也可执行一次性任务：

```bash
python3 coding_agent.py "阅读当前项目并修复测试"
python3 coding_agent.py --workspace /path/to/project "实现一个 TODO 应用"
```

使用 `--workspace` 时会优先读取启动目录的 `.env`，因此可以把凭据留在本项目中，不必复制到目标代码仓库。

交互模式支持 `/clear` 清空历史、`/exit` 退出。

## 功能与设计

- 工具：只暴露一个通用终端。模型使用 `ls`、`rg`、`sed`、Python、测试及构建命令完成所有本地工作，减少工具选择和提示词开销。
- 循环：模型请求工具后，本地执行并把结果按 `tool_call_id` 回传，直到模型给出最终答复。
- 上下文：保存多轮对话、工具调用及 DeepSeek `reasoning_content`；按完整用户回合裁剪，避免破坏工具消息配对。
- 可靠性：终端从指定工作区启动且不继承 Key/Token/Secret 等敏感环境变量；参数和 JSON 均校验；输出截断，超时时终止整个子进程组，API 对限流和临时故障重试，循环轮数有上限。

注意：`terminal` 会执行模型给出的本地命令。请仅在可信代码仓库中运行并在提交前检查改动；它不是操作系统级沙箱。

## 配置

```dotenv
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_API_KEY=your-key
OPENAI_MODEL=deepseek-v4-flash
AGENT_API_TIMEOUT=180
AGENT_MAX_ROUNDS=20
AGENT_MAX_HISTORY_CHARS=300000
```

## 测试

```bash
python3 -m unittest discover -v
```

测试使用脚本化假模型验证工具循环，不消耗 API；实际模型可用一次性任务进行端到端测试。
