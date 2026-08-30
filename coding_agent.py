#!/usr/bin/env python3
"""A small, self-contained coding agent using native model tool calls."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


MAX_TOOL_OUTPUT = 20_000


class AgentError(Exception):
    """A user-facing agent error."""


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise AgentError(f"{name} must be an integer, got {raw!r}.") from exc
    if not minimum <= value <= maximum:
        raise AgentError(f"{name} must be between {minimum} and {maximum}, got {value}.")
    return value


def load_dotenv(path: Path) -> None:
    """Load a small, conventional subset of dotenv without another dependency."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if value[:1] == value[-1:] and value[:1] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


@dataclass(frozen=True)
class Config:
    api_key: str
    base_url: str
    model: str
    timeout: int = 180
    max_rounds: int = 20
    max_history_chars: int = 300_000
    max_output_tokens: int = 32_768

    @classmethod
    def from_env(cls) -> "Config":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise AgentError("OPENAI_API_KEY is missing; put it in .env or the environment.")
        return cls(
            api_key=api_key,
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com").rstrip("/"),
            model=os.getenv("OPENAI_MODEL", "deepseek-v4-flash"),
            timeout=_env_int("AGENT_API_TIMEOUT", 180, 1, 600),
            max_rounds=_env_int("AGENT_MAX_ROUNDS", 20, 1, 100),
            max_history_chars=_env_int("AGENT_MAX_HISTORY_CHARS", 300_000, 10_000, 10_000_000),
            max_output_tokens=_env_int("AGENT_MAX_OUTPUT_TOKENS", 32_768, 1_024, 384_000),
        )


def _clip(text: str, limit: int = MAX_TOOL_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    kept = limit // 2
    return text[:kept] + f"\n... <truncated {len(text) - 2 * kept} chars> ...\n" + text[-kept:]


class TerminalTool:
    """One general-purpose terminal rooted at the selected workspace."""

    def __init__(self, root: Path):
        self.root = root.resolve()

    def run(self, command: str, timeout: int = 60) -> dict[str, Any]:
        if not command.strip():
            raise AgentError("Command cannot be empty.")
        timeout = max(1, min(int(timeout), 300))
        argv = (
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command]
            if os.name == "nt"
            else ["/bin/bash", "-o", "pipefail", "-lc", command]
        )
        sensitive = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
        child_env = {
            key: value
            for key, value in os.environ.items()
            if not any(marker in key.upper() for marker in sensitive)
        }
        child_env["PYTHONUNBUFFERED"] = "1"
        popen_options: dict[str, Any] = {"start_new_session": True} if os.name != "nt" else {
            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP
        }
        process = subprocess.Popen(
            argv,
            cwd=self.root,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=child_env,
            **popen_options,
        )
        try:
            output, _ = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
            raise AgentError(f"Command timed out after {timeout}s.\n{_clip(output or '')}") from exc
        output = output or ""
        return {
            "exit_code": process.returncode,
            "output": _clip(output) if output else "<no output>",
            "output_chars": len(output),
            "truncated": len(output) > MAX_TOOL_OUTPUT,
        }


TOOL_SCHEMAS: list[dict[str, Any]] = [{
    "type": "function",
    "function": {
        "name": "terminal",
        "description": (
            "Run one shell command in the workspace and return combined stdout/stderr plus its exit code. "
            "Use standard CLI programs to inspect, search, read, edit, create, delete, build, and test files. "
            "The shell is bash with pipefail enabled on Linux/macOS and PowerShell on Windows."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The complete shell command to execute."},
                "timeout": {"type": "integer", "minimum": 1, "maximum": 300},
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
}]


class DeepSeekClient:
    def __init__(self, config: Config):
        self.config = config

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "thinking": {"type": "enabled"},
            "reasoning_effort": "high",
            "max_tokens": self.config.max_output_tokens,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.config.base_url + "/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                    raw_response = response.read().decode("utf-8")
                try:
                    data = json.loads(raw_response)
                except json.JSONDecodeError as exc:
                    raise AgentError("API returned invalid JSON.") from exc
                if not isinstance(data, dict) or not isinstance(data.get("choices"), list) or not data["choices"]:
                    raise AgentError("Model returned no choices.")
                return data
            except urllib.error.HTTPError as exc:
                detail = _clip(exc.read().decode("utf-8", errors="replace"), 2000)
                if exc.code not in {408, 429, 500, 502, 503, 504} or attempt == 3:
                    raise AgentError(f"API HTTP {exc.code}: {detail}") from exc
                retry_after = exc.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else 2**attempt
                time.sleep(min(delay, 10))
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt == 3:
                    raise AgentError(f"API request failed: {exc}") from exc
                time.sleep(2**attempt)
        raise AgentError("API request failed after retries.")


SYSTEM_PROMPT = """You are a careful coding agent operating in a local workspace.
Use the terminal tool and standard command-line programs to inspect the project, edit files, and run relevant checks. The terminal starts in the workspace, but it is not an OS sandbox: keep every operation inside the workspace unless the user explicitly asks otherwise. Never inspect or print credentials, .env files, private keys, or secret environment variables. Do not claim a change or test succeeded unless terminal output confirms it. Read existing files before editing them. Keep changes focused and preserve unrelated work. When a command fails, diagnose its output and recover if possible. Once the task is genuinely complete, respond with a concise summary and tests run."""


def print_event(message: str) -> None:
    print(message, flush=True)


class CodingAgent:
    def __init__(self, client: Any, workspace: Path, config: Config, on_event: Callable[[str], None] = print_event):
        self.client = client
        self.config = config
        self.terminal = TerminalTool(workspace)
        self.on_event = on_event
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def _trim_history(self) -> None:
        system, rest = self.messages[0], self.messages[1:]
        segments: list[list[dict[str, Any]]] = []
        for message in rest:
            if message.get("role") == "user" or not segments:
                segments.append([])
            segments[-1].append(message)
        kept: list[list[dict[str, Any]]] = []
        total = len(json.dumps(system, ensure_ascii=False))
        for segment in reversed(segments):
            size = len(json.dumps(segment, ensure_ascii=False))
            if kept and total + size > self.config.max_history_chars:
                break
            kept.append(segment)
            total += size
        self.messages = [system] + [message for segment in reversed(kept) for message in segment]

    def _execute(self, call: dict[str, Any]) -> str:
        function = call.get("function") or {}
        name = function.get("name", "")
        if name != "terminal":
            return json.dumps({"ok": False, "error": f"Unknown tool: {name}"}, ensure_ascii=False)
        try:
            arguments = json.loads(function.get("arguments") or "{}")
            if not isinstance(arguments, dict):
                raise AgentError("Tool arguments must be a JSON object.")
            result = self.terminal.run(**arguments)
            return json.dumps({"ok": True, "result": result}, ensure_ascii=False)
        except (AgentError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
            return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)

    def run(self, user_text: str) -> str:
        self.messages.append({"role": "user", "content": user_text})
        self._trim_history()
        tool_call_count = 0
        input_tokens = 0
        output_tokens = 0
        for round_number in range(1, self.config.max_rounds + 1):
            response = self.client.complete(self.messages, TOOL_SCHEMAS)
            usage = response.get("usage") or {}
            input_tokens += int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
            output_tokens += int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
            choices = response.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                raise AgentError("Model response has an invalid choices field.")
            choice = choices[0]
            message = choice.get("message")
            if not isinstance(message, dict):
                raise AgentError("Model response is missing an assistant message.")
            assistant: dict[str, Any] = {"role": "assistant", "content": message.get("content")}
            if message.get("reasoning_content") is not None:
                assistant["reasoning_content"] = message["reasoning_content"]
            if message.get("tool_calls"):
                assistant["tool_calls"] = message["tool_calls"]
            self.messages.append(assistant)

            calls = message.get("tool_calls") or []
            if not isinstance(calls, list):
                raise AgentError("Model response has an invalid tool_calls field.")
            if not calls:
                finish = choice.get("finish_reason")
                if finish == "length":
                    self.on_event(f"[round {round_number}] output limit reached; asking model to continue")
                    self.messages.append({
                        "role": "user",
                        "content": (
                            "Continue the same task from where you stopped. The previous response reached its "
                            "output limit, so do not restart or repeat completed exploration. Use the terminal "
                            "to finish the implementation and verification, then provide the final answer."
                        ),
                    })
                    continue
                if finish not in {None, "stop"}:
                    raise AgentError(f"Model stopped without completing the task: {finish}.")
                if not isinstance(message.get("content"), str) or not message["content"].strip():
                    raise AgentError("Model stopped without returning final text.")
                self.on_event(
                    f"[done] rounds={round_number} tool_calls={tool_call_count} "
                    f"tokens={input_tokens}in/{output_tokens}out"
                )
                return message.get("content") or "<model returned no final text>"

            for call_number, call in enumerate(calls, 1):
                if not isinstance(call, dict) or not isinstance(call.get("id"), str) or not call["id"]:
                    raise AgentError("Model returned a tool call without a valid id.")
                tool_call_count += 1
                name = (call.get("function") or {}).get("name", "unknown")
                raw_arguments = (call.get("function") or {}).get("arguments", "")
                try:
                    command = json.loads(raw_arguments).get("command", "")
                except (AttributeError, json.JSONDecodeError):
                    command = ""
                preview = " ".join(command.strip().splitlines())
                if len(preview) > 140:
                    preview = preview[:137] + "..."
                label = f"[tool {round_number}.{call_number}] {name}"
                self.on_event(label + (f": {preview}" if preview else ""))
                result = self._execute(call)
                parsed = json.loads(result)
                terminal_result = parsed.get("result") or {}
                if parsed["ok"] and isinstance(terminal_result, dict):
                    suffix = " truncated" if terminal_result.get("truncated") else ""
                    status = f"exit={terminal_result.get('exit_code')} output={terminal_result.get('output_chars')} chars{suffix}"
                else:
                    status = "ok" if parsed["ok"] else f"error: {parsed['error']}"
                self.on_event(f"  -> {status}")
                self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
        raise AgentError(f"Agent stopped after {self.config.max_rounds} tool rounds without a final answer.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A minimal DeepSeek coding agent")
    parser.add_argument("prompt", nargs="*", help="one-shot task; omit for interactive mode")
    parser.add_argument("--workspace", default=".", help="workspace root (default: current directory)")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    launch_directory = Path.cwd().resolve()
    workspace = Path(args.workspace).resolve()
    if not workspace.is_dir():
        print(f"error: workspace is not a directory: {workspace}", file=sys.stderr)
        return 1
    load_dotenv(launch_directory / ".env")
    if workspace != launch_directory:
        load_dotenv(workspace / ".env")
    try:
        config = Config.from_env()
        agent = CodingAgent(DeepSeekClient(config), workspace, config)
        if args.prompt:
            print(agent.run(" ".join(args.prompt)))
            return 0
        print(f"Coding Agent | model={config.model} | workspace={workspace}")
        print("Type /exit to quit, /clear to reset the conversation.")
        while True:
            try:
                text = input("\nyou> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not text:
                continue
            if text in {"/exit", "/quit"}:
                return 0
            if text == "/clear":
                agent.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                print("Conversation cleared.")
                continue
            try:
                print("\nagent> " + agent.run(text))
            except AgentError as exc:
                print(f"\nerror> {exc}", file=sys.stderr)
    except AgentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
