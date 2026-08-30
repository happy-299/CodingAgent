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

    @classmethod
    def from_env(cls) -> "Config":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise AgentError("OPENAI_API_KEY is missing; put it in .env or the environment.")
        return cls(
            api_key=api_key,
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com").rstrip("/"),
            model=os.getenv("OPENAI_MODEL", "deepseek-v4-flash"),
            timeout=int(os.getenv("AGENT_API_TIMEOUT", "180")),
            max_rounds=int(os.getenv("AGENT_MAX_ROUNDS", "20")),
            max_history_chars=int(os.getenv("AGENT_MAX_HISTORY_CHARS", "300000")),
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

    def run(self, command: str, timeout: int = 60) -> str:
        if not command.strip():
            raise AgentError("Command cannot be empty.")
        timeout = max(1, min(int(timeout), 300))
        argv = (
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command]
            if os.name == "nt"
            else ["/bin/bash", "-lc", command]
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
        output = _clip(output)
        return f"exit_code={process.returncode}\n{output or '<no output>'}"


TOOL_SCHEMAS: list[dict[str, Any]] = [{
    "type": "function",
    "function": {
        "name": "terminal",
        "description": (
            "Run one shell command in the workspace and return combined stdout/stderr plus its exit code. "
            "Use standard CLI programs to inspect, search, read, edit, create, delete, build, and test files. "
            "The shell is bash on Linux/macOS and PowerShell on Windows."
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
            "max_tokens": 8192,
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
                    data = json.loads(response.read().decode("utf-8"))
                if not data.get("choices"):
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


class CodingAgent:
    def __init__(self, client: Any, workspace: Path, config: Config, on_event: Callable[[str], None] = print):
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
        for round_number in range(1, self.config.max_rounds + 1):
            response = self.client.complete(self.messages, TOOL_SCHEMAS)
            choice = response["choices"][0]
            message = choice.get("message") or {}
            assistant: dict[str, Any] = {"role": "assistant", "content": message.get("content")}
            if message.get("reasoning_content") is not None:
                assistant["reasoning_content"] = message["reasoning_content"]
            if message.get("tool_calls"):
                assistant["tool_calls"] = message["tool_calls"]
            self.messages.append(assistant)

            calls = message.get("tool_calls") or []
            if not calls:
                finish = choice.get("finish_reason")
                if finish == "length":
                    raise AgentError("Model output hit its token limit before completing the task.")
                return message.get("content") or "<model returned no final text>"

            for call_number, call in enumerate(calls, 1):
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
                status = "ok" if parsed["ok"] else f"error: {parsed['error']}"
                self.on_event(f"  -> {status}")
                self.messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": result})
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
