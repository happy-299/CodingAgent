#!/usr/bin/env python3
"""A small, self-contained coding agent using native model tool calls."""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


MAX_TOOL_OUTPUT = 20_000
PLAN_REVIEW_TERMINAL_INTERVAL = 8
HISTORICAL_REASONING_CHARS = 1_000
RECENT_REASONING_UNITS = 2
SENSITIVE_ENV_PARTS = {"KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "CREDENTIALS"}
SENSITIVE_ENV_SUFFIXES = (
    "APIKEY",
    "ACCESSTOKEN",
    "AUTHTOKEN",
    "CLIENTSECRET",
    "PASSWORD",
    "CREDENTIAL",
    "CREDENTIALS",
)


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
    max_rounds: int = 30
    max_history_chars: int = 160_000
    max_output_tokens: int = 32_768
    reasoning_effort: str = "medium"

    @classmethod
    def from_env(cls) -> "Config":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise AgentError("OPENAI_API_KEY is missing; put it in .env or the environment.")
        reasoning_effort = os.getenv("AGENT_REASONING_EFFORT", "medium").strip().lower()
        if reasoning_effort not in {"low", "medium", "high"}:
            raise AgentError(
                "AGENT_REASONING_EFFORT must be one of low, medium, or high, "
                f"got {reasoning_effort!r}."
            )
        return cls(
            api_key=api_key,
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com").rstrip("/"),
            model=os.getenv("OPENAI_MODEL", "deepseek-v4-flash"),
            timeout=_env_int("AGENT_API_TIMEOUT", 180, 1, 600),
            max_rounds=_env_int("AGENT_MAX_ROUNDS", 30, 1, 100),
            max_history_chars=_env_int("AGENT_MAX_HISTORY_CHARS", 160_000, 10_000, 10_000_000),
            max_output_tokens=_env_int("AGENT_MAX_OUTPUT_TOKENS", 32_768, 1_024, 384_000),
            reasoning_effort=reasoning_effort,
        )


def _clip(text: str, limit: int = MAX_TOOL_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    kept = limit // 2
    return text[:kept] + f"\n... <truncated {len(text) - 2 * kept} chars> ...\n" + text[-kept:]


def _looks_like_verification_command(command: str) -> bool:
    """Recognize common test/verification commands without naming a project."""
    normalized = command.lower()
    return bool(re.search(
        r"(?:pytest|unittest|go\s+test|cargo\s+test|npm\s+(?:run\s+)?test|"
        r"mvn\s+test|gradle\s+test|dotnet\s+test|make\s+test|--self-?test\b|"
        r"py_compile|compileall|cargo\s+check|npm\s+(?:run\s+)?build|"
        r"ruff\s+check|flake8|mypy|eslint|(?:^|\s)tsc(?:\s|$))",
        normalized,
    ))


def _contains_tool_markup(content: str) -> bool:
    """Detect provider-specific textual tool markup in a tool-free response."""
    return bool(re.search(r"<\|?/?(?:DSML|tool_calls?|function_call)\b|<｜｜", content, re.IGNORECASE))


def _is_sensitive_env_name(name: str) -> bool:
    """Recognize credential-like names without rejecting benign substrings."""
    upper_name = name.upper()
    parts = {part for part in re.split(r"[^A-Z0-9]+", upper_name) if part}
    return bool(parts & SENSITIVE_ENV_PARTS) or upper_name.endswith(SENSITIVE_ENV_SUFFIXES)


def _read_captured_output(stream: Any, limit: int = MAX_TOOL_OUTPUT) -> tuple[str, int, bool]:
    stream.flush()
    size = stream.seek(0, os.SEEK_END)
    truncated = size > limit
    if not truncated:
        stream.seek(0)
        data = stream.read()
        return data.decode("utf-8", errors="replace"), size, False
    kept = limit // 2
    stream.seek(0)
    head = stream.read(kept)
    stream.seek(-kept, os.SEEK_END)
    tail = stream.read(kept)
    marker = f"\n... <truncated {size - 2 * kept} bytes> ...\n"
    return (
        head.decode("utf-8", errors="replace") + marker + tail.decode("utf-8", errors="replace"),
        size,
        True,
    )


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
        child_env = {
            key: value
            for key, value in os.environ.items()
            if not _is_sensitive_env_name(key)
        }
        child_env["PYTHONUNBUFFERED"] = "1"
        child_env["CODING_AGENT_WORKSPACE"] = str(self.root)
        popen_options: dict[str, Any] = {"start_new_session": True} if os.name != "nt" else {
            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP
        }
        with tempfile.TemporaryFile(mode="w+b") as capture:
            process = subprocess.Popen(
                argv,
                cwd=self.root,
                stdout=capture,
                stderr=subprocess.STDOUT,
                env=child_env,
                **popen_options,
            )
            try:
                process.wait(timeout=timeout)
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
                process.wait()
                output, _, _ = _read_captured_output(capture)
                raise AgentError(f"Command timed out after {timeout}s.\n{output}") from exc
            output, output_bytes, truncated = _read_captured_output(capture)
            return {
                "exit_code": process.returncode,
                "output": output if output else "<no output>",
                "output_bytes": output_bytes,
                "truncated": truncated,
            }


class PlanState:
    """Ephemeral structured progress state, separate from repository files."""

    STATUSES = {"pending", "in_progress", "completed", "blocked"}

    def __init__(self) -> None:
        self.items: list[dict[str, str]] = []

    def reset(self) -> None:
        self.items = []

    def update(self, items: Any) -> list[dict[str, str]]:
        if not isinstance(items, list) or not 1 <= len(items) <= 8:
            raise AgentError("Plan must contain between 1 and 8 items.")
        normalized: list[dict[str, str]] = []
        active = 0
        seen: set[str] = set()
        for item in items:
            if not isinstance(item, dict):
                raise AgentError("Every plan item must be an object.")
            step = item.get("step")
            status = item.get("status")
            if not isinstance(step, str) or not step.strip() or len(step.strip()) > 300:
                raise AgentError("Every plan step must contain 1 to 300 characters.")
            if status not in self.STATUSES:
                raise AgentError(f"Invalid plan status: {status!r}.")
            clean_step = step.strip()
            if clean_step in seen:
                raise AgentError("Plan steps must be unique.")
            seen.add(clean_step)
            active += status == "in_progress"
            normalized.append({"step": clean_step, "status": status})
        if active > 1:
            raise AgentError("At most one plan item may be in progress.")
        self.items = normalized
        return [dict(item) for item in self.items]

    @property
    def has_open_items(self) -> bool:
        return any(item["status"] in {"pending", "in_progress"} for item in self.items)


class WorkingMemory:
    """Bounded, evidence-based memory used when raw protocol history grows large."""

    def __init__(self, observation_limit: int = 24) -> None:
        self.observation_limit = observation_limit
        self.objective = ""
        self.prior_context = ""
        self.observations: list[dict[str, Any]] = []

    def reset(self, objective: str, prior_messages: list[dict[str, Any]]) -> None:
        self.objective = objective
        self.observations = []
        prior: list[str] = []
        for message in prior_messages[-8:]:
            role = message.get("role")
            content = message.get("content")
            if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
                prior.append(f"{role}: {_clip(content.strip(), 600)}")
        self.prior_context = "\n".join(prior[-4:])

    def observe(self, command: str, result: dict[str, Any]) -> None:
        self.observations.append({
            "command": _clip(command.strip(), 500),
            "exit_code": result.get("exit_code"),
            "output": _clip(str(result.get("output", "")), 1_200),
            "truncated": bool(result.get("truncated")),
        })
        self.observations = self.observations[-self.observation_limit :]

    def render(self, plan: PlanState) -> str:
        sections = ["CURRENT OBJECTIVE:\n" + self.objective]
        if self.prior_context:
            sections.append("RECENT PRIOR CONTEXT:\n" + self.prior_context)
        if plan.items:
            sections.append("CURRENT PLAN:\n" + json.dumps(plan.items, ensure_ascii=False))
        if self.observations:
            sections.append("RECENT VERIFIED OBSERVATIONS:\n" + json.dumps(self.observations, ensure_ascii=False))
        return "\n\n".join(sections)


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
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
    },
    {
        "type": "function",
        "function": {
            "name": "update_plan",
            "description": (
                "Create or replace the concise working plan shown to the user. Track the smallest set of "
                "acceptance outcomes and necessary implementation or verification steps for non-trivial work; "
                "exclude speculative expansion. Update it when evidence changes progress. This only tracks "
                "state and cannot modify files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 8,
                        "items": {
                            "type": "object",
                            "properties": {
                                "step": {"type": "string", "minLength": 1, "maxLength": 300},
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed", "blocked"],
                                },
                            },
                            "required": ["step", "status"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["items"],
                "additionalProperties": False,
            },
        },
    },
]


class DeepSeekClient:
    def __init__(self, config: Config):
        self.config = config

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: Any = "auto",
        on_delta: Callable[[str, str], None] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "model": self.config.model,
            "messages": messages,
            # Once the host has established a successful verification result
            # it sends a tool-free finalization request. Keep that response
            # concise and deterministic; normal execution requests retain
            # configured reasoning and native tool calling.
            "thinking": {"type": "enabled"} if tools else {"type": "disabled"},
            "max_tokens": self.config.max_output_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            payload["reasoning_effort"] = self.config.reasoning_effort
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
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
                    return self._read_stream(response, on_delta)
            except urllib.error.HTTPError as exc:
                detail = _clip(exc.read().decode("utf-8", errors="replace"), 2000)
                if exc.code not in {408, 429, 500, 502, 503, 504} or attempt == 3:
                    raise AgentError(f"API HTTP {exc.code}: {detail}") from exc
                retry_after = exc.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else 2**attempt
                time.sleep(min(delay, 10))
            except (urllib.error.URLError, TimeoutError, http.client.HTTPException, ConnectionError) as exc:
                if attempt == 3:
                    raise AgentError(f"API request failed: {exc}") from exc
                if on_delta:
                    on_delta("status", "Stream interrupted; retrying the response…")
                time.sleep(2**attempt)
        raise AgentError("API request failed after retries.")

    @staticmethod
    def _read_stream(response: Any, on_delta: Callable[[str, str], None] | None) -> dict[str, Any]:
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        finish_reason: Any = None
        usage: dict[str, Any] = {}
        saw_data = False
        saw_done = False

        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data:"):
                continue
            data_text = line[5:].strip()
            if data_text == "[DONE]":
                saw_done = True
                break
            try:
                chunk = json.loads(data_text)
            except json.JSONDecodeError as exc:
                raise http.client.IncompleteRead(data_text.encode("utf-8")) from exc
            saw_data = True
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
            choices = chunk.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            choice = choices[0] if isinstance(choices[0], dict) else {}
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
            delta = choice.get("delta") or {}
            reasoning = delta.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning:
                reasoning_parts.append(reasoning)
                if on_delta:
                    on_delta("reasoning", reasoning)
            content = delta.get("content")
            if isinstance(content, str) and content:
                content_parts.append(content)
                if on_delta:
                    on_delta("content", content)
            for call_delta in delta.get("tool_calls") or []:
                if not isinstance(call_delta, dict):
                    continue
                index = int(call_delta.get("index", 0))
                call = tool_calls.setdefault(
                    index,
                    {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                )
                if call_delta.get("id"):
                    call["id"] = call_delta["id"]
                if call_delta.get("type"):
                    call["type"] = call_delta["type"]
                function_delta = call_delta.get("function") or {}
                if function_delta.get("name"):
                    call["function"]["name"] = function_delta["name"]
                if function_delta.get("arguments"):
                    call["function"]["arguments"] += function_delta["arguments"]

        if not saw_data or not saw_done:
            raise http.client.IncompleteRead(b"stream ended before [DONE]")
        message: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(content_parts) or None,
        }
        if reasoning_parts:
            message["reasoning_content"] = "".join(reasoning_parts)
        if tool_calls:
            message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]
        return {
            "choices": [{"index": 0, "finish_reason": finish_reason, "message": message}],
            "usage": usage,
        }


SYSTEM_PROMPT = """You are a careful coding agent operating in a local workspace.
Translate the user's request into concrete acceptance criteria and keep every action anchored to them. Choose each next action from the workspace and observed evidence rather than a prewritten task-specific playbook.
Begin with the smallest orientation that removes important uncertainty. After that orientation, if the work is non-trivial, create a short outcome-oriented plan before extensive design or implementation; otherwise proceed directly. Planning is your decision, not a mandatory first action. Keep a plan current when evidence changes it, and complete or block every item before finishing.
Keep analysis proportional and concise. Once there is enough information for a safe, useful action, act and observe instead of restating settled requirements, exhaustively enumerating options, or simulating details that can be checked cheaply. Completeness means satisfying all required behavior, not maximizing feature count. Where requirements leave design choices open, choose one coherent minimal design, stop probing alternatives once it is viable, and do not build ancillary subsystems unless an acceptance criterion needs them.
Use the terminal and standard command-line programs to inspect, search, read, edit, create, build, and test. Every terminal call starts at the exact selected workspace root. Use relative paths by default, confirm location only when uncertain, and never replace the current directory with a guessed path. The exact root is also available as CODING_AGENT_WORKSPACE.
Respect the requested scope and preserve unrelated work. Do not change the workspace when the user only authorized inspection or an answer. The terminal is not an OS sandbox, so keep operations inside the workspace unless the user explicitly requests otherwise. Never inspect or print credentials, .env files, private keys, or secret environment variables.
Treat command failures as evidence: check both the implementation and the expectation behind a failing check, diagnose the cause, and recover when possible. Settle the structure before emitting a large edit; after that, prefer focused corrections over rewriting unchanged content. Before finishing, reconcile every acceptance criterion with concrete observed evidence and run the smallest set of checks that covers distinct required behaviors. A successful command proves only the behavior it actually exercised: inspect its output and side effects, and use meaningful assertions that could fail when behavior is wrong. For interactive or long-running flows, verify observable state transitions, output, and side effects; do not use tautological assertions, swallow errors, or treat exit code 0 alone as proof of behavior. Reuse successful evidence while it remains valid; repeat a check only when a relevant change, failure, or specific evidence gap justifies it. Never claim success without supporting terminal output.
Stop using tools when the request is complete or genuinely blocked, not merely because a round budget is near. Then respond with a concise, honest summary of changes, checks, and anything unresolved."""


def print_event(message: str) -> None:
    print(message, flush=True)


class TerminalUI:
    """Small dependency-free terminal renderer; agent logic stays UI-agnostic."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[38;5;45m"
    BLUE = "\033[38;5;75m"
    PURPLE = "\033[38;5;141m"
    GREEN = "\033[38;5;78m"
    YELLOW = "\033[38;5;221m"
    RED = "\033[38;5;203m"

    BANNER = (
        " ██████╗ ██████╗ ██████╗ ██╗███╗   ██╗ ██████╗      █████╗  ██████╗ ███████╗███╗   ██╗████████╗\n"
        "██╔════╝██╔═══██╗██╔══██╗██║████╗  ██║██╔════╝     ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝\n"
        "██║     ██║   ██║██║  ██║██║██╔██╗ ██║██║  ███╗    ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   \n"
        "██║     ██║   ██║██║  ██║██║██║╚██╗██║██║   ██║    ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   \n"
        "╚██████╗╚██████╔╝██████╔╝██║██║ ╚████║╚██████╔╝    ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   \n"
        " ╚═════╝ ╚═════╝ ╚═════╝ ╚═╝╚═╝  ╚═══╝ ╚═════╝     ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   "
    )

    def __init__(self, plain: bool = False, stream: Any = None):
        self.stream = stream or sys.stdout
        self.plain = plain
        self.color = not plain and bool(getattr(self.stream, "isatty", lambda: False)())
        if os.getenv("NO_COLOR") is not None or os.getenv("TERM") == "dumb":
            self.color = False
        self._stream_kind: str | None = None
        self._streamed_content = ""

    def _paint(self, text: str, *styles: str) -> str:
        return "".join(styles) + text + self.RESET if self.color else text

    def _write(self, text: str = "") -> None:
        print(text, file=self.stream, flush=True)

    def _close_stream(self) -> None:
        if self._stream_kind is not None:
            self._write()
            self._stream_kind = None

    def _rule(self) -> str:
        return "─" * min(max(shutil.get_terminal_size((100, 24)).columns - 4, 36), 100)

    def banner(self, model: str, workspace: Path) -> None:
        if self.plain:
            self._write(f"Coding Agent | model={model} | workspace={workspace}")
            return
        self._write(self._paint(self.BANNER, self.BOLD, self.CYAN))
        self._write(self._paint("  CODING AGENT  ·  terminal-first autonomous development", self.BOLD, self.PURPLE))
        self._write(self._paint(self._rule(), self.DIM))
        self._write(f"  {self._paint('MODEL', self.PURPLE, self.BOLD)}      {model}")
        self._write(f"  {self._paint('WORKSPACE', self.BLUE, self.BOLD)}  {workspace}")
        self._write(self._paint(self._rule(), self.DIM))
        self._write(self._paint("  /clear reset conversation   /exit quit", self.DIM))

    def _panel(self, title: str, body: str, color: str) -> None:
        self._write()
        self._write(self._paint(f"◆ {title}", self.BOLD, color))
        for line in (body.rstrip().splitlines() or [""]):
            self._write(self._paint("│ ", self.DIM) + line)

    def event(self, message: str) -> None:
        head, separator, body = message.partition("\n")
        self._close_stream()
        if head == "[model]":
            self._streamed_content = ""
        if self.plain:
            self._write(message)
            return
        if head == "[thinking]":
            self._panel("THINKING", body, self.PURPLE)
        elif head.startswith("[command "):
            self._panel("COMMAND", body, self.CYAN)
        elif head.startswith("[output "):
            self._panel("OUTPUT", body, self.BLUE)
        elif head == "[plan]":
            try:
                items = json.loads(body)
            except json.JSONDecodeError:
                items = []
            self._write()
            self._write(self._paint("◆ PLAN", self.BOLD, self.YELLOW))
            icons = {"completed": "✓", "in_progress": "●", "pending": "○", "blocked": "!"}
            colors = {
                "completed": self.GREEN,
                "in_progress": self.CYAN,
                "pending": self.DIM,
                "blocked": self.RED,
            }
            for item in items if isinstance(items, list) else []:
                status = item.get("status", "pending")
                self._write(self._paint(f"  {icons.get(status, '○')} {item.get('step', '')}", colors.get(status, self.DIM)))
        elif head.startswith("[tool "):
            title = head[1 : head.find("]")].upper()
            tool_name = head.split("]", 1)[1].strip()
            self._write()
            self._write(self._paint(f"● {title}", self.BOLD, self.YELLOW) + f"  {tool_name}")
        elif head.startswith("[result "):
            status = head.split("]", 1)[1].strip()
            color = self.GREEN if "exit=0" in status else self.RED
            self._write(self._paint(f"  ↳ {status}", color))
            if separator and body:
                for line in body.rstrip().splitlines():
                    self._write(self._paint("    │ ", self.DIM) + line)
        elif head.startswith("[done]"):
            self._write(self._paint(f"\n✓ {head[6:].strip()}", self.GREEN, self.BOLD))
        elif head == "[model]":
            self._write(self._paint("\n◌ DeepSeek is thinking…", self.PURPLE, self.BOLD))
        elif head.startswith("[round ") or head.startswith("[round limit]"):
            self._write(self._paint(f"\n! {head.strip('[]')}", self.YELLOW))
        else:
            self._write(message)

    def stream_delta(self, kind: str, text: str) -> None:
        if kind == "status":
            self._close_stream()
            self._write(self._paint(f"\n↻ {text}", self.YELLOW))
            return
        if kind not in {"reasoning", "content"} or not text:
            return
        if self._stream_kind != kind:
            self._close_stream()
            title = "THINKING" if kind == "reasoning" else "AGENT"
            color = self.PURPLE if kind == "reasoning" else self.GREEN
            self._write()
            self._write(self._paint(f"◆ {title} · LIVE", self.BOLD, color))
            self._stream_kind = kind
        if kind == "content":
            self._streamed_content += text
        self.stream.write(text)
        self.stream.flush()

    def answer(self, text: str) -> None:
        self._close_stream()
        if self._streamed_content == text:
            self._streamed_content = ""
            return
        if self.plain:
            self._write(text)
            return
        self._panel("AGENT", text, self.GREEN)

    def error(self, text: str) -> None:
        self._write(self._paint(f"\n✗ ERROR  {text}", self.RED, self.BOLD))

    def prompt(self) -> str:
        return "\n" + self._paint("❯ ", self.CYAN, self.BOLD)


class CodingAgent:
    def __init__(
        self,
        client: Any,
        workspace: Path,
        config: Config,
        on_event: Callable[[str], None] = print_event,
        on_stream: Callable[[str, str], None] | None = None,
    ):
        self.client = client
        self.config = config
        self.terminal = TerminalTool(workspace)
        self.plan = PlanState()
        self.memory = WorkingMemory()
        self.task_start = 1
        self.on_event = on_event
        self.on_stream = on_stream
        self.system_message = {
            "role": "system",
            "content": SYSTEM_PROMPT + f"\n\nSELECTED WORKSPACE ROOT: {self.terminal.root}",
        }
        self.messages: list[dict[str, Any]] = [dict(self.system_message)]

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

    @staticmethod
    def _protocol_units(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        units: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            if role == "assistant":
                if current:
                    units.append(current)
                current = [message]
            elif role == "tool" and current and current[0].get("role") == "assistant":
                current.append(message)
            else:
                if current:
                    units.append(current)
                    current = []
                units.append([message])
        if current:
            units.append(current)
        return units

    @staticmethod
    def _condense_reasoning_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Bound old internal reasoning while preserving every protocol message."""
        reasoning_indexes = [
            index
            for index, message in enumerate(messages)
            if isinstance(message.get("reasoning_content"), str)
        ]
        preserve = set(reasoning_indexes[-RECENT_REASONING_UNITS:])
        condensed: list[dict[str, Any]] = []
        for index, message in enumerate(messages):
            reasoning = message.get("reasoning_content")
            if (
                index not in preserve
                and isinstance(reasoning, str)
                and len(reasoning) > HISTORICAL_REASONING_CHARS
            ):
                message = dict(message)
                message["reasoning_content"] = _clip(reasoning, HISTORICAL_REASONING_CHARS)
            condensed.append(message)
        return condensed

    def _messages_for_model(self) -> list[dict[str, Any]]:
        raw_size = len(json.dumps(self.messages, ensure_ascii=False))
        reasoning_trigger = self.config.max_history_chars // 2
        candidate = self.messages
        if raw_size > reasoning_trigger:
            candidate = self._condense_reasoning_history(self.messages)
        candidate_size = len(json.dumps(candidate, ensure_ascii=False))
        if candidate_size <= self.config.max_history_chars:
            if candidate_size < raw_size:
                self.on_event(
                    f"[context] raw={raw_size} chars reasoning_compacted={candidate_size} chars"
                )
            return candidate
        if raw_size <= self.config.max_history_chars:
            return self.messages
        system = dict(candidate[0])
        objective = dict(candidate[self.task_start])
        memory_text = self.memory.render(self.plan)
        memory_message = {
            "role": "system",
            "content": (
                "The earlier raw interaction was compacted. Treat this bounded working memory as context, "
                "but prefer newer raw tool results when they conflict:\n\n" + memory_text
            ),
        }
        base = [system, objective, memory_message]
        base_size = len(json.dumps(base, ensure_ascii=False))
        budget = max(0, self.config.max_history_chars - base_size)
        kept: list[list[dict[str, Any]]] = []
        used = 0
        for unit in reversed(self._protocol_units(candidate[self.task_start + 1 :])):
            size = len(json.dumps(unit, ensure_ascii=False))
            if kept and used + size > budget:
                break
            kept.append(unit)
            used += size
        compacted = base + [message for unit in reversed(kept) for message in unit]
        self.on_event(
            f"[context] raw={raw_size} chars reasoning_compacted={candidate_size} chars "
            f"compacted={len(json.dumps(compacted, ensure_ascii=False))} chars"
        )
        return compacted

    def _execute(self, call: dict[str, Any]) -> str:
        function = call.get("function") or {}
        name = function.get("name", "")
        try:
            arguments = json.loads(function.get("arguments") or "{}")
            if not isinstance(arguments, dict):
                raise AgentError("Tool arguments must be a JSON object.")
            if name == "terminal":
                result = self.terminal.run(**arguments)
            elif name == "update_plan":
                result = {"items": self.plan.update(arguments.get("items"))}
                self.on_event("[plan]\n" + json.dumps(result["items"], ensure_ascii=False))
            else:
                raise AgentError(f"Unknown tool: {name}")
            return json.dumps({"ok": True, "result": result}, ensure_ascii=False)
        except (AgentError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
            return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)

    def run(self, user_text: str) -> str:
        self._trim_history()
        self.plan.reset()
        self.memory.reset(user_text, self.messages[1:])
        self.messages.append({"role": "user", "content": user_text})
        self.task_start = len(self.messages) - 1
        tool_call_count = 0
        terminal_call_count = 0
        input_tokens = 0
        output_tokens = 0
        completion_reviews = 0
        audit_tool_rounds = 0
        verification_final_requested = False
        successful_verification_seen = False
        planning_nudge_sent = False
        terminal_calls_since_plan_update = 0
        tool_free_final_retries = 0

        def completion_audit_prompt() -> str:
            evidence_note = (
                " At least one relevant verification command has already succeeded. Assess whether "
                "later changes made that evidence stale; if not, reuse it instead of repeating it."
                if successful_verification_seen
                else ""
            )
            return (
                "Perform the final completion audit now. Reconcile every explicit acceptance "
                "criterion, plus workspace instructions that directly govern the requested work, "
                "with concrete evidence already present. Do not broaden the scope or invent hidden "
                "requirements." + evidence_note + " Use a tool only when you identify a concrete "
                "unmet criterion or evidence gap; do not rerun checks merely for reassurance. If the "
                "plan is stale, update it, but do not submit an unchanged plan solely to repeat its "
                "status. Finish with a concise answer when the evidence is sufficient. If genuinely "
                "blocked, preserve the blocker and explain it honestly."
            )

        for round_number in range(1, self.config.max_rounds + 1):
            review_active_at_round_start = completion_reviews > 0
            if round_number == self.config.max_rounds and self.plan.items and completion_reviews == 0:
                completion_reviews = 1
                review_active_at_round_start = True
                self.on_event("[completion review] reserving final round for requirements and evidence")
                self.messages.append({"role": "user", "content": completion_audit_prompt()})
            audit_tool_budget_exhausted = completion_reviews > 0 and audit_tool_rounds >= 2
            if audit_tool_budget_exhausted:
                self.on_event("[completion review] focused-check budget exhausted; requesting final text")
            verification_final = verification_final_requested and not self.plan.items
            if verification_final:
                self.on_event("[verification] successful check confirmed; requesting final text")
            self.on_event("[model]")
            stream_options = {"on_delta": self.on_stream} if self.on_stream else {}
            tool_free_final = verification_final or audit_tool_budget_exhausted
            response = self.client.complete(
                self._messages_for_model(),
                [] if tool_free_final else TOOL_SCHEMAS,
                tool_choice="none" if audit_tool_budget_exhausted else "auto",
                **stream_options,
            )
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
            reasoning = message.get("reasoning_content")
            if not self.on_stream and isinstance(reasoning, str) and reasoning.strip():
                self.on_event("[thinking]\n" + _clip(reasoning.strip(), 8_000))

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
                if (
                    (verification_final or audit_tool_budget_exhausted)
                    and _contains_tool_markup(message["content"])
                    and tool_free_final_retries < 2
                ):
                    tool_free_final_retries += 1
                    self.on_event(
                        "[finalization] provider returned textual tool markup; retrying plain final text"
                    )
                    self.messages.append({
                        "role": "user",
                        "content": (
                            "Your previous response contained provider-specific tool markup, but tool execution is disabled. "
                            "Do not call or describe any tool. Return only a concise natural-language final summary "
                            "of the verified result and any remaining status."
                        ),
                    })
                    continue
                if self.plan.items and completion_reviews == 0:
                    completion_reviews += 1
                    self.on_event("[completion review] auditing requirements, plan, and evidence")
                    self.messages.append({"role": "user", "content": completion_audit_prompt()})
                    continue
                if self.plan.has_open_items:
                    raise AgentError(
                        "Model attempted to finish with pending plan items after the completion audit."
                    )
                self.on_event(
                    f"[done] rounds={round_number} terminal_calls={terminal_call_count} "
                    f"plan_items={len(self.plan.items)} "
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
                event_id = f"{round_number}.{call_number}"
                label = f"[tool {event_id}] {name}"
                self.on_event(label + (f": {preview}" if preview else ""))
                if command:
                    self.on_event(f"[command {event_id}]\n{command}")
                result = self._execute(call)
                parsed = json.loads(result)
                terminal_result = parsed.get("result") or {}
                if name == "terminal" and parsed["ok"] and isinstance(terminal_result, dict):
                    terminal_call_count += 1
                    if self.plan.items:
                        terminal_calls_since_plan_update += 1
                    self.memory.observe(command, terminal_result)
                    if (
                        terminal_result.get("exit_code") == 0
                        and _looks_like_verification_command(command)
                    ):
                        successful_verification_seen = True
                        if not self.plan.items:
                            verification_final_requested = True
                    suffix = " truncated" if terminal_result.get("truncated") else ""
                    status = f"exit={terminal_result.get('exit_code')} output={terminal_result.get('output_bytes')} bytes{suffix}"
                elif name == "update_plan" and parsed["ok"]:
                    terminal_calls_since_plan_update = 0
                    status = f"updated {len(self.plan.items)} items"
                else:
                    status = "ok" if parsed["ok"] else f"error: {parsed['error']}"
                self.on_event(f"[result {event_id}] {status}")
                if name == "terminal" and parsed["ok"] and isinstance(terminal_result, dict):
                    self.on_event(f"[output {event_id}]\n{terminal_result.get('output', '<no output>')}")
                self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
                if (
                    name == "terminal"
                    and parsed["ok"]
                    and isinstance(terminal_result, dict)
                    and verification_final_requested
                    and call is calls[-1]
                ):
                    self.messages.append({
                        "role": "user",
                        "content": (
                            "The requested verification command succeeded and there is no active plan. "
                            "If the user's acceptance criteria are now satisfied, provide the final concise "
                            "summary immediately. Do not run more commands or speculate about unstated "
                            "requirements."
                        ),
                    })
            if self.plan.items and not self.plan.has_open_items and completion_reviews == 0:
                completion_reviews = 1
                self.on_event("[completion review] plan complete; auditing requirements and evidence")
                self.messages.append({"role": "user", "content": completion_audit_prompt()})
            elif (
                self.plan.items
                and completion_reviews == 0
                and terminal_calls_since_plan_update >= PLAN_REVIEW_TERMINAL_INTERVAL
            ):
                terminal_calls_since_plan_update = 0
                self.on_event("[progress checkpoint] reconciling plan and remaining scope")
                self.messages.append({
                    "role": "user",
                    "content": (
                        "Progress checkpoint: several execution steps have occurred since the plan was "
                        "last reconciled. Compare the remaining plan to the original acceptance criteria, "
                        "remove unnecessary scope, update statuses that evidence has resolved, and choose "
                        "the shortest path to verified completion."
                    ),
                })
            elif (
                not self.plan.items
                and not planning_nudge_sent
                and not verification_final_requested
                and terminal_call_count >= 2
            ):
                planning_nudge_sent = True
                self.on_event("[planning checkpoint] deciding whether remaining work needs a plan")
                self.messages.append({
                    "role": "user",
                    "content": (
                        "Based on the initial evidence, decide now whether the remaining work is non-trivial "
                        "enough to benefit from a concise plan. Create one only if it improves execution; "
                        "otherwise proceed directly. In either case, keep the scope to the smallest complete "
                        "solution for the original acceptance criteria."
                    ),
                })
            if review_active_at_round_start:
                audit_tool_rounds += 1
        self.on_event(
            f"[round limit] {self.config.max_rounds} rounds used; disabling tools for final response"
        )
        final_messages = self.messages + [{
            "role": "user",
            "content": (
                "The execution-round limit has been reached. Do not call any more tools. Based only on the "
                "work and tool results already present in the conversation, give a concise, honest final "
                "status. Clearly state anything that remains incomplete or unverified."
            ),
        }]
        self.on_event("[model]")
        stream_options = {"on_delta": self.on_stream} if self.on_stream else {}
        response = self.client.complete(
            final_messages,
            TOOL_SCHEMAS,
            tool_choice="none",
            **stream_options,
        )
        usage = response.get("usage") or {}
        input_tokens += int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
        output_tokens += int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise AgentError("Model did not return a final response after the round limit.")
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict) or message.get("tool_calls"):
            raise AgentError("Model did not respect the tool-free final response request.")
        content = message.get("content")
        if choice.get("finish_reason") != "stop" or not isinstance(content, str) or not content.strip():
            raise AgentError("Model could not produce final text after the round limit.")
        assistant: dict[str, Any] = {"role": "assistant", "content": content}
        if message.get("reasoning_content") is not None:
            assistant["reasoning_content"] = message["reasoning_content"]
            if (
                not self.on_stream
                and isinstance(message["reasoning_content"], str)
                and message["reasoning_content"].strip()
            ):
                self.on_event("[thinking]\n" + _clip(message["reasoning_content"].strip(), 8_000))
        self.messages.extend([final_messages[-1], assistant])
        self.on_event(
            f"[done] rounds={self.config.max_rounds} terminal_calls={terminal_call_count} "
            f"plan_items={len(self.plan.items)} "
            f"tokens={input_tokens}in/{output_tokens}out round_limit=true"
        )
        return content


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A minimal DeepSeek coding agent")
    parser.add_argument("prompt", nargs="*", help="one-shot task; omit for interactive mode")
    parser.add_argument("--workspace", default=".", help="workspace root (default: current directory)")
    parser.add_argument("--plain", action="store_true", help="disable the decorated terminal UI")
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
        ui = TerminalUI(plain=args.plain)
        agent = CodingAgent(
            DeepSeekClient(config),
            workspace,
            config,
            on_event=ui.event,
            on_stream=ui.stream_delta,
        )
        ui.banner(config.model, workspace)
        if args.prompt:
            ui.answer(agent.run(" ".join(args.prompt)))
            return 0
        while True:
            try:
                text = input(ui.prompt()).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not text:
                continue
            if text in {"/exit", "/quit"}:
                return 0
            if text == "/clear":
                agent.messages = [dict(agent.system_message)]
                agent.plan.reset()
                ui.event("Conversation cleared.")
                continue
            try:
                ui.answer(agent.run(text))
            except AgentError as exc:
                ui.error(str(exc))
    except AgentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
