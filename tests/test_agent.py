import http.client
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from coding_agent import (
    AgentError,
    CodingAgent,
    Config,
    DeepSeekClient,
    PlanState,
    SYSTEM_PROMPT,
    TerminalTool,
    TerminalUI,
    TOOL_SCHEMAS,
    _looks_like_verification_command,
)


def config(**changes):
    values = dict(
        api_key="test",
        base_url="https://example.invalid",
        model="test",
        max_rounds=4,
    )
    values.update(changes)
    return Config(**values)


class ScriptedClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []
        self.tool_choices = []

    def complete(self, messages, tools, tool_choice="auto"):
        self.requests.append(json.loads(json.dumps(messages)))
        self.tool_choices.append(tool_choice)
        return next(self.responses)


class TerminalToolTest(unittest.TestCase):
    def test_command_reports_exit_code(self):
        with tempfile.TemporaryDirectory() as directory:
            output = TerminalTool(Path(directory)).run("printf hello", timeout=5)
            self.assertEqual(output["exit_code"], 0)
            self.assertEqual(output["output"], "hello")
            self.assertFalse(output["truncated"])

    def test_command_starts_in_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            output = TerminalTool(Path(directory)).run("printf '%s|' \"$CODING_AGENT_WORKSPACE\"; pwd", timeout=5)
            self.assertIn(str(Path(directory).resolve()), output["output"])

    def test_sensitive_environment_is_not_inherited(self):
        os.environ["CODING_AGENT_TEST_SECRET"] = "must-not-leak"
        os.environ["MONKEY_PATCH"] = "ordinary-value"
        try:
            with tempfile.TemporaryDirectory() as directory:
                output = TerminalTool(Path(directory)).run(
                    "python3 -c \"import os; print(os.getenv('CODING_AGENT_TEST_SECRET'), os.getenv('MONKEY_PATCH'))\"",
                    timeout=5,
                )
            self.assertIn("None", output["output"])
            self.assertIn("ordinary-value", output["output"])
            self.assertNotIn("must-not-leak", output["output"])
        finally:
            os.environ.pop("CODING_AGENT_TEST_SECRET", None)
            os.environ.pop("MONKEY_PATCH", None)

    def test_timeout_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(Exception, "(?s)timed out.*before"):
                TerminalTool(Path(directory)).run("printf before; sleep 5", timeout=1)

    def test_pipeline_propagates_upstream_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            result = TerminalTool(Path(directory)).run("(exit 7) | true", timeout=5)
        self.assertEqual(result["exit_code"], 7)


class AgentLoopTest(unittest.TestCase):
    def test_agent_can_explore_before_deciding_to_plan(self):
        terminal_call = {
            "id": "explore-first",
            "type": "function",
            "function": {"name": "terminal", "arguments": '{"command":"touch explored.txt"}'},
        }
        client = ScriptedClient([
            {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": None, "tool_calls": [terminal_call]}}]},
            {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "Explored first."}}]},
        ])
        with tempfile.TemporaryDirectory() as directory:
            result = CodingAgent(
                client,
                Path(directory),
                config(),
                on_event=lambda _: None,
            ).run("do work")
            self.assertTrue(Path(directory, "explored.txt").exists())
        self.assertEqual(result, "Explored first.")
        self.assertEqual(len(client.requests), 2)
        self.assertTrue(all(choice == "auto" for choice in client.tool_choices))

    def test_two_orientation_calls_trigger_a_planning_decision_checkpoint(self):
        calls = [
            {
                "id": f"orient-{number}",
                "type": "function",
                "function": {"name": "terminal", "arguments": '{"command":"printf ok"}'},
            }
            for number in range(2)
        ]
        client = ScriptedClient([
            {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": None, "tool_calls": calls}}]},
            {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "No plan needed."}}]},
        ])
        with tempfile.TemporaryDirectory() as directory:
            result = CodingAgent(client, Path(directory), config(), on_event=lambda _: None).run("inspect")
        self.assertEqual(result, "No plan needed.")
        checkpoint = client.requests[1][-1]
        self.assertEqual(checkpoint["role"], "user")
        self.assertIn("decide now whether", checkpoint["content"])
        self.assertIn("smallest complete solution", checkpoint["content"])

    def test_native_tool_loop_and_history(self):
        call = {"id": "call_1", "type": "function", "function": {"name": "terminal", "arguments": '{"command":"printf ok > done.txt"}'}}
        client = ScriptedClient([
            {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": None, "reasoning_content": "inspect", "tool_calls": [call]}}]},
            {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "Finished."}}]},
        ])
        with tempfile.TemporaryDirectory() as directory:
            agent = CodingAgent(client, Path(directory), config(), on_event=lambda _: None)
            self.assertEqual(agent.run("create a file"), "Finished.")
            self.assertEqual(Path(directory, "done.txt").read_text(), "ok")
        second_request = client.requests[1]
        self.assertEqual(second_request[-2]["reasoning_content"], "inspect")
        self.assertEqual(second_request[-1]["role"], "tool")
        self.assertTrue(json.loads(second_request[-1]["content"])["ok"])

    def test_unknown_tool_returns_error_to_model(self):
        call = {"id": "bad", "type": "function", "function": {"name": "missing", "arguments": "{}"}}
        client = ScriptedClient([
            {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": None, "tool_calls": [call]}}]},
            {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "Recovered."}}]},
        ])
        with tempfile.TemporaryDirectory() as directory:
            agent = CodingAgent(client, Path(directory), config(), on_event=lambda _: None)
            self.assertEqual(agent.run("test"), "Recovered.")
        self.assertFalse(json.loads(client.requests[1][-1]["content"])["ok"])

    def test_parallel_calls_have_distinct_event_labels(self):
        calls = [
            {"id": "a", "type": "function", "function": {"name": "terminal", "arguments": '{"command":"printf a"}'}},
            {"id": "b", "type": "function", "function": {"name": "terminal", "arguments": '{"command":"printf b"}'}},
        ]
        client = ScriptedClient([
            {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": None, "tool_calls": calls}}]},
            {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "Done."}}]},
        ])
        events = []
        with tempfile.TemporaryDirectory() as directory:
            CodingAgent(client, Path(directory), config(), on_event=events.append).run("test")
        tool_events = [event for event in events if event.startswith("[tool ")]
        self.assertTrue(tool_events[0].startswith("[tool 1.1] terminal: printf a"))
        self.assertTrue(tool_events[1].startswith("[tool 1.2] terminal: printf b"))
        self.assertIn("[command 1.1]\nprintf a", events)
        self.assertIn("[output 1.2]\nb", events)

    def test_length_finish_recovers_and_continues(self):
        call = {"id": "continued", "type": "function", "function": {"name": "terminal", "arguments": '{"command":"printf recovered"}'}}
        client = ScriptedClient([
            {"choices": [{"finish_reason": "length", "message": {"role": "assistant", "content": None, "reasoning_content": "partial reasoning"}}]},
            {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": None, "reasoning_content": "continued reasoning", "tool_calls": [call]}}]},
            {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "Recovered."}}]},
        ])
        events = []
        with tempfile.TemporaryDirectory() as directory:
            result = CodingAgent(client, Path(directory), config(), on_event=events.append).run("complex task")
        self.assertEqual(result, "Recovered.")
        self.assertTrue(any("output limit reached" in event for event in events))
        self.assertTrue(any(event.startswith("[thinking]") for event in events))
        self.assertEqual(client.requests[1][-2]["reasoning_content"], "partial reasoning")
        self.assertEqual(client.requests[1][-1]["role"], "user")
        self.assertIn("Continue the same task", client.requests[1][-1]["content"])

    def test_verification_final_rejects_textual_tool_markup(self):
        call = {
            "id": "verify",
            "type": "function",
            "function": {
                "name": "terminal",
                "arguments": '{"command":"python3 -m unittest"}',
            },
        }
        client = ScriptedClient([
            {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": None, "tool_calls": [call]}}]},
            {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "<｜｜DSML｜｜tool_calls>fake</｜｜DSML｜｜tool_calls>"}}]},
            {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "Verified."}}]},
        ])
        with tempfile.TemporaryDirectory() as directory:
            result = CodingAgent(client, Path(directory), config(), on_event=lambda _: None).run("verify")
        self.assertEqual(result, "Verified.")
        self.assertEqual(len(client.requests), 3)

    def test_textual_tool_markup_is_retried_with_native_tools(self):
        first = {
            "id": "first",
            "type": "function",
            "function": {"name": "terminal", "arguments": '{"command":"printf ready"}'},
        }
        second = {
            "id": "second",
            "type": "function",
            "function": {"name": "terminal", "arguments": '{"command":"touch completed.txt"}'},
        }
        client = ScriptedClient([
            {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": None, "tool_calls": [first]}}]},
            {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "<｜｜DSML｜｜tool_calls>not executed</｜｜DSML｜｜tool_calls>"}}]},
            {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": None, "tool_calls": [second]}}]},
            {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "Finished."}}]},
        ])
        with tempfile.TemporaryDirectory() as directory:
            result = CodingAgent(client, Path(directory), config(), on_event=lambda _: None).run("finish the task")
            self.assertTrue(Path(directory, "completed.txt").exists())
        self.assertEqual(result, "Finished.")
        self.assertEqual(client.tool_choices[1], "auto")


    def test_completed_plan_starts_audit_and_reuses_verification_evidence(self):
        planning = {
            "id": "plan-start",
            "type": "function",
            "function": {
                "name": "update_plan",
                "arguments": json.dumps({
                    "items": [
                        {"step": "Implement and verify", "status": "in_progress"},
                    ]
                }),
            },
        }
        verify = {
            "id": "verify",
            "type": "function",
            "function": {
                "name": "terminal",
                "arguments": json.dumps({
                    "command": "python3 -c \"print('verified')\" --selftest",
                }),
            },
        }
        complete_plan = {
            "id": "plan-complete",
            "type": "function",
            "function": {
                "name": "update_plan",
                "arguments": json.dumps({
                    "items": [
                        {"step": "Implement and verify", "status": "completed"},
                    ]
                }),
            },
        }
        client = ScriptedClient([
            {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": None, "tool_calls": [planning]}}]},
            {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": None, "tool_calls": [verify]}}]},
            {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": None, "tool_calls": [complete_plan]}}]},
            {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "Complete and verified."}}]},
        ])
        events = []
        with tempfile.TemporaryDirectory() as directory:
            result = CodingAgent(
                client,
                Path(directory),
                config(max_rounds=6),
                on_event=events.append,
            ).run("build a feature")
        self.assertEqual(result, "Complete and verified.")
        self.assertEqual(len(client.requests), 4)
        audit = client.requests[3][-1]
        self.assertEqual(audit["role"], "user")
        self.assertIn("already succeeded", audit["content"])
        self.assertIn("do not rerun checks", audit["content"])
        self.assertTrue(any("plan complete" in event for event in events))

    def test_plan_progress_checkpoint_follows_sustained_execution(self):
        planning = {
            "id": "plan-start",
            "type": "function",
            "function": {
                "name": "update_plan",
                "arguments": json.dumps({
                    "items": [{"step": "Deliver the requested outcome", "status": "in_progress"}]
                }),
            },
        }
        terminal_calls = [
            {
                "id": f"work-{number}",
                "type": "function",
                "function": {"name": "terminal", "arguments": '{"command":"printf work"}'},
            }
            for number in range(8)
        ]
        completed = {
            "id": "plan-complete",
            "type": "function",
            "function": {
                "name": "update_plan",
                "arguments": json.dumps({
                    "items": [{"step": "Deliver the requested outcome", "status": "completed"}]
                }),
            },
        }
        client = ScriptedClient([
            {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": None, "tool_calls": [planning]}}]},
            {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": None, "tool_calls": terminal_calls}}]},
            {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": None, "tool_calls": [completed]}}]},
            {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "Done."}}]},
        ])
        events = []
        with tempfile.TemporaryDirectory() as directory:
            result = CodingAgent(
                client,
                Path(directory),
                config(max_rounds=6),
                on_event=events.append,
            ).run("non-trivial work")
        self.assertEqual(result, "Done.")
        checkpoint = client.requests[2][-1]
        self.assertEqual(checkpoint["role"], "user")
        self.assertIn("Progress checkpoint", checkpoint["content"])
        self.assertTrue(any("progress checkpoint" in event for event in events))

    def test_nonzero_exit_and_truncation_are_structured(self):
        with tempfile.TemporaryDirectory() as directory:
            result = TerminalTool(Path(directory)).run(
                "python3 -c \"import sys; print('x' * 25000); sys.exit(7)\"",
                timeout=5,
            )
        self.assertEqual(result["exit_code"], 7)
        self.assertEqual(result["output_bytes"], 25001)
        self.assertTrue(result["truncated"])
        self.assertIn("<truncated", result["output"])

    def test_round_limit_disables_tools_and_requests_final_text(self):
        call = {"id": "loop", "type": "function", "function": {"name": "terminal", "arguments": '{"command":"true"}'}}
        response = {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": None, "tool_calls": [call]}}]}
        final = {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "Stopped safely; work may be incomplete."}}]}
        client = ScriptedClient([response, response, final])
        with tempfile.TemporaryDirectory() as directory:
            agent = CodingAgent(client, Path(directory), config(max_rounds=2), on_event=lambda _: None)
            result = agent.run("loop forever")
        self.assertEqual(result, "Stopped safely; work may be incomplete.")
        self.assertEqual(client.tool_choices[-1], "none")
        self.assertIn("execution-round limit", client.requests[2][-1]["content"])

    def test_round_limit_rejects_another_tool_call(self):
        call = {"id": "loop", "type": "function", "function": {"name": "terminal", "arguments": '{"command":"true"}'}}
        response = {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": None, "tool_calls": [call]}}]}
        client = ScriptedClient([response, response])
        with tempfile.TemporaryDirectory() as directory:
            agent = CodingAgent(client, Path(directory), config(max_rounds=1), on_event=lambda _: None)
            with self.assertRaisesRegex(AgentError, "tool-free"):
                agent.run("loop forever")

    def test_history_trimming_keeps_complete_latest_turn(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = CodingAgent(
                ScriptedClient([]),
                Path(directory),
                config(max_history_chars=220),
                on_event=lambda _: None,
            )
        agent.messages.extend([
            {"role": "user", "content": "old " * 100},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "latest"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "x"}]},
            {"role": "tool", "tool_call_id": "x", "content": "result"},
        ])
        agent._trim_history()
        self.assertEqual([message["role"] for message in agent.messages], ["system", "user", "assistant", "tool"])
        self.assertEqual(agent.messages[1]["content"], "latest")

    def test_content_filter_is_reported_as_error(self):
        client = ScriptedClient([
            {"choices": [{"finish_reason": "content_filter", "message": {"role": "assistant", "content": None}}]},
        ])
        with tempfile.TemporaryDirectory() as directory:
            agent = CodingAgent(client, Path(directory), config(), on_event=lambda _: None)
            with self.assertRaisesRegex(AgentError, "content_filter"):
                agent.run("task")

    def test_missing_tool_call_id_is_rejected(self):
        call = {"type": "function", "function": {"name": "terminal", "arguments": '{"command":"true"}'}}
        client = ScriptedClient([
            {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": None, "tool_calls": [call]}}]},
        ])
        with tempfile.TemporaryDirectory() as directory:
            agent = CodingAgent(client, Path(directory), config(), on_event=lambda _: None)
            with self.assertRaisesRegex(AgentError, "valid id"):
                agent.run("task")

    def test_missing_assistant_message_is_rejected(self):
        client = ScriptedClient([{"choices": [{"finish_reason": "stop"}]}])
        with tempfile.TemporaryDirectory() as directory:
            agent = CodingAgent(client, Path(directory), config(), on_event=lambda _: None)
            with self.assertRaisesRegex(AgentError, "assistant message"):
                agent.run("task")


class ConfigTest(unittest.TestCase):
    def test_invalid_integer_environment_is_user_facing(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test", "AGENT_MAX_ROUNDS": "many"}, clear=True):
            with self.assertRaisesRegex(AgentError, "AGENT_MAX_ROUNDS must be an integer"):
                Config.from_env()

    def test_out_of_range_environment_is_user_facing(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test", "AGENT_MAX_ROUNDS": "0"}, clear=True):
            with self.assertRaisesRegex(AgentError, "between 1 and 100"):
                Config.from_env()

    def test_invalid_reasoning_effort_is_user_facing(self):
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "test", "AGENT_REASONING_EFFORT": "extreme"},
            clear=True,
        ):
            with self.assertRaisesRegex(AgentError, "must be one of low, medium, or high"):
                Config.from_env()

    def test_defaults_balance_reasoning_and_context(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}, clear=True):
            loaded = Config.from_env()
        self.assertEqual(loaded.reasoning_effort, "medium")
        self.assertEqual(loaded.max_rounds, 80)
        self.assertEqual(loaded.max_history_chars, 160_000)


class DeepSeekClientTest(unittest.TestCase):
    def test_configured_reasoning_effort_is_sent_with_tools(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def __iter__(self):
                yield b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n'
                yield b"data: [DONE]\n"

        captured = {}

        def open_request(request, timeout):
            captured.update(json.loads(request.data.decode("utf-8")))
            return Response()

        client = DeepSeekClient(config(reasoning_effort="low"))
        with patch("urllib.request.urlopen", side_effect=open_request):
            client.complete([], TOOL_SCHEMAS)
        self.assertEqual(captured["reasoning_effort"], "low")
        self.assertEqual(captured["thinking"], {"type": "enabled"})

    def test_incomplete_http_response_is_retried(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def __iter__(self):
                chunks = [
                    {"choices": [{"delta": {"reasoning_content": "check "}, "finish_reason": None}]},
                    {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]},
                    {"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 1}},
                ]
                for chunk in chunks:
                    yield ("data: " + json.dumps(chunk) + "\n\n").encode()
                yield b"data: [DONE]\n\n"

        client = DeepSeekClient(config())
        with patch("urllib.request.urlopen", side_effect=[http.client.IncompleteRead(b"partial"), Response()]), patch(
            "time.sleep"
        ):
            response = client.complete([], [])
        self.assertEqual(response["choices"][0]["message"]["content"], "ok")

    def test_stream_reassembles_fragmented_tool_calls(self):
        lines = [
            {
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "terminal", "arguments": '{"command":"pri'},
                        }]
                    },
                    "finish_reason": None,
                }]
            },
            {
                "choices": [{
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'ntf ok"}'}}]},
                    "finish_reason": "tool_calls",
                }]
            },
        ]
        response = DeepSeekClient._read_stream(
            [("data: " + json.dumps(line) + "\n").encode() for line in lines] + [b"data: [DONE]\n"],
            None,
        )
        call = response["choices"][0]["message"]["tool_calls"][0]
        self.assertEqual(call["id"], "call-1")
        self.assertEqual(call["function"]["arguments"], '{"command":"printf ok"}')

class ArchitectureTest(unittest.TestCase):
    def test_terminal_is_the_only_execution_tool(self):
        tool_names = [schema["function"]["name"] for schema in TOOL_SCHEMAS]
        self.assertEqual(tool_names, ["terminal", "update_plan"])
        plan_description = TOOL_SCHEMAS[1]["function"]["description"]
        self.assertIn("only tracks state", plan_description)

    def test_verification_detection_covers_generic_project_checks(self):
        commands = [
            "python3 -m unittest",
            "python3 -m py_compile app.py",
            "python3 app.py --selftest",
            "npm run build",
            "cargo check",
            "ruff check .",
        ]
        self.assertTrue(all(_looks_like_verification_command(command) for command in commands))
        self.assertFalse(_looks_like_verification_command("ls -la"))
        self.assertFalse(_looks_like_verification_command(
            "python3 --version; which python3; which pytest 2>/dev/null || echo no pytest"
        ))
        self.assertFalse(_looks_like_verification_command("pytest --version"))
        self.assertTrue(_looks_like_verification_command("which pytest && pytest -q"))

    def test_system_prompt_contains_no_named_evaluation_scenarios(self):
        for task_name in ("MiniQueue", "2048", "HTTP"):
            self.assertNotIn(task_name, SYSTEM_PROMPT)

    def test_system_prompt_requires_behavioral_interactive_verification(self):
        self.assertIn("observable state transitions", SYSTEM_PROMPT)
        self.assertIn("tautological assertions", SYSTEM_PROMPT)
        self.assertIn("exit code 0 alone", SYSTEM_PROMPT)

    def test_plan_requires_unique_steps_and_one_active_item(self):
        plan = PlanState()
        plan.update([
            {"step": "Inspect", "status": "completed"},
            {"step": "Implement", "status": "in_progress"},
        ])
        self.assertTrue(plan.has_open_items)
        with self.assertRaisesRegex(AgentError, "At most one"):
            plan.update([
                {"step": "A", "status": "in_progress"},
                {"step": "B", "status": "in_progress"},
            ])
        with self.assertRaisesRegex(AgentError, "unique"):
            plan.update([
                {"step": "A", "status": "pending"},
                {"step": "A", "status": "completed"},
            ])

    def test_compaction_preserves_complete_recent_protocol_unit(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = CodingAgent(
                ScriptedClient([]),
                Path(directory),
                config(max_history_chars=500),
                on_event=lambda _: None,
            )
        agent.messages.append({"role": "user", "content": "objective"})
        agent.task_start = 1
        agent.memory.reset("objective", [])
        for number in range(3):
            agent.messages.extend([
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": f"call-{number}", "function": {"name": "terminal", "arguments": "{}"}}],
                },
                {"role": "tool", "tool_call_id": f"call-{number}", "content": "x" * 300},
            ])
        compacted = agent._messages_for_model()
        self.assertLess(len(compacted), len(agent.messages))
        self.assertEqual(compacted[-2]["role"], "assistant")
        self.assertEqual(compacted[-1]["role"], "tool")
        self.assertEqual(compacted[-2]["tool_calls"][0]["id"], compacted[-1]["tool_call_id"])

    def test_reasoning_compaction_preserves_protocol_and_recent_thoughts(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = CodingAgent(
                ScriptedClient([]),
                Path(directory),
                config(max_history_chars=10_000),
                on_event=lambda _: None,
            )
        messages = [{"role": "system", "content": "system"}]
        for number in range(5):
            messages.extend([
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": str(number) * 4_000,
                    "tool_calls": [{
                        "id": f"call-{number}",
                        "type": "function",
                        "function": {"name": "terminal", "arguments": "{}"},
                    }],
                },
                {"role": "tool", "tool_call_id": f"call-{number}", "content": "observed"},
            ])
        condensed = agent._condense_reasoning_history(messages)
        self.assertEqual(len(condensed), len(messages))
        self.assertLess(len(condensed[1]["reasoning_content"]), 4_000)
        self.assertEqual(condensed[-4]["reasoning_content"], "3" * 4_000)
        self.assertEqual(condensed[-2]["reasoning_content"], "4" * 4_000)
        for assistant, tool in zip(condensed[1::2], condensed[2::2]):
            self.assertEqual(assistant["tool_calls"][0]["id"], tool["tool_call_id"])

    def test_model_context_compacts_reasoning_before_dropping_protocol_units(self):
        events = []
        with tempfile.TemporaryDirectory() as directory:
            agent = CodingAgent(
                ScriptedClient([]),
                Path(directory),
                config(max_history_chars=20_000),
                on_event=events.append,
            )
        agent.messages.append({"role": "user", "content": "objective"})
        agent.task_start = 1
        agent.memory.reset("objective", [])
        for number in range(4):
            agent.messages.extend([
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": str(number) * 5_000,
                    "tool_calls": [{
                        "id": f"call-{number}",
                        "type": "function",
                        "function": {"name": "terminal", "arguments": "{}"},
                    }],
                },
                {"role": "tool", "tool_call_id": f"call-{number}", "content": "result"},
            ])
        compacted = agent._messages_for_model()
        self.assertEqual(len(compacted), len(agent.messages))
        self.assertLess(len(json.dumps(compacted)), len(json.dumps(agent.messages)))
        self.assertTrue(any("reasoning_compacted" in event for event in events))


class FakeTTY(io.StringIO):
    def isatty(self):
        return True


class TerminalUITest(unittest.TestCase):
    def test_plain_render_contains_all_demo_sections_without_ansi(self):
        stream = io.StringIO()
        ui = TerminalUI(stream=stream)
        ui.banner("example-model", Path("/workspace"))
        ui.event("[thinking]\nInspect the repository first.")
        ui.event("[tool 1.1] terminal")
        ui.event("[command 1.1]\npython3 -m unittest")
        ui.event("[result 1.1] exit=0 output=3 bytes")
        ui.event("[output 1.1]\nOK")
        ui.answer("Finished.")
        rendered = stream.getvalue()
        for expected in ("CODING AGENT", "THINKING", "COMMAND", "OUTPUT", "AGENT", "Finished."):
            self.assertIn(expected, rendered)
        self.assertNotIn("\033[", rendered)

    def test_streamed_answer_is_not_printed_twice(self):
        stream = io.StringIO()
        ui = TerminalUI(stream=stream)
        ui.event("[model]")
        ui.stream_delta("reasoning", "Inspecting…")
        ui.stream_delta("content", "Finished.")
        ui.event("[done] rounds=1")
        ui.answer("Finished.")
        rendered = stream.getvalue()
        self.assertIn("THINKING · LIVE", rendered)
        self.assertIn("AGENT · LIVE", rendered)
        self.assertEqual(rendered.count("Finished."), 1)

    def test_tty_dashboard_collapses_reasoning_and_keeps_input_bar_fixed(self):
        stream = FakeTTY()
        with patch("coding_agent.shutil.get_terminal_size", return_value=os.terminal_size((80, 20))):
            ui = TerminalUI(stream=stream)
            ui.banner("deepseek-v4-flash", Path("/workspace/demo"))
            ui.event("[model]")
            ui.stream_delta("reasoning", "private reasoning should stay collapsed")
            ui.stream_delta("content", "working")
            ui.prompt()
        frame = stream.getvalue().rsplit("\033[H", 1)[-1]
        self.assertIn("MODEL deepseek-v4-flash", frame)
        self.assertNotIn("MODEL  deepseek-v4-flash", frame)
        self.assertNotIn("newest output stays visible", frame)
        self.assertIn("THINKING · COMPLETE · COLLAPSED", frame)
        self.assertNotIn("private reasoning should stay collapsed", frame)
        self.assertLess(frame.index("THINKING · COMPLETE"), frame.index("AGENT · LIVE"))
        self.assertIn("❯", frame)
        self.assertIn("\033[20;6H\033[?25h", frame)
        self.assertEqual(stream.getvalue().count("\033[2K"), 20 * 5)

    def test_tty_reasoning_expands_inside_bounded_scrolling_pane(self):
        stream = FakeTTY()
        with patch("coding_agent.shutil.get_terminal_size", return_value=os.terminal_size((80, 20))):
            ui = TerminalUI(stream=stream)
            ui.banner("test-model", Path("/workspace"))
            ui.event("[model]")
            ui.stream_delta("reasoning", "\n".join(f"reasoning-{i}" for i in range(10)))
            self.assertTrue(ui.toggle_thinking())
        frame = stream.getvalue().rsplit("\033[H", 1)[-1]
        self.assertIn("THINKING · LIVE", frame)
        self.assertIn("reasoning-9", frame)
        self.assertNotIn("reasoning-0", frame)
        self.assertEqual(frame.count("\033[2K"), 20)

    def test_completed_thinking_card_expands_on_first_toggle(self):
        stream = FakeTTY()
        with patch("coding_agent.shutil.get_terminal_size", return_value=os.terminal_size((80, 20))):
            ui = TerminalUI(stream=stream)
            ui.banner("test-model", Path("/workspace"))
            ui.toggle_thinking()
            ui.event("[model]")
            ui.stream_delta("reasoning", "completed reasoning")
            ui.stream_delta("content", "formal output")
            self.assertTrue(ui.toggle_thinking())
        frame = stream.getvalue().rsplit("\033[H", 1)[-1]
        self.assertIn("THINKING · COMPLETE", frame)
        self.assertIn("completed reasoning", frame)

    def test_tty_tool_events_are_presented_as_one_card(self):
        stream = FakeTTY()
        with patch("coding_agent.shutil.get_terminal_size", return_value=os.terminal_size((80, 20))):
            ui = TerminalUI(stream=stream)
            ui.banner("test-model", Path("/workspace"))
            ui.event("[tool 2.1] terminal: python3 -m unittest")
            ui.event("[command 2.1]\npython3 -m unittest")
            ui.event("[result 2.1] exit=0 output=3 bytes")
            ui.event("[output 2.1]\nOK")
        frame = stream.getvalue().rsplit("\033[H", 1)[-1]
        self.assertIn("TOOL 2.1 · terminal", frame)
        self.assertIn("$ python3 -m unittest", frame)
        self.assertIn("↳ exit=0 output=3 bytes", frame)
        self.assertIn("OK", frame)
        self.assertEqual(frame.count("TOOL 2.1"), 1)
        self.assertEqual(frame.count("python3 -m unittest"), 1)

    def test_tty_system_event_separates_title_from_inline_detail(self):
        stream = FakeTTY()
        with patch("coding_agent.shutil.get_terminal_size", return_value=os.terminal_size((80, 20))):
            ui = TerminalUI(stream=stream)
            ui.banner("test-model", Path("/workspace"))
            ui.event("[planning checkpoint] deciding whether a plan is useful")
        frame = stream.getvalue().rsplit("\033[H", 1)[-1]
        self.assertIn("planning checkpoint: deciding whether a plan is useful", frame)
        self.assertNotIn("checkpoint]", frame)

    def test_tty_activity_scrolls_without_moving_status_footer(self):
        stream = FakeTTY()
        with patch("coding_agent.shutil.get_terminal_size", return_value=os.terminal_size((80, 20))):
            ui = TerminalUI(stream=stream)
            ui.banner("test-model", Path("/workspace"))
            for index in range(10):
                event_id = f"{index + 1}.1"
                ui.event(f"[tool {event_id}] terminal")
                ui.event(f"[command {event_id}]\necho output-{index}")
                ui.event(f"[result {event_id}] exit=0 output=10 bytes")
                ui.event(f"[output {event_id}]\noutput-{index}")
            ui.prompt()
            newest = stream.getvalue().rsplit("\033[H", 1)[-1]
            ui.scroll(100)
            oldest = stream.getvalue().rsplit("\033[H", 1)[-1]
            ui.scroll(-100)
            restored = stream.getvalue().rsplit("\033[H", 1)[-1]
        self.assertIn("output-9", newest)
        self.assertNotIn("output-0", newest)
        self.assertIn("output-0", oldest)
        self.assertNotIn("output-9", oldest)
        self.assertIn("output-9", restored)
        self.assertIn("MODEL test-model", restored)
        self.assertIn("❯", restored)
        self.assertEqual(restored.count("\033[2K"), 20)

    def test_tty_preserves_thinking_output_sequence_across_model_rounds(self):
        stream = FakeTTY()
        with patch("coding_agent.shutil.get_terminal_size", return_value=os.terminal_size((100, 30))):
            ui = TerminalUI(stream=stream)
            ui.banner("test-model", Path("/workspace"))
            ui.event("[model]")
            ui.stream_delta("reasoning", "first reasoning")
            ui.stream_delta("content", "First formal output")
            ui.event("[model]")
            ui.stream_delta("reasoning", "second reasoning")
            ui.stream_delta("content", "Second formal output")
        frame = stream.getvalue().rsplit("\033[H", 1)[-1]
        first_thinking = frame.index("THINKING · COMPLETE")
        first_output = frame.index("First formal output")
        second_thinking = frame.index("THINKING · COMPLETE", first_thinking + 1)
        second_output = frame.index("Second formal output")
        self.assertLess(first_thinking, first_output)
        self.assertLess(first_output, second_thinking)
        self.assertLess(second_thinking, second_output)

    def test_tty_renders_markdown_and_aligns_cjk_borders(self):
        stream = FakeTTY()
        with patch("coding_agent.shutil.get_terminal_size", return_value=os.terminal_size((80, 22))):
            ui = TerminalUI(stream=stream)
            ui.banner("test-model", Path("/工作区/演示"))
            ui.answer("# 总结\n\n**结果**：创建了 `你好.txt`。\n\n- 第一项内容很长，需要正确换行而不能挤坏右侧边框")
        frame = stream.getvalue().rsplit("\033[H", 1)[-1]
        self.assertIn("▌ 总结", frame)
        self.assertIn("• 第一项", frame)
        self.assertNotIn("**结果**", frame)
        self.assertNotIn("`你好.txt`", frame)
        visible_lines = [ui._clean(line) for line in frame.splitlines() if "│" in ui._clean(line)]
        for line in visible_lines:
            self.assertEqual(ui._cell_width(line), 80, line)

    def test_mouse_wheel_sequences_are_recognized(self):
        self.assertEqual(TerminalUI._mouse_button(b"\x1b[<64;10;5M"), 64)
        self.assertEqual(TerminalUI._mouse_button(b"\x1b[<65;10;5M"), 65)
        self.assertIsNone(TerminalUI._mouse_button(b"\x1b[A"))


if __name__ == "__main__":
    unittest.main()
