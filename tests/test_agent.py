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
            for number in range(6)
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
        self.assertEqual(loaded.max_history_chars, 120_000)


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

    def test_system_prompt_contains_no_named_evaluation_scenarios(self):
        for task_name in ("MiniQueue", "2048", "HTTP"):
            self.assertNotIn(task_name, SYSTEM_PROMPT)

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


if __name__ == "__main__":
    unittest.main()
