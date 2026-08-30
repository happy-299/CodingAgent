import json
import os
import tempfile
import unittest
from pathlib import Path

from coding_agent import AgentError, CodingAgent, Config, TerminalTool


def config(**changes):
    values = dict(api_key="test", base_url="https://example.invalid", model="test", max_rounds=4)
    values.update(changes)
    return Config(**values)


class ScriptedClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def complete(self, messages, tools):
        self.requests.append(json.loads(json.dumps(messages)))
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
            output = TerminalTool(Path(directory)).run("pwd", timeout=5)
            self.assertIn(str(Path(directory).resolve()), output["output"])

    def test_sensitive_environment_is_not_inherited(self):
        os.environ["CODING_AGENT_TEST_SECRET"] = "must-not-leak"
        try:
            with tempfile.TemporaryDirectory() as directory:
                output = TerminalTool(Path(directory)).run(
                    "python3 -c \"import os; print(os.getenv('CODING_AGENT_TEST_SECRET'))\"",
                    timeout=5,
                )
            self.assertIn("None", output["output"])
            self.assertNotIn("must-not-leak", output["output"])
        finally:
            os.environ.pop("CODING_AGENT_TEST_SECRET", None)

    def test_timeout_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(Exception, "timed out"):
                TerminalTool(Path(directory)).run("sleep 5", timeout=1)

    def test_pipeline_propagates_upstream_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            result = TerminalTool(Path(directory)).run("(exit 7) | true", timeout=5)
        self.assertEqual(result["exit_code"], 7)


class AgentLoopTest(unittest.TestCase):
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
        self.assertTrue(events[0].startswith("[tool 1.1] terminal: printf a"))
        self.assertTrue(events[2].startswith("[tool 1.2] terminal: printf b"))

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
        self.assertIn("output limit reached", events[0])
        self.assertEqual(client.requests[1][-2]["reasoning_content"], "partial reasoning")
        self.assertEqual(client.requests[1][-1]["role"], "user")
        self.assertIn("Continue the same task", client.requests[1][-1]["content"])

    def test_nonzero_exit_and_truncation_are_structured(self):
        with tempfile.TemporaryDirectory() as directory:
            result = TerminalTool(Path(directory)).run(
                "python3 -c \"import sys; print('x' * 25000); sys.exit(7)\"",
                timeout=5,
            )
        self.assertEqual(result["exit_code"], 7)
        self.assertEqual(result["output_chars"], 25001)
        self.assertTrue(result["truncated"])
        self.assertIn("<truncated", result["output"])

    def test_max_rounds_is_a_hard_stop(self):
        call = {"id": "loop", "type": "function", "function": {"name": "terminal", "arguments": '{"command":"true"}'}}
        response = {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": None, "tool_calls": [call]}}]}
        client = ScriptedClient([response, response])
        with tempfile.TemporaryDirectory() as directory:
            agent = CodingAgent(client, Path(directory), config(max_rounds=2), on_event=lambda _: None)
            with self.assertRaisesRegex(AgentError, "stopped after 2"):
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


if __name__ == "__main__":
    unittest.main()
