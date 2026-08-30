import json
import os
import tempfile
import unittest
from pathlib import Path

from coding_agent import CodingAgent, Config, TerminalTool


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
            self.assertIn("exit_code=0", output)
            self.assertIn("hello", output)

    def test_command_starts_in_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            output = TerminalTool(Path(directory)).run("pwd", timeout=5)
            self.assertIn(str(Path(directory).resolve()), output)

    def test_sensitive_environment_is_not_inherited(self):
        os.environ["CODING_AGENT_TEST_SECRET"] = "must-not-leak"
        try:
            with tempfile.TemporaryDirectory() as directory:
                output = TerminalTool(Path(directory)).run(
                    "python3 -c \"import os; print(os.getenv('CODING_AGENT_TEST_SECRET'))\"",
                    timeout=5,
                )
            self.assertIn("None", output)
            self.assertNotIn("must-not-leak", output)
        finally:
            os.environ.pop("CODING_AGENT_TEST_SECRET", None)

    def test_timeout_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(Exception, "timed out"):
                TerminalTool(Path(directory)).run("sleep 5", timeout=1)


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


if __name__ == "__main__":
    unittest.main()
