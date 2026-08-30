# End-to-end evaluation

The agent was exercised against an isolated Python Todo CLI project using the real `deepseek-v4-flash` API, not a scripted model.

## Initial implementation

Given only a requirements file and tests, the agent inspected the workspace, created the implementation, ran the tests, and performed extra edge-case checks. It completed after 9 tool rounds. Independent verification passed all 3 tests covering persistence, unknown IDs, and malformed JSON.

## Incremental change

A `stats` command and a failing test were then added. Starting from the existing code, the agent:

1. inspected the requirements, implementation, and tests;
2. ran the suite and reproduced the failure;
3. made a focused edit through the terminal;
4. reran the full suite and checked empty/corrupt database behavior.

It completed after 5 tool rounds. Independent verification passed all 4 tests. This second run also verified that `--workspace` loads credentials from the launch directory and that multiple terminal calls in one model response are handled correctly.

The disposable evaluation fixture lived under ignored `tmp/` and is not part of the submitted implementation.
