from aurora.agent import Agent
from aurora.memory import Memory
from aurora.tools import Toolbox

from conftest import ScriptedChat, tool_call, final


def test_agent_writes_file_then_answers(cfg, embedder, tmp_path):
    backend = ScriptedChat(
        [
            tool_call("write_file", {"path": "note.txt", "content": "hello world"}, cid="c1"),
            final("I created note.txt with the requested content."),
        ]
    )
    toolbox = Toolbox.default(cfg.workspace, cfg.knowledge_dir, lambda q, k: [])
    agent = Agent(cfg, backend, toolbox)

    res = agent.run("write note.txt containing hello world")

    assert res.answer
    assert res.tool_calls == 1
    written = (cfg.workspace / "note.txt").read_text()
    assert written == "hello world"
    # a tool step and a final step were emitted
    kinds = [s.kind for s in res.steps]
    assert "tool" in kinds and "final" in kinds


def test_agent_retries_tool_failure_gracefully(cfg, embedder):
    # First call writes to a bad path (will fail path-guard), model recovers.
    backend = ScriptedChat(
        [
            tool_call("write_file", {"path": "../escape.txt", "content": "x"}, cid="c1"),
            tool_call("write_file", {"path": "safe.txt", "content": "ok"}, cid="c2"),
            final("Created safe.txt instead."),
        ]
    )
    toolbox = Toolbox.default(cfg.workspace, cfg.knowledge_dir, lambda q, k: [])
    agent = Agent(cfg, backend, toolbox)
    res = agent.run("save something")
    assert (cfg.workspace / "safe.txt").exists()
    assert not (cfg.workspace / "escape.txt").exists()


def test_agent_respects_step_budget(cfg, embedder):
    from aurora.exceptions import MaxStepsExceeded

    # model keeps calling tools forever; budget forces a summary attempt.
    backend = ScriptedChat([tool_call("list_files", {}, cid="c1")])
    toolbox = Toolbox.default(cfg.workspace, cfg.knowledge_dir, lambda q, k: [])
    agent = Agent(cfg, backend, toolbox, memory=Memory(cfg, "sys"))
    # With max_steps=2 the loop exhausts and _summarize is called. Our scripted
    # model only returns tool calls, so MaxStepsExceeded is expected.
    try:
        agent.run("loop", max_steps=2)
        assert False, "expected MaxStepsExceeded or summary"
    except MaxStepsExceeded:
        pass
