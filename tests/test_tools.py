import json

from aurora.tools import Toolbox


def _box(cfg):
    return Toolbox.default(cfg.workspace, cfg.knowledge_dir, lambda q, k: [])


def test_write_then_read(cfg):
    box = _box(cfg)
    box.dispatch("write_file", {"path": "a.txt", "content": "line1\nline2"})
    r = box.dispatch("read_file", {"path": "a.txt"})
    assert r.ok and "line1" in r.output


def test_path_traversal_blocked(cfg):
    box = _box(cfg)
    r = box.dispatch("write_file", {"path": "../../escape.txt", "content": "x"})
    assert not r.ok
    assert not (cfg.workspace.parent / "escape.txt").exists()


def test_run_command_captures_output(cfg):
    box = _box(cfg)
    r = box.dispatch("run_command", {"command": "echo hi-from-shell"})
    assert r.ok and "hi-from-shell" in r.output


def test_run_python(tmp_path, cfg):
    box = _box(cfg)
    r = box.dispatch("run_python", {"code": "print(21*2)"})
    assert r.ok and "42" in r.output


def test_run_python_blocks_dangerous_import(cfg):
    box = _box(cfg)
    r = box.dispatch("run_python", {"code": "import os\nprint(os.getcwd())"})
    assert not r.ok


def test_list_files(cfg):
    box = _box(cfg)
    (cfg.workspace / "x.txt").write_text("x")
    (cfg.workspace / "sub").mkdir()
    r = box.dispatch("list_files", {"path": "."})
    assert r.ok and "x.txt" in r.output


def test_search_knowledge_returns_hits(cfg):
    hits = [{"text": "Aurora runs locally.", "source": "doc.md", "score": 0.9}]
    box = Toolbox.default(cfg.workspace, cfg.knowledge_dir, lambda q, k: hits)
    r = box.dispatch("search_knowledge", {"query": "local agent", "top_k": 3})
    assert r.ok and "Aurora runs locally." in r.output
    assert r.meta["hits"] == hits
