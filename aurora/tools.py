"""Toolset for the Aurora agent.

Each tool is a small, side-effect-controlled capability. Tools run inside the
agent's workspace and a constrained sandbox so the autonomous loop can act
without unlimited blast radius.
"""

from __future__ import annotations

import ast
import json
import subprocess
import textwrap
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .exceptions import ToolError
from .types import ToolResult, ToolSpec


class Tool:
    """Base class. Subclasses implement ``run`` and expose a ``spec``."""

    name: str = "tool"
    description: str = ""

    def spec(self) -> ToolSpec:
        raise NotImplementedError

    def run(self, **kwargs) -> ToolResult:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class Toolbox:
    """Holds the available tools and dispatches by name."""

    tools: Dict[str, Tool]

    def specs(self) -> List[ToolSpec]:
        return [t.spec() for t in self.tools.values()]

    def dispatch(self, name: str, arguments: dict) -> ToolResult:
        tool = self.tools.get(name)
        if tool is None:
            return ToolResult(ok=False, output=f"unknown tool: {name}")
        try:
            return tool.run(**arguments)
        except ToolError as exc:
            return ToolResult(ok=False, output=f"tool error: {exc}")
        except Exception as exc:  # surface, never crash the loop
            return ToolResult(ok=False, output=f"exception: {type(exc).__name__}: {exc}\n{traceback.format_exc()[-800:]}")

    @classmethod
    def default(cls, workspace: Path, knowledge: Path, retriever: Callable[[str, int], List[dict]]) -> "Toolbox":
        return cls(
            tools={
                "read_file": ReadFile(workspace),
                "write_file": WriteFile(workspace),
                "list_files": ListFiles(workspace),
                "run_command": RunCommand(workspace),
                "run_python": RunPython(),
                "web_fetch": WebFetch(),
                "search_knowledge": SearchKnowledge(retriever),
            }
        )


def _safe_path(workspace: Path, path: str) -> Path:
    """Resolve ``path`` and ensure it stays inside ``workspace`` (path-traversal guard)."""
    ws = workspace.resolve()
    candidate = (ws / path).resolve()
    if candidate != ws and ws not in candidate.parents and candidate.parent != ws:
        # allow files directly in workspace or subdirs
        if ws not in candidate.parents:
            raise ToolError("path escapes workspace")
    return candidate


class ReadFile(Tool):
    name = "read_file"
    description = "Read a UTF-8 text file from the workspace. Returns its content."

    def __init__(self, workspace: Path):
        self.ws = Path(workspace)

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path inside the workspace."}
                },
                "required": ["path"],
            },
        )

    def run(self, path: str) -> ToolResult:
        p = _safe_path(self.ws, path)
        if not p.exists():
            return ToolResult(ok=False, output=f"file not found: {path}")
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return ToolResult(ok=False, output=f"cannot read: {exc}")
        if len(text) > 200_000:
            text = text[:200_000] + "\n...[truncated]"
        return ToolResult(ok=True, output=text)


class WriteFile(Tool):
    name = "write_file"
    description = "Write UTF-8 text content to a file in the workspace, creating parent dirs. Overwrites existing files."

    def __init__(self, workspace: Path):
        self.ws = Path(workspace)

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        )

    def run(self, path: str, content: str) -> ToolResult:
        p = _safe_path(self.ws, path)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        except Exception as exc:
            return ToolResult(ok=False, output=f"cannot write: {exc}")
        return ToolResult(ok=True, output=f"wrote {len(content)} chars to {path}")


class ListFiles(Tool):
    name = "list_files"
    description = "List files and directories under a workspace path (default: root)."

    def __init__(self, workspace: Path):
        self.ws = Path(workspace)

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Relative dir, default '.'"}},
            },
        )

    def run(self, path: str = ".") -> ToolResult:
        base = _safe_path(self.ws, path)
        if not base.exists():
            return ToolResult(ok=False, output=f"not found: {path}")
        entries = []
        for child in sorted(base.iterdir()):
            entries.append(("d" if child.is_dir() else "f") + " " + child.name)
        return ToolResult(ok=True, output="\n".join(entries) or "(empty)")


class RunCommand(Tool):
    name = "run_command"
    description = "Execute a shell command in the workspace (sandboxed, 60s timeout). Captures stdout/stderr."

    def __init__(self, workspace: Path):
        self.ws = Path(workspace)

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run."},
                    "timeout": {"type": "integer", "description": "Seconds, max 120."},
                },
                "required": ["command"],
            },
        )

    def run(self, command: str, timeout: int = 60) -> ToolResult:
        timeout = max(1, min(int(timeout or 60), 120))
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(self.ws),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(ok=False, output=f"command timed out after {timeout}s")
        out = (proc.stdout or "") + (proc.stderr or "")
        if len(out) > 50_000:
            out = out[-50_000:]
        status = f"[exit {proc.returncode}]\n"
        return ToolResult(ok=proc.returncode == 0, output=status + out)


class RunPython(Tool):
    name = "run_python"
    description = "Execute a Python snippet in-process and return stdout. Safe-ish: no network imports by default."

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Python source to execute."}},
                "required": ["code"],
            },
        )

    def run(self, code: str) -> ToolResult:
        src = textwrap.dedent(code)
        try:
            tree = ast.parse(src)
        except SyntaxError as exc:
            return ToolResult(ok=False, output=f"syntax error: {exc}")
        # Disallow obviously dangerous calls.
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    if n.name.split(".")[0] in ("os", "subprocess", "shutil", "socket"):
                        return ToolResult(ok=False, output="blocked import: " + n.name)
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0] in ("os", "subprocess", "shutil", "socket"):
                    return ToolResult(ok=False, output="blocked import: " + node.module)
        g: dict = {}
        try:
            import io
            import contextlib

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                exec(compile(tree, "<agent>", "exec"), g)  # noqa: S102 - deliberate, sandboxed
            return ToolResult(ok=True, output=buf.getvalue() or "(no output)")
        except Exception as exc:
            return ToolResult(ok=False, output=f"{type(exc).__name__}: {exc}")


class WebFetch(Tool):
    name = "web_fetch"
    description = "Fetch a URL and return its textual content (markdown-ish). Best-effort, no JS rendering."

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Absolute http(s) URL."},
                    "max_chars": {"type": "integer", "description": "Truncate to N chars."},
                },
                "required": ["url"],
            },
        )

    def run(self, url: str, max_chars: int = 8000) -> ToolResult:
        try:
            import urllib.request
            import ssl

            req = urllib.request.Request(url, headers={"User-Agent": "Aurora-Agent/0.1"})
            ctx = ssl.create_default_context()
            with urllib.request.urlopen(req, timeout=20, context=ctx) as resp:
                raw = resp.read(500_000)
            text = raw.decode("utf-8", errors="replace")
        except Exception as exc:
            return ToolResult(ok=False, output=f"fetch failed: {exc}")
        # crude tag strip
        import re

        text = re.sub(r"<script[\s\S]*?</script>", " ", text, flags=re.I)
        text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return ToolResult(ok=True, output=text[: int(max_chars)])


class SearchKnowledge(Tool):
    name = "search_knowledge"
    description = "Semantic search over the agent's local knowledge base (RAG). Returns relevant passages."

    def __init__(self, retriever: Callable[[str, int], List[dict]]):
        self.retriever = retriever

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look up."},
                    "top_k": {"type": "integer", "description": "Number of passages."},
                },
                "required": ["query"],
            },
        )

    def run(self, query: str, top_k: int = 4) -> ToolResult:
        hits = self.retriever(query, int(top_k))
        if not hits:
            return ToolResult(ok=True, output="(no relevant passages found)")
        blocks = []
        for i, h in enumerate(hits, 1):
            blocks.append(f"[{i}] (score={h.get('score', 0):.3f}) {h.get('source', '')}\n{h.get('text', '')}")
        return ToolResult(ok=True, output="\n\n".join(blocks), meta={"hits": hits})
