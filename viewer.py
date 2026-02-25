#!/usr/bin/env python3
"""Claude Code Conversation Log Viewer — Terminal UI."""

import json
import os
import re
from datetime import datetime
from pathlib import Path

from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static

# ── Data layer ──────────────────────────────────────────────────────────────

PROJECTS_DIR = Path.home() / ".claude" / "projects"

def _readable_project(dirname: str) -> str:
    """Derive a human-readable project name from the directory name.

    Directory names encode filesystem paths with / → -.
    Use the home dir to strip the known prefix.
    """
    home = str(Path.home()).replace("/", "-")  # e.g. "-groups-scicompsoft-home-rokickik"
    # Strip home prefix + optional "-dev" segment
    name = dirname
    if name.startswith(home):
        name = name[len(home):]
    name = name.lstrip("-")
    if name.startswith("dev-"):
        name = name[4:]
    return name or dirname.lstrip("-")


def discover_conversations() -> list[dict]:
    """Walk ~/.claude/projects/ and return metadata for every conversation JSONL."""
    if not PROJECTS_DIR.is_dir():
        return []
    conversations = []
    for project_dir in sorted(PROJECTS_DIR.iterdir()):
        if not project_dir.is_dir():
            continue
        project_name = _readable_project(project_dir.name)
        for jsonl in sorted(project_dir.glob("*.jsonl")):
            # Skip subagent / memory dirs (they're dirs, not files — but be safe)
            if "subagent" in str(jsonl) or "memory" in str(jsonl):
                continue
            stat = jsonl.stat()
            conversations.append(
                {
                    "project": project_name,
                    "session_id": jsonl.stem,
                    "path": jsonl,
                    "mtime": datetime.fromtimestamp(stat.st_mtime),
                    "size": stat.st_size,
                }
            )
    return conversations


def load_conversation_preview(conv: dict) -> dict:
    """Read a conversation file to extract preview info (title, timestamp, message count)."""
    title = ""
    first_ts = None
    user_count = 0
    assistant_count = 0

    with open(conv["path"]) as f:
        for line in f:
            # Fast path: check type via string search before parsing JSON
            if '"type":"user"' in line or '"type": "user"' in line:
                user_count += 1
                if not title:
                    try:
                        obj = json.loads(line)
                        if obj.get("isMeta"):
                            continue
                        content = obj.get("message", {}).get("content", "")
                        # Skip tool_result lines (user type but list content)
                        if isinstance(content, list):
                            continue
                        # Skip command / system messages
                        if content.startswith("<"):
                            continue
                        if len(content.strip()) > 0:
                            title = content.strip()[:120]
                            ts = obj.get("timestamp")
                            if ts:
                                first_ts = ts
                    except (json.JSONDecodeError, KeyError):
                        pass
            elif '"type":"assistant"' in line or '"type": "assistant"' in line:
                assistant_count += 1

    if first_ts and isinstance(first_ts, str):
        try:
            first_ts = datetime.fromisoformat(first_ts.replace("Z", "+00:00"))
        except ValueError:
            first_ts = None

    conv["title"] = title or "(empty conversation)"
    conv["first_ts"] = first_ts
    conv["msg_count"] = user_count + assistant_count
    return conv


def _extract_text(content) -> str:
    """Extract plain text from message content (string or block list)."""
    if isinstance(content, str):
        # Strip XML-ish tags from user messages
        text = re.sub(r"<[^>]+>", "", content).strip()
        return text
    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")
            if btype == "text":
                parts.append(block.get("text", ""))
            elif btype == "tool_use":
                name = block.get("name", "unknown")
                inp = block.get("input", {})
                # Show a compact summary of tool input
                if isinstance(inp, dict):
                    summary = json.dumps(inp, ensure_ascii=False)
                else:
                    summary = str(inp)
                if len(summary) > 100:
                    summary = summary[:100] + "…"
                parts.append(f"[Tool: {name}] {summary}")
            elif btype == "tool_result":
                result_content = block.get("content", "")
                preview = str(result_content)[:200]
                if len(str(result_content)) > 200:
                    preview += "…"
                tool_id = block.get("tool_use_id", "")[:12]
                parts.append(f"[Result {tool_id}] {preview}")
            # Skip thinking, signature, etc.
        return "\n".join(parts)
    return ""


def load_conversation(path: Path) -> list[dict]:
    """Full parse of a conversation file. Returns list of {role, content_text} dicts."""
    messages = []
    with open(path) as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg_type = obj.get("type")
            if msg_type not in ("user", "assistant"):
                continue
            if obj.get("isMeta"):
                continue
            content = obj.get("message", {}).get("content", "")
            text = _extract_text(content)
            if not text:
                continue
            role = "user" if msg_type == "user" else "assistant"
            messages.append({"role": role, "text": text})
    return messages


# ── TUI layer ───────────────────────────────────────────────────────────────


class ConversationDetail(VerticalScroll):
    """Scrollable view of a single conversation's messages."""

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("q", "back", "Back"),
    ]

    def __init__(self, conv: dict, **kwargs):
        super().__init__(**kwargs)
        self.conv = conv

    def compose(self) -> ComposeResult:
        messages = load_conversation(self.conv["path"])
        if not messages:
            yield Static("[dim]No messages found.[/dim]")
            return
        for msg in messages:
            role = msg["role"]
            text = msg["text"]
            if role == "user":
                # Check if this is a tool result
                if text.startswith("[Result "):
                    yield Static(
                        Text.from_markup(f"[dim]{_escape(text)}[/dim]"),
                        classes="tool-result",
                    )
                else:
                    yield Static(
                        Text.from_markup(
                            f"[bold cyan]▌ User[/bold cyan]\n{_escape(text)}"
                        ),
                        classes="user-msg",
                    )
            else:
                # Assistant — split tool use lines from text
                lines = text.split("\n")
                parts = []
                for ln in lines:
                    if ln.startswith("[Tool: "):
                        parts.append(f"[dim italic]{_escape(ln)}[/dim italic]")
                    elif ln.startswith("[Result "):
                        parts.append(f"[dim]{_escape(ln)}[/dim]")
                    else:
                        parts.append(_escape(ln))
                yield Static(
                    Text.from_markup(
                        f"[bold green]▌ Assistant[/bold green]\n"
                        + "\n".join(parts)
                    ),
                    classes="assistant-msg",
                )

    def action_back(self) -> None:
        self.app.pop_screen()


class ConversationScreen(Screen):
    """Full-screen view of a conversation."""

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("q", "back", "Back"),
    ]

    def __init__(self, conv: dict, **kwargs):
        super().__init__(**kwargs)
        self.conv = conv

    def compose(self) -> ComposeResult:
        title = self.conv.get("title", self.conv["session_id"][:8])
        yield Header()
        yield ConversationDetail(self.conv, id="detail")
        yield Footer()

    def on_mount(self) -> None:
        title = self.conv.get("title", self.conv["session_id"][:8])
        self.title = _truncate(title, 80)
        self.sub_title = self.conv["project"]

    def action_back(self) -> None:
        self.app.pop_screen()


def _escape(text: str) -> str:
    """Escape Rich markup characters in user/assistant text."""
    return text.replace("[", "\\[")


def _truncate(text: str, length: int) -> str:
    if len(text) <= length:
        return text
    return text[: length - 1] + "…"


class LogViewerApp(App):
    """Main application — shows conversation index."""

    TITLE = "Claude Log Viewer"
    CSS = """
    DataTable {
        height: 1fr;
    }
    .user-msg {
        margin: 1 0 0 0;
        padding: 0 1;
        border-left: thick $accent;
    }
    .assistant-msg {
        margin: 1 0 0 0;
        padding: 0 1;
        border-left: thick $success;
    }
    .tool-result {
        margin: 0;
        padding: 0 2;
    }
    #detail {
        padding: 0 1;
    }
    """
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("enter", "open", "Open", show=False),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conversations: list[dict] = []

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable(id="index")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#index", DataTable)
        table.cursor_type = "row"
        table.add_columns("Project", "Title", "Date", "Msgs")
        self.loading = True
        self.call_later(self._load_index)

    def _load_index(self) -> None:
        convs = discover_conversations()
        # Load previews
        for c in convs:
            load_conversation_preview(c)
        # Sort by modification time, newest first
        convs.sort(key=lambda c: c["mtime"], reverse=True)
        self.conversations = convs

        table = self.query_one("#index", DataTable)
        for c in convs:
            date_str = c["mtime"].strftime("%Y-%m-%d %H:%M")
            table.add_row(
                c["project"],
                _truncate(c["title"], 80),
                date_str,
                str(c["msg_count"]),
            )
        self.loading = False

    @on(DataTable.RowSelected)
    def on_row_selected(self, event: DataTable.RowSelected) -> None:
        row_index = event.cursor_row
        if 0 <= row_index < len(self.conversations):
            conv = self.conversations[row_index]
            self.push_screen(ConversationScreen(conv))

    def action_open(self) -> None:
        table = self.query_one("#index", DataTable)
        row_index = table.cursor_row
        if 0 <= row_index < len(self.conversations):
            conv = self.conversations[row_index]
            self.push_screen(ConversationScreen(conv))


if __name__ == "__main__":
    LogViewerApp().run()
