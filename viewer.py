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
from textual.widgets import DataTable, Footer, Header, Input, Static

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
                    rendered = Text.from_markup(f"[dim]{_escape(text)}[/dim]")
                    w = Static(rendered, classes="tool-result")
                else:
                    rendered = Text.from_markup(
                        f"[bold cyan]▌ User[/bold cyan]\n{_escape(text)}"
                    )
                    w = Static(rendered, classes="user-msg")
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
                rendered = Text.from_markup(
                    f"[bold green]▌ Assistant[/bold green]\n"
                    + "\n".join(parts)
                )
                w = Static(rendered, classes="assistant-msg")
            w._plain_text = text
            w._original_renderable = rendered
            yield w

    def _scroll_to_top(self, widget: Static) -> None:
        """Scroll so that widget is at the top of the viewport."""
        self.scroll_to(y=max(0, widget.virtual_region.y), animate=False)

    def highlight(self, term: str) -> None:
        """Highlight all occurrences of term in all message widgets."""
        for w in self.query(Static):
            original = getattr(w, "_original_renderable", None)
            if original is None:
                continue
            highlighted = original.copy()
            highlighted.highlight_words([term], style="on dark_magenta", case_sensitive=False)
            w.update(highlighted)

    def clear_highlights(self) -> None:
        """Restore all widgets to their original (unhighlighted) text."""
        for w in self.query(Static):
            original = getattr(w, "_original_renderable", None)
            if original is None:
                continue
            w.update(original)


class ConversationScreen(Screen):
    """Full-screen view of a conversation."""

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("q", "back", "Back"),
        Binding("u", "next_user", "Next user msg"),
        Binding("s", "search", "Search"),
        Binding("n", "search_next", "Next match"),
        Binding("ctrl+f", "page_down", "Page down", show=False),
        Binding("ctrl+b", "page_up", "Page up", show=False),
    ]

    def __init__(self, conv: dict, search_term: str = "", **kwargs):
        super().__init__(**kwargs)
        self.conv = conv
        self._search_term: str = search_term

    def compose(self) -> ComposeResult:
        yield Header()
        yield ConversationDetail(self.conv, id="detail")
        yield Input(placeholder="Search…", id="search-input")
        yield Footer()

    def on_mount(self) -> None:
        title = self.conv.get("title", self.conv["session_id"][:8])
        self.title = _truncate(title, 80)
        self.sub_title = self.conv["project"]
        self.query_one("#search-input", Input).display = False
        if self._search_term:
            # Defer until after layout so virtual_region positions are valid
            self.set_timer(0.2, lambda: self._do_search(from_top=True))

    def action_back(self) -> None:
        search_input = self.query_one("#search-input", Input)
        if search_input.display:
            search_input.display = False
            self.query_one("#detail", ConversationDetail).focus()
            return
        self.app.pop_screen()

    # ── ctrl+f / ctrl+b: page down / up ──────────────────────────────────

    def action_page_down(self) -> None:
        detail = self.query_one("#detail", ConversationDetail)
        detail.scroll_to(y=detail.scroll_y + detail.size.height, animate=False)

    def action_page_up(self) -> None:
        detail = self.query_one("#detail", ConversationDetail)
        detail.scroll_to(y=max(0, detail.scroll_y - detail.size.height), animate=False)

    # ── u: next user message ────────────────────────────────────────────

    def action_next_user(self) -> None:
        detail = self.query_one("#detail", ConversationDetail)
        user_widgets = detail.query(".user-msg")
        if not user_widgets:
            return
        # Find the first user widget whose top is below the current scroll position
        # (add a small offset so pressing u again skips the current one)
        current_top = detail.scroll_y + 1
        for w in user_widgets:
            if w.virtual_region.y > current_top:
                detail._scroll_to_top(w)
                return
        # Wrap around to first
        detail._scroll_to_top(user_widgets[0])

    # ── s: search ───────────────────────────────────────────────────────

    def action_search(self) -> None:
        search_input = self.query_one("#search-input", Input)
        search_input.value = self._search_term
        search_input.display = True
        search_input.focus()

    @on(Input.Submitted, "#search-input")
    def on_search_submitted(self, event: Input.Submitted) -> None:
        search_input = self.query_one("#search-input", Input)
        search_input.display = False
        detail = self.query_one("#detail", ConversationDetail)
        new_term = event.value.strip()
        if new_term != self._search_term:
            detail.clear_highlights()
        self._search_term = new_term
        if self._search_term:
            detail.highlight(self._search_term)
            self._do_search(from_top=True)
        detail.focus()

    def _do_search(self, from_top: bool = False) -> None:
        if not self._search_term:
            return
        detail = self.query_one("#detail", ConversationDetail)
        term = self._search_term.lower()
        all_widgets = list(detail.query(Static))
        if not all_widgets:
            return
        if from_top:
            # Search from the beginning — match by list order, not position
            detail.highlight(self._search_term)
            for w in all_widgets:
                plain = getattr(w, "_plain_text", "")
                if term in plain.lower():
                    detail._scroll_to_top(w)
                    return
        else:
            # Search after current scroll position
            start_y = detail.scroll_y + 1
            for w in all_widgets:
                plain = getattr(w, "_plain_text", "")
                if w.virtual_region.y > start_y and term in plain.lower():
                    detail._scroll_to_top(w)
                    return
            # Wrap: search from top
            for w in all_widgets:
                plain = getattr(w, "_plain_text", "")
                if term in plain.lower():
                    detail._scroll_to_top(w)
                    return

    # ── n: next search match ────────────────────────────────────────────

    def action_search_next(self) -> None:
        self._do_search(from_top=False)


def _escape(text: str) -> str:
    """Escape Rich markup characters in user/assistant text."""
    return text.replace("[", "\\[")


def _human_size(nbytes: int) -> Text:
    """Format byte count as right-aligned MB string."""
    mb = nbytes / (1024 * 1024)
    return Text(f"{mb:.2f} MB", justify="right")


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
    #search-input, #index-search {
        dock: bottom;
        margin: 0;
        height: auto;
    }
    """
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("enter", "open", "Open", show=False),
        Binding("s", "search", "Search"),
        Binding("escape", "clear_search", "Clear search"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.all_conversations: list[dict] = []
        self.conversations: list[dict] = []
        self._search_term: str = ""

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable(id="index")
        yield Input(placeholder="Search conversations…", id="index-search")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#index", DataTable)
        table.cursor_type = "row"
        table.add_columns("Project", "Size", "Title", "Date", "Msgs")
        self.query_one("#index-search", Input).display = False
        self.loading = True
        self.call_later(self._load_index)

    def _load_index(self) -> None:
        convs = discover_conversations()
        for c in convs:
            load_conversation_preview(c)
        convs.sort(key=lambda c: c["mtime"], reverse=True)
        self.all_conversations = convs
        self.conversations = convs
        self._populate_table(convs)
        self.loading = False

    def _populate_table(self, convs: list[dict]) -> None:
        table = self.query_one("#index", DataTable)
        table.clear()
        for c in convs:
            date_str = c["mtime"].strftime("%Y-%m-%d %H:%M")
            table.add_row(
                c["project"],
                _human_size(c["size"]),
                _truncate(c["title"], 80),
                date_str,
                str(c["msg_count"]),
            )

    def action_search(self) -> None:
        search_input = self.query_one("#index-search", Input)
        search_input.value = self._search_term
        search_input.display = True
        search_input.focus()

    @on(Input.Submitted, "#index-search")
    def on_index_search_submitted(self, event: Input.Submitted) -> None:
        search_input = self.query_one("#index-search", Input)
        search_input.display = False
        self._search_term = event.value.strip()
        if self._search_term:
            self.sub_title = f"Searching for \"{self._search_term}\"…"
            self.call_later(self._run_search)
        else:
            self.sub_title = ""
            self.conversations = self.all_conversations
            self._populate_table(self.conversations)
        self.query_one("#index", DataTable).focus()

    def _run_search(self) -> None:
        term = self._search_term.lower()
        matches = []
        for c in self.all_conversations:
            try:
                with open(c["path"]) as f:
                    for line in f:
                        if term in line.lower():
                            matches.append(c)
                            break
            except OSError:
                continue
        self.conversations = matches
        self._populate_table(matches)
        self.sub_title = f"\"{self._search_term}\" — {len(matches)} result{'s' if len(matches) != 1 else ''}"

    def action_clear_search(self) -> None:
        search_input = self.query_one("#index-search", Input)
        if search_input.display:
            search_input.display = False
            self.query_one("#index", DataTable).focus()
            return
        if self._search_term:
            self._search_term = ""
            self.sub_title = ""
            self.conversations = self.all_conversations
            self._populate_table(self.conversations)

    def _open_conversation(self, conv: dict) -> None:
        self.push_screen(ConversationScreen(conv, search_term=self._search_term))

    @on(DataTable.RowSelected)
    def on_row_selected(self, event: DataTable.RowSelected) -> None:
        row_index = event.cursor_row
        if 0 <= row_index < len(self.conversations):
            self._open_conversation(self.conversations[row_index])

    def action_open(self) -> None:
        table = self.query_one("#index", DataTable)
        row_index = table.cursor_row
        if 0 <= row_index < len(self.conversations):
            self._open_conversation(self.conversations[row_index])


if __name__ == "__main__":
    LogViewerApp().run()
