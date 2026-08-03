# Claude Log Viewer

A terminal UI for browsing Claude Code conversation logs stored in `~/.claude/projects/`.

Subagent transcripts are spliced into the parent conversation at the `Task` call that
spawned them, indented by spawn depth. Every message is timestamped, and any tool call
taking longer than 10s (~p95) gets a `⏱ took Xm Ys` line so slow steps are easy to spot.

## Setup

Requires [pixi](https://pixi.sh):

```sh
pixi install
```

## Usage

```sh
pixi run viewer
```

## Keybindings

### Conversation Index

| Key            | Action                              |
|----------------|-------------------------------------|
| Enter          | Open selected conversation          |
| s              | Search conversations by content     |
| Escape         | Clear search / dismiss search input |
| Click column   | Sort by column (click again to reverse) |
| q              | Quit                                |

### Conversation Detail

| Key     | Action                          |
|---------|---------------------------------|
| Escape  | Back to index                   |
| q       | Back to index                   |
| u       | Jump to next user message       |
| s       | Search for text                 |
| n       | Repeat search (next match)      |
| Ctrl+F  | Page down                       |
| Ctrl+B  | Page up                         |
