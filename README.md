# Claude Log Viewer

A terminal UI for browsing Claude Code conversation logs stored in `~/.claude/projects/`.

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

| Key     | Action                     |
|---------|----------------------------|
| Enter   | Open selected conversation |
| q       | Quit                       |

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
