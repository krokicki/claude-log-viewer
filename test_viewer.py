"""Self-check: subagent transcripts get inlined at their Task tool call."""

import json
import tempfile
from pathlib import Path

from viewer import load_conversation


def _write(path, objs):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(o) for o in objs))


def test_subagent_inlined():
    with tempfile.TemporaryDirectory() as tmp:
        proj = Path(tmp)
        _write(
            proj / "sess.jsonl",
            [
                {"type": "user", "message": {"content": "do the thing"}},
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {"type": "tool_use", "id": "toolu_1", "name": "Task", "input": {}}
                        ]
                    },
                },
                {"type": "assistant", "message": {"content": [{"type": "text", "text": "done"}]}},
            ],
        )
        subs = proj / "sess" / "subagents"
        _write(subs / "agent-x.jsonl", [
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "sub work"}]}}
        ])
        (subs / "agent-x.meta.json").write_text(
            json.dumps({"agentType": "general-purpose", "description": "Task 6",
                        "toolUseId": "toolu_1", "model": "sonnet"})
        )

        msgs = load_conversation(proj / "sess.jsonl")
        roles = [(m["role"], m["depth"]) for m in msgs]
        assert roles == [
            ("user", 0), ("assistant", 0), ("subagent", 0), ("assistant", 1), ("assistant", 0)
        ], roles
        assert "Task 6 — general-purpose, sonnet" == msgs[2]["text"]
        assert msgs[3]["text"] == "sub work"


if __name__ == "__main__":
    test_subagent_inlined()
    print("ok")
