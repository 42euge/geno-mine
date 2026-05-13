"""Correlate skill traces with session transcript segments.

Links structured traces (from geno-trace) to the raw JSONL transcript
segments where the skill was invoked, producing CorrelatedSegments that
downstream stages can classify and convert to training examples.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CLAUDE_PROJECTS = Path.home() / ".claude" / "projects"
TRACES_DIR = Path.home() / ".geno" / "traces"


@dataclass
class Turn:
    line: int
    type: str  # user, assistant_text, tool_call, tool_result
    content: str = ""
    tool: str | None = None
    is_error: bool = False
    is_skill_invocation: bool = False


@dataclass
class CorrelatedSegment:
    trace_id: str
    session_id: str
    skill_name: str
    outcome: str
    transcript_path: str
    line_start: int
    line_end: int
    turns: list[Turn] = field(default_factory=list)
    tool_calls: int = 0
    errors: int = 0
    user_corrections: int = 0


def parse_transcript(path: Path) -> list[Turn]:
    """Parse a Claude Code JSONL transcript into structured turns."""
    turns: list[Turn] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        t = obj.get("type")
        if t == "user":
            msg = obj.get("message", {}).get("content", "")
            content = str(msg)[:2000]
            is_skill = "<command-name>" in content or "<command-message>" in content
            turns.append(Turn(
                line=i, type="user", content=content,
                is_skill_invocation=is_skill,
            ))

        elif t == "assistant":
            blocks = obj.get("message", {}).get("content", "")
            if isinstance(blocks, list):
                for block in blocks:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "tool_use":
                        turns.append(Turn(
                            line=i, type="tool_call",
                            tool=block.get("name"),
                            content=str(block.get("input", {}))[:500],
                        ))
                    elif block.get("type") == "text":
                        turns.append(Turn(
                            line=i, type="assistant_text",
                            content=block["text"][:2000],
                        ))

        elif t == "tool_result":
            turns.append(Turn(
                line=i, type="tool_result",
                is_error=obj.get("is_error", False),
                content=str(obj.get("content", ""))[:1000],
            ))

    return turns


def find_transcript(session_id: str) -> Path | None:
    """Find the JSONL transcript for a session ID."""
    if not CLAUDE_PROJECTS.exists():
        return None
    for project_dir in CLAUDE_PROJECTS.iterdir():
        if not project_dir.is_dir():
            continue
        for jsonl in project_dir.glob("*.jsonl"):
            if session_id in jsonl.stem:
                return jsonl
    return None


def _extract_skill_name(turn: Turn) -> str | None:
    """Extract skill name from a skill invocation turn."""
    content = turn.content
    if "<command-name>" in content:
        start = content.find("<command-name>") + len("<command-name>")
        end = content.find("</command-name>", start)
        if end > start:
            name = content[start:end].strip().strip("/")
            return name
    return None


def _is_correction(turn: Turn) -> bool:
    """Detect if a user turn is correcting the agent."""
    if turn.type != "user":
        return False
    lower = turn.content.lower()
    correction_markers = [
        "no,", "no ", "not that", "wrong", "stop", "don't",
        "instead", "actually", "what i meant", "try again",
    ]
    return any(m in lower for m in correction_markers)


def correlate_traces(
    traces: list[dict],
    *,
    transcript_cache: dict[str, list[Turn]] | None = None,
) -> list[CorrelatedSegment]:
    """Correlate traces with transcript segments.

    For each trace, finds the matching transcript, locates the skill
    invocation, and extracts the segment of turns until the next
    skill invocation or end of session.
    """
    cache = transcript_cache or {}
    segments: list[CorrelatedSegment] = []

    for trace in traces:
        session_id = trace.get("session_id", "")
        if not session_id:
            continue

        transcript_path = find_transcript(session_id)
        if not transcript_path:
            continue

        path_str = str(transcript_path)
        if path_str not in cache:
            cache[path_str] = parse_transcript(transcript_path)
        turns = cache[path_str]

        skill_name = trace.get("skill", {}).get("name", "")
        trace_ts = trace.get("timestamp", "")

        skill_starts = [
            i for i, t in enumerate(turns)
            if t.is_skill_invocation and _extract_skill_name(t) == skill_name
        ]

        if not skill_starts:
            skill_starts = [
                i for i, t in enumerate(turns)
                if t.is_skill_invocation
            ]

        if not skill_starts:
            continue

        start_idx = skill_starts[-1]

        next_skills = [
            i for i in range(start_idx + 1, len(turns))
            if turns[i].is_skill_invocation
        ]
        end_idx = next_skills[0] if next_skills else len(turns)

        segment_turns = turns[start_idx:end_idx]
        tool_calls = sum(1 for t in segment_turns if t.type == "tool_call")
        errors = sum(1 for t in segment_turns if t.type == "tool_result" and t.is_error)
        corrections = sum(1 for t in segment_turns if _is_correction(t))

        segments.append(CorrelatedSegment(
            trace_id=trace.get("id", ""),
            session_id=session_id,
            skill_name=skill_name,
            outcome=trace.get("outcome", {}).get("status", "unknown"),
            transcript_path=path_str,
            line_start=segment_turns[0].line if segment_turns else 0,
            line_end=segment_turns[-1].line if segment_turns else 0,
            turns=segment_turns,
            tool_calls=tool_calls,
            errors=errors,
            user_corrections=corrections,
        ))

    return segments
