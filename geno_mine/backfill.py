"""Retroactively generate skill traces from historical session transcripts."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from geno_mine.correlator import (
    CLAUDE_PROJECTS, TRACES_DIR, Turn,
    parse_transcript, _extract_skill_name, _is_correction,
)

BUILTIN_COMMANDS = {
    "clear", "exit", "compact", "login", "context", "extra-usage",
    "rate-limit-options", "help", "config", "doctor", "mcp", "loop",
    "schedule", "init", "review", "security-review", "simplify",
    "fewer-permission-prompts", "keybindings-help", "update-config",
}


@dataclass
class InvocationSegment:
    skill_name: str
    timestamp: str
    session_id: str
    project: str
    git_branch: str
    turns: list[Turn] = field(default_factory=list)
    transcript_path: str = ""
    line_start: int = 0
    line_end: int = 0


def scan_transcripts(*, since: str | None = None) -> list[Path]:
    """Find all JSONL transcript files, optionally filtering by mtime."""
    if not CLAUDE_PROJECTS.exists():
        return []

    cutoff_ts = 0.0
    if since:
        try:
            dt = datetime.fromisoformat(since)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            cutoff_ts = dt.timestamp()
        except ValueError:
            pass

    files: list[Path] = []
    for project_dir in CLAUDE_PROJECTS.iterdir():
        if not project_dir.is_dir():
            continue
        for jsonl in project_dir.rglob("*.jsonl"):
            if cutoff_ts and jsonl.stat().st_mtime < cutoff_ts:
                continue
            files.append(jsonl)

    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def _extract_metadata(path: Path) -> dict:
    """Extract session_id, project, git_branch from first lines of a transcript."""
    meta = {"session_id": "", "project": "", "git_branch": ""}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not meta["session_id"] and obj.get("sessionId"):
                    meta["session_id"] = obj["sessionId"]
                if not meta["project"] and obj.get("cwd"):
                    meta["project"] = obj["cwd"]
                if not meta["git_branch"] and obj.get("gitBranch"):
                    meta["git_branch"] = obj["gitBranch"]
                if all(meta.values()):
                    break
    except Exception:
        pass
    return meta


def _extract_line_timestamp(path: Path, line_num: int) -> str:
    """Extract timestamp from a specific line in the transcript."""
    try:
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == line_num:
                    obj = json.loads(line.strip())
                    return obj.get("timestamp", datetime.now(timezone.utc).isoformat())
    except Exception:
        pass
    return datetime.now(timezone.utc).isoformat()


def segment_invocations(
    turns: list[Turn],
    transcript_path: Path,
    meta: dict,
    *,
    skill_filter: str | None = None,
) -> list[InvocationSegment]:
    """Split turns at skill invocation boundaries."""
    invocation_indices: list[tuple[int, str]] = []

    for i, turn in enumerate(turns):
        if not turn.is_skill_invocation:
            continue
        name = _extract_skill_name(turn)
        if not name:
            continue
        clean = name.strip("/")
        if clean.startswith("gt-"):
            clean = "geno-" + clean[3:]
        if clean in BUILTIN_COMMANDS:
            continue
        if len(clean) > 60 or " " in clean or "\n" in clean or "`" in clean:
            continue
        if skill_filter and clean != skill_filter:
            continue
        invocation_indices.append((i, clean))

    segments: list[InvocationSegment] = []
    for idx, (start_i, skill_name) in enumerate(invocation_indices):
        if idx + 1 < len(invocation_indices):
            end_i = invocation_indices[idx + 1][0]
        else:
            end_i = len(turns)

        seg_turns = turns[start_i:end_i]
        ts = _extract_line_timestamp(transcript_path, turns[start_i].line)

        segments.append(InvocationSegment(
            skill_name=skill_name,
            timestamp=ts,
            session_id=meta.get("session_id", ""),
            project=meta.get("project", ""),
            git_branch=meta.get("git_branch", ""),
            turns=seg_turns,
            transcript_path=str(transcript_path),
            line_start=seg_turns[0].line if seg_turns else 0,
            line_end=seg_turns[-1].line if seg_turns else 0,
        ))

    return segments


def infer_outcome(segment: InvocationSegment) -> tuple[str, str | None]:
    """Infer success/partial/failure/abandoned from segment content."""
    turns = segment.turns
    if len(turns) < 3:
        return "abandoned", None

    tool_calls = sum(1 for t in turns if t.type == "tool_call")
    errors = sum(1 for t in turns if t.type == "tool_result" and t.is_error)
    corrections = sum(1 for t in turns if _is_correction(t))

    if tool_calls == 0:
        return "abandoned", None

    if errors == 0 and corrections == 0:
        return "success", None

    if errors > 0 and corrections == 0:
        error_ratio = errors / max(tool_calls, 1)
        if error_ratio > 0.5:
            return "failure", "high_error_rate"
        return "partial", None

    if errors > 0 and corrections > 0:
        last_error_idx = max((i for i, t in enumerate(turns) if t.type == "tool_result" and t.is_error), default=0)
        last_correction_idx = max((i for i, t in enumerate(turns) if _is_correction(t)), default=0)
        if last_correction_idx > last_error_idx:
            return "success", None
        return "partial", None

    if corrections > 0 and errors == 0:
        return "success", None

    return "partial", None


def build_trace(segment: InvocationSegment, status: str, error_type: str | None) -> dict:
    """Construct a trace dict matching the geno-trace schema."""
    tool_calls = sum(1 for t in segment.turns if t.type == "tool_call")
    errors = sum(1 for t in segment.turns if t.type == "tool_result" and t.is_error)
    corrections = sum(1 for t in segment.turns if _is_correction(t))

    return {
        "id": f"trace-bf-{uuid4().hex[:12]}",
        "timestamp": segment.timestamp,
        "session_id": segment.session_id,
        "project": segment.project,
        "skill": {
            "name": segment.skill_name,
            "skillset": segment.skill_name.rsplit("-", 2)[0] if "-" in segment.skill_name else segment.skill_name,
            "version": "unknown",
        },
        "outcome": {
            "status": status,
            "error_type": error_type,
            "error_detail": None,
        },
        "metrics": {
            "tool_calls": tool_calls,
            "errors": errors,
            "thrashing_score": 0.0,
            "user_corrections": corrections,
            "duration_turns": len(segment.turns),
        },
        "context": {
            "task_id": None,
            "scope": None,
            "branch": segment.git_branch if segment.git_branch else None,
        },
        "knowledge": {"consumed": [], "produced": []},
        "tags": ["backfill"],
    }


def load_existing_trace_index() -> set[tuple[str, str, str]]:
    """Build dedup index: (session_id, skill_name, timestamp_minute)."""
    index: set[tuple[str, str, str]] = set()
    if not TRACES_DIR.exists():
        return index
    for f in TRACES_DIR.rglob("*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                t = json.loads(line)
                sid = t.get("session_id", "")
                skill = t.get("skill", {}).get("name", "")
                ts = t.get("timestamp", "")[:16]
                if sid and skill:
                    index.add((sid, skill, ts))
            except json.JSONDecodeError:
                continue
    return index


def write_traces_bulk(traces: list[dict]) -> dict[str, int]:
    """Write traces grouped by month to ~/.geno/traces/YYYY/YYYY-MM.jsonl."""
    by_month: dict[str, list[dict]] = {}
    for t in traces:
        ts = t.get("timestamp", "")
        if len(ts) >= 7:
            month = ts[:7]
        else:
            month = datetime.now(timezone.utc).strftime("%Y-%m")
        by_month.setdefault(month, []).append(t)

    counts: dict[str, int] = {}
    for month, items in by_month.items():
        year = month[:4]
        dir_path = TRACES_DIR / year
        dir_path.mkdir(parents=True, exist_ok=True)
        file_path = dir_path / f"{month}.jsonl"
        with open(file_path, "a", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, separators=(",", ":")) + "\n")
        counts[month] = len(items)

    return counts


def backfill(
    *,
    since: str | None = None,
    skill: str | None = None,
    dry_run: bool = False,
) -> dict:
    """Main backfill orchestrator."""
    files = scan_transcripts(since=since)
    if not files:
        return {"transcripts_scanned": 0, "invocations_found": 0, "traces_emitted": 0}

    dedup_index = load_existing_trace_index()
    all_traces: list[dict] = []
    skipped_dedup = 0
    by_skill: dict[str, int] = {}
    by_status: dict[str, int] = {}

    for path in files:
        meta = _extract_metadata(path)
        try:
            turns = parse_transcript(path)
        except Exception:
            continue

        segments = segment_invocations(turns, path, meta, skill_filter=skill)

        for seg in segments:
            ts_prefix = seg.timestamp[:16] if len(seg.timestamp) >= 16 else seg.timestamp
            key = (seg.session_id, seg.skill_name, ts_prefix)
            if key in dedup_index:
                skipped_dedup += 1
                continue

            status, error_type = infer_outcome(seg)
            trace = build_trace(seg, status, error_type)
            all_traces.append(trace)
            dedup_index.add(key)

            by_skill[seg.skill_name] = by_skill.get(seg.skill_name, 0) + 1
            by_status[status] = by_status.get(status, 0) + 1

    if not dry_run and all_traces:
        write_traces_bulk(all_traces)

    return {
        "transcripts_scanned": len(files),
        "invocations_found": len(all_traces) + skipped_dedup,
        "traces_emitted": len(all_traces),
        "traces_skipped_dedup": skipped_dedup,
        "traces_by_skill": by_skill,
        "traces_by_status": by_status,
    }
