"""Example generator — convert classified segments to training examples.

Supports multiple output formats:
- SFT: instruction/response pairs
- DPO: chosen/rejected preference pairs
- Tool traces: full message arrays for tool-use training
- Anthropic: Claude API fine-tuning format
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from geno_mine.correlator import CorrelatedSegment, Turn
from geno_mine.filters import scrub, has_sensitive_content
from geno_mine.signals import ClassifiedSegment


@dataclass
class Example:
    format: str  # sft, dpo, tool_trace, anthropic
    data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def _turns_to_instruction(turns: list[Turn]) -> str:
    """Extract the user instruction from the beginning of a segment."""
    for t in turns:
        if t.type == "user":
            return scrub(t.content[:1000])
    return ""


def _turns_to_response(turns: list[Turn]) -> str:
    """Extract the assistant's response (text portions)."""
    parts = []
    for t in turns:
        if t.type == "assistant_text":
            parts.append(scrub(t.content[:500]))
    return "\n".join(parts[:5])


def _turns_to_messages(turns: list[Turn]) -> list[dict]:
    """Convert turns to a messages array."""
    messages: list[dict] = []
    for t in turns:
        if t.type == "user":
            messages.append({"role": "user", "content": scrub(t.content)})
        elif t.type == "assistant_text":
            messages.append({"role": "assistant", "content": scrub(t.content)})
        elif t.type == "tool_call":
            messages.append({
                "role": "assistant",
                "content": [{"type": "tool_use", "name": t.tool, "input": scrub(t.content)}],
            })
        elif t.type == "tool_result":
            messages.append({
                "role": "user",
                "content": [{"type": "tool_result", "content": scrub(t.content[:500])}],
            })
    return messages


def _find_correction_boundary(turns: list[Turn]) -> int | None:
    """Find the index where the user first corrects the agent."""
    for i, t in enumerate(turns):
        if t.type == "user" and i > 0:
            lower = t.content.lower()
            corrections = ["no,", "no ", "not that", "wrong", "instead", "actually"]
            if any(m in lower for m in corrections):
                return i
    return None


def generate_sft(classified: ClassifiedSegment) -> Example | None:
    """Generate an SFT (instruction/response) example."""
    seg = classified.segment
    if has_sensitive_content(" ".join(t.content for t in seg.turns)):
        return None

    instruction = _turns_to_instruction(seg.turns)
    response = _turns_to_response(seg.turns)
    if not instruction or not response:
        return None

    return Example(
        format="sft",
        data={
            "instruction": instruction,
            "response": response,
        },
        metadata={
            "skill": seg.skill_name,
            "outcome": seg.outcome,
            "tier": classified.tier,
            "source_session": seg.session_id[:12],
            "tool_calls": seg.tool_calls,
        },
    )


def generate_dpo(classified: ClassifiedSegment) -> Example | None:
    """Generate a DPO (preference) example from a correction segment."""
    if not classified.has_preference_pair:
        return None

    seg = classified.segment
    if has_sensitive_content(" ".join(t.content for t in seg.turns)):
        return None

    boundary = _find_correction_boundary(seg.turns)
    if boundary is None:
        return None

    prompt = _turns_to_instruction(seg.turns)
    rejected = _turns_to_response(seg.turns[:boundary])
    chosen = _turns_to_response(seg.turns[boundary:])

    if not prompt or not rejected or not chosen:
        return None

    return Example(
        format="dpo",
        data={
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        },
        metadata={
            "skill": seg.skill_name,
            "correction_type": "user_redirect",
            "source_session": seg.session_id[:12],
        },
    )


def generate_tool_trace(classified: ClassifiedSegment) -> Example | None:
    """Generate a full tool-use trace example."""
    seg = classified.segment
    if has_sensitive_content(" ".join(t.content for t in seg.turns)):
        return None

    messages = _turns_to_messages(seg.turns)
    if len(messages) < 3:
        return None

    return Example(
        format="tool_trace",
        data={"messages": messages},
        metadata={
            "skill": seg.skill_name,
            "outcome": seg.outcome,
            "tier": classified.tier,
            "source_session": seg.session_id[:12],
            "tool_calls": seg.tool_calls,
        },
    )


def generate_anthropic(classified: ClassifiedSegment) -> Example | None:
    """Generate an Anthropic fine-tuning format example."""
    seg = classified.segment
    if has_sensitive_content(" ".join(t.content for t in seg.turns)):
        return None

    messages = _turns_to_messages(seg.turns)
    if len(messages) < 2:
        return None

    return Example(
        format="anthropic",
        data={"messages": messages},
        metadata={
            "skill": seg.skill_name,
            "outcome": seg.outcome,
            "source_session": seg.session_id[:12],
        },
    )


def generate_all(
    classified: list[ClassifiedSegment],
    *,
    formats: list[str] | None = None,
    max_tier: int = 2,
) -> dict[str, list[Example]]:
    """Generate examples in all requested formats."""
    formats = formats or ["sft", "dpo", "tool_trace", "anthropic"]
    generators = {
        "sft": generate_sft,
        "dpo": generate_dpo,
        "tool_trace": generate_tool_trace,
        "anthropic": generate_anthropic,
    }

    results: dict[str, list[Example]] = {f: [] for f in formats}

    for cs in classified:
        if cs.tier > max_tier:
            continue
        for fmt in formats:
            gen = generators.get(fmt)
            if not gen:
                continue
            example = gen(cs)
            if example:
                results[fmt].append(example)

    return results
