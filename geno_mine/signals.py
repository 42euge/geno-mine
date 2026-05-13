"""Signal extractor — classify correlated segments by training value.

Tier 1 (always extract): clean successes, user corrections → preference pairs
Tier 2 (extract with filtering): partial successes, novel tool chains
Tier 3 (skip): high thrashing, infra failures, trivial invocations
"""

from __future__ import annotations

from dataclasses import dataclass

from geno_mine.correlator import CorrelatedSegment


@dataclass
class ClassifiedSegment:
    segment: CorrelatedSegment
    tier: int  # 1, 2, or 3
    reason: str
    has_preference_pair: bool = False


def _thrashing_score(segment: CorrelatedSegment) -> float:
    """Fraction of tool calls that hit the same resource 3+ times."""
    if segment.tool_calls < 3:
        return 0.0
    tool_targets: dict[str, int] = {}
    for t in segment.turns:
        if t.type == "tool_call" and t.tool:
            key = f"{t.tool}:{t.content[:100]}"
            tool_targets[key] = tool_targets.get(key, 0) + 1
    thrashing = sum(1 for v in tool_targets.values() if v >= 3)
    return thrashing / max(len(tool_targets), 1)


def classify(segment: CorrelatedSegment) -> ClassifiedSegment:
    """Classify a segment into a value tier."""
    outcome = segment.outcome
    tc = segment.tool_calls
    errors = segment.errors
    corrections = segment.user_corrections
    thrashing = _thrashing_score(segment)

    if tc < 3:
        return ClassifiedSegment(
            segment=segment, tier=3,
            reason="trivial invocation (<3 tool calls)",
        )

    if thrashing > 0.5 and outcome != "success":
        return ClassifiedSegment(
            segment=segment, tier=3,
            reason=f"high thrashing ({thrashing:.0%}) without resolution",
        )

    infra_markers = ["api outage", "network error", "connection refused", "timeout"]
    for t in segment.turns:
        if t.type == "tool_result" and t.is_error:
            lower = t.content.lower()
            if any(m in lower for m in infra_markers):
                return ClassifiedSegment(
                    segment=segment, tier=3,
                    reason="infrastructure failure (not a skill issue)",
                )

    if outcome == "success" and corrections == 0 and errors == 0:
        return ClassifiedSegment(
            segment=segment, tier=1,
            reason="clean success — positive training example",
        )

    if corrections > 0 and outcome == "success":
        return ClassifiedSegment(
            segment=segment, tier=1,
            reason=f"user correction ({corrections}) followed by success — preference pair",
            has_preference_pair=True,
        )

    if outcome == "success" and errors > 0:
        return ClassifiedSegment(
            segment=segment, tier=1,
            reason="success with recovery from errors — shows resilience",
        )

    if outcome == "partial":
        return ClassifiedSegment(
            segment=segment, tier=2,
            reason="partial success — useful with filtering",
        )

    if outcome == "failure" and corrections > 0:
        return ClassifiedSegment(
            segment=segment, tier=2,
            reason="failure with corrections — negative example for DPO",
            has_preference_pair=True,
        )

    return ClassifiedSegment(
        segment=segment, tier=3,
        reason=f"failure without recovery (outcome={outcome})",
    )


def classify_batch(segments: list[CorrelatedSegment]) -> list[ClassifiedSegment]:
    """Classify a batch of segments and return sorted by tier."""
    classified = [classify(s) for s in segments]
    classified.sort(key=lambda c: c.tier)
    return classified
