"""geno-mine CLI — session mining and dataset generation."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from geno_mine import __version__

TRACES_DIR = Path.home() / ".geno" / "traces"


def _load_traces(*, since: str | None = None, skill: str | None = None, limit: int = 1000) -> list[dict]:
    """Load traces from ~/.geno/traces/."""
    traces = []
    if not TRACES_DIR.exists():
        return traces

    for f in sorted(TRACES_DIR.rglob("*.jsonl"), reverse=True):
        for line in reversed(f.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                t = json.loads(line)
            except json.JSONDecodeError:
                continue
            if skill and t.get("skill", {}).get("name") != skill:
                continue
            if since and t.get("timestamp", "") < since:
                continue
            traces.append(t)
            if len(traces) >= limit:
                return traces
    return traces


def cmd_extract(args: argparse.Namespace) -> int:
    """Run the full mining pipeline: traces → correlate → classify → generate → store."""
    from geno_mine.correlator import correlate_traces
    from geno_mine.signals import classify_batch
    from geno_mine.generator import generate_all
    from geno_mine.store import save

    since = None
    if args.since:
        if args.since.endswith("d"):
            days = int(args.since[:-1])
            since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        else:
            since = args.since

    print(f"loading traces (since={since or 'all'}, skill={args.skill or 'all'})...")
    traces = _load_traces(since=since, skill=args.skill)
    if not traces:
        print("no traces found")
        return 0
    print(f"  found {len(traces)} traces")

    print("correlating with transcripts...")
    segments = correlate_traces(traces)
    print(f"  correlated {len(segments)} segments")

    if not segments:
        print("no segments could be correlated (transcripts may be missing)")
        return 0

    print("classifying segments...")
    classified = classify_batch(segments)
    tier_counts = {1: 0, 2: 0, 3: 0}
    for c in classified:
        tier_counts[c.tier] = tier_counts.get(c.tier, 0) + 1
    print(f"  tier 1: {tier_counts[1]}, tier 2: {tier_counts[2]}, tier 3: {tier_counts[3]}")

    formats = args.format.split(",") if args.format else None
    print(f"generating examples (formats={formats or 'all'}, max_tier={args.max_tier})...")
    examples = generate_all(classified, formats=formats, max_tier=args.max_tier)

    total = sum(len(v) for v in examples.values())
    if total == 0:
        print("no examples generated (all segments were filtered)")
        return 0

    for fmt, items in examples.items():
        if items:
            print(f"  {fmt}: {len(items)} examples")

    if args.dry_run:
        print("(dry run — not saving)")
        return 0

    print("saving to dataset store...")
    paths = save(examples)
    for fmt, path in paths.items():
        print(f"  {fmt}: {path}")

    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    """Show dataset statistics."""
    from geno_mine.store import stats

    s = stats()
    if args.json:
        print(json.dumps(s, indent=2))
        return 0

    print(f"total examples: {s['total_examples']}")
    print(f"dataset versions: {s['versions']}")

    if s["by_format"]:
        print("\nby format:")
        for fmt, count in sorted(s["by_format"].items()):
            print(f"  {fmt}: {count}")

    if s["by_skill"]:
        print("\nby skill:")
        for skill, count in sorted(s["by_skill"].items(), key=lambda x: -x[1]):
            print(f"  {skill}: {count}")

    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Export a dataset version to a specific format."""
    from geno_mine.store import DATASETS_DIR

    fmt_dir = args.format.replace("_", "-")
    version = args.version or "latest"

    if version == "latest":
        manifest_path = DATASETS_DIR / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            version = manifest.get("latest", "")
        if not version:
            print("no datasets found — run `geno-mine extract` first")
            return 1

    src_dir = DATASETS_DIR / fmt_dir / version
    if not src_dir.exists():
        print(f"dataset not found: {src_dir}")
        return 1

    output = Path(args.output) if args.output else Path.cwd() / f"geno-mine-{fmt_dir}-{version}"
    output.mkdir(parents=True, exist_ok=True)

    import shutil
    for f in src_dir.iterdir():
        shutil.copy2(f, output / f.name)

    print(f"exported {fmt_dir}/{version} → {output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="geno-mine", description="Session mining and dataset generation")
    parser.add_argument("--version", action="version", version=f"geno-mine {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_extract = sub.add_parser("extract", help="run the full mining pipeline")
    p_extract.add_argument("--since", help="time window (e.g. '30d', ISO timestamp)")
    p_extract.add_argument("--skill", help="filter to a specific skill")
    p_extract.add_argument("--format", help="comma-separated formats (sft,dpo,tool_trace,anthropic)")
    p_extract.add_argument("--max-tier", type=int, default=2, help="max tier to include (1-3)")
    p_extract.add_argument("--dry-run", action="store_true", help="analyze without saving")
    p_extract.set_defaults(func=cmd_extract)

    p_stats = sub.add_parser("stats", help="show dataset statistics")
    p_stats.add_argument("--json", action="store_true")
    p_stats.set_defaults(func=cmd_stats)

    p_export = sub.add_parser("export", help="export a dataset version")
    p_export.add_argument("--format", required=True, choices=["sft", "dpo", "tool_trace", "anthropic"])
    p_export.add_argument("--version", help="dataset version (default: latest)")
    p_export.add_argument("--output", "-o", help="output directory")
    p_export.set_defaults(func=cmd_export)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
