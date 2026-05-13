"""Dataset store — versioned storage at ~/.geno/datasets/.

Layout:
  ~/.geno/datasets/
  ├── sft/v<YYYYMMDD>/train.jsonl
  ├── dpo/v<YYYYMMDD>/pairs.jsonl
  ├── tool-traces/v<YYYYMMDD>/traces.jsonl
  ├── anthropic/v<YYYYMMDD>/examples.jsonl
  └── manifest.json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from geno_mine.generator import Example

DATASETS_DIR = Path.home() / ".geno" / "datasets"

FORMAT_FILES = {
    "sft": "train.jsonl",
    "dpo": "pairs.jsonl",
    "tool_trace": "traces.jsonl",
    "anthropic": "examples.jsonl",
}


def _version_tag() -> str:
    return f"v{datetime.now(timezone.utc).strftime('%Y%m%d')}"


def save(
    examples: dict[str, list[Example]],
    *,
    version: str | None = None,
) -> dict[str, Path]:
    """Write examples to versioned dataset directories. Returns paths written."""
    version = version or _version_tag()
    paths: dict[str, Path] = {}

    for fmt, items in examples.items():
        if not items:
            continue

        fmt_dir = fmt.replace("_", "-")
        out_dir = DATASETS_DIR / fmt_dir / version
        out_dir.mkdir(parents=True, exist_ok=True)

        filename = FORMAT_FILES.get(fmt, "examples.jsonl")
        out_path = out_dir / filename

        with open(out_path, "w", encoding="utf-8") as f:
            for ex in items:
                line = {**ex.data, "metadata": ex.metadata}
                f.write(json.dumps(line, ensure_ascii=False, separators=(",", ":")) + "\n")

        metadata_path = out_dir / "metadata.json"
        metadata = {
            "format": fmt,
            "version": version,
            "count": len(items),
            "created": datetime.now(timezone.utc).isoformat(),
            "skills": list({ex.metadata.get("skill", "") for ex in items}),
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
            f.write("\n")

        paths[fmt] = out_path

    _update_manifest(examples, version)
    return paths


def _update_manifest(examples: dict[str, list[Example]], version: str) -> None:
    """Update the top-level manifest.json."""
    manifest_path = DATASETS_DIR / "manifest.json"
    manifest: dict[str, Any] = {}

    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            manifest = {}

    if "versions" not in manifest:
        manifest["versions"] = []

    entry = {
        "version": version,
        "created": datetime.now(timezone.utc).isoformat(),
        "formats": {},
    }
    for fmt, items in examples.items():
        if items:
            entry["formats"][fmt] = len(items)

    manifest["versions"].append(entry)
    manifest["latest"] = version

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


def list_versions() -> list[dict]:
    """List all dataset versions from the manifest."""
    manifest_path = DATASETS_DIR / "manifest.json"
    if not manifest_path.exists():
        return []
    try:
        manifest = json.loads(manifest_path.read_text())
        return manifest.get("versions", [])
    except json.JSONDecodeError:
        return []


def stats() -> dict[str, Any]:
    """Compute dataset statistics."""
    result: dict[str, Any] = {"total_examples": 0, "by_format": {}, "by_skill": {}}
    if not DATASETS_DIR.exists():
        result["versions"] = 0
        return result
    versions = list_versions()

    for v in versions:
        for fmt, count in v.get("formats", {}).items():
            result["by_format"][fmt] = result["by_format"].get(fmt, 0) + count
            result["total_examples"] += count

    for fmt_dir in DATASETS_DIR.iterdir():
        if not fmt_dir.is_dir() or fmt_dir.name == "manifest.json":
            continue
        for ver_dir in fmt_dir.iterdir():
            if not ver_dir.is_dir():
                continue
            meta_path = ver_dir / "metadata.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    for skill in meta.get("skills", []):
                        if skill:
                            result["by_skill"][skill] = result["by_skill"].get(skill, 0) + meta.get("count", 0)
                except json.JSONDecodeError:
                    continue

    result["versions"] = len(versions)
    return result
