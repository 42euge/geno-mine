---
name: geno-mine-extract
description: >-
  Run the full session mining pipeline — correlate traces with transcripts,
  classify segments, generate training examples, and save to dataset store.
  Use when user says /geno-mine-extract.
argument-hint: "[--since <days>d] [--skill <name>] [--format sft,dpo] [--dry-run]"
license: MIT
metadata:
  author: 42euge
  version: "0.1.0"
observability:
  success_signal: "dataset saved with >0 examples"
  failure_signals:
    - "no traces found"
    - "no segments correlated"
    - "all segments filtered"
  knowledge_reads:
    - "~/.geno/traces/ (structured skill traces)"
    - "~/.claude/projects/ (session transcripts)"
  knowledge_writes:
    - "~/.geno/datasets/ (training examples)"
---

# Extract Training Examples

Run the full mining pipeline:

1. Load traces from `~/.geno/traces/`
2. Correlate with session transcripts in `~/.claude/projects/`
3. Classify segments by training value (tier 1/2/3)
4. Generate examples in requested formats
5. Apply privacy filters (path scrubbing, secret detection, PII removal)
6. Save to `~/.geno/datasets/`

## Workflow

### 1. Run the pipeline

```bash
geno-mine extract [--since 30d] [--skill <name>] [--format sft,dpo] [--max-tier 2] [--dry-run]
```

Parse `$ARGUMENTS` and pass them to the CLI.

### 2. Review results

Show the user what was generated:
- Number of traces processed
- Segments correlated
- Tier distribution (tier 1/2/3)
- Examples generated per format
- Dataset location

### 3. Suggest next steps

- `geno-mine stats` to see overall dataset statistics
- `geno-mine export --format sft` to export for finetuning
- Run again with different filters to expand the dataset

## Completion

```bash
geno-trace emit \
  --skill geno-mine-extract \
  --status <success|failure> \
  --tool-calls <count> \
  --errors <count> \
  --produced "~/.geno/datasets/"
```
