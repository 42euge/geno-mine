---
name: geno-mine
description: >-
  Session mining and finetuning dataset generation — extract training examples
  from Claude Code session transcripts. Use when user says /geno-mine.
license: MIT
metadata:
  author: 42euge
  version: "0.1.0"
---

# geno-mine

Session mining toolkit for the geno ecosystem. Extracts finetuning datasets
from Claude Code session transcripts by correlating structured skill traces
with raw JSONL transcripts, classifying segments by training value, and
generating examples in multiple formats (SFT, DPO, tool traces, Anthropic).

## Sub-skills

| Skill | Slash command | Purpose |
|-------|---------------|---------|
| geno-mine-extract | /geno-mine-extract | Run the full mining pipeline |
| geno-mine-stats | /geno-mine-stats | Show dataset statistics |
| geno-mine-export | /geno-mine-export | Export datasets to external formats |
