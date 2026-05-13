---
name: geno-mine-export
description: >-
  Export a dataset version to a directory for finetuning. Supports SFT, DPO,
  tool traces, and Anthropic formats. Use when user says /geno-mine-export.
argument-hint: "--format <sft|dpo|tool_trace|anthropic> [--version <tag>] [-o <dir>]"
license: MIT
metadata:
  author: 42euge
  version: "0.1.0"
---

# Export Dataset

Export a mined dataset version to a local directory for use with finetuning pipelines.

## Workflow

```bash
geno-mine export --format <sft|dpo|tool_trace|anthropic> [--version <tag>] [-o <output-dir>]
```

Defaults to the latest version. Copies the dataset files and metadata to the output directory.
