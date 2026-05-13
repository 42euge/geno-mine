# geno-mine — Session Mining and Finetuning Dataset Generation

`geno-mine` extracts finetuning datasets from Claude Code session transcripts. It correlates structured skill traces (from `geno-trace`) with raw JSONL transcripts, classifies segments by training value, applies privacy filters, and generates examples in multiple formats.

## Skills

| Skill | Slash command | Purpose |
|-------|---------------|---------|
| geno-mine | — | Umbrella |
| geno-mine-extract | /geno-mine-extract | Run the full mining pipeline |
| geno-mine-stats | /geno-mine-stats | Show dataset statistics |
| geno-mine-export | /geno-mine-export | Export datasets |

## Repo structure

```
geno-mine/
├── GENO.md                              # agent instructions (this file)
├── SKILL.md -> skills/geno-mine/SKILL.md
├── genotools.yaml                       # install manifest
├── pyproject.toml                       # Python package metadata
├── geno_mine/                           # Python package
│   ├── cli.py                           #   CLI entry points (extract/stats/export)
│   ├── correlator.py                    #   link traces to transcript segments
│   ├── signals.py                       #   classify segment value (tier 1/2/3)
│   ├── generator.py                     #   produce SFT, DPO, tool-trace examples
│   ├── filters.py                       #   privacy: path scrub, secret/PII removal
│   └── store.py                         #   dataset versioning at ~/.geno/datasets/
└── skills/
    ├── geno-mine/SKILL.md               #   umbrella
    ├── geno-mine-extract/SKILL.md       #   full pipeline skill
    ├── geno-mine-stats/SKILL.md         #   statistics skill
    └── geno-mine-export/SKILL.md        #   export skill
```

## Entry point

```toml
[project.scripts]
geno-mine = "geno_mine.cli:main"
```

## Pipeline

```
traces (~/.geno/traces/) ──→ correlator ──→ signals ──→ generator ──→ store
     ↑                           │              │            │           │
 geno-trace emit            parse JSONL    classify     SFT/DPO/    ~/.geno/
 (from skills)              transcripts    tier 1/2/3   tool-trace  datasets/
                                                         + filter
```

## Privacy

All processing is local (zero telemetry). The filter stage:
1. Scrubs absolute paths (`/Users/...` → `~/...`)
2. Detects and redacts API keys, tokens, passwords
3. Replaces emails and phone numbers with placeholders
4. Skips segments that touch `.env`, credentials, `~/.ssh/`
5. Configurable exclusions in `~/.geno/config.yaml` under `mining`
