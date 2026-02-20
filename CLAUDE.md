# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Context Engineering Template for structured feature development using PRPs (Product Requirements Prompts). Separates planning from execution with validation gates.

## Core Workflow

```bash
# 1. Write feature requirements in INITIAL.md, then generate implementation blueprint
/generate-prp INITIAL.md

# 2. Execute the PRP with validation after each step
/execute-prp PRPs/your-feature.md

# 3. Prime context for a new session
/primer
```

### Workflow Separation

- **generate-prp**: Analyzes codebase, creates step-by-step blueprint with validation gates, risk mitigation, and rollback plan. Output: `PRPs/feature_name_prp.md`
- **execute-prp**: Implements one task at a time, runs validation immediately after each step, stops if >50% of gates fail

## Validation Gates (from PRP template)

```bash
# Level 1: Syntax & Style
ruff check src/new_feature.py --fix
mypy src/new_feature.py

# Level 2: Unit Tests
uv run pytest test_new_feature.py -v

# Level 3: Full Suite
uv run pytest tests/ -v
uv run ruff check src/
uv run mypy src/
```

## Project Structure

```
.claude/
├── commands/      # /generate-prp, /execute-prp, /primer
├── agents/        # validation-gates, documentation-manager, code-analyst, etc.
└── skills/        # anthropic-skills, templates
PRPs/
├── templates/     # prp_base.md - template for new PRPs
└── *.md           # Generated implementation blueprints
src/               # Source code
INITIAL.md         # Feature request input
```

## PRP Core Principles

1. **Context is King**: Include ALL necessary documentation, examples, and gotchas
2. **Validation Loops**: Executable tests/lints that can be run and fixed iteratively
3. **Progressive Success**: Start simple, validate, then enhance
4. **Never skip failed validation gates**
