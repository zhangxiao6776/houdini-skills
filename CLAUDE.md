# CLAUDE.md — Houdini Skills Repository

## What This Is

Collection of Claude Code skills for SideFX Houdini. Each skill is a Markdown file that provides deep domain knowledge for a specific Houdini area (OpenCL, VEX, Solaris, etc.). Skills are invoked as slash commands in Claude Code.

## Repository Structure

```
opencl/                  # GPU kernel development (COPs, SOPs, DOPs)
vex/                     # VEX wrangle patterns and recipes
python/                  # Python scripting and HDA development
solaris/                 # USD, LOPs, Karma, lighting
copernicus/              # COP image processing patterns
workflows/               # Cross-domain workflows and automation
```

## Adding a New Skill

1. Create `<category>/<skill_name>.md` with comprehensive content
2. Symlink to `~/.claude/commands/`: `ln -sf <path> ~/.claude/commands/<skill_name>.md`
3. Update README.md table
4. Test by invoking `/<skill_name>` in Claude Code

## Conventions

- Each skill starts with a role prompt ("You are an expert...")
- Structure: basics → advanced patterns → gotchas → complete recipes → debugging
- Include a session log of tested examples with status
- Capture every debugging insight — the goal is never solving the same problem twice
- Keep skills self-contained (no cross-references that require loading another skill)

## GitHub

- Remote: `https://github.com/zhangxiao6776/houdini-skills`
- License: MIT
