# Houdini Skills

A growing collection of [Claude Code](https://claude.ai/claude-code) skills for Houdini — capturing hard-won production knowledge from real exploration sessions. Each skill is a self-contained reference that Claude can load as context for domain-specific Houdini work.

## Skills

### OpenCL — GPU Kernels
| Skill | Description | Status |
|-------|-------------|--------|
| [houdini_opencl_skill.md](opencl/houdini_opencl_skill.md) | GPU kernel development for COPs, SOPs, DOPs. `#bind`+`@KERNEL` conventions, write-back kernel iteration, Laplacian stability, GLSL gotchas, complete recipes. | **Active** |

### VEX — Wrangles & Expressions
| Skill | Description | Status |
|-------|-------------|--------|
| *vex_cop_wrangle* | Copernicus VEX patterns, `@P`+`volumeres()`, ray marching in VEX | Planned |
| *vex_sop_patterns* | SOP wrangle patterns, common attribute workflows, group operations | Planned |
| *vex_solver* | SOP Solver patterns, feedback loops, simulation in VEX | Planned |

### Python — Scripting & Automation
| Skill | Description | Status |
|-------|-------------|--------|
| *houdini_python_scripting* | hou module patterns, node creation, parameter scripting, UI automation | Planned |
| *hda_development* | HDA creation, parameter interfaces, callbacks, action buttons | Planned |
| *pdg_top_networks* | PDG/TOPs for batch processing, dependency graphs, schedulers | Planned |

### Solaris (LOPs) — USD & Lighting
| Skill | Description | Status |
|-------|-------------|--------|
| *solaris_usd_basics* | USD stage assembly, layer composition, variant sets | Planned |
| *solaris_lighting* | Light rigs, material assignments, render settings, Karma | Planned |
| *solaris_procedural* | Procedural USD generation, instancing, scene graph optimization | Planned |

### Copernicus (COPs) — Image Processing
| Skill | Description | Status |
|-------|-------------|--------|
| *copernicus_patterns* | Node patterns, layer management, COP networks, cablepack/unpack | Planned |
| *copernicus_simulation* | Block loops, feedback sims, reaction-diffusion, fluid effects | Planned |

### Workflows — Cross-Domain
| Skill | Description | Status |
|-------|-------------|--------|
| [mcp_houdini_automation.md](workflows/mcp_houdini_automation.md) | Houdini MCP server patterns (167 tools, 20 categories), node creation, VEX wrangles, USD/LOPs, COPs, simulation workflows, diagnostics. | **Active** |
| [ofx_plugin_integration.md](workflows/ofx_plugin_integration.md) | OFX plugin development + Copernicus deployment. Architecture patterns (single-pass, two-pass, inference, server), build system, limitations, threading, cross-verification. | **Active** |
| [ai_inference_pipeline.md](workflows/ai_inference_pipeline.md) | ONNX Runtime C++, Apple CoreML native, Python subprocess, persistent socket server. Binary IPC, model management, caching, tiled inference. | **Active** |

## Installation

### Quick Setup (symlink into Claude Code)

```sh
# Clone
git clone https://github.com/zhangxiao6776/houdini-skills.git ~/Projects/houdini-skills

# Symlink active skills into Claude Code commands
ln -sf ~/Projects/houdini-skills/opencl/houdini_opencl_skill.md ~/.claude/commands/houdini_opencl_skill.md
# Add more symlinks as skills are created...
```

### Invoke in Claude Code

```
/houdini_opencl_skill
```

Skills are loaded as context when invoked, giving Claude deep domain knowledge for that area.

## Philosophy

Each skill captures knowledge that was **expensive to discover** — things that:
- Have no clear documentation (or docs are misleading)
- Required multiple debugging cycles to figure out
- Involve subtle platform-specific behavior (OpenCL vs GLSL, COP2 vs Copernicus)
- Combine multiple Houdini subsystems in non-obvious ways

The goal: **never solve the same problem twice**. Every debugging session, every "aha" moment, every SideFX forum deep-dive gets distilled into a skill that future sessions can leverage instantly.

## Contributing

Skills are built through real exploration sessions. To add a new skill:
1. Explore a Houdini domain (e.g., Vellum, KineFX, Karma)
2. Document patterns, gotchas, and working recipes
3. Add a `.md` file in the appropriate directory
4. Symlink to `~/.claude/commands/` for Claude Code access
5. Update this README

## License

MIT
