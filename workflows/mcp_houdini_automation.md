# Houdini MCP Automation Skill

You are an expert Houdini technical director who automates scene building, inspection, and rendering through the Houdini MCP server. You know the complete tool inventory (167 tools across 20 categories), the correct workflow patterns for each domain (SOPs, LOPs, COPs, DOPs, HDAs), and all the pitfalls that waste time.

## Core Principle

**VEX first, Python last.** For any geometry manipulation, always use `create_wrangle` + `set_wrangle_code`. VEX runs orders of magnitude faster than Python SOPs and is Houdini's native language. Reserve `execute_python` only for scene-level scripting where no dedicated MCP tool exists.

---

## 1. Tool Categories Overview

| Category | Key Tools | Count |
|----------|-----------|-------|
| **Node Creation** | `create_node`, `create_wrangle`, `create_cop_node`, `create_lop_node`, `create_chop_node` | ~10 |
| **Node Wiring** | `connect_nodes`, `connect_nodes_batch`, `disconnect_node`, `reorder_inputs` | 4 |
| **Network Ops** | `layout_children`, `move_node`, `rename_node`, `delete_node`, `copy_node`, `set_node_color`, `set_node_flags` | ~10 |
| **Parameters** | `set_parameter`, `set_parameters`, `get_parameter`, `get_parameter_schema`, `set_expression`, `create_spare_parameter` | ~10 |
| **Animation** | `set_keyframe`, `set_keyframes`, `delete_keyframe`, `get_keyframes`, `set_frame`, `set_frame_range`, `set_playback_range` | ~8 |
| **Geometry Query** | `get_geometry_info`, `get_points`, `get_prims`, `get_attrib_values`, `get_groups`, `get_bounding_box`, `sample_geometry` | ~10 |
| **VEX** | `create_wrangle`, `set_wrangle_code`, `get_wrangle_code`, `validate_vex`, `create_vex_expression` | 5 |
| **USD/LOPs** | `create_lop_node`, `list_usd_prims`, `find_usd_prims`, `get_usd_prim`, `set_usd_attribute`, `get_usd_composition` | ~15 |
| **Materials** | `create_material`, `create_material_network`, `assign_material`, `list_materials`, `get_material_info` | ~6 |
| **Lights** | `create_light`, `create_light_rig`, `list_lights`, `set_light_properties` | 4 |
| **Rendering** | `create_render_node`, `start_render`, `set_render_settings`, `get_render_settings`, `get_render_progress` | ~6 |
| **Viewport** | `render_viewport`, `set_viewport_renderer`, `set_viewport_display`, `set_viewport_direction`, `set_viewport_camera` | ~8 |
| **Screenshots** | `capture_screenshot`, `capture_network_editor`, `render_viewport`, `render_quad_view`, `render_node_network` | 5 |
| **COPs** | `create_cop_node`, `set_cop_flags`, `get_cop_info`, `get_cop_layer`, `get_cop_geometry`, `list_cop_node_types` | ~7 |
| **DOPs** | `get_simulation_info`, `get_dop_object`, `get_dop_field`, `get_dop_relationships`, `reset_simulation`, `step_simulation` | ~8 |
| **HDAs** | `create_hda`, `update_hda`, `install_hda`, `reload_hda`, `get_hda_info`, `get_hda_sections` | ~8 |
| **TOPs/PDG** | `cook_top_node`, `get_top_network_info`, `get_work_item_info`, `generate_static_items`, `dirty_work_items` | ~8 |
| **Scene** | `get_scene_info`, `get_scene_summary`, `load_scene`, `save_scene`, `new_scene`, `import_file`, `export_file` | ~8 |
| **Sim Workflows** | `setup_pyro_sim`, `setup_rbd_sim`, `setup_flip_sim`, `setup_vellum_sim` | 4 |
| **Diagnostics** | `find_error_nodes`, `get_node_errors_detailed`, `get_cook_chain`, `explain_node`, `get_cache_status` | ~6 |

---

## 2. Network Building Patterns

### 2.1 The Controller Null Pattern

**NEVER hardcode tweakable values in VEX or Python.** Always create a controller null with spare parameters so the user can adjust interactively.

```
Step 1: create_node /obj/geo1 null name:CTRL
Step 2: create_spare_parameter /obj/geo1/CTRL count int default:10 min:1 max:100
Step 3: create_spare_parameter /obj/geo1/CTRL size float default:1.0 min:0.1 max:10.0
Step 4: create_spare_parameter /obj/geo1/CTRL seed int default:42
Step 5: set_node_color /obj/geo1/CTRL color:[1,1,0]  (yellow = controller)
```

In VEX, read from the controller:
```vex
int count = chi("../CTRL/count");
float size = chf("../CTRL/size");
int seed = chi("../CTRL/seed");
```

### 2.2 Building SOP Networks

**Pattern: Create nodes, connect, layout regularly.**

```
// Create source geometry
create_node /obj/geo1 box name:source
set_parameter /obj/geo1/source sizex:2 sizey:0.5 sizez:2

// Add processing nodes
create_node /obj/geo1 polyextrude name:extrude
create_node /obj/geo1 mountain name:displace
create_node /obj/geo1 normal name:normals

// Wire them together
connect_nodes /obj/geo1/source /obj/geo1/extrude
connect_nodes /obj/geo1/extrude /obj/geo1/displace
connect_nodes /obj/geo1/displace /obj/geo1/normals

// IMPORTANT: Layout every 5-10 nodes, not just at the end
layout_children /obj/geo1

// Set display flag
set_node_flags /obj/geo1/normals display:true render:true
```

**For sequential chains, use `build_sop_chain`:**
```
build_sop_chain /obj/geo1 [
  {"type": "box", "name": "source", "parms": {"sizex": 2}},
  {"type": "polyextrude", "name": "extrude"},
  {"type": "mountain", "name": "displace"},
  {"type": "normal", "name": "normals"}
]
```

### 2.3 Batch Connections

When wiring many nodes at once:
```
connect_nodes_batch [
  {"from": "/obj/geo1/source", "to": "/obj/geo1/merge", "input": 0},
  {"from": "/obj/geo1/scatter", "to": "/obj/geo1/merge", "input": 1},
  {"from": "/obj/geo1/merge", "to": "/obj/geo1/output"}
]
```

### 2.4 Layout Discipline

**Call `layout_children` every 5-10 nodes.** Don't wait until the end.

```
// After first batch of 5 nodes
layout_children /obj/geo1

// After adding 5 more
layout_children /obj/geo1

// After final nodes
layout_children /obj/geo1
```

This prevents messy overlapping networks that confuse the user.

---

## 3. VEX Wrangle Patterns

### 3.1 Creating and Validating

```
// Create a wrangle (default runs over Points)
create_wrangle /obj/geo1 "v@Cd = set(1,0,0);" name:color_wrangle

// Set code separately (useful for multi-line)
set_wrangle_code /obj/geo1/attribwrangle1 "
float n = noise(@P * chf('../CTRL/freq'));
@Cd = set(n, n*0.5, 1.0 - n);
"

// ALWAYS validate after writing VEX
validate_vex /obj/geo1/attribwrangle1
```

### 3.2 Run-Over Contexts

```
// Points (default)
create_wrangle /obj/geo1 "..." run_over:Points

// Primitives
create_wrangle /obj/geo1 "..." run_over:Primitives

// Detail (single execution, good for aggregation)
create_wrangle /obj/geo1 "..." run_over:Detail

// Vertices
create_wrangle /obj/geo1 "..." run_over:Vertices

// Numbers (no geometry, just math)
create_wrangle /obj/geo1 "..." run_over:Numbers
```

### 3.3 Common VEX Patterns

**Scatter with controller:**
```vex
int count = chi("../CTRL/count");
int seed = chi("../CTRL/seed");
removepoint(0, @ptnum);
for(int i = 0; i < count; i++) {
    vector pos = set(rand(seed + i), rand(seed + i + 1000), rand(seed + i + 2000));
    addpoint(0, pos * chf("../CTRL/size"));
}
```

**Attribute transfer pattern:**
```vex
float n = noise(@P * 5.0);
@Cd = set(n, chramp("color_ramp", n), 1.0 - n);
@pscale = fit(n, 0, 1, chf("../CTRL/min_scale"), chf("../CTRL/max_scale"));
```

---

## 4. USD/LOP Workflows

### 4.1 Scene Assembly

```
// Create a USD scene
create_lop_node /stage/lopnet1 sublayer name:base_layer
create_lop_node /stage/lopnet1 sopimport name:import_geo
set_parameter /stage/lopnet1/import_geo soppath:/obj/geo1/OUT

// Inspect the stage
list_usd_prims /stage/lopnet1/import_geo
get_usd_prim /stage/lopnet1/import_geo /World/geo1
```

### 4.2 Lighting Setup

```
// Quick three-point rig
create_light_rig /stage/lopnet1 preset:three_point

// Or manual light creation
create_light /stage/lopnet1 type:rect name:key_light
set_light_properties /stage/lopnet1/key_light intensity:5.0 color:[1,0.95,0.9]

create_light /stage/lopnet1 type:dome name:env_light
set_light_properties /stage/lopnet1/env_light texture:"$HIP/textures/env.exr"
```

### 4.3 Material Assignment

```
// Create a principled shader
create_material /stage/lopnet1 name:metal_mat base_color:[0.8,0.8,0.85] roughness:0.2 metallic:1.0

// Assign to geometry
assign_material /stage/lopnet1 geometry:/World/geo1 material:/materials/metal_mat
```

---

## 5. COP (Copernicus) Automation

### 5.1 Creating COP Networks

```
// Create a COP network
create_cop_node /img/copnet1 file name:input
set_parameter /img/copnet1/input filename1:"$HIP/textures/input.exr"

// Create an OpenCL COP
create_cop_node /img/copnet1 opencl name:my_kernel
// Set kernel code via parameter (the 'kernel' parm contains the OpenCL source)

// Create a VEX wrangle COP
create_cop_node /img/copnet1 vexgenerate name:my_vex
```

### 5.2 COP Inspection

```
// Get COP resolution, format, layers
get_cop_info /img/copnet1/input

// Read layer data
get_cop_layer /img/copnet1/input layer:C

// List available COP node types
list_cop_node_types
```

### 5.3 COP Output

```
// Render COP to file (use rop_image, NOT rop_comp which doesn't exist in Copernicus)
create_cop_node /img/copnet1 rop_image name:render_out
set_parameter /img/copnet1/render_out outputfilepath:"$HIP/render/output.exr"
```

---

## 6. Simulation Workflows

### 6.1 Use Pre-Built Workflow Tools

**Always prefer workflow tools over manual network building:**

```
// Pyro (smoke, fire)
setup_pyro_sim /obj/geo1/source_geo type:smoke

// RBD (rigid body dynamics)
setup_rbd_sim /obj/geo1/fractured_geo

// FLIP (fluids)
setup_flip_sim /obj/geo1/emitter_geo

// Vellum (cloth, hair, grain, softbody)
setup_vellum_sim /obj/geo1/cloth_geo type:cloth
```

### 6.2 Simulation Inspection

```
// Check sim state
get_simulation_info /obj/dopnet1

// Query specific DOP objects
list_dop_objects /obj/dopnet1
get_dop_object /obj/dopnet1 object:fluid

// Read field data
get_dop_field /obj/dopnet1 object:fluid field:vel

// Step or reset
step_simulation /obj/dopnet1
reset_simulation /obj/dopnet1
```

---

## 7. HDA Development

### 7.1 Creating HDAs

```
// Create HDA from existing subnet
create_hda /obj/geo1/my_subnet name:my_tool label:"My Tool" version:1.0

// Or install an existing HDA
install_hda /path/to/tool.hda

// Update an HDA definition
update_hda /obj/geo1/my_tool1
```

### 7.2 HDA Inspection

```
// Get HDA metadata
get_hda_info /obj/geo1/my_tool1

// List internal sections (extra files, scripts, etc.)
get_hda_sections /obj/geo1/my_tool1

// Read a specific section
get_hda_section_content /obj/geo1/my_tool1 section:PythonModule
```

---

## 8. Viewport and Rendering

### 8.1 Viewport Control

```
// Set renderer for lookdev
set_viewport_renderer renderer:karma_cpu  // or karma_xpu, storm, gl

// Shading mode
set_viewport_display mode:smooth_shaded  // wireframe, shaded, matcap

// Standard views
set_viewport_direction direction:perspective
set_viewport_direction direction:front

// Look through camera
set_viewport_camera camera:/obj/cam1

// Auto-frame
frame_all
frame_selection
```

### 8.2 Screenshots — CRITICAL Visibility Requirements

**`capture_screenshot` and `render_viewport` use Houdini's `viewwrite` internally. This command REQUIRES the SceneViewer pane tab to be the active/visible tab.** If another tab (e.g., CompositorViewer for COPs) is active, the tool reports `"success": true` but **writes no file** — a silent failure with no error message.

**COP/Copernicus images CANNOT be captured via `viewwrite` at all** — it returns `"No viewers found to write"`. The only way to capture COP output is through a `rop_image` render node.

#### Capture by Context

| Context | Method | Notes |
|---------|--------|-------|
| **3D viewport (SOPs)** | `capture_screenshot` or `render_viewport` | SceneViewer must be the **active tab** in its pane |
| **COP/Copernicus output** | `rop_image` node → render to file → Read the file | The ONLY reliable method for COP images |
| **Network editor** | `capture_network_editor /path` | Works independently of tab visibility |
| **All 4 viewports** | `render_quad_view` | Requires SceneViewer visible |

#### Ensuring SceneViewer is Visible (Before SOP Captures)

If captures silently fail, the SceneViewer tab is likely hidden. Switch it to active first:

```python
execute_python """
import hou
desktop = hou.ui.curDesktop()
viewer = desktop.paneTabOfType(hou.paneTabType.SceneViewer)
if viewer:
    viewer.setIsCurrentTab()
"""
```

Then call `capture_screenshot` or `render_viewport` — they will work.

#### Capturing COP Output (The Correct Way)

```
// Create rop_image in the COP network (NOT rop_comp — doesn't exist in Copernicus)
create_cop_node /img/copnet1 rop_image name:render_out
set_parameter /img/copnet1/render_out outputfilepath:"/tmp/cop_output.png"

// Connect to the COP node you want to capture
connect_nodes /img/copnet1/my_opencl_cop /img/copnet1/render_out

// Render it
execute_python "hou.node('/img/copnet1/render_out').render()"

// Now read the image file for visual inspection
```

#### Network Editor Captures (Always Work)

```
// Great for documenting node graphs — no visibility requirement
capture_network_editor /obj/geo1
capture_network_editor /img/copnet1
```

### 8.3 Final Render

```
// Create render node
create_render_node /out type:karma name:final_render
set_render_settings /out/final_render resolution:[1920,1080] samples:128

// Start render
start_render /out/final_render
get_render_progress /out/final_render
```

---

## 9. Diagnostics and Debugging

### 9.1 Error Detection

```
// Find all nodes with errors (do this after building networks!)
find_error_nodes

// Get detailed error info
get_node_errors_detailed /obj/geo1/broken_node

// Trace cook chain to find dependency issues
get_cook_chain /obj/geo1/output

// Explain what a node does (useful for unfamiliar node types)
explain_node /obj/geo1/attribwrangle1
```

### 9.2 Data Inspection

```
// Geometry summary
get_geometry_info /obj/geo1/output
// Returns: point count, prim count, attribute list, bounding box

// Sample specific attributes
get_attrib_values /obj/geo1/output attribute:Cd class:point

// Query points with pagination (for large geo)
get_points /obj/geo1/output start:0 count:100

// List groups
get_groups /obj/geo1/output

// Bounding box
get_bounding_box /obj/geo1/output
```

---

## 10. `execute_python` — Scene-Level Scripting Only

**Use ONLY when no dedicated MCP tool exists.** Examples:

```python
# Setting up a complex node network programmatically
execute_python """
import hou
geo = hou.node('/obj').createNode('geo', 'procedural_city')
# ... complex network setup that would take 50+ MCP calls
"""

# Batch parameter operations
execute_python """
import hou
for node in hou.node('/obj/geo1').children():
    if node.type().name() == 'attribwrangle':
        node.parm('class').set(1)  # run over prims
"""

# Custom UI operations (shelf tools, menus)
execute_python """
import hou
hou.ui.displayMessage('Build complete!', title='Status')
"""
```

**NEVER use `execute_python` for:**
- Creating individual nodes (use `create_node`)
- Setting individual parameters (use `set_parameter`)
- Reading geometry data (use `get_geometry_info`, `get_points`)
- VEX code (use `create_wrangle`)

---

## 11. Standard Workflow Template

For any Houdini automation task, follow this sequence:

```
1. INSPECT — get_scene_info, find_nodes (understand current state)
2. BUILD   — create_node, create_wrangle, connect_nodes (build network)
3. LAYOUT  — layout_children every 5-10 nodes
4. CONTROL — create_spare_parameter on CTRL null (user-facing knobs)
5. WIRE    — set_expression for parameter dependencies
6. VERIFY  — validate_vex, find_error_nodes (catch problems early)
7. VIEW    — set_viewport_renderer, frame_all (let user see results)
8. CAPTURE — capture_screenshot (document the result)
```

---

## 12. MCP Server Lifecycle

### Architecture

```
Claude Code (parent process)
  └── fxhoudinimcp (child process, MCP server)
        └── TCP connection → 127.0.0.1:8100

Houdini FX (separate process)
  └── Listening on port 8100 (started via shelf tool)
```

**The MCP server is a child of Claude Code, NOT Houdini.** It connects to Houdini's internal server on TCP port 8100 (localhost).

### When You Quit Houdini

1. Houdini's port 8100 listener dies
2. The MCP server process **stays alive** (owned by Claude Code)
3. MCP tool calls will fail with connection errors
4. The MCP server process is cleaned up when the Claude Code session ends

### Safe Shutdown

```sh
# Option 1: Just quit Houdini normally
# MCP server harmlessly loses connection, dies with Claude Code session

# Option 2: Kill MCP server explicitly (cleanest)
pkill -f fxhoudinimcp
# Then quit Houdini

# Option 3: After Houdini is already closed
pkill -f fxhoudinimcp
```

No "orphan server" risk — the MCP process is always parented to Claude Code.

### Checking Status

```sh
# Is the MCP server running?
pgrep -l -f fxhoudinimcp

# What port is it using?
lsof -i :8100 -P -n

# Is Houdini still listening?
lsof -i :8100 -P -n | grep LISTEN
```

---

## 13. Security Configuration

The MCP server is security-hardened:

- **Localhost only**: Bound to `127.0.0.1` (no network exposure)
- **Manual start**: Auto-start disabled to prevent Houdini startup hang; start via shelf tool
- **No credentials in repo**: `.mcp.json` is in `.gitignore`
- **No destructive operations**: MCP tools don't delete files or modify system state outside Houdini

---

## 14. Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| Using `execute_python` for geometry | Use `create_wrangle` (VEX) instead |
| Hardcoding values in VEX | Create spare parameters on CTRL null |
| Not calling `layout_children` | Call every 5-10 nodes during build |
| Using `rop_comp` in Copernicus | Use `rop_image` (rop_comp doesn't exist) |
| Forgetting to validate VEX | Call `validate_vex` after `set_wrangle_code` |
| Not checking for errors | Call `find_error_nodes` after building networks |
| Building sims from scratch | Use `setup_pyro_sim`, `setup_rbd_sim`, etc. |
| Single node connections in loop | Use `connect_nodes_batch` for efficiency |
| Not framing viewport | Call `frame_all` after building geometry |
| Auto-start MCP on Houdini launch | Causes startup hang; use manual shelf tool |
| `capture_screenshot` silently fails | SceneViewer must be the **active tab** — switch with `viewer.setIsCurrentTab()` first |
| Trying to screenshot COP output | `viewwrite` can't capture COP viewer — use `rop_image` node to render to file |
| MCP server lingers after Houdini quits | MCP is a child of Claude Code, not Houdini — `pkill -f fxhoudinimcp` to clean up |

---

## Session Log

| Date | Test | Status |
|------|------|--------|
| 2026-03-27 | Full MCP tool inventory audit (167 tools, 20 categories) | Verified |
| 2026-03-27 | OpenCL COP creation + iteration via MCP | Verified |
| 2026-03-27 | Ray march SDF scene built entirely through MCP | Verified |
| 2026-03-27 | Reaction-diffusion sim created through MCP tools | Verified |
| 2026-03-27 | COP network with rop_image output | Verified |
| 2026-03-27 | Houdini scene saved via save_scene | Verified |
| 2026-03-29 | Screenshot failure root cause: SceneViewer must be active tab | Verified |
| 2026-03-29 | COP viewer cannot be captured via viewwrite — use rop_image | Verified |
| 2026-03-29 | MCP server lifecycle: child of Claude Code, TCP to Houdini:8100 | Verified |
