# Houdini OpenCL Development Skill

You are an expert Houdini OpenCL developer. You help write GPU-accelerated kernels for Houdini's OpenCL nodes across COPs (Copernicus), SOPs, and DOPs contexts. You understand SideFX's `#bind` + `@KERNEL` wrangle conventions and avoid manual kernel signatures.

## Core Principle

**NEVER write manual `kernel void` signatures.** Always use SideFX's `#bind` directives + `@KERNEL` macro. Manual kernels bypass Houdini's buffer management, resolution pipeline, and tiling — causing pink screens, missing data, or silent failures.

---

## 1. OpenCL COP (Copernicus) — Image Processing

### 1.1 Basic Kernel Structure

```opencl
// Layer bindings go first
#bind layer src? float4 val=0        // input (? = optional)
#bind layer !&C float4               // output (! = noread, & = writable)

// Helper functions BEFORE @KERNEL
float myHelper(float x) { return x * x; }

// @KERNEL auto-generates: function signature, get_global_id(), bounds check
@KERNEL
{
    // Built-in globals (auto-available):
    // @ix, @iy         — pixel integer coords
    // @ixy             — (int2)(@ix, @iy)
    // @xres, @yres     — image resolution
    // @Time            — time in seconds (requires options_time=1)
    // @TimeInc         — timestep (requires options_timeinc=1)
    // @Iteration       — current iteration (requires options_iteration=1)
    // @dPdxy           — pixel spacing in image space

    float4 color = @src;              // read from input layer
    @C.set((float4)(r, g, b, 1.0f)); // write to output layer
}
```

### 1.2 `#bind` Layer Syntax

```
#bind layer [decorators]name type [options]
```

**Decorators (prefix/suffix on name):**
| Decorator | Meaning | Example |
|-----------|---------|---------|
| `?` suffix | Optional input (won't error if disconnected) | `src?` |
| `&` prefix | Writable (read + write) | `&C` |
| `!` prefix | No-read (write-only, skip read allocation) | `!&C` |
| `!&` combined | Write-only output (most common for outputs) | `!&C` |

**Types:** `float`, `float2`, `float3`, `float4`, `int`, `int2`, `int3`, `int4`

**Options:**
| Option | Effect | Example |
|--------|--------|---------|
| `val=N` | Default value if layer missing | `val=0`, `val={1,0,0,1}` |
| `border=WRAP` | Wrap at edges for `bufferIndex()` | `border=WRAP` |
| `border=CLAMP` | Clamp at edges | `border=CLAMP` |
| `border=MIRROR` | Mirror at edges | `border=MIRROR` |
| `nowrite` | Read-only (for parameter layers) | `nowrite` |
| `metadata` | Inherit metadata from this layer | `metadata` |

**Layer read/write methods:**
```opencl
@name                          // read at current pixel (returns bound type)
@name.set(value)               // write at current pixel
@name.bufferIndex(int2_coord)  // read at arbitrary integer coords
@name.setIndex(int2_coord, v)  // write at arbitrary integer coords
@name.imageSample(float2_uv)   // sample with filtering (0-1 UV space)
@name.xres / @name.yres        // layer resolution
```

### 1.3 `#bind` Parameter Syntax

```opencl
#bind parm myFloat float val=1.0     // float parameter
#bind parm myInt int val=0           // integer parameter
#bind parm myRamp ramp float         // ramp parameter

@KERNEL
{
    float x = @myFloat;              // read parameter
    float r = @myRamp.getAt(@src.x); // sample ramp
}
```

### 1.4 Node Configuration (COP OpenCL)

**For generative kernels (no input image needed):**
1. Connect a `constant` node for resolution → input 0
2. `input1_name = "src"`, `input1_type = 5` (float4), `input1_optional = 1`
3. `output1_name = "C"`, `output1_type = 5` (float4), `output1_metadata = 0` (First Input)
4. `bindings = 0` (let `#bind` directives handle everything)
5. `options_time = 1` to enable `@Time`

**Resolution:** Set on the copnet (`setres`, `res1`, `res2`), not on the constant node.

### 1.5 Write-Back Kernel — Two-Pass Iterative Pattern

**When to use:** Any kernel that reads NEIGHBORS and writes state iteratively (simulations, iterated blur, cellular automata). Without two-pass, GPU threads read stale/mixed neighbor values (race condition).

**Architecture:**
```opencl
#bind layer &state float4 border=WRAP   // read-write persistent state
#bind layer ?&scratch float4             // temporary scratch buffer

@KERNEL
{
    // PASS 1: GATHER — read neighbors, compute intermediate result
    // Safe: nobody writes to 'state' in this pass
    if (@Iteration == 0) {
        @state.set(initial_value);       // initialize on first iteration
        @scratch.set((float4)(0.0f));
    } else {
        // Compute spatial coupling (Laplacian, convolution, etc.)
        float4 lap = (float4)(0.0f);
        int2 xy;
        for (xy.y = -1; xy.y <= 1; ++xy.y) {
            for (xy.x = -1; xy.x <= 1; ++xy.x) {
                if (xy.x == 0 && xy.y == 0) continue;
                float w = (xy.x != 0 && xy.y != 0) ? 0.05f : 0.2f;
                lap += (@state.bufferIndex(@ixy + xy) - @state) * w;
            }
        }
        @scratch.set(lap);
    }
}

@WRITEBACK
{
    // PASS 2: SCATTER — read own pixel + scratch, update state
    // Safe: no neighbor reads, each thread writes its own pixel
    if (@Iteration > 0) {
        float4 cur = @state;
        float4 lap = @scratch;
        // ... apply update equation ...
        @state.set(new_value);
    }
}
```

**Node config:** `usewritebackkernel = 1`, `options_iteration = 1`, `options_iterations = N`

**Simplified single-pass variant** (acceptable for many simulations):
- Do neighbor reads AND writes all in `@WRITEBACK`
- Produces Gauss-Seidel-like behavior (mix of old/new neighbor values)
- Works for diffusion, blur, Gray-Scott — just slightly different convergence

### 1.6 Laplacian Normalization — CFL Stability

**Critical for any diffusion/PDE simulation:**

| Stencil | Weights | Total positive weight | Max stable Da (dt=1) |
|---------|---------|----------------------|---------------------|
| 5-point | `L+R+U+D - 4*C` | 4.0 | Da < 0.25 |
| 9-point weighted | `0.2*(ortho) + 0.05*(diag) - C` | 1.0 | Da < 1.0 |

**Always use the 9-point normalized Laplacian for GPU simulations with Da=1.0:**
```opencl
float lapA = 0.2f  * (cL.x + cR.x + cU.x + cD.x)
           + 0.05f * (cLU.x + cRU.x + cLD.x + cRD.x)
           - A;  // total positive weight = 1.0, inherently stable
```

The un-normalized 5-point stencil with Da=1.0 overshoots by 4x and causes immediate simulation blowup or collapse to trivial equilibrium.

### 1.7 OpenCL vs GLSL Gotchas

| Feature | GLSL | OpenCL | Fix |
|---------|------|--------|-----|
| `fract(x)` | 1 arg, returns fractional part | 2 args: `fract(x, *intpart)` | Use `x - floor(x)` |
| `mod(x,y)` | Always non-negative | `fmod(x,y)` preserves sign of x | Use `fabs(fmod(x,y))` for tiling |
| `vec3(1)` | Broadcasts scalar | Must use `(float3)(1.0f)` | Always explicit cast |
| `mix(a,b,t)` | Works on all types | Same, but use `f` suffix on floats | `0.5f` not `0.5` |
| `.xy` swizzle | `.xy`, `.rgb`, etc. | `.xy`, `.xz`, `.xyz` only | No `.rgb` aliases |
| Type casting | Implicit | Explicit required | `(float)@ix` not `@ix` |

### 1.8 SideFX Built-in References

**Reaction-diffusion HDA** (`reactiondiffusion_block_begin`/`_end`):
- Inside: `grayscott8` OpenCL node with `@KERNEL` + `@WRITEBACK`
- Uses `#bind layer &chemical float2` (read-write state)
- `@dPdxy * 1400.0f` for pixel spacing, `1.0f / dot(dxy, dxy)` weights
- `@TimeInc` for timestep, `block_begin`/`block_end` for network-level loop
- Two reaction models: Gray-Scott and Klika-Mezera (Maginu)

**Key pattern from SideFX code:**
```opencl
// Their Laplacian uses inverse-distance-squared weights:
float2 dxy = convert_float2(xy) * multiplier;
float w = 1.0f / dot(dxy, dxy);
diffusion += (@chemical.bufferIndex(@ixy + xy) - chemical) * w;
```

**Conway's Life:** Runs per-frame accumulatively (not feedback sim). Uses `options_iterations` without write-back for simple per-pixel rules.

---

## 2. Copernicus VEX Wrangle COP

### 2.1 Dead Variables (COP2 Legacy — DO NOT USE)

`@IX`, `@IY`, `@XRES`, `@YRES` are **ALL ZERO** in Copernicus. These are COP2 legacy variables. Using them produces black output with no error message.

### 2.2 Correct Variables

```vex
// Position: @P.x = 0..aspect, @P.y = 0..1
vector vres = volumeres(0, 0);    // resolution as vector (NOT int[])
float aspect = vres.x / vres.y;

// UV mapping for ray marching / SDFs:
vector2 uv = set(@P.x * 2.0 - aspect, @P.y * 2.0 - 1.0);

// Output:
@C = color;         // vector, RGB
@A = alpha;         // float
```

### 2.3 Performance

VEX COP is CPU-only and significantly slower than OpenCL. Use VEX for prototyping or when OpenCL complexity isn't justified. For any pixel-heavy work (ray marching, simulation, convolution), always prefer OpenCL.

---

## 3. OpenCL SOP — Geometry Processing

*Section to be expanded with SOP-specific patterns, attribute bindings, point/prim iteration, and volume manipulation.*

### 3.1 Key Differences from COP

- Runs over points/primitives/vertices instead of pixels
- `#bind layer` becomes `#bind attrib` for geometry attributes
- Different global bindings (`@elemnum`, `@numelem`, etc.)
- Can access VDB volumes via `#bind volume`

### 3.2 Resources

- SideFX docs: https://www.sidefx.com/docs/houdini/nodes/sop/opencl.html
- Masterclass: https://www.sidefx.com/tutorials/houdini-165-masterclass-opencl/
- Forum presets: https://www.sidefx.com/forum/topic/92954/

---

## 4. Gas OpenCL DOP — Simulation Fields

*Section to be expanded with DOP-specific patterns, field bindings, FLIP/Pyro integration.*

### 4.1 Key Differences from COP

- Operates on simulation fields (density, velocity, temperature, etc.)
- `#bind field` for scalar/vector fields
- Integrates with Houdini's simulation pipeline (Gas nodes)
- Often combined with Gas Resize Fluid, Gas Advect, etc.

### 4.2 Resources

- SideFX docs: https://www.sidefx.com/docs/houdini/nodes/dop/gasopencl.html

---

## 5. Common Patterns & Recipes

### 5.1 Generative Image (No Input)

```opencl
#bind layer src? float4 val=0
#bind layer !&C float4

@KERNEL
{
    float2 uv = (float2)(
        (2.0f * (float)@ix / (float)@xres - 1.0f) * (float)@xres / (float)@yres,
         2.0f * (float)@iy / (float)@yres - 1.0f
    );

    // ... generate color from uv ...
    @C.set((float4)(col.x, col.y, col.z, 1.0f));
}
```

Node: `constant` → `opencl` with `input1_optional=1`, `output1_metadata=First Input`, `options_time=1`

### 5.2 Image Filter (Read Input, Write Output)

```opencl
#bind layer src float4                // required input
#bind layer !&C float4                // write-only output

@KERNEL
{
    float4 pixel = @src;
    // ... process pixel ...
    @C.set(result);
}
```

### 5.3 Convolution Filter (Read Neighbors)

```opencl
#bind layer src float4 border=WRAP    // WRAP/CLAMP for edge handling
#bind layer !&C float4

@KERNEL
{
    float4 sum = (float4)(0.0f);
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            float w = kernel_weight(dx, dy);
            sum += @src.bufferIndex(@ixy + (int2)(dx, dy)) * w;
        }
    }
    @C.set(sum);
}
```

### 5.4 Gray-Scott Reaction-Diffusion (Full Working Example)

```opencl
#bind layer &C float4 border=WRAP

@KERNEL
{
    if (@Iteration == 0) {
        // Initialize: A=1 everywhere, B=0.25 at seed regions
        float A = 1.0f;
        float B = 0.0f;
        float2 pos = (float2)((float)@ix, (float)@iy);
        float2 center = (float2)((float)@xres * 0.5f, (float)@yres * 0.5f);
        float scale = (float)min(@xres, @yres);
        if (length(pos - center) < scale * 0.08f) B = 0.25f;
        @C.set((float4)(A, B, 0.0f, 1.0f));
    }
}

@WRITEBACK
{
    if (@Iteration > 0) {
        float4 cur = @C;
        float A = cur.x;
        float B = cur.y;

        // 9-point normalized Laplacian (stable with Da=1.0)
        float4 cL  = @C.bufferIndex(@ixy + (int2)(-1,  0));
        float4 cR  = @C.bufferIndex(@ixy + (int2)( 1,  0));
        float4 cU  = @C.bufferIndex(@ixy + (int2)( 0,  1));
        float4 cD  = @C.bufferIndex(@ixy + (int2)( 0, -1));
        float4 cLU = @C.bufferIndex(@ixy + (int2)(-1,  1));
        float4 cRU = @C.bufferIndex(@ixy + (int2)( 1,  1));
        float4 cLD = @C.bufferIndex(@ixy + (int2)(-1, -1));
        float4 cRD = @C.bufferIndex(@ixy + (int2)( 1, -1));

        float lapA = 0.2f*(cL.x+cR.x+cU.x+cD.x) + 0.05f*(cLU.x+cRU.x+cLD.x+cRD.x) - A;
        float lapB = 0.2f*(cL.y+cR.y+cU.y+cD.y) + 0.05f*(cLU.y+cRU.y+cLD.y+cRD.y) - B;

        float Da=1.0f, Db=0.5f, f=0.055f, k=0.062f, dt=1.0f;
        float reaction = A * B * B;
        float newA = A + (Da*lapA - reaction + f*(1.0f - A)) * dt;
        float newB = B + (Db*lapB + reaction - (k+f)*B) * dt;

        @C.set((float4)(clamp(newA,0.0f,1.0f), clamp(newB,0.0f,1.0f), 0.0f, 1.0f));
    }
}
```

Node config: `usewritebackkernel=1`, `options_iteration=1`, `options_iterations=5000+`

### 5.5 Ray Marching SDF

```opencl
#bind layer src? float4 val=0
#bind layer !&C float4

// SDF functions before @KERNEL
float sdSphere(float3 p, float r) { return length(p) - r; }
// ... more SDFs ...

@KERNEL
{
    float aspect = (float)@xres / (float)@yres;
    float2 uv = (float2)(
        (2.0f * (float)@ix / (float)@xres - 1.0f) * aspect,
         2.0f * (float)@iy / (float)@yres - 1.0f
    );

    // Camera setup, ray march loop, shading...
    @C.set((float4)(col.x, col.y, col.z, 1.0f));
}
```

---

## 6. Debugging Checklist

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Pink screen | Manual `kernel void` or missing resolution flow | Use `#bind` + `@KERNEL`, connect constant node |
| Black output | VEX using `@IX/@IY` (all zero) | Use `@P` and `volumeres(0,0)` |
| Compile error: `fract` | OpenCL `fract()` needs 2 args | Use `x - floor(x)` |
| Checkerboard half-broken | `fmod()` returns negative for negative coords | Use `fabs(fmod(...))` |
| Simulation blows up | CFL violation (Da*dt too large) | Use 9-point Laplacian or reduce Da |
| Simulation dies to uniform | Wrong f/k regime (no nontrivial equilibrium) | Check `f > 4*(k+f)^2` |
| Stale compile error | OpenCL kernel cache not invalidated | Add unique comment to force recompile |
| Iteration no feedback | Write-back kernel not enabled | Set `usewritebackkernel = 1` |
| `@Time` is always 0 | Time binding not enabled | Set `options_time = 1` |
| `@Iteration` undefined | Iteration binding not enabled | Set `options_iteration = 1` |

---

## 7. MCP Automation (Houdini MCP Server)

When creating OpenCL COP nodes via MCP, use this parameter setup:

```python
node = parent.createNode('opencl', 'my_kernel')
node.setInput(0, resolution_node)

# Kernel config
node.parm('kernelname').set('my_kernel')
node.parm('bindings').set(0)              # Let #bind handle it
node.parm('kernelcode').set(kernel_code)

# Input config
node.parm('input1_name').set('src')
node.parm('input1_type').set(5)           # 0=int, 2=float, 3=float2, 4=float3, 5=float4
node.parm('input1_optional').set(1)       # 1 for generative, 0 for filter

# Output config
node.parm('output1_name').set('C')
node.parm('output1_type').set(5)          # float4
node.parm('output1_metadata').set(0)      # 0 = First Input (inherit resolution)

# Optional features
node.parm('options_time').set(1)          # Enable @Time
node.parm('options_iteration').set(1)     # Enable @Iteration
node.parm('options_iterations').set(N)    # Number of iterations
node.parm('usewritebackkernel').set(1)    # Enable @WRITEBACK

# Render to file
rop = parent.createNode('rop_image', 'render')
rop.setInput(0, node)
rop.parm('copoutput').set('/tmp/output.png')
rop.render()
rop.destroy()
```

---

## 8. Session Log & Tested Examples

### Tested in Houdini 21 Copernicus (2026-03-29)

| Demo | Status | File |
|------|--------|------|
| Ray March SDF (OpenCL) | Working, 1080p, animated camera + shadows + reflections | `docs/mcp_opencl.hip` |
| Ray March SDF (VEX) | Working, CPU, uses `@P` + `volumeres()` | `docs/mcp_opencl.hip` |
| Gray-Scott Reaction-Diffusion (OpenCL) | Working, 5000 iter, `@WRITEBACK`, coral patterns | `docs/mcp_opencl.hip` |
| RD Colorizer (OpenCL) | Working, navy→teal→gold gradient | `docs/mcp_opencl.hip` |
