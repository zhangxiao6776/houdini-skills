# OFX Plugin Development & Copernicus Integration Skill

You are an expert OpenFX (OFX) C++ plugin developer who deploys creative image processing plugins into Houdini Copernicus. You understand the full lifecycle: architecture design, coding conventions, build systems, Copernicus-specific limitations, and deployment workflows.

## Core Principle

**OFX plugins are portable C++ shared libraries that run in any OFX host.** But Copernicus has specific limitations (no temporal clip access, no handles). Design plugins to work within these constraints from the start, not as an afterthought.

---

## 1. Plugin Architecture Taxonomy

### 1.1 Single-Pass Stylization (FMModulation, PixelArt, AsciiArt, Dither)

- Per-pixel transform, no cross-pixel dependencies
- Supports tiling (`setSupportsTiles(true)`)
- All work happens in `multiThreadProcessImages()`
- Template: `Processor<PIX, nComponents, maxValue>` handles 8/16/32-bit

```
Source → [per-pixel transform] → Output
```

### 1.2 Two-Pass Statistical (ColorTransfer)

- Pass 1: Gather full-image statistics (single-threaded in `setupAndProcess()`)
- Pass 2: Apply per-pixel transform using stats (multi-threaded)
- **Must** call `setSupportsTiles(false)` — needs full image for stats

```
Source → [gather stats] → [per-pixel transform with stats] → Output
```

### 1.3 Inference-Based (DepthEstimation, DeepBump)

- AI model runs in `setupAndProcess()` (single-threaded)
- Result cached per-frame, applied in `multiThreadProcessImages()`
- Dual-backend: ONNX C++ (production) + Python subprocess (fallback)
- Frame-level caching to avoid redundant inference

```
Source → [inference (cached)] → [output mapping] → Output
```

### 1.4 Persistent Server (VideoDepth)

- Persistent Python server via Unix domain socket
- Server maintains temporal state (KV-cache) across frames
- Auto-start via `fork()`+`setsid()`, auto-shutdown after 5min idle
- Binary IPC protocol for low-latency communication

```
Source → [socket IPC] → Python Server (persistent) → [output mapping] → Output
```

---

## 2. C++ Coding Conventions

### 2.1 File Structure

```cpp
// SPDX-License-Identifier: BSD-3-Clause
// Copyright notice

#ifdef _WIN32
#define NOMINMAX
#endif

#include <algorithm>        // Standard library
#include <cmath>
#include <vector>

#include "ofxsImageEffect.h"   // OFX SDK
#include "ofxsProcessing.H"

#include "utils.h"          // Local shared headers
#include "mat3.h"
```

### 2.2 Parameter Constants

```cpp
static const char* kParamOmega     = "omega";        // DoubleParam
static const char* kParamColorMode = "colorMode";    // ChoiceParam
static const char* kParamInvert    = "invert";       // BooleanParam
static const char* kParamOpacity   = "opacity";      // DoubleParam
```

### 2.3 Class Hierarchy

```
MyPluginBase (inherits OFX::ImageProcessor)
  - Holds all parameter values as members
  - setParams() to receive values from plugin class

MyPluginProcessor<PIX, nComponents, maxValue> (inherits Base)
  - multiThreadProcessImages() — per-pixel work
  - Uses toFloat<>/fromFloat<> from utils.h

MyPlugin (inherits OFX::ImageEffect)
  - Holds OFX::Param* pointers
  - setupAndProcess() — reads params, creates processor, calls process()
  - render() — dispatches by bit depth and component count
  - describeInContext() — defines UI parameters
```

### 2.4 Pixel Type Handling

```cpp
// From utils.h — convert any pixel type to float [0,1]:
template<int maxValue>
static inline float toFloat(auto v) {
    return (maxValue == 1) ? static_cast<float>(v)
                           : static_cast<float>(v) / static_cast<float>(maxValue);
}
```

### 2.5 Adding a New Parameter (11-Step Checklist)

1. Add `static const char* kParam...` constant
2. Add member variable to Base class
3. Initialize in Base constructor
4. Add to `setParams()` signature and body
5. Add `OFX::XxxParam*` member to Plugin class
6. Initialize to `0` in Plugin constructor
7. `fetchXxxParam()` in Plugin constructor body
8. `getValueAtTime()` in `setupAndProcess()`
9. Pass to `processor.setParams()`
10. Use in processing functions
11. Define UI in `describeInContext()` with labels, hints, defaults, ranges, parent group

---

## 3. Build System (CMake)

### 3.1 Basic Build

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DPLUGIN_INSTALLDIR="$HOME/OFX/Plugins"
cmake --build build --config Release --parallel
cmake --install build
```

### 3.2 Build with ONNX Runtime

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DONNXRUNTIME_DIR=~/onnxruntime
cmake --build build --config Release --parallel
```

### 3.3 Build Single Plugin

```sh
cmake --build build --config Release --target DepthEstimation --parallel
```

### 3.4 CMake Structure

- OpenFX SDK fetched via `FetchContent` (no vendoring)
- Each plugin is a `MODULE` library target
- Conditional ONNX Runtime linking via `DEPTH_HAS_ONNX` define
- macOS universal binary (x86_64 + arm64)
- Install creates `.ofx.bundle` directory structure

---

## 4. Houdini Copernicus OFX Limitations

### 4.1 CRITICAL: No Temporal Clip Access

```cpp
// DO NOT USE — Copernicus silently passes through source image:
desc.setTemporalClipAccess(true);           // BROKEN
srcClip->setTemporalClipAccess(true);       // BROKEN
srcClip_->fetchImage(time);  // for time != args.time — returns wrong data or null
```

**There is NO error message.** The plugin loads, `render()` is called, your code runs, but the host discards the output and returns the input image unchanged.

**Workaround:** Process one frame at a time. Use persistent external state (Python server, disk cache) for cross-frame coherence.

### 4.2 Other Unsupported Features

- OFX handles (roto tools, on-screen widgets)
- Neat Video plugins specifically
- Custom overlays

### 4.3 Per-Frame Animation in Copernicus

Houdini's COP graph is pull-based. A node only re-cooks if an input or parameter changes. For plugins that must update every frame (even with still input):

```cpp
// Step 1: Set frame-varying flag
void MyPlugin::getClipPreferences(OFX::ClipPreferencesSetter& cp) {
    cp.setOutputFrameVarying(true);
}

// Step 2: Add a Frame parameter the user sets to $F
// This forces Houdini to re-cook every frame
static const char* kParamFrame = "frame";
// In describeInContext():
IntParamDescriptor* frame = desc.defineIntParam(kParamFrame);
frame->setLabel("Frame");
frame->setHint("Set to $F in Houdini to force per-frame update");
frame->setDefault(0);

// Step 3: Convert time to frame number (Houdini passes seconds, not frames)
double fps = srcClip_->getFrameRate();
if(fps <= 0.0) fps = 24.0;
int frame = static_cast<int>(std::round(args.time * fps)) + frameParam;
```

---

## 5. Deployment to Houdini

### 5.1 Bundle Structure

```
MyPlugin.ofx.bundle/
  Contents/
    MacOS/
      MyPlugin.ofx                     # Plugin binary
    Frameworks/
      libonnxruntime.dylib             # Optional: ONNX Runtime (macOS)
    Resources/
      model.onnx                       # Optional: AI model weights
      model.mlmodelc/                  # Optional: CoreML compiled model
      infer_script.py                  # Optional: Python fallback
    Info.plist
```

### 5.2 Deploy Commands

```sh
# Build
cmake --build build --config Release --target MyPlugin --parallel

# Copy binary to bundle
cp build/MyPlugin.ofx ~/OFX_Plugins/MyPlugin.ofx.bundle/Contents/MacOS/

# Copy resources if needed
cp src/scripts/my_script.py ~/OFX_Plugins/MyPlugin.ofx.bundle/Contents/Resources/

# RESTART HOUDINI (no hot-reload for OFX plugins)
```

### 5.3 Plugin Path Discovery

```cpp
// CORRECT: dladdr() finds the plugin's own .ofx path
#include <dlfcn.h>
static std::string getPluginDir() {
    Dl_info info;
    if(dladdr(reinterpret_cast<void*>(&getPluginDir), &info) && info.dli_fname) {
        std::string path(info.dli_fname);
        size_t pos = path.rfind('/');
        if(pos != std::string::npos) return path.substr(0, pos);
    }
    return ".";
}

// WRONG: _NSGetExecutablePath() returns Houdini's path, not the plugin's
```

### 5.4 macOS Universal Binary + ONNX Runtime

```sh
# Create fat ONNX Runtime library
lipo -create onnxruntime-arm64/lib/libonnxruntime.dylib \
             onnxruntime-x86_64/lib/libonnxruntime.dylib \
     -output ~/onnxruntime/lib/libonnxruntime.dylib
```

---

## 6. Multi-Threading Patterns

### 6.1 Standard Per-Pixel (Tiling OK)

```cpp
void multiThreadProcessImages(OfxRectI procWindow) override {
    for(int y = procWindow.y1; y < procWindow.y2; y++) {
        PIX* dst = static_cast<PIX*>(_dstImg->getPixelAddress(procWindow.x1, y));
        const PIX* src = static_cast<const PIX*>(_srcImg->getPixelAddress(procWindow.x1, y));
        for(int x = procWindow.x1; x < procWindow.x2; x++) {
            // per-pixel transform
            dst += nComponents;
            src += nComponents;
        }
    }
}
```

### 6.2 Error Diffusion (Thread-Safe)

For algorithms that push error to neighboring pixels:
- Allocate local error rows sized to the thread's chunk
- Add padding (`chunkWidth + 4`) to handle overflow
- Accept chunk boundary resets (adds to the digital aesthetic)
- **NEVER** use a global error buffer without locks

### 6.3 Two-Pass with Stats

```cpp
void setupAndProcess(/* ... */) {
    // Pass 1: Single-threaded stats gathering (non-templated)
    Stats stats;
    gatherStats(srcImg, refImg, depth, nComp, stats);

    // Pass 2: Multi-threaded pixel transform
    processor.setStats(stats);
    processor.process();
}
```

---

## 7. Shared Headers

### 7.1 `utils.h` — Pixel Conversion

```cpp
#include "utils.h"
// toFloat<maxValue>(pixel)   — any pixel type to [0,1] float
// fromFloat<maxValue>(float) — [0,1] float back to pixel type
// luminance(r, g, b)         — ITU-R BT.601 luminance
// clampVal(v, lo, hi)        — clamped value
```

### 7.2 `mat3.h` — Linear Algebra

```cpp
#include "mat3.h"
// Mat3 — 3x3 matrix struct
// mat3_eigen_symmetric()     — Jacobi eigenvalue decomposition
// mat3_sqrt(), mat3_inv_sqrt() — matrix square root via eigendecomp
// mat3_multiply(), mat3_transpose(), mat3_inverse()
```

### 7.3 `fft.h` — FFT (Used by DeepBump)

```cpp
#include "fft.h"
// fft_1d()     — radix-2 Cooley-Tukey 1D FFT
// ifft_1d()    — inverse 1D FFT
// fft_2d()     — 2D FFT via row/column decomposition
// ifft_2d()    — inverse 2D FFT
```

---

## 8. Cross-Verification Methodology

When porting algorithms from Python to C++:

1. **Python reference** (`tests/test_*.py`) — compute with known inputs, print ALL intermediates
2. **C++ test** (`tests/test_*.cpp`) — identical inputs, compare step by step
3. **Threshold**: `1e-10` for exact math, `1e-6` for float precision
4. **Build**: `c++ -std=c++17 -O2 -o tests/test_foo tests/test_foo.cpp`

**Key rule:** Compare intermediate values, not just final outputs. The first divergence point reveals the bug.

---

## 9. Numerical Stability Patterns

| Pattern | Value | Why |
|---------|-------|-----|
| Jacobi convergence | `offDiag < 1e-15` | Stop when negligible |
| MKL invD epsilon | `1e-15` | Prevent divide-by-zero |
| Reinhard std guard | `1e-10` | Skip normalization for uniform images |
| LMS log10 clamp | `1e-10` | Prevent log(0) = -inf |
| Matrix inverse singular | `det < 1e-30` | Return identity |
| Eigenvalue clamp | `< 0 -> 0` | Covariance matrices are PSD |

---

## 10. Complete Plugin Portfolio

| Plugin | Type | Architecture | Key Feature |
|--------|------|-------------|-------------|
| **FMModulation** | Stylization | Single-pass, tiling | FM synthesis image effects |
| **PixelArt** | Stylization | Single-pass, tiling | Downsampling + palette quantization |
| **AsciiArt** | Stylization | Single-pass, tiling | Character rendering from luminance |
| **Dither** | Stylization | Single-pass, tiling | 1-bit dithering effects |
| **ColorTransfer** | Statistical | Two-pass, no tiles | HM, Reinhard, MKL, MVGD methods |
| **DepthEstimation** | Inference | Dual-backend + CoreML | DA V2 Small/Large, Depth Pro |
| **DeepBump** | Hybrid | ONNX + algorithmic | Normals, Height (FFT), Curvature |
| **VideoDepth** | Server | Persistent Python socket | VDA temporal coherence |

---

## 11. Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| `setTemporalClipAccess(true)` in Copernicus | Silent passthrough — never use it |
| `_NSGetExecutablePath()` for plugin path | Returns Houdini path — use `dladdr()` |
| `args.time` is frames | It's **seconds** — multiply by frame rate |
| ONNX CoreML EP for ViT models | Fragments into 100+ partitions — use native CoreML |
| `struct.pack("Bii")` in Python IPC | Adds padding — always use `"<Bii"` prefix |
| Global error buffer in multi-threading | Race conditions — use per-thread local buffers |
| OFX hot-reload in Houdini | Not supported — must restart Houdini |
| `std` as variable name | Shadows `std::` namespace — use `stdDev` |
| Building sims with temporal access | Process one frame at a time, external state for coherence |

---

## Session Log

| Date | Plugin | Test | Status |
|------|--------|------|--------|
| 2026-01 | FMModulation | Full parameter sweep in Copernicus | Verified |
| 2026-01 | PixelArt | Palette quantization modes | Verified |
| 2026-01 | AsciiArt | Character rendering, $F animation | Verified |
| 2026-02 | ColorTransfer | All 4 methods cross-verified vs Python | Verified |
| 2026-02 | Dither | Atkinson error diffusion thread safety | Verified |
| 2026-03 | DepthEstimation | ONNX + CoreML + Python, 3 models | Verified |
| 2026-03 | VideoDepth | Socket IPC, temporal coherence | Verified |
| 2026-03 | DeepBump | Normals + Height + Curvature, 6 bug fixes | Verified |
