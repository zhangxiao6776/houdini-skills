# AI Inference Pipeline Skill — Houdini OFX

You are an expert at integrating AI/ML models into Houdini via OFX plugins. You understand four distinct inference backends (ONNX Runtime C++, Apple CoreML native, Python subprocess, persistent Python server), binary IPC protocols, model management, caching strategies, and cross-platform deployment.

## Core Principle

**Every inference plugin supports at least two backends.** ONNX Runtime C++ for production speed, Python subprocess as a universal fallback. On macOS, native CoreML bypasses ONNX Runtime's fragmented execution providers for massive GPU speedups. For temporal/video models, a persistent Python server maintains cross-frame state.

---

## 1. Backend Architecture Overview

```
                  ┌─────────────────────────────────────┐
                  │        OFX Plugin (C++)              │
                  │  ┌──────────────────────────────┐   │
                  │  │ Backend Selection (Auto)      │   │
                  │  └──┬───────┬──────┬────────┬───┘   │
                  │     │       │      │        │        │
                  │  ┌──▼──┐ ┌─▼───┐ ┌▼─────┐ ┌▼─────┐ │
                  │  │ONNX │ │Core │ │Python│ │Socket│ │
                  │  │ C++ │ │ ML  │ │Subpr.│ │Server│ │
                  │  └─────┘ └─────┘ └──────┘ └──────┘ │
                  └─────────────────────────────────────┘
```

### Backend Selection Logic

```cpp
enum BackendEnum { eBackendAuto = 0, eBackendPython = 1, eBackendONNX = 2 };

// Auto mode: Try best available, fall back gracefully
bool useONNX = false;
#ifdef DEPTH_HAS_ONNX
if(backendChoice == eBackendAuto || backendChoice == eBackendONNX) useONNX = true;
#endif

if(useONNX) {
    success = runONNXInference(...);
    if(!success && backendChoice == eBackendAuto)
        success = runPythonInference(...);  // fallback
} else {
    success = runPythonInference(...);
}
```

---

## 2. ONNX Runtime C++ Backend

### 2.1 Three-Level Caching

```cpp
// Level 1: Ort::Env singleton (process-wide, thread-safe)
static Ort::Env& getOrtEnv() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MyPlugin");
    return env;
}

// Level 2: Ort::Session cached by model path (expensive ~1s to create)
struct OnnxSessionCache {
    std::mutex mutex;
    std::unique_ptr<Ort::Session> session;
    std::string modelPath;
};
static OnnxSessionCache& getSessionCache() {
    static OnnxSessionCache cache;
    return cache;
}

// Level 3: Inference result cached per-frame
// Keyed on (args.time, model_selection, backend, sourceUID)
```

### 2.2 Session Configuration

```cpp
Ort::SessionOptions opts;
opts.SetIntraOpNumThreads(0);  // 0 = auto-detect (don't hardcode)
opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
```

### 2.3 Tensor Creation and Inference

```cpp
// Prepare input
std::vector<float> inputData(1 * 3 * inputSize * inputSize);
// Fill with ImageNet-normalized, bilinear-resized image...

std::vector<int64_t> inputShape = {1, 3, inputSize, inputSize};
auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
    memInfo, inputData.data(), inputData.size(),
    inputShape.data(), inputShape.size());

// Run inference
const char* inputNames[]  = {"pixel_values"};
const char* outputNames[] = {"predicted_depth"};
auto results = session->Run(Ort::RunOptions{nullptr},
    inputNames, &inputTensor, 1, outputNames, 1);

// Extract output
auto& outTensor = results[0];
const float* outData = outTensor.GetTensorData<float>();
auto outShape = outTensor.GetTensorTypeAndShapeInfo().GetShape();
```

### 2.4 ImageNet Normalization

Most vision models expect ImageNet normalization:

```cpp
static const float kImageNetMean[3] = {0.485f, 0.456f, 0.406f};  // RGB
static const float kImageNetStd[3]  = {0.229f, 0.224f, 0.225f};  // RGB

// Normalize: pixel = (pixel - mean) / std
// NOTE: use 'stdDev' not 'std' to avoid shadowing std:: namespace
for(int c = 0; c < 3; c++) {
    float val = rgbPixel[c];
    inputData[c * H * W + y * W + x] = (val - kImageNetMean[c]) / kImageNetStd[c];
}
```

### 2.5 Bilinear Resize (Half-Pixel Aligned)

Matching PyTorch's `align_corners=False`:

```cpp
float sy = (static_cast<float>(dy) + 0.5f) * srcH / dstH - 0.5f;
float sx = (static_cast<float>(dx) + 0.5f) * srcW / dstW - 0.5f;
// Then bilinear interpolation with clamped integer coords
```

### 2.6 Tiled Inference (DeepBump Pattern)

For models that process fixed-size tiles (e.g., 256x256 MobileNetV2):

```
Image → Split into overlapping tiles → Inference per tile → Pyramidal blend → Reassemble
```

Key parameters:
- **Tile size**: Model's native input (256x256 for DeepBump)
- **Overlap**: Configurable (Small=43px, Medium=64px, Large=128px)
- **Padding**: Wrap padding for seamless textures
- **Blending**: Pyramidal weight (linear ramp from edge to center) in overlap region

---

## 3. Apple CoreML Native Backend

### 3.1 Why Not ONNX CoreML EP?

ONNX Runtime's CoreML Execution Provider fragments Vision Transformer models into 100+ partitions. Each partition boundary requires a CPU↔GPU copy. Result: **slower than pure CPU**.

```
ONNX Runtime CPU:        ~200ms  (baseline)
ONNX CoreML EP:          ~370ms  (SLOWER — 107 partition copies)
CoreML native (GPU/ANE): ~26ms   (7.7x faster than CPU)
```

### 3.2 Architecture

```
depthestimation.cpp → coreml_inference.h (C++ API)
                    → coreml_inference.mm (Obj-C++ implementation)
```

The `.mm` file uses ARC (`-fobjc-arc`) and CoreML framework API directly:

```objc
// Model loading (cached in static array with mutex)
NSURL* modelURL = [NSURL fileURLWithPath:...];
MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
config.computeUnits = MLComputeUnitsAll;  // GPU + ANE
MLModel* model = [MLModel modelOfContentsOfURL:modelURL
                                 configuration:config error:&error];

// Inference with MLMultiArray input
MLMultiArray* inputArray = [[MLMultiArray alloc]
    initWithShape:@[@1, @3, @(inputSize), @(inputSize)]
    dataType:MLMultiArrayDataTypeFloat32 error:&error];

// Fill input array with ImageNet-normalized data
id<MLFeatureProvider> prediction = [model predictionFromFeatures:input error:&error];
MLMultiArray* output = [prediction featureValueForName:@"predicted_depth"].multiArrayValue;
```

### 3.3 Model Formats

| Format | Extension | Usage |
|--------|-----------|-------|
| ONNX | `.onnx` | Cross-platform, ONNX Runtime |
| CoreML compiled | `.mlmodelc/` | macOS GPU/ANE, pre-compiled |
| CoreML package | `.mlpackage/` | Source format, needs compilation |

**Pre-compile for deployment:**
```python
import coremltools as ct
model = ct.convert(torch_model, ...)
model.save("model.mlpackage")
# Then: xcrun coremlc compile model.mlpackage .
# Produces model.mlmodelc/
```

### 3.4 Cross-Platform Guards

```cpp
// In header:
#ifdef __APPLE__
bool coremlModelAvailable(const std::string& onnxModelPath);
bool runCoreMLInference(const float* rgb, int w, int h, ...);
#endif

// In plugin:
#ifdef __APPLE__
if(deviceChoice != eDeviceCPU && coremlModelAvailable(onnxPath)) {
    success = runCoreMLInference(rgb, w, h, depthOut, ...);
}
#endif
if(!success) {
    success = runONNXInference(...);  // CPU fallback
}
```

### 3.5 Model Cache (ARC-Managed)

```objc
// Static model cache with mutex protection
static const int kMaxModels = 4;
static __strong MLModel* sModelCache[kMaxModels] = {};
static std::string sModelPaths[kMaxModels] = {};
static std::mutex sModelMutex;

// ARC automatically retains/releases MLModel pointers
// No manual memory management needed
```

---

## 4. Python Subprocess Backend

### 4.1 Binary IPC Protocol

**Text/JSON is too slow for image data.** Use binary over stdin/stdout:

```
C++ → Python stdin:
  [int32 width] [int32 height] [float32 RGB pixels, H*W*3]

Python → C++ stdout:
  [int32 width] [int32 height] [float32 depth pixels, H*W]
```

### 4.2 C++ Side (Writing)

```cpp
float* buf = new float[srcW * srcH * 3];
// Extract RGB from OFX image into buf...

FILE* pipe = popen(cmd.c_str(), "r+");
fwrite(&srcW, sizeof(int32_t), 1, pipe);
fwrite(&srcH, sizeof(int32_t), 1, pipe);
fwrite(buf, sizeof(float), srcW * srcH * 3, pipe);
fflush(pipe);

// Read result
int32_t outW, outH;
fread(&outW, sizeof(int32_t), 1, pipe);
fread(&outH, sizeof(int32_t), 1, pipe);
// Validate: reject > 65536 or <= 0 to prevent OOM
std::vector<float> depth(outW * outH);
fread(depth.data(), sizeof(float), outW * outH, pipe);
```

### 4.3 Python Side (Reading)

```python
import struct
import sys
import numpy as np

# CRITICAL: Use '<' prefix to avoid alignment padding
data = sys.stdin.buffer.read(8)
w, h = struct.unpack('<ii', data)

# Read image
rgb = np.frombuffer(sys.stdin.buffer.read(w * h * 3 * 4), dtype=np.float32)
rgb = rgb.reshape(h, w, 3)

# Run inference...
depth = model.infer(rgb)

# Write result
sys.stdout.buffer.write(struct.pack('<ii', w, h))
sys.stdout.buffer.write(depth.astype(np.float32).tobytes())
sys.stdout.buffer.flush()
```

### 4.4 The `struct.pack` Alignment Trap

```python
struct.pack("Bii", 0, 100, 100)    # 12 bytes! (1 + 3 padding + 4 + 4)
struct.pack("<Bii", 0, 100, 100)    # 9 bytes  (1 + 4 + 4, no padding)
```

**Rule:** Any format string mixing different-sized types MUST use `<` or `>` prefix.

### 4.5 Python Virtual Environment

```sh
python3 -m venv ~/depth_venv
source ~/depth_venv/bin/activate
pip install torch torchvision transformers pillow numpy
```

The Python script auto-selects device:
```python
if torch.backends.mps.is_available():
    device = torch.device("mps")     # Apple Silicon GPU
elif torch.cuda.is_available():
    device = torch.device("cuda")    # NVIDIA GPU
else:
    device = torch.device("cpu")
```

---

## 5. Persistent Python Server Backend (VideoDepth)

### 5.1 Architecture

```
C++ Plugin ←→ Unix Domain Socket ←→ Python Server (persistent)
                                      ├── VDA model (loaded once)
                                      ├── Temporal KV-cache
                                      └── Auto-shutdown (5min idle)
```

### 5.2 Binary Command Protocol

```
Client → Server:
  [uint8 command] [payload...]

Commands:
  CMD_INFER    (0): [int32 w] [int32 h] [float32 RGB, H*W*3]
  CMD_RESET    (1): (no payload) — clear temporal cache
  CMD_CONFIG   (2): [model_name] [device] — reconfigure server
  CMD_SHUTDOWN (3): (no payload) — graceful shutdown
  CMD_PING     (4): (no payload) — health check

Server → Client:
  [uint8 status] [payload...]
  status 0 = OK, then: [int32 w] [int32 h] [float32 depth, H*W]
```

### 5.3 Server Lifecycle

```cpp
// Auto-start via fork() + setsid()
pid_t pid = fork();
if(pid == 0) {
    setsid();  // Become session leader (survives plugin unload)
    // Redirect stdin to /dev/null, stdout to stderr
    execlp(pythonPath, pythonPath, scriptPath,
           "--socket", socketPath,
           "--model", model,
           "--device", device, nullptr);
}

// Wait for server readiness (poll socket up to 30s)
for(int i = 0; i < 300; i++) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if(isServerRunning(socketPath)) break;
}
```

### 5.4 Socket IPC Helpers

```cpp
static bool sendExact(int fd, const void* data, size_t len) {
    const char* ptr = static_cast<const char*>(data);
    size_t sent = 0;
    while(sent < len) {
        ssize_t n = write(fd, ptr + sent, len - sent);
        if(n <= 0) return false;
        sent += static_cast<size_t>(n);
    }
    return true;
}

static bool recvExact(int fd, void* data, size_t len) {
    char* ptr = static_cast<char*>(data);
    size_t received = 0;
    while(received < len) {
        ssize_t n = read(fd, ptr + received, len - received);
        if(n <= 0) return false;
        received += static_cast<size_t>(n);
    }
    return true;
}
```

### 5.5 Server Health Check

```cpp
static bool isServerRunning(const char* socketPath) {
    int fd = connectToSocket(socketPath);
    if(fd < 0) return false;

    uint8_t cmd = kCmdPing;
    if(!sendExact(fd, &cmd, 1)) { close(fd); return false; }

    uint8_t status;
    if(!recvExact(fd, &status, 1)) { close(fd); return false; }

    close(fd);
    return (status == kStatusOK);
}
```

### 5.6 Temporal KV-Cache (VDA)

Video Depth Anything maintains temporal coherence through a sliding window KV-cache:

- **First frame**: Duplicated to `INFER_LEN=32` slots for cache initialization
- **Subsequent frames**: Cache built from first 2 + last 29 entries, evicts after `gap=41`
- **Reset**: `CMD_RESET` clears both server KV-cache and plugin-side frame cache

```python
# Server-side cache management
if frame_count == 0:
    # First frame: duplicate to fill all 32 cache slots
    cache = init_cache_from_single_frame(frame)
else:
    # Sliding window: keep first 2, last 29, evict old middle
    cache = update_cache(cache, frame, gap=41)
```

### 5.7 Windows Fallback

No Unix domain sockets on Windows. Fall back to subprocess-per-frame:

```cpp
#ifdef _WIN32
// Stateless subprocess (no persistent server)
success = runPythonSubprocess(rgb, w, h, depthOut);
#else
// Persistent server with Unix domain socket
success = runServerInference(rgb, w, h, depthOut);
#endif
```

---

## 6. Inference Result Caching

### 6.1 Cache Key Design

Only re-run inference when inputs that affect the model output change:

| Cache Key Field | Detects Change In |
|----------------|-------------------|
| `cachedTime_` | Frame / timeline position |
| `cachedModel_` | Model selection parameter |
| `cachedBackend_` | Backend switch (ONNX/Python/Auto) |
| `cachedSourceUID_` | Input image changes (disconnect, upstream) |
| `cachedDepth_.empty()` | First render after load |

### 6.2 Implementation

```cpp
// In plugin class:
OfxTime     cachedTime_    = -1e30;
int         cachedModel_   = -1;
int         cachedBackend_ = -1;
std::string cachedSourceUID_;
std::vector<float> cachedDepth_;
std::mutex  cacheMutex_;

// In setupAndProcess():
{
    std::lock_guard<std::mutex> lock(cacheMutex_);
    std::string srcUID = src.get() ? src->getUniqueIdentifier() : "";

    bool needInference = (cachedTime_ != args.time
                       || cachedModel_ != modelVal
                       || cachedBackend_ != backendVal
                       || srcUID.empty()
                       || cachedSourceUID_ != srcUID
                       || cachedDepth_.empty());

    if(needInference && src.get()) {
        // Run inference...
        cachedTime_ = args.time;
        cachedModel_ = modelVal;
        cachedBackend_ = backendVal;
        cachedSourceUID_ = srcUID;
    }
}
```

### 6.3 Key Rules

- `srcUID.empty()` → always re-run (host didn't set UID, be conservative)
- Output-only params (near/far clip, opacity, output mode) do NOT invalidate cache
- Different backends may produce slightly different float results → invalidate on switch
- Use `kOfxImagePropUniqueIdentifier` (host assigns new UID when source changes)

---

## 7. Model Management

### 7.1 Multi-Model Support

```cpp
struct ModelConfig {
    int         inputSize;
    const char* filename;
    float       normMean[3];
    float       normStd[3];
    bool        isMetricDepth;
};

static const ModelConfig kModelConfigs[] = {
    // DA V2 Small (bundled, 99MB)
    { 518, "depth_anything_v2_small.onnx",
      {0.485f,0.456f,0.406f}, {0.229f,0.224f,0.225f}, false },
    // DA V2 Large (fp16, 669MB)
    { 518, "depth_anything_v2_large_fp16.onnx",
      {0.485f,0.456f,0.406f}, {0.229f,0.224f,0.225f}, false },
    // Depth Pro (fp16, 1.9GB)
    { 1536, "depth_pro_fp16.onnx",
      {0.5f,0.5f,0.5f}, {0.5f,0.5f,0.5f}, true },
};
```

### 7.2 ONNX Model Export

```python
import torch

model = load_pretrained_model()
dummy_input = torch.randn(1, 3, 518, 518)

torch.onnx.export(model, dummy_input, "model.onnx",
                  input_names=["pixel_values"],
                  output_names=["predicted_depth"],
                  dynamic_axes={"pixel_values": {0: "batch"}},
                  opset_version=17)
```

### 7.3 Bundle Layout

```
Plugin.ofx.bundle/
  Contents/
    MacOS/Plugin.ofx
    Frameworks/libonnxruntime.dylib
    Resources/
      model_small.onnx          # Bundled (small enough)
      model_small.mlmodelc/     # Pre-compiled CoreML
      model_large_fp16.onnx     # Downloaded on demand
      infer_script.py           # Python fallback
```

### 7.4 Cross-Verification

Compare ONNX C++ vs PyTorch reference:

```sh
python3 tests/test_depth_crossverify.py
```

Expected results (DA V2 Small):
- Correlation: ~0.997
- Max absolute difference: ~0.043
- ONNX is ~7x faster than PyTorch on CPU

---

## 8. Output Modes

Most depth/inference plugins support multiple output visualizations:

```cpp
enum OutputModeEnum {
    eOutputNormalized = 0,  // Linear 0-1 (P2/P98 percentile)
    eOutputInverse    = 1,  // 1/depth (emphasizes foreground)
    eOutputRaw        = 2,  // Auto-normalized raw model output
    eOutputDisparity  = 3   // pow(1-norm, 2) (perceptual depth)
};
```

**Raw mode gotcha:** Model outputs are NOT [0,1]. DA V2 outputs ~0-500. Always auto-normalize with P2/P98 percentiles to avoid binary clipping.

**Disparity mode gotcha:** Simple `1/depth` produces washed-out or over-compressed results. Use `pow(1 - normalized, 2)` power curve instead.

---

## 9. Adding a New Inference Plugin (Template)

1. **Prototype Python script** — `scripts/<model>_infer.py` with binary IPC
2. **Create plugin skeleton** — Copy DepthEstimation structure
3. **Implement Python subprocess** — `runPythonInference()` with stdin/stdout binary
4. **Export ONNX model** — `torch.onnx.export()` with dynamic batch
5. **Cross-verify** — ONNX output vs PyTorch reference (correlation > 0.99)
6. **Add ONNX C++ backend** — `runONNXInference()` with session caching
7. **Add CoreML backend** (macOS) — Pre-compile `.mlmodelc`, ARC wrapper
8. **CMake integration** — Conditional `HAS_ONNX`, install rules for model + runtime
9. **Bundle** — Model in Resources/, runtime lib in Frameworks/
10. **Deploy** — Copy to `~/OFX_Plugins/`, restart Houdini

---

## 10. Performance Reference

| Model | Backend | Time | Speedup |
|-------|---------|------|---------|
| DA V2 Small (518x518) | ONNX CPU | ~200ms | baseline |
| DA V2 Small (518x518) | ONNX CoreML EP | ~370ms | 0.5x (SLOWER) |
| DA V2 Small (518x518) | CoreML native GPU | ~26ms | 7.7x |
| VDA Small (518x518) | Python MPS server | ~50ms/frame | — |
| VDA Small (518x518) | Python subprocess | ~3-4s/frame | — |
| DeepBump normals (256x256 tiles) | ONNX CPU | ~100ms/tile | — |

---

## 11. Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| ONNX CoreML EP for ViT | 107 partitions, slower than CPU — use native CoreML |
| `struct.pack` without `<` | Adds alignment padding — always use `"<Bii"` |
| `_NSGetExecutablePath()` | Returns host (Houdini) path — use `dladdr()` |
| `args.time` treated as frames | It's seconds — multiply by `srcClip_->getFrameRate()` |
| Raw depth output clipped binary | Auto-normalize with P2/P98 percentiles |
| `1/depth` for disparity | Washed-out — use `pow(1-norm, 2)` curve |
| Hardcoded thread count | Use `SetIntraOpNumThreads(0)` for auto-detect |
| Missing dimension validation | Reject width/height > 65536 or <= 0 |
| `std` as variable name | Shadows `std::` namespace — use `stdDev` |
| ARC not enabled for .mm files | Causes dangling MLModel* pointers |

---

## Session Log

| Date | Backend | Test | Status |
|------|---------|------|--------|
| 2026-02 | ONNX C++ | DA V2 Small inference + caching | Verified |
| 2026-02 | Python subprocess | Binary IPC protocol | Verified |
| 2026-03 | CoreML native | DA V2 Small/Large + Depth Pro GPU/ANE | Verified |
| 2026-03 | Persistent server | VDA socket IPC, temporal KV-cache | Verified |
| 2026-03 | Tiled inference | DeepBump 256x256 MobileNetV2 | Verified |
| 2026-03 | Multi-model | 3 depth models with ModelConfig lookup | Verified |
| 2026-03 | Cross-verification | ONNX vs PyTorch correlation ~0.997 | Verified |
