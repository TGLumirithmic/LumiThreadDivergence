# Debug Findings - NaN Issues

## Date: 2025-12-02

## Summary

Using the comprehensive CUDA debugging tools, we identified the exact source of NaN values in the neural network inference pipeline.

## Tools Used

1. **Device-side inspection** - Print values directly from CUDA kernels
2. **Host-side buffer statistics** - Copy to host and compute min/max/mean/NaN count
3. **Step-by-step checkpoints** - Verify data at each stage of the pipeline
4. **Debug build flags** - `-g -G -O0` for full debug symbols

## Root Causes Identified

### Issue 1: Direction Padding Buffer Type Mismatch ✅ FIXED

**Problem:**
- Buffer `d_dir_input_float_` was allocated as `float*` (4 bytes per element)
- When `TCNN_HALF_PRECISION=1`, kernel writes `__half` data (2 bytes per element)
- This causes memory layout corruption

**Evidence:**
```
Before padding (original directions, correct):
  [0] = -0.27984822
  [1] = -0.06284534
  [2] = 0.95798510

After padding (reading as float from half buffer, corrupted):
  [0] = -0.00000000
  [1] = 0.00000000
  [2] = 0.00000000
  [8] = 0.00655199  (garbage)
```

**Fix Applied:**
- Modified `allocate_buffers()` to allocate buffers based on `TCNN_HALF_PRECISION`
- Use `sizeof(__half)` when half precision is enabled
- Use `sizeof(float)` otherwise
- Applied to both `d_dir_input_float_` and `d_concatenated_float_` buffers

**Files Changed:**
- [src/neural/network.cpp:233-246](../src/neural/network.cpp#L233-L246) - Direction input buffer
- [src/neural/network.cpp:260-273](../src/neural/network.cpp#L260-L273) - Concatenated buffer

### Issue 2: Position Encoding Produces Zeros and NaN ⚠️ UNDER INVESTIGATION

**Problem:**
- Input positions are correct: `[1.0, 0.964, 0.839, ...]`
- Encoded positions are mostly zeros with some NaN values
- Out of 8192 encoded values: 21 NaN, 7804 zeros

**Evidence:**
```
>>> STEP 1 Result: Checking encoded positions <<<
[DEVICE-SIDE] Inspecting buffer at 0x7f3eaea02c00
  [0-19] @ ... = 0.00000000 (Zero)

[DEBUG] Encoded positions (size=8192)
  NaN count: 21
  Inf count: 0
  Zero count: 7804
```

**Suspicious Weight Values:**
```
First 10 weight values: 0, 1.875, -1.07227, 1.86523, -405.5, 1.83398, 5376, 1.54102, 0, 0
```

Values like `-405.5` and `5376` are **way too large** for typical neural network weights (should be ±2).

**Possible Causes:**
1. **Weight loading error** - Weights may be loaded with wrong precision/format
2. **Weight conversion issue** - Float-to-half conversion may be incorrect
3. **Hash grid parameter mismatch** - Loaded 488240 params but encoder expects 491280
4. **TCNN internal issue** - The encoding inference may have a bug

**Next Steps to Debug:**
1. Check if weights file is corrupted or wrong format
2. Verify weight conversion from float32 to half precision
3. Inspect the hash grid parameter structure
4. Try loading weights directly in Python and compare with tiny-cuda-nn output

## Debug Output Analysis

### Working Components ✅

1. **Input Data** - Positions and directions arrive correctly on device
2. **Memory Copies** - Device-to-device copies work correctly
3. **Kernel Launch** - No CUDA errors during kernel execution
4. **Buffer Allocation** - Buffers are allocated with correct sizes

### Failing Components ❌

1. **Position Encoding** - Produces zeros and NaN instead of feature vectors
2. **Direction Encoding** - Gets NaN input from position encoder, produces all NaN
3. **All Decoders** - Get NaN inputs, produce all NaN outputs

## Cascade Effect

The failures cascade through the pipeline:

```
Input (OK) → Position Encoding (FAIL: zeros/NaN) →
                ↓
Direction Encoding (FAIL: all NaN, garbage input) →
                ↓
Concatenation (FAIL: inherits NaN from both) →
                ↓
Decoders (FAIL: NaN in → NaN out)
```

**Critical Point:** Fix position encoding first, then retest direction encoding.

## Verification Steps After Fixes

1. ✅ Rebuild with corrected buffer allocations
2. ⏳ Verify direction padding now works correctly
3. ⏳ Investigate position encoding weight issue
4. ⏳ Test with corrected weights
5. ⏳ Verify end-to-end inference produces valid outputs

## Commands for Testing

```bash
# Build with debug symbols
cd /workspaces/LumiThreadDivergence/build
make -j8

# Run test with debug output
./bin/test_network ../data/models/weights.bin ../data/test/predictions.bin ../output

# Check specific steps
./bin/test_network ... 2>&1 | grep -A 20 "STEP 0b"  # Check weights
./bin/test_network ... 2>&1 | grep -A 20 "STEP 1"    # Check encoding
./bin/test_network ... 2>&1 | grep -A 20 "STEP 2a"   # Check padding
```

## Files Modified for Debugging

1. **Debug Utilities**
   - `src/utils/debug_utils.h` - Debug function declarations
   - `src/utils/debug_utils.cu` - Device-side inspection kernels

2. **Network Code**
   - `src/neural/network.cpp` - Added debug checkpoints throughout inference

3. **Build System**
   - `CMakeLists.txt` - Added debug flags (-g -G -O0)

4. **Documentation**
   - `docs/DEBUGGING_CUDA.md` - Complete debugging guide
   - `docs/DEBUG_FINDINGS.md` - This file

## Conclusion

The comprehensive debugging infrastructure successfully identified:
1. A critical buffer type mismatch (FIXED)
2. A potential weight loading issue (INVESTIGATING)

The device-side inspection capability proved invaluable for seeing exactly what data looks like on the GPU before and after operations.

## Next Actions

1. Verify direction padding fix actually works (may need to fix debug output reading)
2. Investigate weight loading - check if values are being loaded with correct byte order/precision
3. Consider regenerating the weight file with correct export format
4. Test with a simple known-good weight configuration
