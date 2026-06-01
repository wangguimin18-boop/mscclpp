# MSCCL++ Agent Instructions

## Build

CMake >= 3.25, C++17 / CUDA 17 / HIP 17. Version is read from `VERSION` file (MAJOR.MINOR.PATCH format).

```bash
# NVIDIA (auto-detect GPU)
cmake -B build -DMSCCLPP_BUILD_TESTS=ON
cmake --build build -j

# NVIDIA (no GPU on machine, bypass check)
cmake -B build -DMSCCLPP_BYPASS_GPU_CHECK=ON -DMSCCLPP_USE_CUDA=ON -DMSCCLPP_BUILD_TESTS=ON

# AMD ROCm (no GPU, bypass check)
cmake -B build -DMSCCLPP_BYPASS_GPU_CHECK=ON -DMSCCLPP_USE_ROCM=ON -DMSCCLPP_BUILD_TESTS=ON
```

**IBVerbs is required by default.** Add `-DMSCCLPP_USE_IB=OFF` if libibverbs is unavailable.

Build outputs: `build/lib/` (libraries), `build/bin/` (executables).

## Lint / Format

```bash
# Check C++ formatting (dry run, CI mode)
bash tools/lint.sh cpp dry

# Apply C++ formatting
bash tools/lint.sh cpp

# Check Python formatting
bash tools/lint.sh py dry

# Apply Python formatting
bash tools/lint.sh py
```

- C++: clang-format, **Google style with ColumnLimit 120** (`.clang-format`)
- Python: **black with line-length 120** (`pyproject.toml`)

## Tests

### C++ tests

Custom GTest-like framework (`test/framework.hpp`), **not** Google Test. Key macros: `TEST()`, `EXPECT_*`, `ASSERT_*`, `PERF_TEST()`, `CUDA_CHECK()`.

- **Single-process**: `./build/bin/unit_tests [--filter=TestSuiteName] [--exclude-perf-tests]`
- **Multi-process**: `mpirun -np 2 ./build/bin/mp_unit_tests [--filter=TestSuiteName]`

Adding a new test requires adding the source file to the appropriate `CMakeLists.txt` (`test/unit/` or `test/mp_unit/`). No separate registration step — `TEST()` auto-registers.

### Python tests

Require MPI and GPU. `conftest.py` initializes MPI and assigns GPUs.

```bash
pip install -e ".[test]"
# Run with MPI:
mpirun -np 2 pytest python/test/
```

## Architecture

- `src/core/` — Core library: `libmscclpp` (shared) + `libmscclpp_static` (static). Object library `mscclpp_obj` is shared between both.
- `src/ext/collectives/` — Collective algorithm implementations (controlled by `MSCCLPP_BUILD_EXT_COLLECTIVES`)
- `src/ext/nccl/` — NCCL-compatible interfaces (controlled by `MSCCLPP_BUILD_EXT_NCCL`)
- `include/mscclpp/` — Public C++/CUDA headers. `version.hpp` is generated from `version.hpp.in` + `VERSION` file.
- `python/csrc/` — Nanobind (v1.9.2) Python bindings, produces `_mscclpp` module
- `python/mscclpp/` — Python package with `core/`, `ext/`, `language/`, `utils.py`

## ROCm Quirks

When building for ROCm:
- `.cu` files are compiled as CXX (not CUDA) — `set_source_files_properties(LANGUAGE CXX)`
- `__HIP_PLATFORM_AMD__` is defined automatically
- `CXX=/opt/rocm/bin/hipcc` must be set
- Default GPU archs: `gfx90a;gfx941;gfx942`

## Python Wheel Build

```bash
# Build wheel (pip install with cmake args)
CMAKE_ARGS="-DMSCCLPP_BYPASS_GPU_CHECK=ON -DMSCCLPP_USE_CUDA=ON" pip install .
```

Uses scikit-build-core + setuptools-scm. Version comes from git via setuptools-scm (`version_scheme: no-guess-dev`).

## Key CMake Options

| Option | Default | Notes |
|---|---|---|
| `MSCCLPP_BUILD_TESTS` | OFF | Must be ON for testing |
| `MSCCLPP_BUILD_PYTHON_BINDINGS` | ON | |
| `MSCCLPP_USE_IB` | ON | Requires libibverbs-dev |
| `MSCCLPP_USE_GDRCOPY` | ON | Auto-disabled on ROCm |
| `MSCCLPP_BYPASS_GPU_CHECK` | OFF | Set ON + USE_CUDA/USE_ROCM on machines without GPUs |
| `MSCCLPP_GPU_ARCHS` | auto | Comma/space/semicolon-delimited list; "native" for detected GPU |