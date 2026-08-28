# Building amigo with pixi

[pixi](https://pixi.sh) creates a reproducible, project-local conda environment
from `pixi.toml` and `pixi.lock`. It handles the dependencies that are otherwise
awkward to install by hand — MPI, BLAS/LAPACK, METIS and MUMPS — on Linux,
macOS (Apple Silicon) and Windows.

Nothing is installed system-wide: the environment lives in `.pixi/envs/` inside
the repository, and `pixi.lock` pins exact package versions for every platform,
so a checkout builds the same way on every machine.

## Install pixi

```bash
# Linux / macOS
curl -fsSL https://pixi.sh/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm -useb https://pixi.sh/install.ps1 | iex"
```

## Quick start

From the repository root:

```bash
pixi install           # create the environment (first run downloads packages)
pixi run install-amigo # clone a2d if needed, then build and install amigo
pixi run cart-pole     # build and solve the cart-pole example
```

`pixi run install-amigo` performs an editable install (`pip install -e .`), so
Python changes take effect immediately; re-run it after changing C++ sources.

## Platform prerequisites

| Platform | What you need beforehand |
| --- | --- |
| `linux-64` | Nothing. The compiler (`cxx-compiler`), OpenMPI, OpenBLAS, METIS and MUMPS all come from conda-forge. |
| `osx-arm64` | Nothing. Clang, `llvm-openmp`, OpenMPI, OpenBLAS, METIS and MUMPS come from conda-forge. |
| `win-64` | **Visual Studio Build Tools with the C++ workload.** conda-forge cannot ship MSVC, so it must be installed system-wide. |

### Installing MSVC on Windows

amigo's Python extension has to be built with the same compiler and C runtime
as CPython itself, which on Windows means MSVC. conda-forge cannot redistribute
it, so it is the one dependency pixi cannot provide. You do **not** need the
full Visual Studio IDE — the free standalone Build Tools are enough.

Run this from PowerShell:

```powershell
winget install --id Microsoft.VisualStudio.2022.BuildTools -e --accept-package-agreements --accept-source-agreements --override "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

What to expect:

- The `--override` string **must stay on one line.** If your terminal wraps it
  across lines, `--add` arrives without its workload and the installer exits
  with code 87 (`Installer failed with exit code: 87`).
- `Microsoft.VisualStudio.Workload.VCTools` is the C++ build-tools workload;
  `--includeRecommended` pulls in the MSVC toolchain and the Windows SDK, both
  of which CMake needs. Without the SDK you get a compiler that cannot link.
- It downloads roughly 2 GB and takes several minutes with no progress output
  because of `--quiet`. It is not hung — `--wait` keeps winget blocked until
  the install finishes.
- Windows may prompt for elevation. A reboot is not usually required
  (`--norestart`), but restart the terminal so the new tools are picked up.

Verify it landed:

```powershell
winget list --id Microsoft.VisualStudio.2022.BuildTools
```

You do not need a "Developer Command Prompt" or any `vcvarsall.bat` setup —
CMake locates MSVC on its own through the Visual Studio generator, so
`pixi run install-amigo` works from an ordinary shell. A successful configure
prints something like:

```
-- The CXX compiler identification is MSVC 19.44.35228.0
-- Check for working CXX compiler: .../VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe
```

If you already have Visual Studio 2019 or 2022 Community/Professional with the
"Desktop development with C++" workload, that works too — nothing extra to
install.

### Intel macOS

`osx-64` is not in the platform list, because CI only tests Apple Silicon. To
add it, copy each `osx-arm64` block in `pixi.toml`, add `"osx-64"` to
`workspace.platforms`, and re-run `pixi lock`.

## Environments

`pixi.toml` defines three environments. Select one with `-e`; the default
environment needs no flag.

| Environment | Command | Contents |
| --- | --- | --- |
| `default` | `pixi run <task>` | Everything needed to build, install and run amigo: compiler toolchain, CMake, MPI, BLAS/LAPACK, METIS, MUMPS, and amigo's Python dependencies. |
| `dev` | `pixi run -e dev <task>` | `default` plus pytest, smt, black, pre-commit and OpenMDAO. Use this to run the test suite or the OpenMDAO examples. |
| `cuda` | `pixi run -e cuda <task>` | `default` plus the dev tools, the CUDA 12 toolkit and cuDSS. **linux-64 only**, and requires an NVIDIA driver supporting CUDA 12. |

Examples:

```bash
pixi run -e dev test                     # run the full test suite
pixi run -e cuda install-amigo-cuda      # build with CUDA + cuDSS
pixi shell -e dev                        # drop into an interactive shell
```

`pixi shell` activates the environment in your current terminal, which is
useful for running ad-hoc scripts or pointing an IDE at
`.pixi/envs/<env>/python`.

## Tasks

| Task | What it does |
| --- | --- |
| `a2d` | Clones [smdogroup/a2d](https://github.com/smdogroup/a2d) into `../a2d` if it is not already there. `CMakeLists.txt` expects the headers at that path. Every install task depends on this. |
| `install-amigo` | Editable install with METIS enabled, CUDA disabled, and OpenMP on (off on Windows, matching CI). |
| `install-amigo-cuda` | `cuda` environment only: editable install with CUDA, cuDSS and `CMAKE_CUDA_ARCHITECTURES=native`. |
| `test` | Runs `pytest tests/ -v`, installing amigo first. Use with `-e dev`. |
| `cart-pole` | Builds and solves `examples/trajectory/cart/cart_pole.py --build`. |
| `lint` / `format` | `black --check .` / `black .` |
| `clean` | Removes build directories and `__pycache__`. |

Because tasks declare their dependencies, `pixi run test` will install amigo
(and clone a2d) automatically if that has not happened yet.

## What pixi handles for you

- **MPI** — OpenMPI on Linux/macOS, MS-MPI on Windows, with `mpi4py` built
  against it. mpi4py's headers are needed at build time by `amigo/amigo.cpp`.
- **BLAS / LAPACK** — OpenBLAS on Linux/macOS; MKL on Windows, which
  `CMakeLists.txt` auto-detects at `<sys.prefix>/Library/lib/mkl_rt.lib`.
- **METIS** — `METIS_ROOT` is exported into the environment, so
  `cmake/Modules/FindMETIS.cmake` picks it up without any `-D` flags.
- **MUMPS** — `mumps-seq` from conda-forge, with `MUMPS_LIB_DIR` pointing at
  the environment's library directory. This replaces the manual
  coin-or/ThirdParty-Mumps build described in the main README. Not available
  on Windows (see below).
- **CMake and the compiler at run time** — `model.build_module()` compiles
  generated C++ while your script runs, so CMake, Ninja, pybind11 and a
  compiler are part of the runtime environment, not just the build.
- **a2d headers** — fetched by the `a2d` task.

## Platform notes

**Windows: no MUMPS.** conda-forge has no reliable MUMPS build for Windows, so
the `mumps` solver option is unavailable — the same limitation as the Windows CI
job, which skips `tests/functional`. If you build coin-or/ThirdParty-Mumps
under MSYS2 as described in the main README, amigo's loader finds it in
`~/mumps-coinor/bin` or in `%CONDA_PREFIX%\Library\bin`.

**Windows: OpenMP is off.** `install-amigo` passes
`-DAMIGO_ENABLE_OPENMP=OFF` on win-64 to match CI. Remove that flag from the
`[target.win-64.tasks.install-amigo]` command if you want to try it.

**Windows: `scripts/pixi-activate.bat`.** conda-forge's `msmpi` package sets
`MSMPI_INC` and friends from `%PREFIX%`, a variable that only exists during a
conda *build*. Under pixi it expands to nothing, leaving
`MSMPI_INC=\Library\include`, and CMake's `FindMPI` then constructs a broken
`MPI::MPI_CXX` target that makes every `try_compile` fail. The activation
script re-derives those paths from `%CONDA_PREFIX%`. It has to be a script
rather than an `activation.env` entry, because pixi does not expand
`$CONDA_PREFIX` inside `activation.env` values on Windows.

**Paths are passed through the environment, not the command line.** Windows
paths contain backslashes, and CMake treats those as escape characters in
`-D` arguments, so the install tasks deliberately pass no paths.

## Maintaining the manifest

```bash
pixi add scipy                # add a conda dependency
pixi add --pypi some-package  # add a PyPI-only dependency
pixi update                   # refresh pixi.lock
pixi list                     # show what is installed
```

Commit `pixi.toml` **and** `pixi.lock` — the lock file is what makes builds
reproducible. `.pixi/` is generated and should stay out of git.

## Troubleshooting

**`CMAKE_CXX_COMPILER not set` / `'nmake' '-?' failed`** (Windows) — the VS
Build Tools are missing or lack the C++ workload. See
[Installing MSVC on Windows](#installing-msvc-on-windows), then re-run
`pixi run install-amigo`.

**`Imported target "MPI::MPI_CXX" includes non-existent path "/Library/include"`**
(Windows) — the activation script did not run. Confirm that
`scripts/pixi-activate.bat` exists and that `pixi run python -c "import os;
print(os.environ['MSMPI_INC'])"` prints an absolute path.

**`METIS not found: disabling METIS support`** — this is an amigo issue, not a
pixi one. `CMakeLists.txt` pre-creates the `METIS_INCLUDE_DIR` and
`METIS_LIBRARY` cache entries as empty strings, and `find_path`/`find_library`
skip searching when their result variable is already set, so `FindMETIS` can
never succeed on its own. amigo builds and runs correctly without METIS; the
effect is on ordering performance for large problems.

**Stale build after changing C++ sources** — run `pixi run clean` followed by
`pixi run install-amigo`.
