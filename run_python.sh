#!/bin/sh
# Reuse scikit-learn's bundled OpenMP runtime on macOS; no global installation.
set -eu
cd "$(dirname "$0")"
project_python="$PWD/.venv/bin/python"
if [ "$(uname -s)" = "Darwin" ]; then
  project_omp_dir="$("$project_python" -c 'import pathlib, sklearn; print(pathlib.Path(sklearn.__file__).parent / ".dylibs")')"
  if [ -f "$project_omp_dir/libomp.dylib" ]; then
    export DYLD_LIBRARY_PATH="$project_omp_dir${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
  fi
fi
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export IPYTHONDIR="$PWD/.cache/ipython"
export JUPYTER_RUNTIME_DIR="$PWD/.cache/jupyter-runtime"
export PYTHONUNBUFFERED=1
exec "$project_python" "$@"
