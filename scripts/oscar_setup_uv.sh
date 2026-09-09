#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BOOTSTRAP_DIR="${PROJECT_ROOT}/.uv-bootstrap"
UV_VERSION="0.12.11"

cd "${PROJECT_ROOT}"
module load python/3.11.11-5e66
PYTHON_BIN="$(command -v python)"

if [[ ! -x "${BOOTSTRAP_DIR}/bin/uv" ]] || \
   [[ "$("${BOOTSTRAP_DIR}/bin/uv" --version 2>/dev/null || true)" != "uv ${UV_VERSION}"* ]]; then
  "${PYTHON_BIN}" -m venv --clear "${BOOTSTRAP_DIR}"
  "${BOOTSTRAP_DIR}/bin/python" -m pip install --disable-pip-version-check "uv==${UV_VERSION}"
fi

UV_BIN="${BOOTSTRAP_DIR}/bin/uv"
"${UV_BIN}" venv --clear --python "${PYTHON_BIN}" .venv
"${UV_BIN}" sync --frozen --extra test --extra analysis
"${UV_BIN}" run --frozen --no-sync python -m pytest
"${UV_BIN}" run --frozen --no-sync python -c \
  'import torch,torchvision,wandb; print(f"python_runtime={torch.__config__.show().splitlines()[0]}"); print(f"torch={torch.__version__}"); print(f"torchvision={torchvision.__version__}"); print(f"torch_cuda={torch.version.cuda}"); print(f"wandb={wandb.__version__}")'

echo "Oscar UV environment is ready at ${PROJECT_ROOT}/.venv"
