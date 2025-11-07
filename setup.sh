# from the repo root
set -euo pipefail

# 0) Make a project + venv (Python 3.10)
uv init --bare
uv venv --python 3.10

# 1) Add Torch (CUDA 12.4 index) — recorded in pyproject + lock
uv pip install --index https://download.pytorch.org/whl/cu124 torch==2.4.0

# 2) Add FlashAttention wheel — recorded as a direct/path dep
FLASH_ATTENTION_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
uv pip install "$FLASH_ATTENTION_URL"   # (alternatively: download and `uv add ./flash_attn-...whl`)

# 3) Add your local packages **in editable mode**
uv pip install --editable AgentGym-RL
uv pip install --editable AgentGym/agentenv

# 4) Pin extra runtime deps
uv pip install transformers==4.51.3

