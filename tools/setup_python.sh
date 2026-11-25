#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
#  Studio Toolbox — one-time Python setup, on a Mac with nothing on it.
#
#  Everything the app needs, with no Homebrew, no Xcode, no password:
#    1. Miniforge (conda) into ~/.ai_toolbox/miniforge3, unless a conda
#       already exists on this Mac (then that one is used)
#    2. a conda env "studio-toolbox" with Python 3.13 and ffmpeg
#    3. pip: torch, opencv-python, numpy, pillow, OpenEXR, fal-client,
#       onnxruntime, transformers
#    4. ~/.ai_toolbox/python = that env's Python, which the app's
#       launcher reads first -- so from then on the app starts from
#       the Dock on its own.
#
#  Run by the app's launcher (in a Terminal window, so the progress is
#  visible) when it finds no usable Python, or by double-clicking
#  "Set Up Python.command" in the installer. Safe to run again: each
#  step is skipped when it is already done.
#
#  Usage: setup_python.sh [/path/to/Studio-Toolbox.app]
#         (the app is opened at the end when a path is given)
# ─────────────────────────────────────────────────────────────────────
set -u

APP="${1:-}"
ROOT="$HOME/.ai_toolbox"
MF="$ROOT/miniforge3"
ENV_NAME="studio-toolbox"
PY_VER="3.13"
PIP_PKGS=(torch torchvision opencv-python numpy pillow OpenEXR fal-client
          onnxruntime transformers)
NEEDS="torch, cv2, numpy, PIL, OpenEXR"

bold() { printf '\n\033[1m%s\033[0m\n' "$*"; }
die()  { printf '\n\033[31m%s\033[0m\n' "$*"; echo; read -r -p "Press Return to close." _; exit 1; }

mkdir -p "$ROOT"
echo "╔══════════════════════════════════════════════╗"
echo "║   Studio Toolbox — Python setup              ║"
echo "╚══════════════════════════════════════════════╝"
echo "This takes 10–20 minutes and a few GB of disk (mostly torch)."
echo "Nothing outside your home folder is touched."

# ── 1. conda ─────────────────────────────────────────────────────────
CONDA=""
for c in "$MF/bin/conda" "$HOME/miniforge3/bin/conda" "$HOME/miniconda3/bin/conda" \
         "$HOME/anaconda3/bin/conda" "$HOME/mambaforge/bin/conda" \
         "/opt/homebrew/Caskroom/miniconda/base/bin/conda" \
         "/opt/homebrew/Caskroom/miniforge/base/bin/conda" \
         "/opt/miniconda3/bin/conda" "/opt/anaconda3/bin/conda" \
         "${CONDA_EXE:-}"; do
    if [ -n "$c" ] && [ -x "$c" ]; then CONDA="$c"; break; fi
done

if [ -z "$CONDA" ]; then
    bold "1/4  Installing Miniforge (conda) into $MF"
    case "$(uname -m)" in
        arm64)  ARCH="arm64" ;;
        x86_64) ARCH="x86_64" ;;
        *) die "Unsupported CPU: $(uname -m)" ;;
    esac
    URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-$ARCH.sh"
    INST="$ROOT/Miniforge3-installer.sh"
    echo "Downloading $URL"
    curl -fL --progress-bar -o "$INST" "$URL" || die "Download failed. Is the Mac online?"
    bash "$INST" -b -p "$MF" || die "Miniforge install failed."
    rm -f "$INST"
    CONDA="$MF/bin/conda"
else
    bold "1/4  conda found: $CONDA"
fi

# The env lives beside whichever conda is used.
CONDA_ROOT="$(cd "$(dirname "$CONDA")/.." && pwd)"
PREFIX="$CONDA_ROOT/envs/$ENV_NAME"
PY="$PREFIX/bin/python3"

# ── 2. env with Python + ffmpeg ──────────────────────────────────────
if [ -x "$PY" ]; then
    bold "2/4  env exists: $PREFIX"
else
    bold "2/4  Creating env $ENV_NAME (Python $PY_VER + ffmpeg)"
    "$CONDA" create -y -p "$PREFIX" -c conda-forge "python=$PY_VER" ffmpeg \
        || die "conda create failed."
fi
if [ ! -x "$PREFIX/bin/ffmpeg" ]; then
    echo "Adding ffmpeg to the env…"
    "$CONDA" install -y -p "$PREFIX" -c conda-forge ffmpeg || echo "ffmpeg install failed — the Setup tab can retry."
fi

# ── 3. packages ──────────────────────────────────────────────────────
bold "3/4  Installing Python packages (torch is the big one, be patient)"
"$PY" -m pip install --upgrade pip >/dev/null 2>&1 || true
"$PY" -m pip install --upgrade "${PIP_PKGS[@]}" || die "pip install failed."

# ── 4. verify + remember ─────────────────────────────────────────────
bold "4/4  Checking"
if "$PY" -c "import torch, cv2, numpy, PIL, OpenEXR" 2>/dev/null; then
    echo "$PY" > "$ROOT/python"
    echo "OK — $NEEDS all import."
    echo "Remembered $PY in $ROOT/python"
else
    die "The env is missing something ($NEEDS). Scroll up for the error."
fi

echo
echo "✓ Python setup done."
if [ -n "$APP" ] && [ -d "$APP" ]; then
    echo "Opening Studio Toolbox…"
    open "$APP"
else
    echo "Open Studio Toolbox from Applications. Then on its Setup & Install"
    echo "tab press Install for the parts you use (RIFE, Real-ESRGAN, SAM 3)."
fi
echo
read -r -p "Press Return to close this window." _
