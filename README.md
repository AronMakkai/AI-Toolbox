# Studio Toolbox

A Python/Tkinter desktop application for VFX workflows on macOS.

Studio Toolbox is AI-Toolbox without the ADW modules (ADW-Blur,
ADW-CurveLock, ADW-AreaLock, ADW-Interpolate). It lives on the
`studio-toolbox` branch of this repository; `main` is AI-Toolbox.
Project files (`.adwproj`) and Lens Studio presets are interchangeable
between the two: an AI-Toolbox project opens here with its ADW sections
ignored.

## Overview

Studio Toolbox is a VFX processing suite featuring:

- **Lens Studio** — Optical effects (chromatic aberration, bloom, vignette, grain)
- **Expansion Studio** — SDR-to-HDR tone mapping with AI upsampling
- **Upscale Studio** — Video upscaling (SeedVR2, FlashVSR, ESRGAN)
- **Paint Studio** — Per-frame hand painting and smudging
- **Fix Missing Frames** — rebuild dropped, black or held frames with RIFE or Wan VACE
- Plus: MOV ↔ EXR conversion and SAM 3 segmentation

## Getting Started

See [HANDOVER.md](HANDOVER.md) for:
- Project structure and module reference
- Testing procedures and requirements
- Color handling details
- Known limitations and unverified features

## Requirements

- **Python 3.13** (Miniconda recommended)
- **macOS** (Apple Silicon or Intel)
- See HANDOVER.md for dependencies and testing setup

## Development

Follow the testing routine documented in HANDOVER.md section "Testing":

```bash
PYTHONPATH=_stub python3.13 test_organic_masks.py > n.log 2>&1
grep -E "^(FAIL|ERROR):" n.log | sort > n.txt
comm -13 tests_baseline.txt n.txt  # must print nothing
```

Maintain:
- ✅ Exact pyflakes count at **22 warnings** on this branch (64 on `main`)
- ✅ Test failures matching the baseline (28 expected failures)
- ✅ Full test coverage for changes

## Building

macOS .dmg:
```bash
bash build_dmg.sh
```

Windows launcher: `Studio-Toolbox.bat`
