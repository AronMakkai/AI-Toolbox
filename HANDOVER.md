# AI-Toolbox — handover

Written 5 October 2026 for whoever (person or Claude session) picks this
repository up next. Owner: Aron Makkai. It replaces a long chat in which
the app was developed by sending files back and forth; from here on, work
is committed to this repository.

## What this is

AI-Toolbox is a Python/Tkinter desktop app for VFX work, run on macOS. It
is one very large file, `ai_toolbox.py` (about 51,000 lines, one `App`
class), plus one test file.

| File | What it is |
|---|---|
| `ai_toolbox.py` | The whole app. |
| `test_organic_masks.py` | The whole test suite (176 test classes; the name is historical). |
| `_stub/torch/` | A fake `torch` so the app imports for tests without a real install. |
| `tests_baseline.txt` | The 28 tests that already fail in a Linux sandbox (see Testing). |
| `build_dmg.sh` | Builds the macOS .dmg. |
| `AI-Toolbox.bat` | Windows launcher. |

## Modules (tabs, in order)

The names on screen were changed; the code keeps the old keys and
prefixes. Use this table before searching.

| Tab | Tab key | Code prefix |
|---|---|---|
| Setup & Install | `setup` | `_setup_*` |
| MOV ↔ EXR | `exr` | `_exr_*`, `_e2m_*`, `_m2e_*` |
| Upscale Studio | `upstudio` | `_ups_*` |
| Expansion Studio | `upscale` | `_usc_*` |
| Lens Studio (was ADW-Organic) | `organic` | `_organic_*` |
| Paint Studio (was ADW-Paint) | `smudge` | `_smudge_*`, `sm_*` |
| Fix Missing Frames, CurveLock, AreaLock, Blur, Interpolate, SAM 3 | | |

- **Expansion Studio** turns SDR into HDR (tone map, Topaz, Wan inpaint via
  fal). Writes EXR (ACES2065-1 or Linear Rec.709), or an HDR10 movie. Can
  write every ticked model as separate EXR layers ("Multi model output")
  and include `BlowMask` / `CrunchMask` channels.
- **Upscale Studio** is the same workflow for upscaling: SeedVR2
  (`fal-ai/seedvr/upscale/video`), FlashVSR
  (`fal-ai/flashvsr/upscale/video`) and local ESRGAN. SeedVR2 is capped
  (`UPS_MAX_SHORT_SIDE`), with a popup stating the resulting resolution.
- **Lens Studio** applies lens and sensor effects in linear float. Output
  goes to `ADW/Lens_Studio_vNN/Lens_Studio.NNNNN.exr`.
- **Paint Studio** is hand paint/smudge per frame.

## Colour handling

- Every studio has an "Input colour" choice: Auto, display-encoded
  (sRGB / Rec.709), Linear Rec.709, ACES2065-1, HDR PQ / Rec.2020. Auto
  reads the EXR header, a sidecar `_adw_colorspace.json` in the folder, or
  the movie's transfer tag. Shared resolver: `_resolve_input_colour`.
- Internally everything is scene-linear Rec.709 float.
  `_lin709_to_aces` / `_aces_to_lin709` convert at the edges.
- ACES EXRs must be written through `_exr_write_planes(..., aces=True)`,
  which uses the newer `OpenEXR.File` API. The legacy Header API writes
  the chromaticities as zeros.
- Each output folder gets a sidecar via `_adw_write_colorspace` so the
  next module does not have to guess.
- HDR reference white is `HDR_SDR_WHITE = 100` nits.
- ffmpeg zscale wants `tin=iec61966-2-1`, not `tin=srgb`.

## fal

- All calls go through `_fal_call_with_watchdog`. While waiting on fal the
  progress bar freezes with a red/black fal mark, and Preview becomes
  Cancel preview.
- **No real fal call has ever been run from the development sandbox**, only
  mocks. Anything touching fal needs checking on Aron's machine.

## Testing — the routine Aron expects on every change

1. Ground the fix in the code: read what is there before changing it.
2. Run the full suite and compare failure NAMES against the baseline, not
   just the counts:

   ```
   PYTHONPATH=_stub python3.12 test_organic_masks.py > n.log 2>&1
   grep -E "^(FAIL|ERROR):" n.log | sort > n.txt
   comm -13 tests_baseline.txt n.txt     # must print nothing
   ```

   Baseline tail is `FAILED (failures=1, errors=27, skipped=13)`. Those 28
   are sandbox limits (mostly H.264 / codec tests needing an ffmpeg build
   the sandbox lacks), not regressions. The suite takes several minutes.
3. `python3.12 -m pyflakes ai_toolbox.py | wc -l` must stay at **64**.
4. Verify in the real app, not only in unit tests. Headless:

   ```
   PYTHONPATH=_stub xvfb-run -a -s "-screen 0 1600x1000x24" python3.12 drive.py
   ```

   where the script builds `App()`, steps through the UI with `app.after`,
   prints what it sees, and takes screenshots with ImageMagick `import`.
5. Add tests for the change.
6. Summarise in plain prose: what changed, what was tested, and what was
   NOT tested.

### Things that waste time if you do not know them

- A bare `_Stub(App)` without Tk recurses in `__getattr__`. Give `App` a
  class-level default for any new attribute read via `getattr`.
- Stub classes that are not `App` subclasses need `staticmethod(...)`
  around borrowed static/class methods.
- Many tests assert on exact source text. Changing a line can break a test
  whose intent is still met; update the test to the new intent.
- Calling Tk from a worker thread segfaults under Xvfb.
- `sam_runner` FileNotFoundError noise at startup in the sandbox is
  unrelated.
- A `ttk.Combobox` is a `tk.Entry` subclass. Hotkey guards must use
  `_typing_in_field(w)`, not `isinstance(w, tk.Entry)`.
- The sandbox ffmpeg has no `gbrpf16le`, so 16-bit half MOV → EXR is
  untested there.
- The source uses real `·`, `×`, `—`, `→` characters; scripted
  search-and-replace must match those, not `\u` escapes.

## Most recent changes (2–5 October 2026)

- Lens Studio: Input colour setting covering everything Expansion Studio
  writes, plus "Load Movie…" for the HDR10 movie.
- Lens Studio: CANCEL RENDER while rendering; a cancelled render keeps its
  frames, writes no movie and is not handed on.
- Lens Studio: "Expansion adjustment" (Highlights slider, reads
  `BlowMask`) recalculates on mouse release from memory, and only inside
  the ROI when one is set.
- Lens Studio: frames named `Lens_Studio.NNNNN`, keeping the source's
  frame numbers.
- Expansion Studio: Shadows tab left, Highlights right; blow/crunch masks
  available with every model.
- All viewers: resolution and file bit depth shown lower-left.

## Not verified

- Everything above was tested on Linux under Xvfb. None of the recent work
  has been confirmed on macOS by the developer side.
- Real fal calls (see above).
- A real mouse drag on the Highlights slider, and a click on the CANCEL
  RENDER button itself (both were driven programmatically).
- A plain Rec.709 movie through Lens Studio's Load Movie.

## Open offers (suggested, not requested)

- Name Expansion Studio / Upscale Studio frames `Module.NNNNN` too; they
  still write bare `00001.exr`. Paint Studio, CurveLock and AreaLock keep
  the source's name.
- Rename Paint Studio's output folder (`ADW-Paint_vNN`) and its
  "RUN ADW-PAINT" button.
- Rename the EXR → MOV "Linear → Rec.709" style option names.
- ACES reading in CurveLock, AreaLock and Blur.
- Pass the mask channels through Lens Studio's output.
- Keep source highlights when Expansion Studio's input is already HDR.
- Fixed-size aspect-fit in Upscale Studio.
