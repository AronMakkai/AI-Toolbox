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
| Detail Studio | `detail` | `_det_*` |
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
- **Detail Studio** (added 23 Oct 2026) is Upscale Studio's workflow
  for texture at the input resolution: model tiles (ESRGAN / SeedVR2 /
  FlashVSR), Preview & Compare (Wipe / Diff / Detail on black), the
  frequency graph (limit line at 0.5/k = the downscale), per-model
  settings (`det_seedvr_noise_var`, `det_flash_accel_var`), the shared
  cache button and Multi model output (EXR layers). Each frame is
  downscaled (½ or ¼, `_det_downscale`), upscaled back, and the invented
  high band laid over the original (`_det_blend`: Detail only keeps the
  original's low band; Protect keeps the plate's own HF where its local
  energy matches the upscaler's; Strength blends). A model runs once per
  frame (`_det_mem` / fal cache); the sliders re-blend instantly
  (`_det_reblend` → `_det_view`). It calls Upscale Studio's helpers
  (`_ups_read_source`, `_ups_scene`, `_ups_out_px`, `_ups_fal_upscale`,
  `_ups_fal_args`, `_rife_upscale`, `_usc_write_frame`,
  `_usc_write_exr_layers`, `_ups_spectrum`, `_ups_render`) with its own
  state passed as parameters (`mode=`, `hdr=`, `sig=`, `read=`,
  `fal_kw=`, `noise=`, `accel=`) — never through `_ups_*` state. Paid models get the 1/k frames staged under
  `~/.ai_toolbox/detail_stage/<hash>/` so the fal cache hits on re-runs.
  Output `ADW/DETAIL_vNN/`, EXR tagged scene-linear.
- **Lens Studio** applies lens and sensor effects in linear float. Output
  goes to `ADW/Lens_Studio_vNN/Lens_Studio.NNNNN.exr`.
- **Paint Studio** is hand paint/smudge per frame.

## SAM 3

- The full run (`_sam_run`) writes `App.SAM_RUNNER_SRC` to
  `~/RifeApp/sam_runner.py` and executes it in the RifeApp venv, once per
  object, parsing `[i/total]` lines for progress. The runner is readable
  source, not a one-line string.
- Engines, chosen by `--temporal` (the Frame Linking checkbox): `1` is
  SAM 3's video tracker (`Sam3TrackerVideoModel`, streaming session),
  prompted once on `--prompt_frame` (the frame shown when Run is
  pressed) and tracked forward to the end, then backward to the start
  in a second session. `0` is the old per-frame engine, also the
  automatic fallback when the venv's transformers has no
  `Sam3TrackerVideoModel` (the Setup tab's SAM 3 row says so).
- Text prompts: with the video engine the detector (`Sam3Model`) finds
  the concept on the prompt frame and its box becomes the tracker's
  prompt; with points as well, the detector's box is added to them.
- `_stub/transformers` is a fake SAM 3 with the documented API shapes
  (a model that follows a bright blob); `TestSamVideoTracking` runs the
  real runner against it. The real model has never been run from the
  sandbox: the streaming API calls are taken from the transformers
  v5.20 docs for `Sam3TrackerVideoModel` and need checking on the Mac.
- The single-frame preview (`sam_server.py`) is unchanged and still
  per frame.
- Input colour (combo under the folder card, same choices as Upscale
  Studio: Auto / sRGB / Linear / ACES2065-1 / HDR PQ). `_sam_resolve_input`
  sets `_sam_in_mode` via `_resolve_input_colour`; `_sam_display_rgb8`
  reads EXR/DPX as float and converts (`_scene_from_raw` +
  `_organic_linear_to_display`) for the viewer, `_load_frame_bgr` (the
  preview) and the per-frame fallback; the full run passes `--colour` to
  the runner, which has the same maths (`float_rgb_to_display_u8`,
  checked pixel-for-pixel in `TestSamInputColour`) and reads EXR floats
  via OpenEXR -> cv2 -> ffmpeg `gbrpf32le`, since pip OpenCV usually has
  the EXR codec compiled out. PNG/JPG/TIF are display pictures whatever
  the choice says.

## Anamorphic (Upscale Studio and Detail Studio)

- Checkbox + squeeze factor (`UPS_SQUEEZES`: 1.33 / 1.5 / 1.8 / 2.0) in
  the Input card of both studios (`_anam_row`); `ups_anam_var` /
  `ups_squeeze_var` and `det_anam_var` / `det_squeeze_var`. An EXR with
  `pixelAspectRatio` > 1 ticks it and sets the factor on load
  (`_anam_autodetect`, `_exr_pixel_aspect`); a non-EXR source leaves the
  choice alone.
- Desqueeze happens in the reader: `_ups_read_source(..., squeeze=)`
  widens the base (and the HDR gain map) by the factor, so the
  upscalers, the viewer and the graph all see square pixels. Both runs
  call `_resqueeze` on the pixels to write (after `_ups_out_px`, i.e. a
  squeeze of linear light) and pass `pixel_aspect=` to the writers, which
  tag the EXR (`_exr_write_planes`, `_organic_write_exr`,
  `_usc_write_exr_layers`, `_usc_write_frame`).
- Output size: `_ups_model_size` = `_usc_target` on the desqueezed
  picture. A fixed preset (UHD) means 3840 wide AS SEEN, so the file is
  3840 / squeeze wide and keeps its aspect; a factor is a factor either
  way. Cache signatures carry `|sq=`.

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
- `_sam_runner_path` now creates `~/RifeApp` if missing, so the
  startup FileNotFoundError noise in the sandbox is gone.
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
