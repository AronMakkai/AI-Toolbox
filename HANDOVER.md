# AI-Toolbox — handover

Written 5 October 2026 for whoever (person or Claude session) picks this
repository up next. Owner: Aron Makkai. It replaces a long chat in which
the app was developed by sending files back and forth; from here on, work
is committed to this repository.

## Studio Toolbox (branch `studio-toolbox`)

On 5 October 2026 the app was branched into **Studio Toolbox**: the
same code with the four ADW modules removed — ADW-Blur (`rife`),
ADW-CurveLock (`curve`), ADW-AreaLock (`area`) and ADW-Interpolate
(`interp`). `main` stays AI-Toolbox. Everything below applies to both
branches except where noted.

What was removed: the four `_tab_*` builders, their `_rife_*`,
`_curve_*`, `_area_*`, `_interp_*`, `_ip_*` methods, the Blur run/pick/
PNG-cache cluster (`_rrun`, `_rpick*`, `_build_png_cache*`,
`_adw_generate_extra_pairs`, `_dual_timeline`, …), the project save/
load sections for `area`, `curve` and `rife` (an old project still
loads; those sections are ignored), their manual pages in Setup, their
demo steps, the Blur preset map, and their test classes.

What stayed, because other modules call it: the RIFE engine
(`_rife_inference_script`, `_rife_run_pair_stepped`, `_rife_upscale` —
Fix Missing Frames and Upscale Studio), every `_vace_*` helper and the
`VACE_*` constants (Fix Missing Frames, Expansion Studio, Upscale
Studio), `_ip_pv_frame` / `_ip_source_is_encoded` / `_interp_luma_small`
(the colour-aware frame reader and motion scan used by Fix Missing
Frames and Paint Studio), and `_fal_*`. The Wan request settings that
used to be Interpolate-tab widgets (`ip_vace_*_var`) are now created in
`_vace_init_shared_settings`, called from `__init__` before `_build`.

Branch numbers: pyflakes is **22** (was 64 on `main`); the full suite
on this branch has 178 classes. `tests_baseline.txt` is per branch.

The `ADW/` output root and the `ADW-Paint_vNN` folder name are
unchanged on purpose — they are the on-disk convention existing
projects rely on. Renaming them is in Open offers below.

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

## Batch: all shots in this folder (Upscale, Detail, Expansion, MOV→EXR)
- One toolkit, `_batch_*`, inserted before `_anam_row`. `BATCH_MODULES` maps a tab key to the module's running flag, accept, run, log, cancel flag and kind (`seq`/`mov`). `_batch_row(parent, key)` builds the tick box + Shot dropdown + count label; state in `self._batch[key]` (a property over `__dict__["_batch_d"]`, stub-safe).
- Hooks: each module's accept starts with `if self._batch_wants(key, path): return` (sets the root, scans shots, loads the picked shot through the same accept with `_batch_loading` set); each Run starts with `if self._batch_wants_run(key): return` (confirms, then `_batch_next` loads shot i, calls the module's own Run and polls its running flag every 400 ms; a Run that does not start counts as failed; the module's cancel flag ends the queue). `_e_accept` was split out of `_epick`; `_erun` now keeps `_e_running`.
- Output while a batch runs: `_adw_versioned_output_dir` / `_adw_peek_next_version_dir` defer to `_batch_output_dir` → one version folder per module per batch with a subfolder per shot (`ADW/UPSCALE_v03/sh010`). The version folder is itself a batch root for the next module. `_adw_propagate_output` and `_batch_reveal` only fire on the last shot; `_slow_model_notice` once per batch.
- `_batch_confirm` (None → messagebox) is injected by tests. Tests: `TestBatchShots`.

## Colourspaces — one list, every module (28 Oct 2026)
- `CS_KEYS` linear/srgb/rec709/pq/acescg/aces; `CS_NAMES` ("Scene-linear Rec.709", "Display sRGB", "Display Rec.709", "Display PQ / Rec.2020", "Working ACEScg (AP1)", "Archive ACES2065-1 (AP0)"); `CS_IN_CHOICES` = Auto + names, `CS_OUT_CHOICES` = Same as input + names; `CS_LEGACY` maps every old dropdown string; `_cs_key(choice)`; `CS_IN_TIP` / `CS_OUT_TIP` are the ⓘ texts; `_cs_combo` builds a dropdown.
- Hub is scene-linear Rec.709 (1.0 = diffuse white): `_cs_to_lin709(a, key)`, `_cs_from_lin709(lin, key)`, `_cs_convert(a, src, dst)`. PQ = ST.2084 with Rec.2020 primaries, 1.0 linear = `HDR_SDR_WHITE` (100 nits). ACEScg via `_M_709_TO_AP1`. `CS_NUMPY_SRC` is the same maths as plain numpy for venv scripts (SAM EXR merge); a test checks parity.
- `_cs_out_key(choice, in_key, movie)` — Same as input = the input's key; a movie only holds a display space (rec709 fallback). `_cs_sidecar(key)` → (colorspace, primaries) for `_adw_write_colorspace`; `_cs_tagged_key(folder)` reads it back (primaries now also "acescg"/"pq"/"srgb"/"rec709"). EXR writers take `tag=` (chromaticities for aces / acescg, `ACES_AP1_CHROMA`).
- Upscale / Detail: `UPS_IN_CHOICES = CS_IN_CHOICES`; `_resolve_input_colour` resolves via `_cs_key` + tagged key; modes now include rec709 / acescg; `_ups_eotf(mode, explicit)` — sRGB curve only when Display sRGB was chosen by hand; `_ups_out_px(..., out=key)`; `ups_cs_out_var` / `det_cs_out_var` + "Writes →" labels (`_ups_cs_out_key`, `_det_cs_out_key`, `_cs_out_note`).
- Expansion: `usc_cs_out_var` (default Archive ACES2065-1) replaces the EXR colour combo; `_usc_cs_out_key(exr)` — Same as input on an expanding run = linear EXR / PQ movie; the HDR write converts `_lin` with `_cs_from_lin709`; `_usc_exr_aces()` kept.
- Lens: `LENS_IN_CHOICES = CS_IN_CHOICES`; `_organic_source_cs` holds rec709 / acescg on top of the old flags; `_organic_in_key`, `_organic_cs_out_key(fmt)`, `organic_cs_out_var`; `_organic_write_frame(..., cs=)`; `_organic_load_linear(..., cs=)`.
- MOV→EXR: `_m2e_keys(csp, out, transfer)`, `_m2e_plan(in, out)` → (zscale vf or "", mid_key); everything from mid_key to out_key is `_cs_convert` in the app (one loop replaced the linearise + ACES loops); `_m2e_src_transfer` probed on pick.
- Paint: `sm_cs_out_var`; `_smudge_natural_out`, `_smudge_cs_out_key`, `_smudge_aces_out(lin, dst, g=)` rewrites the ffmpeg EXR in the output space; a Run with a different Output re-encodes copied frames too (`_reencode`).
- Fix Missing Frames: `ff_cs_var` is now Input colour (Auto…), `ff_cs_out_var`; `_ff_target_convention` returns (ext, CS key); `_ff_write_repaired_frame` takes the key; the copy step converts EXRs when Output differs.
- SAM 3: `sam_cs_out_var`; the EXR merge script gets `cs_in` / `cs_out` and converts RGB with `CS_NUMPY_SRC`; the SAM runner's `float_rgb_to_display_u8` knows rec709 / acescg / pq-2020.
- ADW modules (AI-Toolbox only): one shared row `_adw_cs_row` bound to `adw_cs_in_var` / `adw_cs_out_var`; Input drives `adw_source_encoded_var` and `_adw_gamma_encode` converts aces/acescg/pq to 709 linear first; `_adw_cs_finish(out_dir, log)` converts written EXRs after the render (called from the four `*_after_run_sendto` / `_vace_apply_done`).
- Tests: `TestColourspaceToolkit`; the old name-based tests were rewritten.

## Output names and the Rec.709 preview MOV (28 Oct 2026)
- `_adw_out_stem(name, frames, i)` names every studio frame `<Module>.<plate frame>` (`UPS_OUT_NAME` Upscale_Studio, `DET_OUT_NAME` Detail_Studio, `USC_OUT_NAME` Expansion_Studio, Lens unchanged); movies encode via `_adw_glob_input(folder)` (ffmpeg glob) so numbering need not start at 1.
- `_adw_rec709_mov(out_dir, name, fps, log_w, key, frames)` reads the written frames back, converts the folder's colourspace to Display Rec.709 and encodes `<name>_Rec709.mov` (ProRes 422 HQ, bt709 tags). `_mov709_row` is the "+ Rec.709 MOV" tick box in the three studios' Output cards (`ups_mov709_var` / `det_mov709_var` / `usc_mov709_var`, ignored when Write is a MOV); Lens's and Paint's "Out .MOV" boxes were relabelled and Lens now encodes through the same helper (its old `_organic_encode_mov` is unused). Tests: `TestStudioOutputNames`, `TestRec709PreviewMov`.

## LTX Alpha (SAM 3's second matte source, 30 Oct 2026)
- `fal_ltx_alpha/app.py` + `DEPLOY.md`: a fal Serverless app (`fal deploy app.py::LTXAlpha --app-name ltx-alpha`) that clones Lightricks' LTX-2 repo, downloads LTX-2.5 + the gated `LTX-2.5-22b-IC-LoRA-Alpha-Gen` LoRA to `/data` (HF_TOKEN secret), and runs `python -m ltx_pipelines.alpha_gen --video-conditioning guide.mp4 1.0 --lora <lora> 1.0 --prompt ""` per request; endpoint `<user>/ltx-alpha/alpha`, input `video_url, num_frames, width, height, frame_rate, seed`, output `alpha` (grey mp4) + `premult`. Untested on fal from here (no GPU/account): the first deploy is a joint test.
- App side (`_ltx_*`, before `_sam_write_multichannel_exr`): app id in `~/.ai_toolbox/ltx_app_id` (Setup → fal card → "LTX Alpha app id"); `_ltx_plan(n, chunk, overlap)` → overlapping 8k+1 chunks ≤145; `_ltx_send_size` (fit ≤1920×1088, ×32); `_ltx_grid_up` pads a short chunk by repeating the last frame; `_ltx_chunk` = cache (`~/.ai_toolbox/ltx_cache/<key>/NNNNN.png`, key from file names/sizes/mtimes + size + seed + input colour) or `_fal_video_submit(app_id + "/alpha", ...)` through the SAM standing panel (`sam_fal_panel`/`_sam_fal_state`); `_ltx_matte_all` crossfades chunks (`_ltx_crossfade`) and writes RGBA PNGs (alpha = matte) to `<folder>/_ltx_alpha_tmp`, optionally limited by `_ltx_limit_mask` (SAM mask dilated + softened).
- SAM tab: "Matte source" card with two tiles (`sam_src_sam_var`, `sam_src_ltx_var`, `_sam_src_changed`, `_sam_run_label`), LTX options (`sam_ltx_chunk/overlap/size/seed/limit_var`, `_ltx_update_note`), Preview (`_sam_preview_ltx` → `_sam_show_ltx_alpha` into the wipe viewer). `_sam_run`: LTX-only runs a no-op in place of the SAM subprocess; after the SAM passes `_ltx_matte_all` runs; `_sam_write_multichannel_exr(..., ltx_dir=)` — the merge script makes A = the LTX matte and adds an `ltx` channel. `_sam_cancel_run` sets `_ip_vace_cancel` for an LTX job. Tests: `TestLtxAlpha`.

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

- `_fal_call_with_watchdog(..., alive=)`: the timeout (`VACE_FAL_TIMEOUT_S`,
  30 min) is measured from the last sign of life when the caller passes
  `alive` (a callable returning that time), not from the start — SeedVR2
  renders that kept reporting for 40 min were being killed at 30.
  `VACE_FAL_HARD_CAP_S` (4 h) still ends a job that talks forever.
  `_fal_video_submit` feeds `alive` from every queue update / log line,
  logs the request id and queue moves, and relays phase / queue / req /
  msg to `on_panel`.
- Upscale Studio and Detail Studio have Expansion Studio's standing fal
  panel (`ups_fal_panel` / `det_fal_panel`, under Run): `_ups_fal_upscale`
  takes `panel=(label, state_attr)`, calls `_fal_panel_begin` with the
  `_vace_estimate` for the span, notes every phase/update, records the
  render time (`_vace_record_render`) and ends with ✓ / the failure.

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
