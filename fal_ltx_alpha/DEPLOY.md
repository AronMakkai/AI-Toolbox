# LTX Alpha on fal — deploy once, then the Toolbox calls it

The alpha-matte IC-LoRA for LTX-2.5 (hair, fur, smoke, glass) runs on an
H100-class GPU, so it cannot run on the Mac. This folder is a fal
Serverless app that runs Lightricks' own `ltx_pipelines.alpha_gen` and
returns the matte. You deploy it under your fal account; the Toolbox
sends clips to it with the fal key you already have.

## Once

1. **Serverless access** — on fal.ai go to Serverless → Get started and
   request access (custom apps need it; the hosted models do not).
2. **Licence** — on Hugging Face, accept the LTX-2 community licence on
   both `Lightricks/LTX-2.5` and `Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen`
   (free under $10M revenue, paid above).
3. **Tools** — on the Mac:
   ```
   pip install fal
   fal auth login
   fal secrets set HF_TOKEN=hf_...          # a Read token from huggingface.co/settings/tokens
   ```
4. **Deploy** — from this folder:
   ```
   fal deploy app.py::LTXAlpha --app-name ltx-alpha
   ```
   It prints the app id, e.g. `aron/ltx-alpha`. The first request downloads
   ~60 GB of weights to the app's `/data` volume (10–20 min); every run
   after that starts in about a minute from cold, instantly while warm.
5. **Toolbox** — Setup tab → fal.ai card → paste the app id into
   *LTX Alpha app id* and Save. The LTX Alpha tile in SAM 3 is now live.

## Cost

Billed per second while a worker is up (H100 ≈ $1.9/h at the time of
writing): a 145-frame full-HD chunk is a few minutes of GPU. `keep_alive`
is 300 s, so back-to-back chunks reuse the warm worker; the Toolbox
caches every chunk, so a re-run with the same settings is free.

## Test

```
fal run app.py::LTXAlpha   # local dev server, or:
curl -X POST https://fal.run/<you>/ltx-alpha/alpha -H "Authorization: Key $FAL_KEY" \
  -H "Content-Type: application/json" -d '{"video_url":"https://.../clip.mp4","num_frames":97,"width":1280,"height":736}'
```

## Limits (the model's)

145 frames per chunk (the Toolbox chunks longer clips and crossfades
the seams), 1920×1088, sizes padded to multiples of 32, frame counts on
the 8k+1 grid, no subject control (use *Limit to SAM object* in the
Toolbox for multi-subject shots).
