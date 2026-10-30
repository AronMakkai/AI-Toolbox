"""LTX Alpha on fal Serverless — the alpha-matte IC-LoRA for LTX-2.5,
deployed once under your own fal account and called by the Toolbox.

Input:  a video URL (RGB, any size; the Toolbox sends it already padded
        to multiples of 32 and a frame count on the 8k+1 grid)
Output: the alpha matte as a video (grey: white = foreground) plus a
        premultiplied preview, both on fal's CDN.

Deploy (see DEPLOY.md):  fal deploy app.py::LTXAlpha --app-name ltx-alpha
The Toolbox then calls  <your-username>/ltx-alpha/alpha
"""
import os
import subprocess
import tempfile
from pathlib import Path

import fal
from fal.toolkit import File, download_file
from pydantic import BaseModel, Field

WEIGHTS = Path("/data/ltx-alpha")      # fal's persistent volume
LTX_REPO = "https://github.com/Lightricks/LTX-2.git"
HF_BASE = "Lightricks/LTX-2.5"
HF_LORA = "Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen"
LORA_FILE = "ltx-2.5-22b-ic-lora-alpha-gen-0.9.safetensors"


class AlphaRequest(BaseModel):
    video_url: str = Field(description="RGB clip; the matte is made for it")
    num_frames: int = Field(default=121, description="8k+1, max 145")
    width: int = Field(default=1920, description="multiple of 32, max 1920")
    height: int = Field(default=1088, description="multiple of 32, max 1088")
    frame_rate: int = Field(default=24)
    seed: int = Field(default=42)
    strength: float = Field(default=1.0, description="IC-LoRA guide strength")


class AlphaResponse(BaseModel):
    alpha: File = Field(description="the matte, grey video")
    premult: File = Field(description="the clip premultiplied by the matte")
    num_frames: int


class LTXAlpha(fal.App, keep_alive=300, max_concurrency=1):
    machine_type = ["GPU-H100", "GPU-B200"]
    requirements = ["uv", "huggingface_hub[cli]", "opencv-python-headless",
                    "numpy", "imageio[ffmpeg]"]

    def setup(self) -> None:
        """Clone the LTX-2 pipelines and fetch the weights once; the
        volume keeps them for every later run. HF_TOKEN must be a fal
        secret (fal secrets set HF_TOKEN=hf_...) that has accepted the
        LTX-2 licence on both repos."""
        WEIGHTS.mkdir(parents=True, exist_ok=True)
        self.repo = WEIGHTS / "LTX-2"
        if not (self.repo / "pyproject.toml").exists():
            subprocess.run(["git", "clone", "--depth", "1", LTX_REPO,
                            str(self.repo)], check=True)
            subprocess.run(["uv", "sync"], cwd=self.repo, check=True)
        env = {**os.environ, "HF_TOKEN": os.environ.get("HF_TOKEN", "")}
        self.models = WEIGHTS / "models"
        if not (self.models / "ltx-2.5").exists():
            subprocess.run(["hf", "download", HF_BASE, "--local-dir",
                            str(self.models / "ltx-2.5")], check=True, env=env)
        self.lora = self.models / LORA_FILE
        if not self.lora.exists():
            subprocess.run(["hf", "download", HF_LORA, LORA_FILE,
                            "--local-dir", str(self.models)], check=True,
                           env=env)
        self.python = self.repo / ".venv" / "bin" / "python"

    @fal.endpoint("/alpha")
    def alpha(self, req: AlphaRequest) -> AlphaResponse:
        import cv2
        import numpy as np
        work = Path(tempfile.mkdtemp())
        src = work / "guide.mp4"
        download_file(req.video_url, target_dir=work, force_filename="guide.mp4")
        out = work / "alpha_rgb.mp4"
        n = max(9, min(req.num_frames, 145))
        n = ((n - 1) // 8) * 8 + 1
        w = max(32, min(req.width, 1920)) // 32 * 32
        h = max(32, min(req.height, 1088)) // 32 * 32
        cmd = [str(self.python), "-m", "ltx_pipelines.alpha_gen",
               "--checkpoint-path", str(self.models / "ltx-2.5"),
               "--lora", str(self.lora), str(req.strength),
               "--video-conditioning", str(src), str(req.strength),
               "--prompt", "",
               "--height", str(h), "--width", str(w),
               "--num-frames", str(n), "--frame-rate", str(req.frame_rate),
               "--seed", str(req.seed),
               "--quantization", "fp8-cast",
               "--output-path", str(out)]
        subprocess.run(cmd, cwd=self.repo, check=True,
                       env={**os.environ,
                            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
        # The model returns the matte as an RGB clip (grey); keep it as
        # a clean 4:4:4 grey movie and make the premult beside it.
        alpha = work / "alpha.mp4"
        premult = work / "premult.mp4"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(out),
                        "-vf", "format=gray", "-c:v", "libx264", "-crf", "8",
                        "-pix_fmt", "yuv444p", str(alpha)], check=True)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src),
                        "-i", str(out), "-filter_complex",
                        "[1:v]format=gray,scale=iw:ih[a];"
                        "[0:v][a]scale2ref[v][a2];[v][a2]blend=all_mode=multiply",
                        "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p",
                        str(premult)], check=False)
        return AlphaResponse(alpha=File.from_path(alpha),
                             premult=File.from_path(premult) if premult.exists()
                             else File.from_path(alpha),
                             num_frames=n)
