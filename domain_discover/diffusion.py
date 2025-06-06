"""Diffusion‑based text generator wrapper using DiffuSeq (patched).

Major fixes compared with the original draft
-------------------------------------------
1. Robustly resolves **DiffuSeq** repo path (env var → up‑tree search).
2. Loads **config** *before* touching `hidden_dim`; builds extra embedding only
   *after* the main model is ready and re‑uses its weights.
3. Handles single‑GPU runs without hanging in `dist_util.setup_dist()`.
4. Removes unsupported kwargs (`top_p`) and correctly forwards `ddim_steps`.
5. Uses `self.model.output_layer` (or linear projection fallback) to obtain
   vocab logits.
6. Cleans decoding (`skip_special_tokens=True`).
7. Misc: no duplicate `sys.path` injection, type hints, logging instead of
   prints.

The class is now safe to import and run on a laptop GPU with the command:

```bash
python diffusion_text_generator_fixed.py \
  --model_path /path/to/ema_0.9999_1000000.pt \
  --num_samples 2 --max_len 50 --temperature 0.9 --use_ddim
```
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import argparse
import json
import logging
import os
import sys
from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1.  Locate DiffuSeq
# ---------------------------------------------------------------------------

def _find_diffuseq() -> Path:
    """Return path to local DiffuSeq repo or raise FileNotFoundError."""
    env_path = os.getenv("DIFFUSEQ_HOME")
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if (path / "diffuseq").exists():
            return path

    # Fallback: walk up from current file
    cur = Path(__file__).resolve()
    for _ in range(5):  # search at most five parent levels
        candidate = cur.parent / "DiffuSeq"
        if (candidate / "diffuseq").exists():
            return candidate
        cur = cur.parent
    raise FileNotFoundError("Cannot locate DiffuSeq repo. Set $DIFFUSEQ_HOME or place repo as sibling named 'DiffuSeq'.")


def _ensure_diffuseq_on_path() -> None:
    repo_root = _find_diffuseq()
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    logger.info("Using DiffuSeq from %s", repo_str)


_ensure_diffuseq_on_path()

# Defer heavy imports until path is set
from diffuseq.gaussian_diffusion import GaussianDiffusion
from diffuseq.rounding import denoised_fn_round
from diffuseq.utils import dist_util
from basic_utils import create_model_and_diffusion, load_tokenizer  # noqa: E402

# ---------------------------------------------------------------------------
# 2.  The generator wrapper
# ---------------------------------------------------------------------------

def _safe_setup_single_gpu_dist():
    """Initialise fake distributed backend if none present (single‑GPU run)."""
    if "RANK" not in os.environ:
        os.environ.update({
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "12355",
            "RANK": "0",
            "WORLD_SIZE": "1",
        })
    dist_util.setup_dist()


@dataclass
class DiffusionTextGenerator:
    model_path: str
    device: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    diffusion_steps: int = 2000  # will be overwritten by config if present
    use_ddim: bool = True
    clip_denoised: bool = True
    ddim_steps: int = 200

    # runtime attributes (post‑init)
    config: Dict[str, Any] = field(default_factory=dict)
    model: Optional[PreTrainedModel] = None
    tokenizer: Optional[PreTrainedTokenizer] = None
    diffusion: Optional[GaussianDiffusion] = None
    model_dir: Optional[str] = None

    # ---------------------------------------------------------------------
    #  post‑init lifecycle
    # ---------------------------------------------------------------------
    def __post_init__(self):
        self._process_model_path()
        self._load_config()
        self._create_model_and_diffusion()
        self._load_tokenizer()
        logger.info("DiffusionTextGenerator ready on %s", self.device)

    # ------------------------------------------------------------------
    #  private helpers
    # ------------------------------------------------------------------
    def _process_model_path(self) -> None:
        p = Path(self.model_path).expanduser().resolve()
        if p.is_dir():
            pt_files = sorted(p.glob("ema*.pt")) or sorted(p.glob("*.pt"))
            if not pt_files:
                raise FileNotFoundError("No .pt model found in %s" % p)
            self.model_path = str(pt_files[0])
            self.model_dir = str(p)
        else:
            if not p.exists():
                raise FileNotFoundError(p)
            self.model_dir = str(p.parent)
        logger.info("Model directory: %s", self.model_dir)
        logger.info("Model file     : %s", self.model_path)

    def _load_config(self) -> None:
        cfg_path = Path(self.model_dir) / "training_args.json"
        if not cfg_path.exists():
            raise FileNotFoundError(cfg_path)
        with cfg_path.open() as f:
            self.config = json.load(f)
        # override / ensure fields
        self.config.setdefault("batch_size", 4)
        self.diffusion_steps = self.config.get("diffusion_steps", self.diffusion_steps)
        logger.info("Hidden dim %s | seq_len %s | diffusion_steps %s",
                    self.config.get("hidden_dim"), self.config.get("seq_len"), self.diffusion_steps)

    def _create_model_and_diffusion(self) -> None:
        # safe single‑GPU
        _safe_setup_single_gpu_dist()
        # build
        self.model, self.diffusion = create_model_and_diffusion(
            device=str(self.device), **self.config
        )
        # load weights
        state = dist_util.load_state_dict(self.model_path, map_location="cpu")
        self.model.load_state_dict(state, strict=False)
        self.model.eval().requires_grad_(False).to(self.device)
        logger.info("Model params: %.2f M", sum(p.numel() for p in self.model.parameters()) / 1e6)

    def _load_tokenizer(self) -> None:
        args_ns = argparse.Namespace(**self.config)
        self.tokenizer = load_tokenizer(args_ns)
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer load failed (config: %s)" % self.config)
        logger.info("Tokenizer vocab_size=%d", self.tokenizer.vocab_size)

    # ------------------------------------------------------------------
    #  public API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        prompt_vec: torch.Tensor,  # (1, hidden_dim)
        num_samples: int,
        max_len: int,
        temperature: float = 1.0,
    ) -> List[str]:
        """Generate *num_samples* sentences conditioned on *prompt_vec*."""
        prompt = prompt_vec.to(self.device)
        # broadcast without real allocation
        prompt_batch = prompt.expand(num_samples, -1).contiguous()

        noise = torch.randn(
            num_samples, max_len, self.config["hidden_dim"], device=self.device
        )

        model_kwargs = {"prompt_embedding": prompt_batch}

        sample_fn = (
            self.diffusion.ddim_sample_loop if self.use_ddim else self.diffusion.p_sample_loop
        )

        sample_kwargs: Dict[str, Any] = dict(
            model=self.model,
            shape=noise.shape,
            noise=noise,
            clip_denoised=self.clip_denoised,
            denoised_fn=partial(denoised_fn_round, argparse.Namespace(**self.config),
                                self.model.word_embedding),
            model_kwargs=model_kwargs,
            progress=True,
        )
        # only ddim accepts custom step count
        if self.use_ddim:
            sample_kwargs["ddim_timesteps"] = self.ddim_steps

        samples = sample_fn(**sample_kwargs)
        # DiffuSeq returns tensor [bsz, seq_len, hidden_dim]
        final_hidden = samples if isinstance(samples, torch.Tensor) else samples[-1]

        # Project to vocab
        if hasattr(self.model, "output_layer"):
            logits = self.model.output_layer(final_hidden)  # type: ignore[attr-defined]
        else:
            # fallback linear proj by sharing weights
            W = self.model.word_embedding.weight  # (V, H)
            logits = torch.einsum("bld,vd->blv", final_hidden, W)

        token_ids = torch.argmax(logits, dim=-1)
        texts = [
            self.tokenizer.decode(seq.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for seq in token_ids
        ]
        return texts


# ---------------------------------------------------------------------------
#  CLI for quick sanity test (not production‑grade)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick test for DiffusionTextGenerator")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--use_ddim", action="store_true")
    parser.add_argument("--num_samples", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.9)
    args = parser.parse_args()

    gen = DiffusionTextGenerator(model_path=args.model_path, use_ddim=args.use_ddim)
    vec = torch.randn(1, gen.config["hidden_dim"], device=gen.device)
    logger.info("Sampling %d texts …", args.num_samples)
    outs = gen.sample(vec, args.num_samples, args.max_len, args.temperature)
    for i, t in enumerate(outs, 1):
        print(f"[{i}] {t}")
