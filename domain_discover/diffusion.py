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
import re

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
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from wordnet_conditioner import TextDataset
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
    # 首先检查环境变量
    env_path = os.getenv("DIFFUSEQ_HOME")
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if (path / "diffuseq").exists():
            return path

    # 检查项目根目录中的DiffuSeq目录
    # 在Domain-Inference项目中，DiffuSeq位于项目根目录下
    cur = Path(__file__).resolve()
    project_root = None
    
    # 向上查找项目根目录（包含DiffuSeq的目录）
    for _ in range(5):  # 最多向上查找5层
        if (cur / "DiffuSeq").exists() or (cur.parent / "DiffuSeq").exists():
            project_root = cur if (cur / "DiffuSeq").exists() else cur.parent
            break
        if cur.name == "Domain-Inference":
            project_root = cur
            break
        cur = cur.parent
    
    # 如果找到了项目根目录，检查DiffuSeq是否存在
    if project_root:
        diffuseq_path = project_root / "DiffuSeq"
        if (diffuseq_path / "diffuseq").exists() or (diffuseq_path / "DiffuSeq" / "diffuseq").exists():
            return diffuseq_path if (diffuseq_path / "diffuseq").exists() else diffuseq_path / "DiffuSeq"
            
    # 直接尝试硬编码路径
    hardcoded_path = Path("/home/dora/Domain-Inference/DiffuSeq/DiffuSeq")
    if hardcoded_path.exists() and (hardcoded_path / "diffuseq").exists():
        return hardcoded_path
        
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
    step: int = 50
    ddim_steps: int = 200
    top_p: float = 0.0
    clamp_step: int = 0
    batch_size: int = 4
    seq_len: int = 128
    split: str = 'valid'
    hidden_dim: Optional[int] = 128
    output_dir: Optional[str] = None
    world_size: int = 1
    rank: int = 0

    # runtime attributes (post‑init)
    config: Dict[str, Any] = field(default_factory=dict)
    model: Optional[PreTrainedModel] = None
    tokenizer: Optional[PreTrainedTokenizer] = None
    diffusion: Optional[GaussianDiffusion] = None
    model_dir: Optional[str] = None
    model_emb: Optional[nn.Embedding] = None

    # ---------------------------------------------------------------------
    #  post‑init lifecycle
    # ---------------------------------------------------------------------
    def __post_init__(self):
        self._process_model_path()
        self._load_config()
        self._create_model_and_diffusion()
        self._load_tokenizer()
        self._create_model_embedding()
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
        
    def _create_model_embedding(self) -> None:
        """Create model embedding layer that shares weights with the model."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded before creating embedding")
            
        self.model_emb = nn.Embedding(
            num_embeddings=self.tokenizer.vocab_size,
            embedding_dim=self.config.get("hidden_dim", 128),
            _weight=self.model.word_embedding.weight.clone().cpu()
        ).eval().requires_grad_(False)

    def _load_data_text(self, conditioner, loop=False):
        # conditioner 可以是一个词或词列表
        if isinstance(conditioner, str):
            words = [conditioner]
        else:
            words = conditioner

        seq_len = self.seq_len
        vocab = self.tokenizer

        # 将每个词转成 token ids（注意encode_token期望接收句子列表）
        input_ids_list = []
        for word in words:
            # 确保word是一个句子，而不仅仅是一个词
            word_sentence = word  # 单个词也可以视为句子
            ids = vocab.encode_token([word_sentence])[0]  # 获取第一个句子的编码
            input_ids_list.append(ids)

        # Padding 函数（剪切或补 pad）
        def pad_to_length(seq, pad_id, target_len):
            if len(seq) >= target_len:
                return seq[:target_len]
            else:
                return seq + [pad_id] * (target_len - len(seq))

        # 执行 Padding 和 mask 构造
        input_ids_padded = []
        input_mask = []

        for ids in input_ids_list:
            padded = pad_to_length(ids, vocab.pad_token_id, seq_len)
            # 创建掩码：1表示真实token，0表示padding
            mask = [1 if i < len(ids) else 0 for i in range(seq_len)]
            input_ids_padded.append(padded)
            input_mask.append(mask)

        training_data = {
            'input_ids': input_ids_padded,          # List[List[int]]  shape: [N, seq_len]
            'input_mask': input_mask            
        }

        dataset = TextDataset(
            text_datasets=training_data,
            data_args=None,
            model_emb=self.model_emb
        )

        sampler = DistributedSampler(dataset)
        data_loader = DataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=4,
        )

        if loop:
            return self._infinite_loader(data_loader)
        else:
            return iter(data_loader)

    def _infinite_loader(self, data_loader):
        while True:
            yield from data_loader

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

    def _pad_to_max_len(self, input_ids, max_len, pad_token_id=0):
        B, L = input_ids.shape
        pad_len = max_len - L

        new_ids  = torch.full((B, max_len), pad_token_id, device=input_ids.device, dtype=input_ids.dtype)
        new_mask = torch.zeros((B, max_len), device=input_ids.device, dtype=torch.long)

        new_ids[:, :L] = input_ids  # 将原始input_ids复制到new_ids的前L个位置
        new_mask[:, :L] = 1  # input_mask: 1 = token, 0 = padding

        return new_ids, new_mask


    @torch.no_grad()
    def generate_from_conditioner(
        self,
        conditioner,
        num_samples: int = 1,
        condition_len: int = 0,
    ) -> List[str]:
        """Generate text samples conditioned on a WordNetConditioner or word list using DiffuSeq.
        
        This method implements the core functionality from sample_seq2seq.py.
        
        Args:
            conditioner: The WordNetConditioner instance or a list of words to condition on
            num_samples: Number of text samples to generate
            condition_len: Length of the condition (0 means unconditional)
            
        Returns:
            List of generated text samples
        """
        # 处理不同类型的conditioner
        # 如果是单词列表，使用SimpleEmbedder处理
        ## 先只用一个word
        conditioner = conditioner[0]                 # 只用第 1 个词
        data_valid  = self._load_data_text(conditioner)

        # ------------ 2. 构造待生成序列 -----------------

        cond  = next(data_valid)[1]             # 这里取到 {'input_ids': ..., 'input_mask': ...}
        # 确保输入数据是张量类型

        input_ids = cond['input_ids']
        input_mask = cond['input_mask']
            
        # 移动到指定设备
        input_ids_x = input_ids.to(self.device)   # (1, L0)
        x_start = self.model.get_embeds(input_ids_x)
        input_ids_mask = input_mask.to(self.device)   # (1, L0)

        noise = torch.randn_like(x_start)
        input_ids_mask = torch.broadcast_to(input_ids_mask.unsqueeze(dim=-1), x_start.shape).to(dist_util.dev())
        x_noised = torch.where(input_ids_mask == 0, x_start, noise)

        model_kwargs = {}
        self.use_ddim = True
        step_gap = self.diffusion_steps//self.step
        sample_fn = (
            self.diffusion.p_sample_loop if not self.use_ddim else self.diffusion.ddim_sample_loop
        )

        sample_shape = (x_start.shape[0], self.seq_len, self.hidden_dim)

        samples      = sample_fn(
                            self.model, 
                            sample_shape,
                            noise=x_noised,
                            clip_denoised=self.clip_denoised,
                            model_kwargs=model_kwargs,
                            top_p=self.top_p,
                            clamp_step=self.clamp_step,
                            clamp_first=True,
                            denoised_fn=partial(denoised_fn_round, self, self.model_emb),
                            mask=input_ids_mask, 
                            x_start=x_start,
                            gap=step_gap,
                        )
        sample  = samples[-1]

        # ------------ 5. 解码 -----------------
        logits = self.model.get_logits(sample)                # (B, gen_len, |V|)
        best   = logits.argmax(-1)                            # (B, gen_len)
        # 解码并清理文本
        texts = []
        for ids in best:
            # 跳过全是0或padding的序列
            if torch.all(ids == 0) or torch.all(ids == self.tokenizer.pad_token_id):
                texts.append("生成失败：全是填充token")
                continue
                
            # 使用decode_token解码，注意不要提前调用tolist()
            text = self.tokenizer.decode_token(ids)
            # 清理文本
            # text = self._clean_generated_text(text)
            texts.append(text)
            
        return texts

    def _clean_generated_text(self, text):
        """清理生成的文本，去除特殊标记和多余空格"""
        # 去除类似 [unused176] 的特殊标记
        text = re.sub(r'\[unused\d+\]', '', text)
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        return text


# ---------------------------------------------------------------------------
#  CLI for quick sanity test (not production‑grade)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick test for DiffusionTextGenerator")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--use_ddim", action="store_true")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.9)
    args = parser.parse_args()

    gen = DiffusionTextGenerator(model_path=args.model_path, use_ddim=args.use_ddim)
    vec = torch.randn(1, gen.config["hidden_dim"], device=gen.device)
    logger.info("Sampling %d texts …", args.num_samples)
    outs = gen.sample(vec, args.num_samples, args.max_len, args.temperature)
    for i, t in enumerate(outs, 1):
        print(f"[{i}] {t}")
