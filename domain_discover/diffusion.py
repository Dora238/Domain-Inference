"""Diffusion-based text generator wrapper using DiffuSeq."""

from typing import List, Optional, Dict, Any, Union, Tuple
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoTokenizer
from dataclasses import dataclass, field
import os
import sys
import numpy as np
from functools import partial
from pathlib import Path


# 确保 DiffuSeq 在 Python 路径中
def ensure_diffuseq_path():
    """确保 DiffuSeq 在 Python 路径中，如果不在则添加"""
    diffuseq_path = Path(__file__).parent.parent / "DiffuSeq"
    if not diffuseq_path.exists():
        raise ImportError(
            "DiffuSeq 仓库不存在。请先克隆 DiffuSeq 仓库: "
            "git clone https://github.com/Shark-NLP/DiffuSeq.git"
        )
    diffuseq_path_str = str(diffuseq_path)
    if diffuseq_path_str not in sys.path:
        sys.path.append(diffuseq_path_str)


# 导入 argparse
import argparse

# 导入 DiffuSeq 相关模块
try:
    ensure_diffuseq_path()
    from diffuseq.gaussian_diffusion import GaussianDiffusion, get_named_beta_schedule
    from diffuseq.rounding import denoised_fn_round
    from diffuseq.text_datasets import load_data_text
    from diffuseq.utils import dist_util
    from diffuseq.utils.nn import mean_flat
    # 导入 basic_utils
    sys.path.append(str(Path(__file__).parent.parent / "DiffuSeq"))
    from basic_utils import create_model_and_diffusion, load_tokenizer
except ImportError as e:
    print(f"导入 DiffuSeq 失败: {e}")
    print("请确保已克隆 DiffuSeq 仓库并安装其依赖")
    raise


@dataclass
class DiffusionTextGenerator:
    """基于 DiffuSeq 的文本生成器包装类。
    
    该类封装了预训练的 DiffuSeq 模型，提供了一个简化的接口，
    用于基于可学习提示向量的条件文本生成。
    
    Args:
        model_path: DiffuSeq 模型路径，可以是模型文件或模型目录
        device: 计算设备
        diffusion_steps: 扩散步数
        use_ddim: 是否使用 DDIM 采样（更快）
        clip_denoised: 是否裁剪去噪结果
    """
    model_path: str
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    diffusion_steps: int = 2000  # 默认值改为2000，与 QQP 模型匹配
    use_ddim: bool = True
    clip_denoised: bool = True
    ddim_steps: int = 200  # DDIM 采样步数，比完整扩散步数小
    
    # 运行时初始化的组件
    model: Optional[PreTrainedModel] = None
    tokenizer: Optional[PreTrainedTokenizer] = None
    diffusion: Optional[GaussianDiffusion] = None
    model_emb: Optional[nn.Embedding] = None
    config: Dict[str, Any] = field(default_factory=dict)
    model_dir: str = None  # 模型目录
    
    def __post_init__(self):
        """初始化 DiffuSeq 模型、分词器和扩散过程"""
        # 确保 DiffuSeq 在路径中
        ensure_diffuseq_path()
        
        # 处理模型路径
        self._process_model_path()
        
        # 加载配置
        self._load_config()
        
        # 创建模型和扩散过程
        self._create_model_and_diffusion()
        
        # 加载分词器
        self._load_tokenizer()
        
        # 创建词嵌入层（用于解码）
        self.model_emb = nn.Embedding(
            num_embeddings=self.tokenizer.vocab_size,
            embedding_dim=self.config["hidden_dim"],
            _weight=self.model.word_embedding.weight.clone()
        ).to(self.device).eval().requires_grad_(False)
        
        print(f"DiffusionTextGenerator 初始化完成，模型位于: {self.device}")
        
    def _process_model_path(self):
        """处理模型路径，确定模型目录和模型文件路径"""
        model_path = Path(self.model_path)
        
        # 如果是目录，则寻找 .pt 文件
        if model_path.is_dir():
            self.model_dir = str(model_path)
            # 寻找 ema 模型文件
            pt_files = list(model_path.glob("ema*.pt"))
            if not pt_files:
                raise FileNotFoundError(f"在目录 {self.model_dir} 中未找到 .pt 模型文件")
            self.model_path = str(pt_files[0])  # 使用第一个找到的 .pt 文件
        else:
            # 如果是文件，则使用其父目录作为模型目录
            if not model_path.exists():
                raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
            self.model_dir = str(model_path.parent)
        
        print(f"模型目录: {self.model_dir}")
        print(f"模型文件: {self.model_path}")
    
    def _load_config(self):
        """加载模型配置"""
        import json
        
        # 获取配置文件路径
        config_path = os.path.join(self.model_dir, "training_args.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # 设置批处理大小
        self.config["batch_size"] = 4  # 默认批处理大小
        
        # 确保扩散步数与配置一致
        if "diffusion_steps" in self.config:
            self.diffusion_steps = self.config["diffusion_steps"]
            
        # 打印主要配置信息
        print(f"模型隐藏维度: {self.config.get('hidden_dim', 'unknown')}")
        print(f"序列长度: {self.config.get('seq_len', 'unknown')}")
        print(f"扩散步数: {self.diffusion_steps}")
        print(f"噪声调度: {self.config.get('noise_schedule', 'unknown')}")
    
    def _create_model_and_diffusion(self):
        """创建模型和扩散过程"""
        # 创建模型和扩散过程
        self.model, self.diffusion = create_model_and_diffusion(**self.config)
        
        # 初始化 dist_util
        dist_util.setup_dist()
        
        # 加载模型权重
        self.model.load_state_dict(
            dist_util.load_state_dict(self.model_path, map_location="cpu")
        )
        
        # 设置模型为评估模式并移动到指定设备
        self.model.eval().requires_grad_(False).to(self.device)
        
        # 打印模型参数数量
        pytorch_total_params = sum(p.numel() for p in self.model.parameters())
        print(f"模型参数数量: {pytorch_total_params}")
    
    def _load_tokenizer(self):
        """加载分词器"""
        # 创建参数对象
        args = argparse.Namespace(**self.config)
        
        # 加载分词器
        self.tokenizer = load_tokenizer(args)
        
        # 检查分词器是否成功加载
        if self.tokenizer is None:
            raise ValueError("分词器加载失败")
        
        print(f"分词器加载成功，词汇表大小: {self.tokenizer.vocab_size}")
        
        # 确保分词器有必要的方法
        if not hasattr(self.tokenizer, 'decode'):
            print("警告: 分词器没有 decode 方法，尝试使用 decode_token")
            if not hasattr(self.tokenizer, 'decode_token'):
                raise AttributeError("分词器既没有 decode 也没有 decode_token 方法")
            # 添加 decode 方法
            self.tokenizer.decode = self.tokenizer.decode_token
    
    @torch.no_grad()
    def sample(
        self,
        prompt_vec: torch.Tensor,
        num_samples: int,
        max_len: int,
        temperature: float = 1.0
    ) -> List[str]:
        """基于提示向量生成文本样本。
        
        Args:
            prompt_vec: 条件向量 (1 x hidden_dim)
            num_samples: 生成的文本数量
            max_len: 最大序列长度
            temperature: 采样温度 (越高 = 越多样化)
            
        Returns:
            生成的文本字符串列表
        """
        # 扩展提示向量为批次大小
        batch_prompts = prompt_vec.expand(num_samples, -1).to(self.device)
        
        # 创建初始噪声
        noise = torch.randn(
            (num_samples, max_len, self.config["hidden_dim"]),
            device=self.device
        )
        
        # 设置模型参数
        model_kwargs = {"prompt_embedding": batch_prompts}
        
        # 选择采样函数
        sample_fn = self.diffusion.ddim_sample_loop if self.use_ddim else self.diffusion.p_sample_loop
        
        # 设置 top_p 采样参数
        top_p = None
        if temperature < 1.0:
            top_p = temperature  # 使用温度作为 top_p 值
        
        # 执行采样过程
        samples = sample_fn(
            model=self.model,
            shape=(num_samples, max_len, self.config["hidden_dim"]),
            noise=noise,
            clip_denoised=self.clip_denoised,
            denoised_fn=partial(denoised_fn_round, argparse.Namespace(**self.config), self.model_emb),
            model_kwargs=model_kwargs,
            top_p=top_p,
            progress=True
        )
        
        # 获取最终样本
        final_sample = samples[-1]
        
        # 获取模型输出的 logits
        logits = self.model.get_logits(final_sample)  # bsz, seqlen, vocab
        
        # 获取最可能的 token
        token_ids = torch.argmax(logits, dim=-1)
        
        # 解码为文本
        texts = []
        for seq in token_ids:
            tokens = self.tokenizer.decode(seq.tolist())
            texts.append(tokens)
        
        return texts


# 添加缺失的 argparse 导入
import argparse


if __name__ == "__main__":
    # 简单单元测试
    import argparse
    
    # 检查 DiffuSeq 是否已安装
    try:
        ensure_diffuseq_path()
        print("DiffuSeq 路径已添加到 Python 路径")
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)
    
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="测试 DiffusionTextGenerator")
    parser.add_argument("--model_path", type=str, required=True, help="DiffuSeq 模型路径")
    parser.add_argument("--use_ddim", action="store_true", help="使用 DDIM 采样")
    parser.add_argument("--num_samples", type=int, default=2, help="生成的样本数量")
    parser.add_argument("--max_len", type=int, default=50, help="最大序列长度")
    parser.add_argument("--temperature", type=float, default=0.9, help="采样温度")
    args = parser.parse_args()
    
    # 创建生成器
    generator = DiffusionTextGenerator(
        model_path=args.model_path,
        use_ddim=args.use_ddim
    )
    
    # 创建随机提示向量
    prompt_vec = torch.randn(1, generator.config["hidden_dim"]).to(generator.device)
    
    # 生成文本
    print(f"生成 {args.num_samples} 个文本样本...")
    texts = generator.sample(
        prompt_vec=prompt_vec,
        num_samples=args.num_samples,
        max_len=args.max_len,
        temperature=args.temperature
    )
    
    # 打印生成的文本
    print("生成的文本:")
    for i, text in enumerate(texts):
        print(f"{i+1}. {text}")
