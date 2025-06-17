"""Command-line interface for domain discovery pipeline."""

import argparse
import json
import os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer, GPT2Model

from wordnet_conditioner import WordNetConditioner
from diffusion import DiffusionTextGenerator
from blackbox import BlackBox
from pipeline import optimise_prompt_vector
from optimizer import optimize_x_start, calculate_diversity
from collections import OrderedDict

# Import DiffuSeq utilities
import sys

# 确保DiffuSeq在路径中
def setup_diffuseq_path():
    """确保DiffuSeq路径正确添加到sys.path中"""
    # 尝试多种可能的路径
    possible_paths = [
        Path(__file__).parent.parent / 'DiffuSeq',
        Path(__file__).parent.parent / 'DiffuSeq' / 'DiffuSeq',
        Path('/home/dora/Domain-Inference/DiffuSeq'),
        Path('/home/dora/Domain-Inference/DiffuSeq/DiffuSeq')
    ]
    
    for path in possible_paths:
        if (path / 'diffuseq').exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
            return True
            
    return False

# 设置路径
if not setup_diffuseq_path():
    print("警告：无法找到DiffuSeq路径，请确保DiffuSeq已正确安装")

# 导入DiffuSeq工具
try:
    from basic_utils import add_dict_to_argparser, load_defaults_config
except ImportError:
    print("错误：无法导入DiffuSeq工具，请检查DiffuSeq安装路径")
    
    # 提供简单的替代函数，以防导入失败
    def add_dict_to_argparser(parser, default_dict):
        for k, v in default_dict.items():
            v_type = type(v)
            parser.add_argument(f"--{k}", default=v, type=v_type)
            
    def load_defaults_config():
        return {"hidden_dim": 128, "seq_len": 128}


def create_argparser():
    # 基础参数
    defaults = dict(
        model_path='/home/dora/Domain-Inference/model/diffuseq_qqp_h128_lr0.0001_t2000_sqrt_lossaware_seed102_test_ori20221113-20:27:29',
        step=0,
        out_dir='output',
        top_p=0.9,
        target_label=0,
        eta=0.85,
        max_steps=8000,
        batch_size=1
    )
    
    # 解码特定参数
    decode_defaults = dict(
        split='valid',
        clamp_step=0,
        seed2=105,
        clip_denoised=True,
        use_ddim=True,
        ddim_steps=200,
        condition_len=5
    )
    
    # 加载DiffuSeq默认配置
    try:
        defaults.update(load_defaults_config())
    except Exception as e:
        print(f"Warning: Could not load DiffuSeq defaults: {e}")
    
    # 更新解码参数
    defaults.update(decode_defaults)
    
    # 创建解析器
    parser = argparse.ArgumentParser(description="Domain Discovery Pipeline")
    
    # 添加子解析器
    add_dict_to_argparser(parser, defaults)
    
    # 添加基本参数
    parser.batch_size = 1
    parser.add_argument(
        "--blackbox_model_name",
        type=str,
        # required=True,
        default="s-nlp/roberta_toxicity_classifier",
        help="HuggingFace model name for black-box classifier,s-nlp/roberta_toxicity_classifier,j-hartmann/emotion-english-distilroberta-base"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        # required=True,
        default='/home/dora/Domain-Inference/domain_discover/output',
        help="Directory to save results"
    )
    
    # 添加x_start优化相关参数
    parser.add_argument(
        "--optimize_x_start", 
        action="store_true",
        default=True,
        help="是否优化x_start以满足成功率和多样性要求"
    )
    parser.add_argument(
        "--target_success_rate", 
        type=float, 
        default=0.8,
        help="目标成功率阈值，范围[0,1]，表示black_box返回1的比例"
    )
    parser.add_argument(
        "--diversity_weight", 
        type=float, 
        default=0.3,
        help="多样性权重，控制优化过程中多样性的重要程度"
    )
    parser.add_argument(
        "--max_iterations", 
        type=int, 
        default=50,
        help="x_start优化的最大迭代次数"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=0.01,
        help="x_start优化的学习率"
    )
    parser.add_argument(
        "--gradient_method", 
        type=str,
        choices=["finite_diff", "random", "diversity", "batch"],
        default="nes_prior",
        help="梯度估计方法: finite_diff(有限差分法), random(随机坐标下降), diversity(考虑多样性), batch(批处理)"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=10,
        help="每次评估生成的样本数量"
    )
    return parser


def sample_seq2seq_command(args):
    """处理 seq2seq 采样子命令"""
    from seq2seq_sampler import sample_seq2seq
    output_path = sample_seq2seq(args)
    print(f"采样完成，结果保存在: {output_path}")

def run_domain_discovery(args):
    """运行域发现管道"""
    # Setup device

    blackbox_model_names = ["j-hartmann/emotion-english-distilroberta-base", "s-nlp/roberta_toxicity_classifier",
                            "nlptown/bert-base-multilingual-uncased-sentiment","mrm8488/bert-tiny-finetuned-sms-spam-detection",
                            "skandavivek2/spam-classifier","wesleyacheng/sms-spam-classification-with-bert","jackhhao/jailbreak-classifier",
                            "lordofthejars/jailbreak-classifier","Necent/distilbert-base-uncased-detected-jailbreak","hallisky/sarcasm-classifier-gpt4-data"]
    for blackbox_model_name in blackbox_model_names:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
    
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
        # Initialize components
        generator = DiffusionTextGenerator(
            model_path=args.model_path,
            device=device,
            use_ddim=args.use_ddim,
            clip_denoised=args.clip_denoised,
            ddim_steps=args.ddim_steps,
            top_p=args.top_p,
            clamp_step=args.clamp_step,
            batch_size=args.batch_size,
            seq_len=args.seq_len if hasattr(args, 'seq_len') else 128,
            split=args.split,
            output_dir=args.output_dir,
            world_size=1,
            rank=0
        )
        
        black_box = BlackBox(model_name=blackbox_model_name)
        
        # 获取模型的hidden_dim
        hidden_dim = generator.config.get("hidden_dim", 128)
        print(f"Using hidden_dim: {hidden_dim}")
        
        wordnet_conditioner = WordNetConditioner(hidden_dim=hidden_dim, init_method='wordnet', black_model=black_box, max_words=5000, min_words_per_category=20, visual=True).to(device)
        # for label, words in wordnet_conditioner.word_dict.items():
        #     print(f"Label: {label}")
        #     print(f"Words: {words}")
            


def main():
    parser = create_argparser()
    args = parser.parse_args()
    
    # 根据子命令执行相应的功能
    if hasattr(args, 'command') and args.command == "seq2seq":
        sample_seq2seq_command(args)
    else:
        # 默认运行域发现管道
        # 如果没有指定命令或命令是"run"
        run_domain_discovery(args)


if __name__ == "__main__":
    main()
