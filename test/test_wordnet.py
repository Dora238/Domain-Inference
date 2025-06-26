"""Command-line interface for domain discovery pipeline."""

import argparse
import json
import os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer, GPT2Model

from wordnet_conditioner import WordNetConditioner
from diffusion import DiffusionTextGenerator
from classifier import Classifier
from optimizer import ExpansionDirectionOptimizer
from collections import OrderedDict

import sys

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
        "--classifier_name",
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

    classifier_names = ["j-hartmann/emotion-english-distilroberta-base", "s-nlp/roberta_toxicity_classifier",
                            "nlptown/bert-base-multilingual-uncased-sentiment","mrm8488/bert-tiny-finetuned-sms-spam-detection",
                            "skandavivek2/spam-classifier","wesleyacheng/sms-spam-classification-with-bert","jackhhao/jailbreak-classifier",
                            "lordofthejars/jailbreak-classifier","Necent/distilbert-base-uncased-detected-jailbreak","hallisky/sarcasm-classifier-gpt4-data"]
    for classifier_name in classifier_names:
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
        
        classifier = Classifier(model_name=classifier_name)
        
        # 获取模型的hidden_dim
        hidden_dim = generator.config.get("hidden_dim", 128)
        print(f"Using hidden_dim: {hidden_dim}")
        
        wordnet_conditioner = WordNetConditioner(hidden_dim=hidden_dim, init_method='wordnet', classifier=classifier, max_words=5000, min_words_per_category=20, visual=True).to(device)
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
