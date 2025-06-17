"""Command-line interface for domain discovery pipeline."""

import argparse
import json
import os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from wordnet_conditioner import WordNetConditioner
from diffusion import DiffusionTextGenerator
from blackbox import BlackBox
from pipeline import optimise_prompt_vector
from optimizer import optimize_x_start, calculate_diversity
from collections import OrderedDict
from generater import optimize_soft_prompt_dynamic, optimize_dimension
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
        default="j-hartmann/emotion-english-distilroberta-base",
        help="HuggingFace model name for black-box classifier,s-nlp/roberta_toxicity_classifier,j-hartmann/emotion-english-distilroberta-base, nlptown/bert-base-multilingual-uncased-sentiment"
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
    
    black_box = BlackBox(model_name=args.blackbox_model_name)
    
    # 获取模型的hidden_dim
    hidden_dim = generator.config.get("hidden_dim", 128)
    print(f"Using hidden_dim: {hidden_dim}")
    
    wordnet_conditioner = WordNetConditioner(hidden_dim=hidden_dim, init_method='wordnet', black_model=black_box, max_words=5000, min_words_per_category=20, load_from_wordnet=True).to(device)
    sorted_embeddings_words_dict = OrderedDict(sorted(wordnet_conditioner.embeddings_words_dict.items(), key=lambda x: int(x[0])))

    for label, words in sorted_embeddings_words_dict.items():
        print(f"\n🔺 Generating for label: {label+1} from wordnet: {words}")
        # sorted_embeddings_words_dict[label] = words
        z, pca, valid_words = optimize_dimension(label, words, black_box)
        best_z, samples = optimize_soft_prompt_dynamic(
            target_label   = label,
            black_box      = black_box,          # 你的分类器
            prefix_len     = 8,
            initial_words  = words,          # ← 新增入口
            # prompt_text    = " Please write a short English sentence:",
            sigma0         = 0.04,
            iterations     = 25,
            samples_per_eval = 15,
            temperature    = 0.7,
            top_p          = 0.8,
            mu             = 0.0,
        )
    # 生成示例文本
    for label, words in sorted_embeddings_words_dict.items():
        print(f"\n🔺 Generating for label: {label+1} from wordnet: {words}")
        args.target_label = label
        conditioner = words[0]  # 只用第一个词
        data_valid = generator._load_data_text(conditioner)
        cond = next(data_valid)[1]

        input_ids_x = cond['input_ids'].to(generator.device)  # shape: (1, L)
        input_ids_mask = cond['input_mask'].to(generator.device)  # shape: (1, L)

        x_start = generator.model.get_embeds(input_ids_x)  # shape: (1, L, D)

        # 生成初始样本
        initial_samples = generator.generate_from_conditioner(x_start, input_ids_x, input_ids_mask, num_samples=args.num_samples)
        print("\n初始样本:")
        for i, t in enumerate(initial_samples):
            print(f"[Sample {i+1}]: {t}")
        
        # 评估初始样本
        initial_labels = black_box.predict(initial_samples)
        initial_success_rate = sum(initial_label == args.target_label for initial_label in initial_labels) / len(initial_labels)
        initial_diversity = calculate_diversity(initial_samples)
        print(f"初始成功率: {initial_success_rate:.4f}, 初始多样性: {initial_diversity:.4f}")
        
        # 如果启用了x_start优化
        if args.optimize_x_start:
            print("\n开始优化x_start...")
            
            # 优化x_start
            optimized_x_start = optimize_x_start(
                initial_x_start=x_start,
                black_box=black_box,
                generator=generator,
                input_ids=input_ids_x,
                input_mask=input_ids_mask,
                num_samples=args.num_samples,
                target_label=args.target_label,
                eta=args.target_success_rate,
                max_iterations=args.max_iterations,
                learning_rate=args.learning_rate,
                diversity_weight=args.diversity_weight,
                gradient_method=args.gradient_method,
                verbose=True
            )
            
            # 使用优化后的x_start生成样本
            optimized_samples = generator.generate_from_conditioner(
                optimized_x_start, input_ids, input_mask, num_samples=10
            )
            
            # 评估优化后的样本
            optimized_labels = black_box.predict(optimized_samples)
            optimized_success_rate = sum(label==target_label for label in optimized_labels) / len(optimized_labels)
            optimized_diversity = calculate_diversity(optimized_samples)
            
            print("\n优化后的样本:")
            for i, (sample, label) in enumerate(zip(optimized_samples, optimized_labels)):
                print(f"[Sample {i+1}] {'✓' if label == 1 else '✗'}: {sample}")
        
        print(f"\n优化结果统计:")
        print(f"- 初始成功率: {initial_success_rate:.4f} -> 优化后成功率: {optimized_success_rate:.4f}")
        print(f"- 初始多样性: {initial_diversity:.4f} -> 优化后多样性: {optimized_diversity:.4f}")
        
        # 保存优化结果
        results_path = output_dir / "optimization_results.json"
        results = {
            "initial_success_rate": float(initial_success_rate),
            "initial_diversity": float(initial_diversity),
            "optimized_success_rate": float(optimized_success_rate),
            "optimized_diversity": float(optimized_diversity),
            "initial_samples": initial_samples,
            "initial_labels": initial_labels,
            "optimized_samples": optimized_samples,
            "optimized_labels": optimized_labels,
            "optimization_params": {
                "target_success_rate": args.target_success_rate,
                "diversity_weight": args.diversity_weight,
                "max_iterations": args.max_iterations,
                "learning_rate": args.learning_rate,
                "gradient_method": args.gradient_method,
                "num_samples": args.num_samples
            }
        }
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"优化结果已保存到: {results_path}")
    
    # 运行原有的优化（如果需要）
    if args.max_steps > 0 and 'wordnet_conditioner' in locals():
        print("\nStarting prompt vector optimization...")
        optimise_prompt_vector(
            generator=generator,
            bb=black_box,
            prompt_vec=wordnet_conditioner,
            target_label=args.target_label,
            eta=args.target_success_rate,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            output_dir=output_dir
        )
        print(f"\nOptimization complete. Results saved to: {output_dir}")


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
