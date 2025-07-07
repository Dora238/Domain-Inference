"""测试脚本：比较MultiExpansion原本的扩展方向与随机生成的正交向量对alpha的影响"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys
from tqdm import tqdm
import json
from collections import OrderedDict
from config import PROJECT_ROOT, SRC_PATH

sys.path.append(str(PROJECT_ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# 导入项目模块
from domain_infer.wordnet_init import WordNetConditioner
from domain_infer.classifier import Classifier
from domain_infer.optimizer import ExpansionDirectionOptimizer, MultiExpansion
from domain_infer.generater import T5Generator
import torch.nn as nn

# 创建一个新的扩展模块，覆盖forward方法以返回随机正交向量
class RandomExpansion(nn.Module):
    def __init__(self, directions):
        super().__init__()
        self.directions = directions
        
    def forward(self, Z):
        return self.directions

def create_argparser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(description="测试不同扩展方向生成方法对alpha的影响")
    
    # 基本参数
    parser.add_argument("--classifier_name", type=str, 
                        default="j-hartmann/emotion-english-distilroberta-base", 
                        help="分类器模型名称")
    parser.add_argument("--output_dir", type=str, 
                        default=f'{PROJECT_ROOT}/output/direction_test2', 
                        help="结果保存目录")
    parser.add_argument("--num_directions", type=int, default=50,
                        help="生成的扩展方向数量")
    parser.add_argument("--target_success_rate", type=float, default=0.5,
                        help="目标成功率阈值")
    parser.add_argument("--alpha_max", type=float, default=8.0,
                        help="二分法搜索的alpha的最大值")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    args = parser.parse_args()
    args.num_directions = int(args.num_directions // args.target_success_rate)
    
    return args


def load_generator():
    """加载T5生成器"""
    cfg = {
        "model_name": "humarin/chatgpt_paraphraser_on_T5_base",
        "peft_model_path": "Dora238/prefix-paraphraser",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    gen = T5Generator(**cfg)
    return gen


def generate_random_orthogonal_vectors(shape, device):
    """
    生成随机正交向量
    
    Args:
        shape: 形状 (n, L, D)，与MultiExpansion输出相同
        device: 计算设备
        
    Returns:
        正交化的随机向量，形状为 (n, L, D)
    """
    n, L, D = shape
    # 生成随机向量
    vectors = torch.randn(n, L, D, device=device)
    
    # 对每个位置L进行Gram-Schmidt正交化
    for l in range(L):
        # 提取当前位置的所有向量
        current_vectors = vectors[:, l, :]  # shape: (n, D)
        
        # Gram-Schmidt正交化
        for i in range(n):
            # 对当前向量进行归一化
            current_vectors[i] = current_vectors[i] / torch.norm(current_vectors[i])
            
            # 对后续向量进行正交化
            if i < n - 1:
                for j in range(i + 1, n):
                    # 计算投影
                    proj = torch.dot(current_vectors[j], current_vectors[i]) * current_vectors[i]
                    # 减去投影
                    current_vectors[j] = current_vectors[j] - proj
        
        # 将正交化的向量放回原始张量
        vectors[:, l, :] = current_vectors
    
    # 归一化
    vectors = torch.nn.functional.normalize(vectors, p=2, dim=-1)
    return vectors


def test_direction_methods(args, classifier, t5_generator):
    """
    测试不同扩展方向生成方法对alpha的影响
    
    Args:
        args: 命令行参数
        classifier: 分类器
        t5_generator: T5生成器
    
    Returns:
        结果字典
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化WordNet条件生成器获取初始句子
    wordnet_conditioner = WordNetConditioner(
        init_method='wordnet', 
        classifier=classifier,
        max_words=5000,
        min_words_per_category=20,
        t5_generator=t5_generator,
        initial_sentence_from_wordnet=False
    ).to(device)
    
    word_dict = wordnet_conditioner.word_dict
    word_dict = OrderedDict(sorted(word_dict.items()))
    
    # 结果存储
    results = {}
    
    # 对每个标签进行测试
    i = 0
    max_outer_steps = 10
    max_binary_steps = 15
    for label, entry in tqdm(word_dict.items(), desc="测试标签"):
        # i += 1
        # if i == 2:
        #     break
        label_results = {}
        
        # 获取初始句子和嵌入
        initial_sentence = entry['center_sentence']
        print(f"标签: {label}, 句子: {initial_sentence}")
        
        initial_embedding, attention = t5_generator.encode(initial_sentence)
        Z = initial_embedding.squeeze(0)  # shape: [L, D]
        
        # test random expansion
        print("测试随机正交向量方法...")
        random_directions = generate_random_orthogonal_vectors((args.num_directions, Z.shape[0], Z.shape[1]), device)
        
        random_expansion = RandomExpansion(random_directions).to(device)
        optimizer_random = ExpansionDirectionOptimizer(
            decoder=t5_generator,
            classifier=classifier,
            expansion_module=random_expansion,
            eta=args.target_success_rate,
            alpha_max=args.alpha_max,
            max_binary_steps=max_binary_steps,
        )
        alpha_random, embedding_random, direction_random, successful_mask_random = optimizer_random.optimise(
            Z, target_label=int(label),max_outer_steps=max_outer_steps
        )
        
        # test nn expansion
        nn_expansion_module = MultiExpansion(num_directions=args.num_directions).to(device)
        
        optimizer_nn = ExpansionDirectionOptimizer(
            decoder=t5_generator,
            classifier=classifier,
            expansion_module=nn_expansion_module,
            eta=args.target_success_rate,
            alpha_max=args.alpha_max,
            max_binary_steps=max_binary_steps,
        )
        
        # 测试原始MultiExpansion方法
        print("测试原始MultiExpansion方法...")
        alpha_nn, embedding_nn, direction_nn, successful_mask = optimizer_nn.optimise(
            Z, target_label=int(label), max_outer_steps=max_outer_steps
        )
        
        # 生成文本进行比较
        text_nn = t5_generator.generate_from_hidden_state(embedding_nn.unsqueeze(0))
        text_random = t5_generator.generate_from_hidden_state(embedding_random.unsqueeze(0))
        
        # compare cos sim
        cos_sim_nn = torch.nn.functional.cosine_similarity(embedding_nn, Z, dim=1)
        cos_sim_random = torch.nn.functional.cosine_similarity(embedding_random, Z, dim=1)
        cos_sim_generator = torch.nn.functional.cosine_similarity(embedding_nn, embedding_random, dim=1)
        print(f"cos sim nn: {cos_sim_nn}")
        print(f"cos sim random: {cos_sim_random}")
        print(f"cos sim generator: {cos_sim_generator}")
        
        embedding_nn_success = embedding_nn[successful_mask]
        direction_nn_success = direction_nn[successful_mask]
        embedding_random_success = embedding_random[successful_mask_random]
        direction_random_success = direction_random[successful_mask_random]
        
        # save embedding and direction
        torch.save(Z, args.output_dir / f"embedding_init_{label}.pt")
        torch.save(embedding_nn_success, args.output_dir / f"embedding_nn_success_{label}.pt")
        torch.save(direction_nn_success, args.output_dir / f"direction_nn_success_{label}.pt")
        torch.save(embedding_random_success, args.output_dir / f"embedding_random_success_{label}.pt")
        torch.save(direction_random_success, args.output_dir / f"direction_random_success_{label}.pt")
        
        # 存储结果
        label_results = {
            "initial_sentence": initial_sentence,
            "nn": {
                "alpha": float(alpha_nn),
                "text": text_nn
            },
            "random": {
                "alpha": float(alpha_random),
                "text": text_random
            }
        }
        
        results[label] = label_results

        # 打印比较结果
        print(f"标签 {label} 的比较结果:")
        print(f"  nn alpha: {alpha_nn}")
        print(f"  random alpha: {alpha_random}")
        print(f"  nn text: {text_nn}")
        print(f"  random text: {text_random}")
        print("-" * 50)
    
    return results


def visualize_results(results, output_dir):
    """
    Visualize test results
    
    Args:
        results: Test results dictionary
        output_dir: Output directory
    """
    labels = list(results.keys())
    nn_alphas = [results[label]["nn"]["alpha"] for label in labels]
    random_alphas = [results[label]["random"]["alpha"] for label in labels]
    
    # Draw alpha comparison chart
    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    index = np.arange(len(labels))
    
    plt.bar(index, nn_alphas, bar_width, label='MultiExpansion')
    plt.bar(index + bar_width, random_alphas, bar_width, label='Random Orthogonal')
    
    plt.xlabel('Labels')
    plt.ylabel('Alpha Value')
    plt.title('Alpha Value Comparison of Different Expansion Direction Methods')
    plt.xticks(index + bar_width / 2, labels)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(output_dir / "alpha_comparison.png")
    plt.close()
    
    # Calculate average values and differences
    avg_nn = np.mean(nn_alphas)
    avg_random = np.mean(random_alphas)
    
    with open(output_dir / "summary.txt", "w") as f:
        f.write(f"MultiExpansion Average Alpha: {avg_nn}\n")
        f.write(f"Random Orthogonal Vectors Average Alpha: {avg_random}\n")
        f.write(f"Difference: {avg_nn - avg_random}\n")


def main():
    """主函数"""
    # 解析命令行参数
    args = create_argparser()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 创建输出目录
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("正在初始化组件...")
    
    # 初始化组件
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = Classifier(model_name=args.classifier_name)
    t5_generator = load_generator()
    
    # 运行测试
    results = test_direction_methods(args, classifier, t5_generator)
    
    # 保存结果
    with open(args.output_dir / "direction_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # 可视化结果
    visualize_results(results, args.output_dir)
    
    print("测试完成！")


if __name__ == "__main__":
    main()