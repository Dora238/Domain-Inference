"""
优化器模块 - 用于优化x_start以满足成功率和多样性要求

此模块提供了用于估计梯度和优化x_start的函数，以便生成既满足黑盒分类器成功率要求
又具有多样性的文本样本。
"""

import torch
import random
import numpy as np
from typing import List, Dict, Any, Optional, Union, Callable
import torch.optim as optim
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_distances
from transformers import AutoTokenizer, AutoModel, logging as transformers_logging
import torch.nn.functional as F
import warnings

# 抑制警告信息
warnings.filterwarnings('ignore')
transformers_logging.set_verbosity_error()  # 只显示错误信息，不显示警告


def calculate_bert_diversity(samples: List[str], model_name='bert-base-uncased') -> float:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    with torch.no_grad():
        inputs = tokenizer(samples, padding=True, truncation=True, return_tensors='pt')
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # 句子表示

    distances = cosine_distances(embeddings, embeddings)
    # 上三角去重（只保留非对角）
    upper_triangle = distances[np.triu_indices_from(distances, k=1)]
    return upper_triangle.mean()

def calculate_distinct_n(samples: List[str], n: int = 2) -> float:
    all_ngrams = set()
    total_ngrams = 0
    for sample in samples:
        tokens = sample.split()
        ngrams = zip(*[tokens[i:] for i in range(n)])
        ngrams = list(ngrams)
        total_ngrams += len(ngrams)
        all_ngrams.update(ngrams)
    return len(all_ngrams) / total_ngrams if total_ngrams > 0 else 0.0

def calculate_self_bleu(samples: List[str], max_n: int = 4) -> float:
    """
    计算 Self-BLEU。每个句子用其他句子作为参考，计算 BLEU 分数的平均值。
    BLEU 分数越低表示生成文本越多样。
    """
    if len(samples) <= 1:
        return 0.0

    # 设置权重
    weights = {
        1: (1.0, 0, 0, 0),
        2: (0.5, 0.5, 0, 0),
        3: (1/3, 1/3, 1/3, 0),
        4: (0.25, 0.25, 0.25, 0.25)
    }[max_n]

    smoothing = SmoothingFunction().method1
    scores = []

    for i, hypothesis in enumerate(samples):
        references = samples[:i] + samples[i+1:]
        references = [word_tokenize(ref) for ref in references]
        hypothesis_tokens = word_tokenize(hypothesis)
        score = sentence_bleu(references, hypothesis_tokens, weights=weights, smoothing_function=smoothing)
        scores.append(score)

    return float(np.mean(scores))

def calculate_jaccard_diversity(samples_text: List[str]) -> float:
    """
    计算文本样本集合的多样性
    
    Args:
        samples_text: 文本样本列表
        
    Returns:
        多样性得分，值越高表示多样性越大
    """
    if len(samples_text) <= 1:
        return 0.0
    
    # 对每个样本进行分词
    tokenized_samples = [set(sample.split()) for sample in samples_text]
    
    # 计算平均词汇重叠率
    overlap_ratios = []
    for i in range(len(tokenized_samples)):
        for j in range(i+1, len(tokenized_samples)):
            set_i = tokenized_samples[i]
            set_j = tokenized_samples[j]
            if not set_i or not set_j:
                continue
                
            overlap = len(set_i.intersection(set_j))
            union = len(set_i.union(set_j))
            
            # Jaccard距离 = 1 - Jaccard相似度
            if union > 0:
                overlap_ratios.append(1 - overlap / union)
    
    # 返回平均Jaccard距离作为多样性度量
    return sum(overlap_ratios) / len(overlap_ratios) if overlap_ratios else 0.0


def calculate_diversity(samples_text: List[str]) -> float:
    weights = {
        'jaccard': 0.3,
        'self_bleu': 0.2,
        'bert': 0.3,
        'distinct': 0.2
    }
    jaccard_diversity =calculate_jaccard_diversity(samples_text)
    self_bleu = calculate_self_bleu(samples_text)
    bert_diversity = calculate_bert_diversity(samples_text)
    distinct_n = calculate_distinct_n(samples_text)
    diversity = weights['jaccard'] * jaccard_diversity + weights['self_bleu'] * (1-self_bleu) + weights['bert'] * bert_diversity + weights['distinct'] * distinct_n
    return diversity

def compute_blackbox_score(x_embed, black_box, generator, input_ids, input_mask, target_label, num_samples, diversity_weight):
    with torch.no_grad():
        texts = generator.generate_from_conditioner(x_embed, input_ids, input_mask, num_samples)
        labels = black_box.predict(texts)
        succ = sum(label == target_label for label in labels) / num_samples
        div = calculate_diversity(texts)
        return succ + diversity_weight * div



def estimate_gradient_with_nes_prior(x, black_box, generator, input_ids, input_mask, target_label,
                                     num_samples=20, sigma=0.05, alpha=0.7, k=30, 
                                     diversity_weight=0.5, prior_vector=None):
    grad_est = torch.zeros_like(x)

    for _ in range(k):
        noise = torch.randn_like(x)  # [1, d]
        if prior_vector is not None:
            prior_vector = F.normalize(prior_vector, dim=-1)
            noise = alpha * prior_vector + (1 - alpha**2)**0.5 * noise

        u = F.normalize(noise, dim=-1)
        x_pos = x + sigma * u
        x_neg = x - sigma * u

        score_pos = compute_blackbox_score(x_pos, black_box, generator, input_ids, input_mask, target_label, num_samples, diversity_weight)
        score_neg = compute_blackbox_score(x_neg, black_box, generator, input_ids, input_mask, target_label, num_samples, diversity_weight)

        g = (score_pos - score_neg) / (2 * sigma)
        grad_est += g * u

    grad_est /= k
    return F.normalize(grad_est, dim=-1)  # or apply clip



def estimate_gradient_with_diversity(x_start, black_box, generator, input_ids, input_mask, 
                                    num_samples, diversity_weight=0.5, num_coords=20, epsilon=1e-4):
    """
    估计同时考虑成功率和多样性的梯度
    
    Args:
        x_start: 当前嵌入向量 (1, hidden_dim)
        black_box: 黑盒分类器
        generator: 文本生成器
        input_ids: 输入ID
        input_mask: 输入掩码
        num_samples: 每次评估生成的样本数
        diversity_weight: 多样性权重
        num_coords: 每次扰动的维度数量
        epsilon: 扰动大小
        
    Returns:
        估计的梯度向量 (1, hidden_dim)
    """
    gradient = torch.zeros_like(x_start)
    
    # 计算当前x_start的评分
    with torch.no_grad():
        samples_text = generator.generate_from_conditioner(
            x_start, input_ids, input_mask, num_samples=num_samples
        )
        labels = black_box.predict(samples_text)
        original_success_rate = sum(label == target_label for label in labels) / num_samples
        original_diversity = calculate_diversity(samples_text)
        original_score = original_success_rate + diversity_weight * original_diversity
    
    # 随机选择num_coords个维度进行扰动
    all_dims = list(range(x_start.shape[1]))
    selected_dims = random.sample(all_dims, min(num_coords, len(all_dims)))
    
    for i in selected_dims:
        # 创建扰动向量
        perturb = torch.zeros_like(x_start)
        perturb[0, i] = epsilon
        
        # 正向扰动
        with torch.no_grad():
            x_plus = x_start + perturb
            samples_text = generator.generate_from_conditioner(
                x_plus, input_ids, input_mask, num_samples=num_samples
            )
            labels = black_box.predict(samples_text)
            plus_success_rate = sum(labels) / num_samples
            plus_diversity = calculate_diversity(samples_text)
            plus_score = plus_success_rate + diversity_weight * plus_diversity
        
        # 计算该维度的梯度
        gradient[0, i] = (plus_score - original_score) / epsilon
    
    return gradient




def optimize_x_start(initial_x_start, black_box, generator, input_ids, input_mask, 
                    num_samples, eta,target_label, max_iterations=50, learning_rate=0.01, 
                    diversity_weight=0.3, gradient_method="finite_diff", verbose=True):
    """
    优化x_start以满足成功率和多样性要求
    
    Args:
        initial_x_start: 初始嵌入向量 (1, hidden_dim)
        black_box: 黑盒分类器
        generator: 文本生成器
        input_ids: 输入ID
        input_mask: 输入掩码
        num_samples: 每次评估生成的样本数
        eta: 目标成功率阈值
        max_iterations: 最大迭代次数
        learning_rate: 学习率
        diversity_weight: 多样性权重
        gradient_method: 梯度估计方法，可选值："finite_diff", "random", "diversity", "batch"
        verbose: 是否打印详细信息
        
    Returns:
        优化后的x_start
    """
    x_start = initial_x_start.clone()
    
    # 选择梯度估计方法
    if  gradient_method == "diversity":
        gradient_estimator = lambda x,target_label: estimate_gradient_with_diversity(
            x, black_box, generator, input_ids, input_mask, num_samples,target_label, 
            diversity_weight=diversity_weight
        )
    elif gradient_method == "nes_prior":
        gradient_estimator = lambda x,target_label: estimate_gradient_with_nes_prior(
            x, black_box, generator, input_ids, input_mask, target_label,
            num_samples=num_samples, diversity_weight=diversity_weight
        )
    else:
        raise ValueError(f"未知的梯度估计方法: {gradient_method}")
    
    # 使用Adam优化器
    x_start_param = torch.nn.Parameter(x_start)
    optimizer = optim.Adam([x_start_param], lr=learning_rate)
    
    best_x_start = x_start.clone()
    best_score = -float('inf')
    
    # 创建进度条
    iterator = tqdm(range(max_iterations)) if verbose else range(max_iterations)
    
    for iteration in iterator:
        # 生成样本
        with torch.no_grad():
            samples_text = generator.generate_from_conditioner(
                x_start_param, input_ids, input_mask, num_samples=num_samples
            )
            
            # 评估样本
            labels = black_box.predict(samples_text)
            success_rate = sum(label==target_label for label in labels) / num_samples
            diversity = calculate_diversity(samples_text)
            
            # 计算综合得分
            score = success_rate + diversity_weight * diversity
            
            if verbose:
                print(f"迭代 {iteration}: 成功率 = {success_rate:.4f}, 多样性 = {diversity:.4f}, 得分 = {score:.4f}")
            
            # 更新最佳结果
            if score > best_score:
                best_score = score
                best_x_start = x_start_param.data.clone()
            
            # 检查是否达到目标
            if success_rate >= eta:
                if verbose:
                    print(f"达到目标成功率 {eta}，提前结束优化")
                return x_start_param.data
        
        # 估计梯度
        gradient = gradient_estimator(x_start_param.data,target_label)
        
        # 更新x_start
        optimizer.zero_grad()
        # 使用负梯度，因为我们要最大化目标函数
        x_start_param.grad = -gradient
        optimizer.step()
    
    if verbose:
        print(f"优化完成，返回最佳结果 (得分: {best_score:.4f})")
    
    # 返回最佳结果
    return best_x_start


if __name__ == "__main__":
    # 简单的单元测试
    print("优化器模块加载成功")
