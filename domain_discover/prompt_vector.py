"""Learnable prompt vector for domain discovery."""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Dict, Tuple, Set, Union
from nltk.corpus import wordnet as wn
import nltk
import random
from collections import defaultdict
from pathlib import Path
from blackbox import BlackBox
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

class SimpleEmbedder:
    """简单的文本嵌入模型，生成128维嵌入向量
    
    这个类使用预训练的Sentence-BERT模型生成嵌入向量，
    并通过线性层将其投影到128维，以与diffusion模型兼容。
    
    Args:
        target_dim: 目标嵌入维度，默认为128
        use_cache: 是否缓存嵌入结果
    """
    
    def __init__(self, target_dim: int = 128, use_cache: bool = True):
        self.target_dim = target_dim
        self.use_cache = use_cache
        self.cache = {}
        
        # 延迟加载模型，仅在需要时加载
        self._model = None
        self._projection = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _load_model(self):
        """加载预训练的Sentence-BERT模型"""
        # 使用较小的模型以提高效率
        self._model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        # 创建投影层
        input_dim = self._model.get_sentence_embedding_dimension()
        self._projection = nn.Linear(input_dim, self.target_dim).to(self.device)
        # 随机初始化投影层
        nn.init.xavier_uniform_(self._projection.weight)
        nn.init.zeros_(self._projection.bias)
        print(f"加载了Sentence-BERT模型并创建了从{input_dim}到{self.target_dim}的投影层")

    def encode(self, text: Union[str, List[str]]) -> torch.Tensor:
        """将文本转换为指定维度的嵌入向量
        
        Args:
            text: 输入文本或文本列表
            
        Returns:
            形状为(batch_size, target_dim)的嵌入向量
        """
        # 如果是单个字符串，转换为列表
        if isinstance(text, str):
            texts = [text]
            single_input = True
        else:
            texts = text
            single_input = False
        
        # 检查缓存
        if self.use_cache:
            # 尝试从缓存中获取嵌入
            cached_embeddings = []
            texts_to_encode = []
            indices = []
            
            for i, t in enumerate(texts):
                if t in self.cache:
                    cached_embeddings.append(self.cache[t])
                else:
                    texts_to_encode.append(t)
                    indices.append(i)
            
            # 如果所有文本都在缓存中
            if len(texts_to_encode) == 0:
                embeddings = torch.stack(cached_embeddings)
                if single_input:
                    return embeddings[0].unsqueeze(0)  # 添加batch维度
                return embeddings
        else:
            texts_to_encode = texts
            indices = list(range(len(texts)))
        
        # 延迟加载模型
        if self._model is None:
            self._load_model()
        
        # 编码文本
        with torch.no_grad():
            # 使用Sentence-BERT获取嵌入
            embeddings = self._model.encode(texts_to_encode, convert_to_tensor=True)
            # 确保embeddings在正确的设备上
            embeddings = embeddings.to(self.device)
            # 投影到目标维度
            embeddings = self._projection(embeddings)
        
        # 更新缓存
        if self.use_cache:
            for t, emb in zip(texts_to_encode, embeddings):
                self.cache[t] = emb
            
            # 将新编码的嵌入与缓存的嵌入合并
            all_embeddings = [None] * len(texts)
            for i, emb in enumerate(cached_embeddings):
                # 确保缓存的嵌入也在正确的设备上
                all_embeddings[i] = emb.to(self.device)
            for i, idx in enumerate(indices):
                all_embeddings[idx] = embeddings[i]
            
            embeddings = torch.stack(all_embeddings)
        
        # 如果是单个输入，返回单个嵌入
        if single_input:
            return embeddings[0].unsqueeze(0)  # 添加batch维度
        
        return embeddings


class PromptVector(nn.Module):
    """Learnable prompt vector for steering text generation.
    
    This module wraps a single learnable embedding vector that will be
    optimized to guide the diffusion model toward generating text with
    desired properties.
    
    Args:
        hidden_dim: Dimension of the embedding vector (matches model)
        init_scale: Scale for random initialization
        init_method: 初始化方法，可选 'random'(随机初始化) 或 'wordnet'(基于WordNet初始化)
        embedding_model: 用于将词汇转换为嵌入的模型（如果init_method='wordnet'）
        max_words: WordNet遍历的最大词汇数
        min_words_per_category: 每个类别最少保留的词汇数量
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,  # 修改为与 diffusion 模型匹配的维度
        init_scale: float = 0.02,
        init_method: str = 'wordnet',
        embedding_model = None,
        max_words: int = 5000,
        min_words_per_category: int = 20,
        black_model = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.init_method = init_method
        self.black_model = black_model
        
        if init_method == 'random':
            # 随机初始化
            self.embedding = nn.Parameter(
                torch.randn(1, hidden_dim) * init_scale
            )
        elif init_method == 'wordnet':
            # 基于WordNet初始化
            self.embedding_model = SimpleEmbedder()
            self.embeddings_dict, self.embeddings_words_dict = self.initialize_from_wordnet(black_model)
    
    def clone_detached(self) -> torch.Tensor:
        """Return a detached copy of the prompt vector.
        
        Returns:
            Tensor of shape (1, hidden_dim), detached from computation graph
        """
        return self.embedding.detach().clone()
    
    def initialize_from_wordnet(self, black_model = None) -> None:
        """从WordNet的词汇初始化prompt vector
        
        如果提供了black_model和target_label，则查找黑盒模型输出为指定标签的词汇嵌入
        否则通过对特定语义类别的词汇进行嵌入并平均，来初始化prompt vector
        
        Args:
            category: 语义类别名称，如果为None则随机选择一个类别
            sample_size: 从类别中采样的词汇数量
            black_model: 黑盒模型，用于预测词汇的标签
            target_label: 目标标签，如果指定则查找黑盒模型输出为该标签的词汇
        """
        
        # 获取WordNet词汇
        word_dict = traverse_wordnet(
            black_model=black_model,
            max_words=5000,
            min_words_per_label=20
            )
            
        # 如果找到了目标标签的词汇
        embeddings_dict = {}
        embeddings_words_dict = {}
        for label, words in word_dict.items():
            print(f"找到 {len(words)} 个标签为 {label} 的词汇")
            
            embeddings = []
            word_list = []

            for word in words:
                with torch.no_grad():
                    embedding = self.embedding_model.encode(word)
                    if isinstance(embedding, torch.Tensor):
                        if embedding.numel() == self.hidden_dim:  # 确保维度匹配
                            embeddings.append(embedding.reshape(1, -1))
                            word_list.append(word)

            if len(embeddings) < 2:
                print(f"⚠️ 类别 {label} 中有效嵌入词汇数量不足，跳过聚类")
                continue

            embeddings = torch.cat(embeddings, dim=0).cpu().numpy()

            # 设置聚类个数（你可以调整）
            n_clusters = min(5, len(embeddings))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(embeddings)

            print(f"标签 {label} 的聚类结果（共 {n_clusters} 类）：")
            for cluster_id in range(n_clusters):
                indices = [i for i, pred in enumerate(kmeans.labels_) if pred == cluster_id]
                cluster_words = [word_list[i] for i in indices]
                print(f"  🔸 Cluster {cluster_id}: {', '.join(cluster_words[:5])}...")
                embeddings_dict[label] = embeddings[indices]
                embeddings_words_dict[label] = cluster_words
        ## 返回每个label对应的聚类的embeddings
        return embeddings_dict, embeddings_words_dict

def traverse_wordnet(black_model=None, max_words=5000, min_words_per_label=20):
    """遍历WordNet词汇，找到黑盒模型预测为目标标签的词汇
    
    Args:
        black_model: 黑盒模型，用于预测词汇的标签
        max_words: 最大处理词汇数量
        min_words_per_label: 每个标签最少保留的词汇数量
        
    Returns:
        如果target_label为None，返回{label: [words]}的字典
        如果target_label不为None，返回符合目标标签的词汇列表
    """
    
    # 获取所有同义词集
    all_synsets = list(wn.all_synsets())
    print(f"WordNet中共有 {len(all_synsets)} 个同义词集")
    
    label_to_words = defaultdict(set)
    count = 0

    for synset in all_synsets[:max_words]:
        for lemma in synset.lemma_names():
            word = lemma.replace('_', ' ')  # 将下划线替换为空格，使其更自然
            try:
                pred_label = black_model.predict([word])[0]
                label_to_words[pred_label].add(word)
            except Exception as e:
                print(f"⚠️ Prediction failed for word: {word} | Error: {e}")
                continue

            count += 1
            if count % 200 == 0:
                current_status = {k: len(v) for k, v in label_to_words.items()}
                print(f"Processed {count} samples; label stats: {current_status}")

        # 如果所有标签都满足最小词数要求，则提前终止
        if all(len(label_to_words[label]) >= min_words_per_label for label in range(black_model.num_labels)):
            break

    # 最终过滤不满足要求的标签
    filtered_result = {
        label: words for label, words in label_to_words.items()
        if len(words) >= min_words_per_label
    }

    print(f"最终保留标签数: {len(filtered_result)}")
    return filtered_result


if __name__ == "__main__":

    # 测试随机初始化
    print("\n测试随机初始化...")
    black_model = BlackBox("j-hartmann/emotion-english-distilroberta-base")
    prompt = PromptVector(hidden_dim=128, init_method='wordnet', black_model=black_model)  # 修改为与 diffusion 模型匹配的维度
    embeddings_dict = prompt.embeddings_dict
    words_dict = prompt.embeddings_words_dict
    # print(embeddings_dict)
    print(words_dict)
    
