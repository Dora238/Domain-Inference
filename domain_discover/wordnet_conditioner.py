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
from sklearn.metrics import pairwise_distances_argmin_min
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer, T5EncoderModel, GPT2Tokenizer, GPT2Model, AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_score
import os
from collections import Counter

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
            embeddings = self._model.encode(texts_to_encode, convert_to_tensor=True, show_progress_bar=False)
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


class WordNetConditioner(nn.Module):
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
        visual: bool = False,
        black_model = None,
        load_from_wordnet = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.init_method = init_method
        self.black_model = black_model
        self.visual = visual
        self.load_from_wordnet = load_from_wordnet

        if init_method == 'random':
            # 随机初始化
            self.embedding = nn.Parameter(
                torch.randn(1, hidden_dim) * init_scale
            )
        elif init_method == 'wordnet':
            # 基于WordNet初始化
            self.embedding_model = SimpleEmbedder()
            self.embeddings_dict, self.embeddings_words_dict = self.initialize_from_wordnet(self.black_model)
    
    def clone_detached(self) -> torch.Tensor:
        """Return a detached copy of the prompt vector.
        
        Returns:
            Tensor of shape (1, hidden_dim), detached from computation graph
        """
        return self.embedding.detach().clone()
    
    def get_embeddings(self, word_dict, model_name="T5"):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        embedding_records = []
        if model_name == "Bert":
            model
            
        if model_name == "qwen":

            model_name = "Qwen/Qwen1.5-7B-Chat"

            # 加载 tokenizer 和模型
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=device,
                trust_remote_code=True
            )
            model.eval()

            embedding_records = []

            for label, words in word_dict.items():
                for word in words:
                    inputs = tokenizer(word, return_tensors="pt").to(device)
                    input_ids = inputs["input_ids"][0]

                    # 排除特殊 token（如 pad、eos，如果存在）
                    special_ids = set([
                        tokenizer.pad_token_id,
                        tokenizer.eos_token_id,
                        tokenizer.bos_token_id,
                        tokenizer.unk_token_id
                    ])
                    valid_mask = torch.tensor(
                        [(tok_id.item() not in special_ids) for tok_id in input_ids],
                        device=input_ids.device
                    )
                    valid_indices = valid_mask.nonzero(as_tuple=True)[0]

                    if len(valid_indices) == 0:
                        continue

                    with torch.no_grad():
                        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
                        last_hidden = outputs.hidden_states[-1][0]  # shape: [seq_len, hidden_size]
                        word_embedding = last_hidden[valid_indices].mean(dim=0)

                    embedding_records.append({
                        "embedding": word_embedding.cpu().float().numpy(),
                        "label": label,
                        "word": word
                    })

        if model_name == "T5":
            tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
            encoder = T5EncoderModel.from_pretrained("google/flan-t5-large").to(device)
            encoder.eval()

            for label, words in word_dict.items():
                for word in words:
                    inputs = tokenizer(word, return_tensors="pt").to(device)
                    input_ids = inputs["input_ids"][0]
                    valid_mask = (input_ids != tokenizer.pad_token_id) & (input_ids != tokenizer.eos_token_id)
                    valid_indices = valid_mask.nonzero(as_tuple=True)[0]
                    if len(valid_indices) == 0:
                        continue
                    with torch.no_grad():
                        outputs = encoder(**inputs)
                        last_hidden = outputs.last_hidden_state[0]
                        word_embedding = last_hidden[valid_indices].mean(dim=0)
                    embedding_records.append({
                        "embedding": word_embedding.cpu().numpy(),
                        "label": label,
                        "word": word
                    })

        elif model_name == "GPT2":
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            model = GPT2Model.from_pretrained("gpt2").to(device)
            model.eval()

            for label, words in word_dict.items():
                for word in words:
                    inputs = tokenizer(word, return_tensors="pt", add_special_tokens=False).to(device)
                    input_ids = inputs["input_ids"][0]
                    if input_ids.shape[0] == 0:
                        continue
                    with torch.no_grad():
                        outputs = model(**inputs)
                        hidden_states = outputs.last_hidden_state[0]
                        word_embedding = hidden_states.mean(dim=0)
                    embedding_records.append({
                        "embedding": word_embedding.cpu().numpy(),
                        "label": label,
                        "word": word
                    })

        return embedding_records



    def visual_embedding(self, embedding_records, model_name="T5"):

        # 提取矩阵与标签
        X = torch.tensor([item["embedding"] for item in embedding_records])
        labels = [item["label"] for item in embedding_records]
        label_set = sorted(set(labels))
        colors = plt.cm.get_cmap("tab10", len(label_set))
        score = silhouette_score(X, labels)
        print(f"{model_name} → Silhouette Score: {score:.4f}")  
        # ========= 2D PCA ==========
        pca_2d = PCA(n_components=2)
        X_pca_2d = pca_2d.fit_transform(X)
        plt.figure(figsize=(10, 8))
        for label in label_set:
            idxs = [i for i, l in enumerate(labels) if l == label]
            plt.scatter(X_pca_2d[idxs, 0], X_pca_2d[idxs, 1], label=f"Class {label}", alpha=0.7, s=60)
        plt.title("PCA (2D) of {model_name} Encoder Hidden States")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"/home/dora/Domain-Inference/domain_discover/output/{model_name}_wordnet_pca_2d_{self.black_model.model_name}.png")
        plt.show()

        # ========= 3D PCA ==========
        pca_3d = PCA(n_components=3)
        X_pca_3d = pca_3d.fit_transform(X)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for label in label_set:
            idxs = [i for i, l in enumerate(labels) if l == label]
            ax.scatter(X_pca_3d[idxs, 0], X_pca_3d[idxs, 1], X_pca_3d[idxs, 2], label=f"Class {label}", alpha=0.7, s=30)
        ax.set_title("PCA (3D) of {model_name} Encoder Hidden States")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.legend()
        plt.savefig(f"/home/dora/Domain-Inference/domain_discover/output/{model_name}_wordnet_pca_3d_{self.black_model.model_name}.png")
        plt.show()

        # ========= 2D t-SNE ==========
        tsne_2d = TSNE(n_components=2, perplexity=30, init='random', random_state=42)
        X_tsne_2d = tsne_2d.fit_transform(X)
        plt.figure(figsize=(10, 8))
        for label in label_set:
            idxs = [i for i, l in enumerate(labels) if l == label]
            plt.scatter(X_tsne_2d[idxs, 0], X_tsne_2d[idxs, 1], label=f"Class {label}", alpha=0.7, s=60)
        plt.title("t-SNE (2D) of {model_name} Encoder Hidden States")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"/home/dora/Domain-Inference/domain_discover/output/{model_name}_wordnet_tsne_2d_{self.black_model.model_name}.png")
        plt.show()

        # ========= 3D t-SNE ==========
        tsne_3d = TSNE(n_components=3, perplexity=30, init='random', random_state=42)
        X_tsne_3d = tsne_3d.fit_transform(X)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for label in label_set:
            idxs = [i for i, l in enumerate(labels) if l == label]
            ax.scatter(X_tsne_3d[idxs, 0], X_tsne_3d[idxs, 1], X_tsne_3d[idxs, 2], label=f"Class {label}", alpha=0.7, s=30)
        ax.set_title("t-SNE (3D) of {model_name} Encoder Hidden States")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_zlabel("Dim 3")
        ax.legend()
        plt.savefig(f"/home/dora/Domain-Inference/domain_discover/output/{model_name}_wordnet_tsne_3d_{self.black_model.model_name}.png")
        plt.show()
        return model_name, score

    def cluster_wordnet(self, word_dict):
        embeddings_dict = {}
        embeddings_words_dict = {}

        for label, words in word_dict.items():
            print(f"找到 {len(words)} 个标签为 {label} 的词汇")

            embeddings = []
            word_list = []

            for word in words:
                with torch.no_grad():
                    embedding = self.embedding_model.encode(word)
                    if isinstance(embedding, torch.Tensor) and embedding.numel() == self.hidden_dim:
                        embeddings.append(embedding.reshape(1, -1))
                        word_list.append(word)

            if len(embeddings) < 2:
                print(f"⚠️ 类别 {label} 中有效嵌入词汇数量不足，跳过聚类")
                continue

            embeddings = torch.cat(embeddings, dim=0).cpu().numpy()

            n_clusters = min(5, len(embeddings))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(embeddings)

            print(f"标签 {label} 的聚类结果（共 {n_clusters} 类）：")

            center_embeddings = []
            center_words = []

            for cluster_id in range(n_clusters):
                indices = [i for i, pred in enumerate(kmeans.labels_) if pred == cluster_id]
                cluster_embeddings = embeddings[indices]
                cluster_words = [word_list[i] for i in indices]

                # 找出最靠近聚类中心的词
                center = kmeans.cluster_centers_[cluster_id].reshape(1, -1)
                closest_idx, _ = pairwise_distances_argmin_min(center, cluster_embeddings)
                center_word = cluster_words[closest_idx[0]]
                center_embed = cluster_embeddings[closest_idx[0]]

                center_words.append(center_word)
                center_embeddings.append(center_embed)

                print(f"  🔸 Cluster {cluster_id}: {center_word}")

        # 保存最终结果：每个label下的中心词和其对应的embedding
        embeddings_dict[label] = np.stack(center_embeddings, axis=0)
        embeddings_words_dict[label] = center_words

        return embeddings_dict, embeddings_words_dict

    
    def initialize_from_wordnet(self, black_model=None) -> None:
        """
        从WordNet词汇初始化prompt vector，保留每个label下每个聚类中心最靠近的词。
        
        Args:
            black_model: 可选的黑盒模型，用于辅助选择词汇。
        Returns:
            word_dict: 每个label下的词汇列表
        """

        word_dict = self.traverse_wordnet(
            black_model=black_model,
            max_words=5000,
            min_words_per_label=20
        )
        pca_visual = self.visual

        if pca_visual:
            for model_name in ["T5", "GPT2"]:
                embeddings = self.get_embeddings(word_dict, model_name=model_name)
                model_name, score = self.visual_embedding(embeddings, model_name)
        cluster = False
        if cluster:
            embeddings_dict, embeddings_words_dict = self.cluster_wordnet(word_dict)
            return embeddings_dict, embeddings_words_dict
        else:
            return None, word_dict

    def traverse_wordnet(self, black_model=None, max_words=5000, min_words_per_label=20):
        """遍历WordNet词汇，找到黑盒模型预测为目标标签的词汇
        
        Args:
            black_model: 黑盒模型，用于预测词汇的标签
            max_words: 最大处理词汇数量
            min_words_per_label: 每个标签最少保留的词汇数量
            
        Returns:
            如果target_label为None，返回{label: [words]}的字典
            如果target_label不为None，返回符合目标标签的词汇列表
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # gpt_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        # tokenizer.pad_token = tokenizer.eos_token
        model_name = "Qwen/Qwen1.5-7B-Chat"

        # 加载 tokenizer 和模型（记得 trust_remote_code=True）
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,         
            device_map=device,                
            trust_remote_code=True
        )
        words_dict = defaultdict(list)
        # 获取所有同义词集
        all_synsets = list(wn.all_synsets())
        print(f"WordNet中共有 {len(all_synsets)} 个同义词集")
        word_count = 0
        break_outer_loop = False
        if self.load_from_wordnet:
            for synset in all_synsets[:max_words]:
                for lemma in synset.lemma_names():
                    prompt = (
                        f"Write one short English sentence using the word '{lemma}'. Only output the sentence.\n"
                    )
                    good_sentences = []
                    pred_labels = []
                    if word_count % 100 == 0:
                        print(f"已收集 {word_count} 个词汇")
                    for _ in range(10):
                        inputs = tokenizer(prompt, return_tensors="pt").to(device)
                        output_ids = model.generate(
                            **inputs,
                            max_length=64,
                            temperature=0.8,
                            top_p=0.95,
                            do_sample=True,
                            pad_token_id=tokenizer.eos_token_id
                        )
                        # 解码并去除 prompt
                        full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                        if prompt.strip() in full_output:
                            sentence = full_output.replace(prompt.strip(), "").strip()
                        else:
                            sentence = full_output.strip()

                        if not sentence:
                            continue  # skip empty generations

                        # 记录句子和预测标签
                        pred_label = black_model.predict([sentence])[0]
                        good_sentences.append(sentence)
                        pred_labels.append(pred_label)

                    # 统计标签分布
                    label_counter = Counter(pred_labels)
                    majority_label, count = label_counter.most_common(1)[0]

                    # 如果70%以上属于同一个标签，则保留
                    if count >= 7:
                        words_dict[majority_label].append(lemma)
                        word_count += 1
                        # 检查是否有任何标签下的词数超过5个
                        if all(len(words) > 10 for words in words_dict.values()):
                            break_outer_loop = True
                            break

                        if break_outer_loop:
                            break  # 跳出当前的外层循环
                if break_outer_loop:
                    break
            
            output_path = "/home/dora/Domain-Inference/domain_discover/data_from_wordnet/words_by_label.txt"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                for label, lemmas in words_dict.items():
                    f.write(f"Label {int(label)} ({len(lemmas)} words):\n")
                    for lemma in lemmas:
                        f.write(f"  - {lemma}\n")
                    f.write("\n")  # 空行分隔标签块

            return words_dict
        else:
            with open("/home/dora/Domain-Inference/domain_discover/data_from_wordnet/words_by_label.txt", "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("Label"):
                        parts = line.split()
                        current_label = int(parts[1])
                        words_dict[current_label] = []
                    elif line.startswith("- "):  # "  - lemma"
                        lemma = line[2:].strip()
                        words_dict[current_label].append(lemma)
            return words_dict



class TextDataset(Dataset):
    def __init__(self, text_datasets, data_args, model_emb=None):
        super().__init__()
        input_ids = text_datasets['input_ids']
        input_mask = text_datasets['input_mask']
        
        # 构造 list of dicts，每条样本是一个字典
        self.text_datasets = [
            {'input_ids': input_ids[i], 'input_mask': input_mask[i]}
            for i in range(len(input_ids))
        ]
        self.length = len(self.text_datasets)
        self.data_args = data_args
        self.model_emb = model_emb

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with torch.no_grad():
            input_ids = self.text_datasets[idx]['input_ids']
            input_mask = self.text_datasets[idx]['input_mask']
            
            # 直接使用CPU处理，避免多进程中的CUDA问题
            input_tensor = torch.tensor(input_ids)
            
            # 使用CPU计算
            hidden_state = self.model_emb(input_tensor)
            arr = torch.tensor(hidden_state.detach().numpy(), dtype=torch.float32)

            out_kwargs = {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'input_mask': torch.tensor(input_mask, dtype=torch.long),
            }

            return arr, out_kwargs




if __name__ == "__main__":

    # 测试随机初始化
    print("\n测试随机初始化...")
    black_model = BlackBox("j-hartmann/emotion-english-distilroberta-base")
    prompt = WordNetConditioner(hidden_dim=128, init_method='wordnet', black_model=black_model)  # 修改为与 diffusion 模型匹配的维度
    embeddings_dict = prompt.embeddings_dict
    words_dict = prompt.embeddings_words_dict
    # print(embeddings_dict)
    print(words_dict)
    
