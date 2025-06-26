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
from domain_infer.classifier import Classifier
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer, T5EncoderModel, GPT2Tokenizer, GPT2Model, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_score
import os
from collections import Counter
import json
import re

class WordNetConditioner(nn.Module):
    """Learnable prompt vector for steering text generation.
    
    This module wraps a single learnable embedding vector that will be
    optimized to guide the diffusion model toward generating text with
    desired properties.
    
    Args:
        init_scale: Scale for random initialization
        init_method: 初始化方法，可选 'random'(随机初始化) 或 'wordnet'(基于WordNet初始化)
        embedding_model: 用于将词汇转换为嵌入的模型（如果init_method='wordnet'）
        max_words: WordNet遍历的最大词汇数
        min_words_per_category: 每个类别最少保留的词汇数量
    """
    
    def __init__(
        self,
        init_scale: float = 0.02,
        init_method: str = 'wordnet',
        embedding_model = None,
        max_words: int = 5000,
        min_words_per_category: int = 20,
        visual: bool = False,
        classifier = None,
        initial_sentence_from_wordnet = True,
        t5_generator = None,
    ):
        super().__init__()
        self.init_method = init_method
        self.classifier = classifier
        self.t5_generator = t5_generator
        self.visual = visual
        self.initial_sentence_from_wordnet = initial_sentence_from_wordnet
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.load_initial_path = f'{self.project_root}/data/initial_from_wordnet'
        if init_method == 'wordnet':
            # 基于WordNet初始化
            self.embeddings_dict, self.embeddings_words_dict = self.initialize_from_wordnet(self.classifier)

    def clean_sentences(self, sentences):
        cleaned_sentences = []
        for sent in sentences:
            sent = sent.strip()
            # 移除常见前缀
            prefixes = [
            "Sentence: ",
            "Sure, here's a sentence: ",
            "Here's a sentence: ",
            "Here is a sentence: ",
            "I'll create a sentence: "
        ]
        
        for prefix in prefixes:
            if sent.startswith(prefix):
                sent = sent[len(prefix):]
                break
        
        # 移除引号
        if sent.startswith('"') and sent.endswith('"'):
            sent = sent[1:-1]
        
        if sent and sent not in cleaned_sentences:
            cleaned_sentences.append(sent)
        return cleaned_sentences
    
    def get_embeddings(self, word_dict, model_name="Bert"):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        embedding_records = []
        if model_name == "Bert":
            model_name = "bert-base-uncased"
            model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            def get_bert_cls(text, model, tokenizer):
                inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
                with torch.no_grad():
                    outputs = model.bert(**inputs)
                return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()  # 取 [CLS]
            embedding_records = []
            for label, words in word_dict.items():
                for word in words:
                    try:
                        embedding = get_bert_cls(word, model, tokenizer)
                        embedding_records.append({
                            "embedding": embedding,
                            "label": label,
                            "word": word
                        })
                    except Exception as e:
                        print(f"跳过：{word} - {str(e)}")
            
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
        # ========== 处理数据 ==========
        X = torch.tensor([item["embedding"] for item in embedding_records])
        labels = [item["label"] for item in embedding_records]
        label_set = sorted(set(labels))
        colors = plt.cm.get_cmap("tab10", len(label_set))
        X = X.squeeze().cpu().numpy()  # 转为 numpy 并确保是 2D

        # ========== Silhouette Score ==========
        score = silhouette_score(X, labels)
        print(f"{model_name} → Silhouette Score: {score:.4f}")  

        # ========== 2D PCA ==========
        pca_2d = PCA(n_components=2)
        X_pca_2d = pca_2d.fit_transform(X)
        plt.figure(figsize=(10, 8))
        for label in label_set:
            idxs = [i for i, l in enumerate(labels) if l == label]
            plt.scatter(X_pca_2d[idxs, 0], X_pca_2d[idxs, 1], label=f"Class {label}", alpha=0.7, s=60)
        plt.title(f"PCA (2D) of {model_name} Encoder Hidden States")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.project_root}/output/visual_word_embedding/{model_name}_wordnet_pca_2d_{self.classifier.model_name}.png")
        plt.show()

        # ========== 3D PCA ==========
        pca_3d = PCA(n_components=3)
        X_pca_3d = pca_3d.fit_transform(X)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for label in label_set:
            idxs = [i for i, l in enumerate(labels) if l == label]
            ax.scatter(X_pca_3d[idxs, 0], X_pca_3d[idxs, 1], X_pca_3d[idxs, 2], label=f"Class {label}", alpha=0.7, s=30)
        ax.set_title(f"PCA (3D) of {model_name} Encoder Hidden States")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.legend()
        plt.savefig(f"{self.project_root}/output/visual_word_embedding/{model_name}_wordnet_pca_3d_{self.classifier.model_name}.png")
        plt.show()

        # ========== t-SNE (2D) ← 先 PCA50 ==========
        pca_tsne = PCA(n_components=50)
        X_reduced_tsne = pca_tsne.fit_transform(X)
        tsne_2d = TSNE(n_components=2, perplexity=30, init='random', random_state=42)
        X_tsne_2d = tsne_2d.fit_transform(X_reduced_tsne)
        plt.figure(figsize=(10, 8))
        for label in label_set:
            idxs = [i for i, l in enumerate(labels) if l == label]
            plt.scatter(X_tsne_2d[idxs, 0], X_tsne_2d[idxs, 1], label=f"Class {label}", alpha=0.7, s=60)
        plt.title(f"t-SNE (2D) of {model_name} Encoder Hidden States")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.project_root}/output/visual_word_embedding/{model_name}_wordnet_tsne_2d_{self.classifier.model_name}.png")
        plt.show()

        # ========== t-SNE (3D) ← 先 PCA50 ==========
        tsne_3d = TSNE(n_components=3, perplexity=30, init='random', random_state=42)
        X_tsne_3d = tsne_3d.fit_transform(X_reduced_tsne)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for label in label_set:
            idxs = [i for i, l in enumerate(labels) if l == label]
            ax.scatter(X_tsne_3d[idxs, 0], X_tsne_3d[idxs, 1], X_tsne_3d[idxs, 2], label=f"Class {label}", alpha=0.7, s=30)
        ax.set_title(f"t-SNE (3D) of {model_name} Encoder Hidden States")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_zlabel("Dim 3")
        ax.legend()
        plt.savefig(f"{self.project_root}/output/visual_word_embedding/{model_name}_wordnet_tsne_3d_{self.classifier.model_name}.png")
        plt.show()

        return model_name, score

    def generate_sentences(self, lemma, max_sentences=20):
        prompt = f"Write one short English sentence using the word '{lemma}'. Only output the sentence.\n"
        sentences = []
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(
            **inputs,
            num_beams=max_sentences,
            num_beam_groups=max_sentences,
            max_length=64,
            temperature=0.7,
            # top_p=0.95,
            do_sample=False,
            diversity_penalty=3.0,
            no_repeat_ngram_size=2,
            repetition_penalty=10.0,
            pad_token_id=self.tokenizer.eos_token_id,
            num_return_sequences=max_sentences,
        )
        full_output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        sentences = [full_output.replace(prompt.strip(), "").strip() for full_output in full_output]
        sentences = [s for s in sentences if s] 
        return sentences

    def evaluate_sentences(self, sentences):
        labels = [self.classifier.predict([s])[0] for s in sentences]
        counter = Counter(labels)
        majority_label, count = counter.most_common(1)[0]
        return majority_label, count, labels
    
    def collect_words(self, max_words=500, label_word_limit=10):

        words_dict = defaultdict(list)              # 记录每个 label 对应的词
        best_count_by_label = defaultdict(int)      # 记录每个 label 最好的 count
        word_count = 0

        for synset in self.all_synsets[:max_words]:
            for lemma in synset.lemma_names():
                if word_count % 100 == 0:
                    print(f"已收集 {word_count} 个词汇")

                # 1. 生成多个句子
                sentences = self.generate_sentences(lemma, max_sentences=label_word_limit)

                # 2. 用 black_box 分类
                majority_label, count, _ = self.evaluate_sentences(sentences)

                # 3. 动态更新逻辑
                if count > best_count_by_label[majority_label]:
                    best_count_by_label[majority_label] = count

                    # 记录 lemma 及其句子
                    words_dict[majority_label].append({
                        "lemma": lemma,
                        "sentences": sentences
                    })

                    word_count += 1
                    print(f"更新 label {majority_label} 的最优 count 为 {count}，加入新词：{lemma}")

                    if all(len(words) > label_word_limit for words in words_dict.values()):
                        return words_dict

        return words_dict

    def collect_sentences(
        self,
        max_words: int = 500,
        label_word_limit: int = 20,
    ):
        """
        - words_dict[label]     → 历史最优演进轨迹（每刷新一次就 append 一条）
        - best_dict[label]      → 始终保存当前“最高 count”的记录
        - 每 100 个 lemma（word_count % 100 == 10）存一次快照
        """
        words_dict      = defaultdict(list)   # 历史
        best_dict       = {}                  # 当前最佳
        best_count      = defaultdict(int)
        word_count      = 0

        os.makedirs(self.load_initial_path, exist_ok=True)

        for synset in self.all_synsets[:max_words]:
            for lemma in synset.lemma_names():

                # 0) 进度快照 --------------------------------------------------------
                if word_count % 100 == 10:
                    print(f"已评估 {word_count} 个 lemma，保存快照…")
                    snap_base = f"{self.classifier.model_name}_{word_count}"
                    with open(f"{self.load_initial_path}/words_dict_{snap_base}.json", "w", encoding="utf-8") as f:
                        json.dump(words_dict, f, ensure_ascii=False, indent=2)
                    with open(f"{self.load_initial_path}/best_dict_{snap_base}.json", "w", encoding="utf-8") as f:
                        json.dump(best_dict, f, ensure_ascii=False, indent=2)

                # 1) 生成 & 评价 ------------------------------------------------------
                sentences = self.generate_sentences(lemma, max_sentences=label_word_limit)
                majority_label, count, _ = self.evaluate_sentences(sentences)

                # 2) 是否刷新最佳 ------------------------------------------------------
                if count > best_count[majority_label]:
                    best_count[majority_label] = count

                    # (a) 更新 words_dict（历史）
                    words_dict[majority_label].append({
                        "lemma": lemma,
                        "count": count,
                        "sentences": sentences,
                    })

                    # (b) 更新 best_dict（当前最佳）
                    best_dict[majority_label] = {
                        "lemma": lemma,
                        "count": count,
                        "sentences": sentences,
                    }

                    print(f"[label {majority_label}] ✨ 新最优 count={count}, lemma='{lemma}'")

                word_count += 1

                # 3) 提前结束：每个 label 已达到阈值 ------------------------------
                if best_dict and all(
                    rec["count"] >= label_word_limit for rec in best_dict.values()
                ):
                    print("所有 label 均达到目标计数，提前结束。")
                    return words_dict, best_dict

        return words_dict, best_dict

    def save_words(self, words_dict, output_path_txt, output_path_json=None):
        import json
        os.makedirs(os.path.dirname(output_path_txt), exist_ok=True)

        # 写入 .txt 可读性文件
        with open(output_path_txt, "w", encoding="utf-8") as f:
            for label, items in words_dict.items():
                f.write(f"Label {int(label)} ({len(items)} words):\n")
                for item in items:
                    f.write(f"  - {item['lemma']}\n")
                    for s in item["sentences"]:
                        f.write(f"      > {s}\n")
                f.write("\n")

        # 可选：保存为 JSON（结构化分析）
        if output_path_json:
            with open(output_path_json, "w", encoding="utf-8") as f_json:
                json.dump(words_dict, f_json, ensure_ascii=False, indent=2)
    
    def initialize_from_wordnet(self, classifier=None) -> None:
        """
        从WordNet词汇初始化prompt vector，保留每个label下每个聚类中心最靠近的词。
        
        Args:
            classifier: 可选的黑盒模型，用于辅助选择词汇。
        Returns:
            word_dict: 每个label下的词汇列表
        """
        if self.initial_sentence_from_wordnet:
            word_dict = self.traverse_wordnet()
        else:
            word_dict = self.load_from_json()
        for label, entry in word_dict.items():
            sentences = entry.get('sentences', [])
            if not sentences:
                continue

            embeddings = []
            clean_sentences = self.clean_sentences(sentences)
            for sent in clean_sentences:
                with torch.no_grad():
                    encoder_outputs, attention = self.t5_generator.encode(sent)
                    hidden_states = encoder_outputs  # [1, L, D]
                    sent_embed = hidden_states.mean(dim=1).squeeze(0)  # [D]
                embeddings.append(sent_embed)  # 保持原始 Tensor，不 .cpu().detach()

            if embeddings:
                embs = torch.stack([x.detach().cpu() for x in embeddings], dim=0)  # [N, D]
                center = embs.mean(dim=0, keepdim=True)                            # [1, D]
                distances = torch.norm(embs - center, dim=1)                       # [N]
                closest_idx = torch.argmin(distances).item()

                # 从原始 embeddings 中取值（未 detach）
                closest_embedding = embeddings[closest_idx]

                word_dict[label]['center_sentence'] = sentences[closest_idx]
                word_dict[label]['embedding_center'] = closest_embedding  # 原始 tensor，未转 list

        self.word_dict = word_dict
        pca_visual = self.visual

        if pca_visual:
            for model_name in ["T5", "GPT2", "Bert"]:
                embeddings = self.get_embeddings(word_dict, model_name=model_name)
                model_name, score = self.visual_embedding(embeddings, model_name)
        cluster = False
        if cluster:
            embeddings_dict, embeddings_words_dict = self.cluster_wordnet(word_dict)
            return embeddings_dict, embeddings_words_dict
        else:
            return None, word_dict
    

    def load_from_json(self):
        # 获取所有符合模式的文件名
        pattern = re.compile(rf"best_dict_{re.escape(self.classifier.model_name)}_(\d+)\.json")
        max_word_count = -1
        best_file = None

        for fname in os.listdir(self.load_initial_path):
            match = pattern.match(fname)
            if match:
                word_count = int(match.group(1))
                if word_count > max_word_count:
                    max_word_count = word_count
                    best_file = fname

        if best_file is None:
            raise FileNotFoundError(f"No matching best_dict_*.json found for model {self.classifier.model_name} in {self.load_initial_path}")

        json_path = os.path.join(self.load_initial_path, best_file)
        with open(json_path, "r", encoding="utf-8") as f:
            word_dict = json.load(f)
        return word_dict

    def load_local_wordnet(self):
        txt_path=f'{self.load_initial_path}/words_by_label.txt'
        words_dict = defaultdict(list)
        current_label = None
        current_lemma = None
        current_sentences = []

        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip()

                if line.startswith("Label"):
                    # 存储上一个 lemma 的句子（如果存在）
                    if current_label is not None and current_lemma is not None:
                        words_dict[current_label].append({
                            "lemma": current_lemma,
                            "sentences": current_sentences
                        })
                    # 重置
                    parts = line.split()
                    current_label = int(parts[1])
                    current_lemma = None
                    current_sentences = []

                elif line.startswith("- "):
                    # 如果有未写入的 lemma，也写入
                    if current_lemma is not None:
                        words_dict[current_label].append({
                            "lemma": current_lemma,
                            "sentences": current_sentences
                        })
                    # 新 lemma 开始
                    current_lemma = line[2:].strip()
                    current_sentences = []

                elif line.startswith("> "):
                    sentence = line[2:].strip()
                    current_sentences.append(sentence)

            # 文件结尾的最后一个 lemma
            if current_label is not None and current_lemma is not None:
                words_dict[current_label].append({
                    "lemma": current_lemma,
                    "sentences": current_sentences
                })

        return words_dict

    def traverse_wordnet(self, load_word_wordnet=False):
        """遍历WordNet词汇，找到黑盒模型预测为目标标签的词汇
        
        Args:
            load_word_wordnet: 是否加载已有的词汇
            
        Returns:
            如果target_label为None，返回{label: [words]}的字典
            如果target_label不为None，返回符合目标标签的词汇列表
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = "Qwen/Qwen1.5-7B-Chat"

        # 加载 tokenizer 和模型（记得 trust_remote_code=True）
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,         
            device_map=device,                
            trust_remote_code=True
        )
        self.model.eval()
        # 获取所有同义词集
        self.all_synsets = list(wn.all_synsets())
        print(f"WordNet中共有 {len(self.all_synsets)} 个同义词集")
        if load_word_wordnet:
            words_dict = self.collect_words(
                max_words=1000,
                label_word_limit=10
            )
            self.save_words(words_dict, f"{self.load_initial_path}/words_by_label_{self.classifier.model_name}.txt")

            return words_dict
        else:
            words_dict, best_dict = self.collect_sentences(max_words=1000, label_word_limit=20)
            self.save_words(words_dict, f"{self.load_initial_path}/sentence_by_label_{self.classifier.model_name}.txt")
            best_dict_path = f"{self.load_initial_path}/best_dict_{self.classifier.model_name}_final.json"
            self.save_words(best_dict, best_dict_path)
            return words_dict


if __name__ == "__main__":

    # 测试随机初始化
    print("\n测试随机初始化...")
    classifier = Classifier("j-hartmann/emotion-english-distilroberta-base")
    prompt = WordNetConditioner(hidden_dim=128, init_method='wordnet', classifier=classifier)  # 修改为与 diffusion 模型匹配的维度
    embeddings_dict = prompt.embeddings_dict
    words_dict = prompt.embeddings_words_dict
    
    # print(embeddings_dict)
    print(words_dict)
    
