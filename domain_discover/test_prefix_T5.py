import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5Tokenizer, T5EncoderModel
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn.metrics import silhouette_score, pairwise_distances, accuracy_score, precision_recall_fscore_support
from scipy.spatial.distance import pdist, squareform
from scipy.stats import ttest_ind
from generater_T5 import T5Generator
import json
import os
import pandas as pd
# 设置 device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载黑盒模型
spam_model_name = "mrm8488/bert-tiny-finetuned-sms-spam-detection"
spam_tokenizer = AutoTokenizer.from_pretrained(spam_model_name)
spam_model = AutoModelForSequenceClassification.from_pretrained(spam_model_name).to(device).eval()

cfg = {
    "model_name": "humarin/chatgpt_paraphraser_on_T5_base",
    "peft_model_path": "/home/dora/Domain-Inference/domain_discover/prefix_paraphraser/checkpoint-65500",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}
t5_model = T5Generator(**cfg)
t5_tokenizer = t5_model.tokenizer
# 加载数据集

dataset_name = "ucirvine/sms_spam"
dataset = load_dataset(dataset_name, split="train[:1000]")

# 嵌入列表
t5_embeddings, labels = [], []

# 将不可序列化的numpy数组转换为列表
def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        # 确保字典的键也被转换为可序列化类型
        return {str(convert_to_serializable(k)): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def get_t5_encoder_hidden(text, model, tokenizer):
    last_hidden_state, attention_mask = model.encode(text)
    hidden = last_hidden_state  # [batch_size, seq_len, hidden_dim]
    return hidden.mean(dim=1).squeeze().cpu().numpy()  # 可选 mean pooling 或取第一个 token


# 遍历数据
for sample in tqdm(dataset):
    text, label = sample["sms"], sample["label"]
    try:
        t5_emb = get_t5_encoder_hidden(text, t5_model, t5_tokenizer)
        t5_embeddings.append(t5_emb)
        labels.append(label)
    except Exception as e:
        print(f"跳过：{text[:30]} - {str(e)}")

# 特征分析函数
def analyze_hidden_states(embeddings, labels, title):
    embeddings = np.array(embeddings)
    if embeddings.ndim == 3:
        embeddings = embeddings.squeeze(axis=1)
    assert embeddings.ndim == 2, f"Expect 2D array, got shape {embeddings.shape}"
    
    # 转换为numpy数组并分离不同标签的样本
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    label_embeddings = {}
    for label in unique_labels:
        label_embeddings[label] = embeddings[labels == label]
    
    # 1. 计算轮廓系数
    silhouette = silhouette_score(embeddings, labels)
    print(f"{title} Silhouette Score: {silhouette:.4f}")
    
    # 2. 计算类内距离和类间距离
    intra_class_distances = {}
    for label, embs in label_embeddings.items():
        if len(embs) > 1:  # 确保有足够的样本计算距离
            distances = pdist(embs, metric='euclidean')
            intra_class_distances[label] = np.mean(distances)
            print(f"Label {label} 类内平均距离: {intra_class_distances[label]:.4f}")
    
    # 类间距离
    inter_class_distances = {}
    for i, label1 in enumerate(unique_labels):
        for label2 in unique_labels[i+1:]:
            dist = pairwise_distances(label_embeddings[label1], label_embeddings[label2], metric='euclidean')
            inter_class_distances[(label1, label2)] = np.mean(dist)
            print(f"Label {label1} 和 Label {label2} 类间平均距离: {inter_class_distances[(label1, label2)]:.4f}")
    
    # 3. 计算每个维度的区分度
    dim_discriminative_power = []
    feature_dim = embeddings.shape[1]
    for dim in range(feature_dim):
        dim_values = {label: embs[:, dim] for label, embs in label_embeddings.items()}
        # 使用t检验评估每个维度的区分能力
        if len(unique_labels) == 2:  # 二分类情况
            t_stat, p_value = ttest_ind(dim_values[0], dim_values[1], equal_var=False)
            dim_discriminative_power.append((dim, abs(t_stat), p_value))
    
    # 按区分能力排序并打印前10个最具区分性的维度
    if len(unique_labels) == 2:
        sorted_dims = sorted(dim_discriminative_power, key=lambda x: x[1], reverse=True)
        print("\n最具区分性的10个维度:")
        for i, (dim, t_stat, p_value) in enumerate(sorted_dims[:10]):
            print(f"维度 {dim}: t统计量 = {t_stat:.4f}, p值 = {p_value:.6f}")
    
    # 4. 计算标签聚类紧密度比率 (类内距离与类间距离的比率)
    if len(unique_labels) == 2 and 0 in intra_class_distances and 1 in intra_class_distances:
        avg_intra_dist = np.mean(list(intra_class_distances.values()))
        avg_inter_dist = np.mean(list(inter_class_distances.values()))
        compactness_ratio = avg_intra_dist / avg_inter_dist
        print(f"\n聚类紧密度比率 (类内/类间): {compactness_ratio:.4f}")
        print(f"较低的比率表示更好的聚类效果，理想值接近0")

    label_centers = {
        label: np.mean(embs, axis=0)
        for label, embs in label_embeddings.items()
    }
    label_dim_variances = {}

    for label, embs in label_embeddings.items():  # embs shape: [N_i, D]
        center = label_centers[label]             # shape: [D]
        # 计算每个维度的方差（样本对该维度减去中心后的平方差）
        var_per_dim = np.var(embs - center, axis=0)
        label_dim_variances[label] = var_per_dim
    df_intra_variance = pd.DataFrame.from_dict(label_dim_variances, orient='index')
    df_intra_variance.columns = [f"Dim_{i}" for i in range(df_intra_variance.shape[1])]
    df_intra_variance.index.name = "Label"

    # 打印
    print(df_intra_variance.head())

    return {
        'silhouette': silhouette,
        'intra_class_distances': intra_class_distances,
        'inter_class_distances': inter_class_distances,
        'discriminative_dims': sorted_dims[:10] if len(unique_labels) == 2 else None,
        'compactness_ratio': compactness_ratio if len(unique_labels) == 2 else None,
        'label_centers': label_centers
    }

# 降维和可视化函数
def reduce_and_plot(embeddings, labels, title, dim=2):
    """降维并可视化嵌入向量
    
    参数:
    - embeddings: 嵌入向量列表
    - labels: 标签列表
    - title: 图表标题
    - dim: 降维维度，支持2或3
    
    返回:
    - analysis_results: 特征分析结果
    """
    # 首先进行特征分析
    analysis_results = analyze_hidden_states(embeddings, labels, title)
    
    embeddings = np.array(embeddings)
    if embeddings.ndim == 3:
        embeddings = embeddings.squeeze(axis=1)
    
    # PCA降维
    pca = PCA(n_components=50)
    pca_result = pca.fit_transform(embeddings)
    
    # 输出PCA解释的方差比例
    explained_variance = pca.explained_variance_ratio_
    print(f"\nPCA解释的前10个成分的方差比例:")
    for i, var in enumerate(explained_variance[:10]):
        print(f"成分 {i+1}: {var:.4f}")
    print(f"前50个成分累计解释方差比例: {sum(explained_variance):.4f}")
    
    # t-SNE降维到指定维度
    tsne = TSNE(n_components=dim, random_state=42, perplexity=30)
    tsne_result = tsne.fit_transform(pca_result)

    # 可视化
    unique_labels = np.unique(labels)
    colors = ['blue' if l == 0 else 'red' for l in labels]
    
    if dim == 2:
        # 2D可视化
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=colors, alpha=0.6)
        plt.title(f"t-SNE of {title} Embeddings (2D)\nSilhouette Score: {analysis_results['silhouette']:.4f}")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 为每个类别添加中心点
        for label in unique_labels:
            mask = np.array(labels) == label
            center = np.mean(tsne_result[mask], axis=0)
            plt.scatter(center[0], center[1], s=100, c='k', marker='*')
            plt.annotate(f"Label {label} Center", (center[0], center[1]), 
                        textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.legend(['ham (0)', 'spam (1)'])
        plt.savefig(f"Prefix_T5_{title}_tsne_2d.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    elif dim == 3:
        # 3D可视化
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制3D散点图
        scatter = ax.scatter(
            tsne_result[:, 0], 
            tsne_result[:, 1], 
            tsne_result[:, 2], 
            c=colors, 
            alpha=0.6
        )
        
        ax.set_title(f"t-SNE of {title} Embeddings (3D)\nSilhouette Score: {analysis_results['silhouette']:.4f}")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_zlabel("Dimension 3")
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 为每个类别添加中心点
        for label in unique_labels:
            mask = np.array(labels) == label
            center = np.mean(tsne_result[mask], axis=0)
            ax.scatter(center[0], center[1], center[2], s=100, c='k', marker='*')
            ax.text(center[0], center[1], center[2], f"Label {label} Center", 
                    fontsize=9, ha='center')
        
        # 添加图例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='ham (0)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='spam (1)'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='black', markersize=10, label='Center')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # 保存并显示图形
        plt.savefig(f"Prefix_T5_{title}_tsne_3d.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    else:
        raise ValueError("降维维度只支持2或3")
    
    return analysis_results

# 使用黑盒模型评估生成的embedding
def evaluate_generated_embeddings(embeddings, target_label, model, tokenizer, decode_func=None):
    """
    评估生成的embedding是否符合目标标签
    
    参数:
    - embeddings: 生成的embedding数组
    - target_label: 目标标签
    - model: 黑盒模型
    - tokenizer: 黑盒模型的tokenizer
    - decode_func: 将embedding解码为文本的函数，如果为None，则使用t5_model生成
    
    返回:
    - 评估结果字典
    """
    results = {
        'generated_texts': [],
        'predictions': [],
        'confidence_scores': [],
        'success_rate': 0.0
    }
    
    # 如果没有提供解码函数，使用T5Generator的generate_from_hidden_state函数从embedding生成文本

    # 初始化T5Generator
    generator = T5Generator(model_name="humarin/chatgpt_paraphraser_on_T5_base", device=device)
    
    def default_decode(emb):
        # 将单个embedding转换为适当的形状 [1, seq_len, hidden_size]
        if len(emb.shape) == 1:
            # 如果是一维向量，添加batch和seq_len维度
            emb = emb.unsqueeze(0).unsqueeze(0)
        elif len(emb.shape) == 2:
            # 如果是二维向量 [seq_len, hidden_size]，添加batch维度
            emb = emb.unsqueeze(0)
            
        # 创建attention mask (全1)
        attention_mask = torch.ones((1, emb.shape[1]), dtype=torch.long, device=device)
        
        # 使用generate_from_hidden_state生成文本
        texts = generator.generate_from_hidden_state(
            encoder_hidden_states=emb,
            encoder_attention_mask=attention_mask,
            max_length=64,
            num_beams=4,
            num_beam_groups=2,
            num_return_sequences=1,
            diversity_penalty=1.5,
            repetition_penalty=1.2,
            length_penalty=1.0
        )
        
        # 返回生成的第一个文本
        return texts[0] if texts else ""
    
    decode_func = default_decode
    
    # 对每个生成的embedding进行评估
    for i, emb in enumerate(embeddings):
        # 将embedding解码为文本
        text = decode_func(emb)
        results['generated_texts'].append(text)
        
        # 使用黑盒模型进行预测
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 获取预测结果
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        pred_label = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_label].item()
        
        results['predictions'].append(pred_label)
        results['confidence_scores'].append(confidence)
    
    # 计算成功率
    correct_preds = [1 if pred == target_label else 0 for pred in results['predictions']]
    results['success_rate'] = sum(correct_preds) / len(correct_preds) if correct_preds else 0.0
    
    return results

# 测试不同生成方法的效果
def test_center_generation_methods(label_centers, intra_class_distances, device=None, n_samples=10, noise_scales=[0.5]):
    """
    测试不同生成方法的效果
    
    参数:
    - label_centers: 每个标签的中心向量字典
    - intra_class_distances: 每个标签的类内距离字典
    - device: 运行设备
    - n_samples: 每种方法生成的样本数量
    - noise_scales: 噪声尺度列表，用于测试不同噪声水平
    
    返回:
    - 测试结果字典
    """
    methods = ['gaussian', 'uniform', 'scaled']
    results = {}
    
    # 确保device参数有效
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for label, center in label_centers.items():
        label_results = {}
        intra_dist = intra_class_distances.get(label, 1.0)  # 默认值为1.0
        
        for method in methods:
            method_results = {}
            
            # 测试不同噪声尺度
            for scale in noise_scales:
                print(f"\n测试 Label {label}, Method {method}, Scale {scale}:")
                
                # 调整噪声尺度
                scaled_dist = intra_dist * scale
                
                # 生成embedding
                generated_embeddings = generate_embeddings_from_center(
                    center, scaled_dist, n_samples=n_samples, method=method
                )
                
                # 转换为torch tensor
                if not isinstance(generated_embeddings, torch.Tensor):
                    generated_embeddings = torch.tensor(generated_embeddings, device=device)
                
                # 评估生成的embedding
                eval_results = evaluate_generated_embeddings(
                    generated_embeddings, label, spam_model, spam_tokenizer
                )
                
                method_results[f"scale_{scale}"] = eval_results
                print(f"Label {label}, Method {method}, Scale {scale}: Success Rate = {eval_results['success_rate']:.4f}")
            
            # 保存该方法的所有尺度结果
            label_results[method] = method_results
        
        results[label] = label_results
    
    return results


# 基于标签中心生成新的embedding
def generate_embeddings_from_center(center, intra_class_distance, n_samples=5, method='gaussian'):
    """
    基于标签中心生成新的embedding
    
    参数:
    - center: 标签中心向量
    - intra_class_distance: 类内平均距离
    - n_samples: 要生成的样本数量
    - method: 生成方法 ('gaussian', 'uniform', 'scaled')
    
    返回:
    - 生成的embedding数组
    """
    dim = len(center)
    std = intra_class_distance / np.sqrt(dim)  # 估计每个维度的标准差
    
    if method == 'gaussian':
        # 使用高斯分布生成
        noise = np.random.normal(0, std, size=(n_samples, dim))
    elif method == 'uniform':
        # 使用均匀分布生成
        noise = np.random.uniform(-std*1.5, std*1.5, size=(n_samples, dim))
    elif method == 'scaled':
        # 使用缩放的高斯分布，保持向量长度相似
        raw_noise = np.random.normal(0, 1, size=(n_samples, dim))
        # 归一化并缩放
        norms = np.linalg.norm(raw_noise, axis=1, keepdims=True)
        noise = raw_noise / norms * std * np.sqrt(dim)
    
    # 生成新的embedding
    new_embeddings = center + noise
    
    return new_embeddings



# 可视化并分析模型结果
dataset_name = dataset_name.replace("/", "_")
analysis_results = reduce_and_plot(t5_embeddings, labels, f"Prefix T5 {dataset_name}", dim=2)
analysis_results = reduce_and_plot(t5_embeddings, labels, f"Prefix T5 {dataset_name}", dim=3)

# res = test_center_generation_methods(analysis_results['label_centers'], analysis_results['intra_class_distances'],n_samples=5)
# print(res)

# 保存分析结果到JSON文件
# serializable_results = convert_to_serializable(analysis_results)
# with open(f"Prefix_T5_{dataset_name}_analysis.json", "w") as f:
#     json.dump(serializable_results, f, indent=2)

# print(f"\n分析结果已保存到 Prefix_T5_{dataset_name}_analysis.json")


