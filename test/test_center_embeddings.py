import torch
import numpy as np
from tqdm import tqdm
import json
from test_prefix_T5 import (
    load_dataset, get_hidden_states, load_model,
    generate_embeddings_from_center, evaluate_generated_embeddings,
    test_center_generation_methods
)

if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据集
    dataset_name = "ucirvine_sms_spam"
    texts, labels = load_dataset(dataset_name)
    
    # 加载模型
    model_name = "humarin/chatgpt_paraphraser_on_T5_base"
    t5_model, tokenizer = load_model(model_name, device)
    
    # 获取hidden states
    hidden_states, labels = get_hidden_states(t5_model, tokenizer, texts, labels, device)
    
    # 计算每个标签的中心和类内距离
    label_centers = {}
    intra_class_distances = {}
    
    for label in set(labels):
        # 获取该标签的所有样本
        label_indices = [i for i, l in enumerate(labels) if l == label]
        label_embeddings = torch.tensor(np.array([hidden_states[i] for i in label_indices]), device=device)
        
        # 计算中心
        center = torch.mean(label_embeddings, dim=0)
        label_centers[label] = center
        
        # 计算类内距离
        distances = torch.norm(label_embeddings - center.unsqueeze(0), dim=1)
        intra_class_distances[label] = torch.mean(distances).item()
        print(f"Label {label} - 中心向量形状: {center.shape}, 类内平均距离: {intra_class_distances[label]:.4f}")
    
    # 测试从标签中心生成embedding
    print("\n开始测试从标签中心生成embedding...")
    test_results = test_center_generation_methods(
        label_centers, 
        intra_class_distances,
        n_samples=5
    )
    
    # 保存测试结果
    with open(f"center_generation_test_results.json", "w") as f:
        # 转换结果为可序列化格式
        serializable_results = {}
        for label, methods in test_results.items():
            serializable_results[str(label)] = {}
            for method, results in methods.items():
                serializable_results[str(label)][method] = {
                    'success_rate': results['success_rate'],
                    'predictions': results['predictions'],
                    'confidence_scores': [float(score) for score in results['confidence_scores']],
                    'generated_texts': results['generated_texts']
                }
        
        json.dump(serializable_results, f, indent=2)
    
    print("\n测试结果已保存到 center_generation_test_results.json")
