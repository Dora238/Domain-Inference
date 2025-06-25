"""Command-line interface for domain discovery pipeline."""

import argparse
from pathlib import Path
import torch
from wordnet_conditioner import WordNetConditioner
from blackbox import BlackBox
from optimizer import ExpansionDirectionOptimizer
from generater_T5 import T5Generator
from collections import OrderedDict
from tqdm import tqdm

def create_argparser():
    
    # 创建解析器
    parser = argparse.ArgumentParser(description="Domain Discovery Pipeline")
    
    # 添加基本参数
    parser.add_argument("--blackbox_model_name", type=str, default="j-hartmann/emotion-english-distilroberta-base", help="HuggingFace model name for black-box classifier,s-nlp/roberta_toxicity_classifier,j-hartmann/emotion-english-distilroberta-base, nlptown/bert-base-multilingual-uncased-sentiment")
    parser.add_argument("--output_dir", type=str, default='/home/dora/Domain-Inference/domain_discover/output', help="Directory to save results")
    
    # 添加x_start优化相关参数
    parser.add_argument("--target_success_rate", type=float, default=0.9, help="目标成功率阈值，范围[0,1]，表示black_box返回1的比例")
    parser.add_argument("--max_iterations", type=int, default=50, help="x_start优化的最大迭代次数")
    parser.add_argument("--num_samples", type=int, default=10, help="每次评估生成的样本数量")
    parser.add_argument("--alpha_max", type=float, default=4.0, help="二分法搜索的alpha的最大值")
    return parser


def load_generator():
    cfg = {
        "model_name": "humarin/chatgpt_paraphraser_on_T5_base",
        "peft_model_path": "/home/dora/Domain-Inference/domain_discover/prefix_paraphraser/checkpoint-65500",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    gen = T5Generator(**cfg)
    return gen

def run_domain_discovery(args):
    """运行域发现管道"""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    black_box = BlackBox(model_name=args.blackbox_model_name)
    t5_generator = load_generator()
    optimizer = ExpansionDirectionOptimizer(decoder=t5_generator, black_box=black_box, eta=args.target_success_rate)

    # Load WordNet conditioner
    wordnet_conditioner = WordNetConditioner(init_method='wordnet', black_model=black_box, max_words=5000, min_words_per_category=20, initial_sentence_from_wordnet=False, t5_generator=t5_generator).to(device)
    word_dict = wordnet_conditioner.word_dict
    word_dict = OrderedDict(sorted(word_dict.items()))
    best_embeddings = []
    best_directions = []
    best_alphas = []
    best_alpha_metrics = []
    best_texts = []
    for label, entry in tqdm(word_dict.items()):
        initial_sentence = entry['center_sentence']
        print(f"Label: {label}, Sentence: {initial_sentence}")
        initial_embedding, attention = t5_generator.encode(initial_sentence)

        Z = initial_embedding  # shape: [L, 768]
        best_alpha, best_embedding, best_direction = optimizer.optimise(Z, target_label=int(label))
        # if best_alpha_metric < best_alpha/best_embedding.shape[0]:
        #     best_alpha_metric = best_alpha/best_embedding.shape[0]
        #     best_embedding = best_embedding
        #     best_direction = best_direction
        #     best_alpha = best_alpha
        best_text = t5_generator.generate_from_hidden_state(best_embedding.unsqueeze(0))
        print(f"Best alpha: {best_alpha}, Best embedding: {best_embedding.shape}, Best direction: {best_direction.shape}, Best alpha metric: {best_alpha_metric}, Best  text: {best_text}")
        best_embeddings.append(best_embedding)
        best_directions.append(best_direction)
        best_alphas.append(best_alpha)
        best_alpha_metrics.append(best_alpha_metric)
        best_texts.append(best_text)

    best_embeddings = torch.stack(best_embeddings, dim=0)
    best_directions = torch.stack(best_directions, dim=0)
    best_alphas = torch.stack(best_alphas, dim=0)
    best_alpha_metrics = torch.stack(best_alpha_metrics, dim=0)
    best_texts = torch.stack(best_texts, dim=0)

    # 记录最好的embedding, direction, alpha, alpha_metric, text
    with open(output_dir / "best_embedding.json", "w") as f:
        json.dump({
            "best_embeddings": best_embeddings.tolist(),
            "best_directions": best_directions.tolist(),
            "best_alphas": best_alphas.tolist(),
            "best_alpha_metrics": best_alpha_metrics.tolist(),
            "best_texts": best_texts.tolist()
        }, f)
    


def main():
    parser = create_argparser()
    args = parser.parse_args()
    run_domain_discovery(args)


if __name__ == "__main__":
    main()
