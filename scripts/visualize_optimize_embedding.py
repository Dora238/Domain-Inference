import torch
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

# Assuming config.py exists at the project root and defines PROJECT_ROOT
# If not, we can define it manually:

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))


output_dir  = PROJECT_ROOT / "output/direction_test"
output_dir.mkdir(parents=True, exist_ok=True)
vis_dir = output_dir / "visualizations"
vis_dir.mkdir(exist_ok=True)

direction_nn_path = output_dir / "direction_nn_0.pt"
direction_random_path = output_dir / "direction_random_0.pt"
embedding_nn_path = output_dir / "embedding_nn_0.pt"
embedding_random_path = output_dir / "embedding_random_0.pt"

if not all([p.exists() for p in [direction_nn_path, direction_random_path, embedding_nn_path, embedding_random_path]]):
    print("Error: Not all required .pt files found in output/direction_test.")
    print("Please run scripts/test_direction.py first to generate them.")
    sys.exit(1)

with open(direction_nn_path, "rb") as f:
    direction_nn = torch.load(f)

with open(direction_random_path, "rb") as f:
    direction_random = torch.load(f)
    
with open(embedding_nn_path, "rb") as f:
    embedding_nn = torch.load(f)
    
with open(embedding_random_path, "rb") as f:
    embedding_random = torch.load(f)
    

# visual embedding with expansion direction
def visualize_embedding_with_direction(embedding, direction, title, save_path):
    """
    Visualizes embeddings and their multiple expansion directions using PCA.

    Args:
        embedding (torch.Tensor): Shape [L, D]
        direction (torch.Tensor): Shape [n, L, D] or [L, D]
        title (str): Plot title.
        save_path (str): Path to save the plot.
    """
    # 句子池化
    if embedding.dim() == 2:
        sent_emb = embedding.mean(dim=0, keepdim=True)  # [1, D]
    elif embedding.dim() == 3:
        sent_emb = embedding.mean(dim=1)  # [M, D]
    else:
        raise ValueError("embedding shape不支持")

    # 标准化方向 shape [n, D] 或 [M, n, D]
    if direction.dim() == 3:
        dir_mean = direction.mean(dim=1)  # [n, D]
        dir_mean = dir_mean.unsqueeze(0)  # [1, n, D]
    elif direction.dim() == 2:
        dir_mean = direction.unsqueeze(0)  # [1, n, D]
    elif direction.dim() == 4:
        dir_mean = direction.mean(dim=2)  # [M, n, D]
    else:
        raise ValueError("direction shape不支持")

    # 计算所有扩展点
    all_points = [sent_emb]
    for i in range(dir_mean.shape[0]):  # M
        all_points.append(sent_emb[i:i+1] + dir_mean[i])  # [n, D]
    all_points = torch.cat(all_points, dim=0).cpu().numpy()  # [(1+n)*M, D]

    # PCA降到2维
    pca = PCA(n_components=2)
    all_points_2d = pca.fit_transform(all_points)

    plt.figure(figsize=(10, 8))
    base_idx = 0
    for i in range(sent_emb.shape[0]):
        center = all_points_2d[base_idx]
        plt.scatter(center[0], center[1], c='blue', s=80, label='Sentence Embedding' if i==0 else None)
        for j in range(dir_mean.shape[1]):
            arrow_end = all_points_2d[base_idx + 1 + j]
            plt.arrow(center[0], center[1],
                      arrow_end[0] - center[0], arrow_end[1] - center[1],
                      head_width=0.03, head_length=0.05, fc='red', ec='red', alpha=0.6)
        base_idx += 1 + dir_mean.shape[1]
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")


print("Generating visualization for NN-based expansion direction...")
# Visualize for NN
visualize_embedding_with_direction(
    embedding_nn,
    direction_nn,
    title="Embeddings with NN-based Expansion Direction",
    save_path=vis_dir / "embedding_with_nn_direction.png"
)

print("Generating visualization for random expansion direction...")
# Visualize for Random
visualize_embedding_with_direction(
    embedding_random,
    direction_random,
    title="Embeddings with Random Expansion Direction",
    save_path=vis_dir / "embedding_with_random_direction.png"
)

print("\nVisualizations saved in:", vis_dir)
print(f"You can view them at: {vis_dir.resolve()}")
