import torch
import torch.nn as nn

class ExpansionModule(nn.Module):
    def __init__(self, hidden_size=768, hidden_factor=2):
        """
        Token-wise residual MLP expansion module.
        Args:
            hidden_size: 输入和输出的维度，通常为 768。
            hidden_factor: 中间层宽度比例，例如 2 → 中间层为 1536。
        """
        super().__init__()
        inner_dim = hidden_size * hidden_factor
        self.adapter = nn.Sequential(
            nn.Linear(hidden_size, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, hidden_size)
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, Z):
        """
        Z: [L, D] token embeddings
        return: Δ ∈ [L, D] token-wise perturbation
        """
        # return self.dropout(self.adapter(Z)) + 0.01 * Z
        return self.dropout(self.adapter(Z))


class ExpansionDirectionOptimizer:
    def __init__(self, 
                 decoder,
                 black_box,
                 label,
                 eta=0.9,
                 alpha_min=0.0,
                 alpha_max=2.0,
                 eps=1e-3,
                 max_binary_steps=20):
        """
        Args:
            expansion_module: a module that takes Z [L,D] and outputs Δ [L,D]
            decoder: should have generate_from_hidden(Z') → str
            black_box: should have predict_proba(text)[label] → float
            label: target label
            eta: minimum confidence threshold
            alpha_min, alpha_max: initial search range for α
            eps: binary search precision
            max_binary_steps: max binary search iterations
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.expansion_module = ExpansionModule().to(device)
        self.decoder = decoder
        self.black_box = black_box
        self.label = int(label)
        self.eta = eta
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.eps = eps
        self.max_binary_steps = max_binary_steps

    def binary_search_alpha(self, Z, delta):
        """
        二分法搜索 α 使得 Pr ≥ η，返回最小 α（扰动强度）及生成文本与得分
        """
        low = self.alpha_min
        high = self.alpha_max
        best_alpha = None
        best_text = None
        best_prob = 0.0

        for _ in range(self.max_binary_steps):
            mid = (low + high) / 2
            Z_prime = Z + mid * delta
            gen_texts = self.decoder.generate_from_hidden_state(Z_prime, num_beams=10,num_beam_groups=5, num_return_sequences=10)
            gen_texts = [text.strip() for text in gen_texts]
            probs = 0
            for gen_text in gen_texts:
                prob = (self.black_box.predict(gen_text)[0] == self.label)
                probs += prob
            probs /= len(gen_texts)
            if probs >= self.eta:
                best_alpha = mid
                best_text = gen_texts[0]
                best_prob = probs
                low = mid
            else:
                high = mid

            if high - low < self.eps:
                break

        return best_alpha, best_text, best_prob

    def optimize(self, Z, num_directions=100):
        """
        尝试 num_directions 个 Δ，找出能支持最大 α 的方向

        Args:
            Z: torch.Tensor [L, D] 初始 embedding
            num_directions: 生成的扰动方向数

        Returns:
            best_delta, best_alpha, best_text, best_prob
        """
        best_alpha = -1
        best_delta = None
        best_text = None
        best_prob = 0.0

        with torch.no_grad():
            for _ in range(num_directions):
                self.expansion_module.eval()
                delta = self.expansion_module(Z)                # Δ ∈ [L, D]
                delta = delta / (delta.norm(dim=-1, keepdim=True) + 1e-8)  # optional normalize

                alpha, text, prob = self.binary_search_alpha(Z, delta)
                if alpha is not None and alpha > best_alpha:
                    best_alpha = alpha
                    best_delta = delta
                    best_text = text
                    best_prob = prob

        return best_delta, best_alpha, best_text, best_prob
