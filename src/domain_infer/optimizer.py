import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from tqdm import tqdm

# NOTE: Avoid using Greek letters.

# NOTE: Why use NN to generate expansion directions?
# 1. Is it chosen for parallalization?
# 2. Is it trained in future?
class MultiExpansion(nn.Module):
    def __init__(self, hidden_size=768, hidden_factor=2, num_directions=50, dropout_p=0.0):
        """
        Generates n token-wise expansion directions per input embedding Z.

        Args:
            hidden_size (int): Input/output dimension of token embeddings (e.g., 768).
            hidden_factor (int): Width multiplier for hidden layer.
            num_directions (int): Number of expansion directions to generate.
            dropout_p (float): Dropout probability. Set to 0.0 if module is frozen.
        """
        super().__init__()
        inner_dim = hidden_size * hidden_factor
        self.num_directions = num_directions
        self.hidden_size = hidden_size

        self.projector = nn.Sequential(
            nn.Linear(hidden_size, inner_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(inner_dim, hidden_size * num_directions)
        )

        # Freeze parameters and set eval mode
        self._freeze()

    def _freeze(self):
        """Freeze all parameters and set to eval mode."""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, Z):
        """
        Args:
            Z: Tensor of shape [L, D], token embeddings from encoder

        Returns:
            directions: Tensor of shape [n, L, D], where each n is a normalized expansion vector
        """
        L, D = Z.shape  # Z: [L, D]
        out = self.projector(Z)                     # [L, D × n]
        out = out.view(L, self.num_directions, D)   # [L, n, D]
        out = out.permute(1, 0, 2).contiguous()     # [n, L, D]
        out = F.normalize(out, p=2, dim=-1)         # normalize each direction
        return out



# NOTE: What if best alpha is larger than alpha_max? 
class ExpansionDirectionOptimizer:
    """
    ① 内层：在固定 Z 上二分搜索 α，使 ≥ η 比例方向保持 target_label。
       - 同时测试 ±α·Δ，若 –α 成功则反转 Δ 的符号。
    ② 外层：若仍有失败方向，则沿 “失败方向均值的反向” 微调 Z，
       不断抬高可行的最大 α。
    整个过程仅依赖 hard-label black-box + decoder.generate。
    """

    # ------------------------- init --------------------------
    def __init__(self,
                 decoder,                   # exposes generate_from_hidden_state(Z) -> [str]
                 classifier,                 # exposes predict(text) -> int
                #  target_label: int,
                 expansion_module: Optional[nn.Module] = None,
                 eta: float = 0.9,
                 alpha_min: float = 0.0,
                 alpha_max: float = 2.0,
                 eps: float = 1e-3,
                 max_binary_steps: int = 10,
                 num_beams: int = 8,
                 device: Optional[str] = None):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.decoder      = decoder
        self.classifier    = classifier
        # self.target_label = int(target_label)

        self.eta              = eta
        self.alpha_min        = alpha_min
        self.alpha_max        = alpha_max
        self.eps              = eps
        self.max_binary_steps = max_binary_steps
        self.num_beams        = num_beams

        # 冻结 (或自带) expansion module
        self.expansion_module = expansion_module or MultiExpansion()
        self.expansion_module = self.expansion_module.to(self.device)
        self.expansion_module.eval()

    # ========================================================
    # --------- 1. 评估单个方向是否成功 (±α) -------------------
    @torch.no_grad()
    def _dir_success(self, Z: torch.Tensor, d: torch.Tensor, alpha: float, target_label: int) -> Tuple[bool, bool]:
        """
        Z:(L,D), d:(L,D)  →  返回 (plus_success, minus_success)
        """
        # seqs是包含label 0和1的的样本
        seqs = []
        # +α
        self.num_beams=4
        seqs.append(
            self.decoder.generate_from_hidden_state(
                (Z + alpha * d).unsqueeze(0),
                num_beams     = self.num_beams,
                num_return_sequences = self.num_beams,
                num_beam_groups      = max(1, self.num_beams // 2),
            )
        )
        # –α
        seqs.append(
            self.decoder.generate_from_hidden_state(
                (Z - alpha * d).unsqueeze(0),
                num_beams     = self.num_beams,
                num_return_sequences = self.num_beams,
                num_beam_groups      = max(1, self.num_beams // 2),
            )
        )

        results = []
        for seq_batch in seqs:        # 两次 (+α, -α)
            vote = 0
            for out in seq_batch:
                pred = self.classifier.predict(out)
                vote += int(pred == target_label)
            results.append((vote / self.num_beams) >= self.eta)
        return results[0], results[1]

    # --------- 2. 二分 α（带符号翻转 + 失败掩码）-------------
    @torch.no_grad()
    def _binary_search_alpha(self, Z: torch.Tensor, directions: torch.Tensor, target_label: int
                             ) -> Tuple[float, torch.Tensor, torch.Tensor]:
        """
        返回：
            best_alpha                 (float, -1 若 α_min 也失败)
            signed_dirs  (n,L,D)       —— 若 –α 成功，则已取 -d
            failed_mask  (n,) bool     —— 在 best_alpha 上仍失败的方向
        """
        n, L, _ = directions.shape
        low, high      = self.alpha_min, self.alpha_max
        best_alpha     = -1.0
        best_signed    = directions.clone()
        best_failed    = torch.ones(n, dtype=torch.bool, device=Z.device)

        sqrt_L = L ** 0.5

        for _ in range(self.max_binary_steps):
            mid_alpha  = (low + high) / 2
            success_cnt = 0
            signed_now  = directions.clone()   # 每轮临时
            failed_now  = torch.zeros(n, dtype=torch.bool, device=Z.device)

            # ---- 逐方向评估 (+α / -α) ----------------------
            for i in range(n):
                plus_ok, minus_ok = self._dir_success(Z, directions[i], mid_alpha, target_label)

                if plus_ok:
                    success_cnt += 1
                elif minus_ok:
                    success_cnt += 1
                    signed_now[i].mul_(-1)       # 方向翻转
                else:
                    failed_now[i] = True         # 双向都失败

            ratio = success_cnt / n
            # NOTE: Would ratio fluctuates too much in repeting experiments?
            if ratio >= self.eta:                # 合格 → 加大 alpha
                best_alpha  = mid_alpha
                best_signed = signed_now.clone()
                best_failed = failed_now.clone()
                low         = mid_alpha
            else:                                # 不合格 → 缩小 alpha
                high = mid_alpha

            if high - low < self.eps:
                break

        return best_alpha, best_signed, best_failed

    # --------- 3. 外层 hill-climb：用失败均值反推 Z ----------
    @torch.no_grad()
    def optimise(self,
                 Z_init: torch.Tensor,
                 target_label: int,
                 max_outer: int = 30,
                 gamma_init: float = 0.08,
                 tol: float = 1e-3,
                 min_step: float = 1e-4
                 ) -> Tuple[float, float, torch.Tensor]:
        """
        返回：
            alpha_raw, alpha_scaled (= α/√L), signed_dirs
        """
        Z_best   = Z_init.squeeze(0).to(self.device).clone()
        gamma    = gamma_init

        # 初始方向
        dirs_best = self.expansion_module(Z_best)
        α_best, signed_best, failed_best = self._binary_search_alpha(Z_best, dirs_best, target_label)

        if α_best < 0:                       # α_min 都失败
            return -1.0, -1.0, signed_best
        expansion_dir = None
        # ------------ 外层迭代:优化偏倚步长 ------------------------------
        for _ in tqdm(range(max_outer)): 
            if (~failed_best).all():         # 所有方向成功 ⇒ 局部最优
                break
            if gamma < min_step:
                break

            # 1) 失败方向均值
            mean_fail = signed_best[failed_best].mean(dim=0)
            mean_fail = mean_fail / mean_fail.norm()    # 单位化

            # 2) 反向位移
            Z_cand = Z_best - gamma * mean_fail

            # 3) 在新 Z 上重新生成方向并二分 α
            dirs_cand = self.expansion_module(Z_cand)
            α_cand, signed_cand, failed_cand = self._binary_search_alpha(Z_cand, dirs_cand, target_label)

            # 4) 接受-拒绝 + 步长自适应
            if α_cand > α_best + tol:
                Z_best, α_best = Z_cand, α_cand
                signed_best, failed_best = signed_cand, failed_cand
                gamma *= 1.2     # 放大步长
                expansion_dir = dirs_cand
            else:
                gamma *= 0.6     # 缩小步长
        print(f"Best alpha: {α_best}")
        print(f"Best embedding: {Z_best}")
        return α_best, Z_best, expansion_dir