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
    ① 内层：在固定 Z 上二分搜索 alpha，使 ≥ η 比例方向保持 target_label。
       - 同时测试 ±alpha·Δ，若 –alpha 成功则反转 Δ 的符号。
    ② 外层：若仍有失败方向，则沿 “失败方向均值的反向” 微调 Z，
       不断抬高可行的最大 alpha。
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
        # self.eta = -1
        self.alpha_min        = alpha_min
        self.alpha_max        = alpha_max
        self.eps              = eps
        self.max_binary_steps = max_binary_steps
        self.num_beams        = num_beams

        # 冻结 (或自带) expansion module
        if expansion_module is not None:    
            self.expansion_module = expansion_module
        else:
            self.expansion_module = MultiExpansion()
        self.expansion_module = self.expansion_module.to(self.device)
        self.expansion_module.eval()

    # ========================================================
    # --------- 1. 评估单个方向是否成功 (±alpha) -------------------
    @torch.no_grad()
    def _dir_success(self, Z: torch.Tensor, d: torch.Tensor, alpha: float, target_label: int) -> Tuple[bool, bool]:
        """
        Z:(L,D), d:(L,D)  →  返回 (plus_success, minus_success)
        """
        # seqs是包含label 0和1的的样本
        seqs = []
        # +alpha
        self.num_beams=4
        seqs.append(
            self.decoder.generate_from_hidden_state(
                (Z + alpha * d).unsqueeze(0),
                num_beams     = self.num_beams,
                num_return_sequences = self.num_beams,
                num_beam_groups      = max(1, self.num_beams // 2),
            )
        )
        # –alpha
        seqs.append(
            self.decoder.generate_from_hidden_state(
                (Z - alpha * d).unsqueeze(0),
                num_beams     = self.num_beams,
                num_return_sequences = self.num_beams,
                num_beam_groups      = max(1, self.num_beams // 2),
            )
        )

        results = []
        for seq_batch in seqs:        # 两次 (+alpha, -alpha)
            vote = 0
            for out in seq_batch:
                pred = self.classifier.predict(out)
                vote += int(pred == target_label)
            results.append((vote / self.num_beams) >= self.eta)
        return results[0], results[1]

    # --------- 2. 二分 alpha（带符号翻转 + 失败掩码）-------------
    @torch.no_grad()
    def _binary_search_alpha(self, Z: torch.Tensor, directions: torch.Tensor, target_label: int
                             ) -> Tuple[float, torch.Tensor, torch.Tensor, int]:
        """
        返回：
            best_alpha                 (float, -1 若 alpha_min 也失败)
            signed_dirs  (n,L,D)       —— 若 –alpha 成功，则已取 -d
            failed_mask  (n,) bool     —— 在 best_alpha 上仍失败的方向
            fail_cnt     (int)         —— 仍失败的方向数量
        """
        n, L, _ = directions.shape
        low, high = self.alpha_min, self.alpha_max
        
        # Initialize with failure case
        best_alpha = -1.0
        best_signed = directions.clone()
        best_failed = torch.ones(n, dtype=torch.bool, device=Z.device)
        best_fail_cnt = n * 2  # Max possible fails

        for _ in range(self.max_binary_steps):
            mid_alpha = (low + high) / 2
            if mid_alpha == low or mid_alpha == high: # Avoid infinite loop
                break

            success_cnt = 0
            fail_cnt = 0
            signed_now = directions.clone()
            failed_now = torch.zeros(n, dtype=torch.bool, device=Z.device)

            for i in range(n):
                plus_ok, minus_ok = self._dir_success(Z, directions[i], mid_alpha, target_label)

                if plus_ok or minus_ok:
                    success_cnt += 1
                    if plus_ok and minus_ok:
                        
                        # Both directions work, implies Z is already good.
                        # We can treat this as a success without needing a direction update.
                        signed_now[i].mul_(0)
                        # fail_cnt is not incremented (0 fails)
                    elif minus_ok: # Only minus succeeded
                        signed_now[i].mul_(-1)
                        fail_cnt += 1 # plus failed
                    elif plus_ok: # Only plus succeeded
                        signed_now[i].mul_(1)
                        fail_cnt += 1 # minus failed
                    else: # Both failed
                        failed_now[i] = True
                    signed_now[i].mul_(0) # Not strictly needed but clear
                    fail_cnt += 2

            ratio = success_cnt / n
            if ratio >= self.eta:  # Success rate is good enough, try larger alpha
                print(f"Ratio: {ratio:.2f}, low: {low:.4f}, high: {high:.4f}")  
                best_alpha = mid_alpha
                best_signed = signed_now.clone()
                best_failed = failed_now.clone()
                best_fail_cnt = fail_cnt
                low = mid_alpha
            else:  # Success rate is too low, need smaller alpha
                high = mid_alpha

            if high - low < self.eps:
                break

        return best_alpha, best_signed, best_failed, best_fail_cnt

    # --------- 3. 外层 hill-climb：用失败均值反推 Z ----------
    @torch.no_grad()
    def optimise(self,
                 Z_init: torch.Tensor,
                 target_label: int,
                 max_outer_steps: int = 30,
                 gamma_init: float = 0.08,
                 tol: float = 1e-3,
                 min_step: float = 1e-4
                 ) -> Tuple[float, torch.Tensor, torch.Tensor]:
        """
        通过迭代优化，寻找一个更优的词向量 Z，并返回最后一次迭代的结果。

        返回 (Tuple[float, torch.Tensor, torch.Tensor]):
            - best_alpha (float): 最后一次迭代找到的最佳缩放系数 alpha。
            - Z (torch.Tensor): 经过 max_outer_steps 次优化后的最终词向量。
            - directions (torch.Tensor): 基于最终的 Z 生成的扩展方向。
        """
        Z_best = Z_init.squeeze(0).to(self.device).clone()
        gamma = gamma_init

        # --- 初始评估 ---
        dirs_best = self.expansion_module(Z_best)
        alpha_best, signed_best, failed_best, best_fail_cnt = self._binary_search_alpha(Z_best, dirs_best, target_label)

        if alpha_best < 0:
            print("Initial alpha is negative, optimization failed at start.")
            return -1.0, Z_best, dirs_best
        
        expansion_dir = dirs_best

        # --- 外层迭代: 优化 Z ---
        for i in tqdm(range(max_outer_steps), desc="Optimizing Z"):
            successful_mask = ~failed_best
            # if successful_mask.all():
            #     print(f"\nAll directions succeeded at step {i+1}. Reached local optimum.")
            #     break
            
            if gamma < min_step:
                print(f"\nLearning rate (gamma) too small at step {i+1}. Stopping.")
                break
            
            if not successful_mask.any():
                print(f"\nWarning: No successful directions found at step {i+1}. Cannot update Z.")
                break
            
            # --- 1. 计算更新方向 ---
            avg_successful_dirs = signed_best[successful_mask].mean(dim=0)
            
            # --- 2. 生成候选 Z 并评估 ---
            Z_cand = Z_best + gamma * avg_successful_dirs
            dirs_cand = self.expansion_module(Z_cand)
            alpha_cand, signed_cand, failed_cand, fail_cnt = self._binary_search_alpha(Z_cand, dirs_cand, target_label)

            # --- 3. 接受-拒绝 + 步长自适应 ---
            if (alpha_cand > alpha_best + tol) or (abs(alpha_cand - alpha_best) < tol and fail_cnt < best_fail_cnt):
                print(f"\nAccepted new Z at step {i+1}. Alpha: {alpha_best:.4f} -> {alpha_cand:.4f}, Fails: {best_fail_cnt} -> {fail_cnt}")
                Z_best        = Z_cand
                alpha_best    = alpha_cand
                signed_best   = signed_cand
                failed_best   = failed_cand
                expansion_dir = dirs_cand
                best_fail_cnt = fail_cnt
                gamma *= 1.2
            else:
                print(f"\nRejected new Z at step {i+1}. Alpha: {alpha_best:.4f} -> {alpha_cand:.4f}, Fails: {best_fail_cnt} -> {fail_cnt}")
                
                gamma *= 0.6
                
        print(f"\nOptimization finished. Best alpha: {alpha_best:.4f}")
        return alpha_best, Z_best, expansion_dir