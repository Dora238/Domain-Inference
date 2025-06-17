import torch, random
import numpy as np
import cma
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer
from typing import Callable, List, Tuple, Optional
from torch import nn
## 验证 training set 的embedding space 
## 预训练 模型的encoder
## 生成能力
# ---------- 核心类：软前缀 + 冻结 GPT-2 ----------
class SoftPromptGPT2(nn.Module):
    def __init__(
        self,
        base_model: str = "gpt2",
        prefix_len: int = 10,
        initial_words: Optional[List[str]] = None,   # ← 新增：给一组词
        random_scale: float = 0.02,                  # 初始化时的小噪声
        device: Optional[str] = None,
    ):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gpt2 = GPT2LMHeadModel.from_pretrained(base_model).to(self.device)
        self.gpt2.eval().requires_grad_(False)
        self.tokenizer = GPT2Tokenizer.from_pretrained(base_model)

        d_model = self.gpt2.config.n_embd
        wte = self.gpt2.transformer.wte                    # 词向量表 [V, d]

        # -------- ① 计算 init_embed --------
        if initial_words:   
            initial_word_single = [initial_words[0]]                               # 若给了初始词
            ids = []
            for w in initial_word_single:
                ids.extend(
                    self.gpt2.transformer.wte.weight.new_tensor(
                        self._tokenize_no_special(w), dtype=torch.long
                    ).tolist()
                )
            ids = torch.tensor(ids, dtype=torch.long, device=self.device)
            base_vec = wte(ids).mean(dim=0)                # [d]
            # 把 base_vec 复制 prefix_len 次并加微小高斯噪声
            init_embed = base_vec.repeat(prefix_len, 1) + random_scale * torch.randn(prefix_len, d_model, device=self.device)
        else:                                              # 否则随机
            init_embed = torch.randn(prefix_len, d_model, device=self.device) * 0.02

        # -------- ② 注册为可学习参数 --------
        self.prefix = nn.Parameter(init_embed, requires_grad=False)
        self.wte = wte                                     # 保存引用

    @torch.no_grad()
    def generate(
        self,
        prefix: torch.Tensor,
        max_length: int = 30,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> str:
        """
        prefix + 明文 prompt  → 生成一句文本
        """
        # 1) 明文 prompt → embedding
        prompt_variants = [
            "Please write an English sentence about: ",
            "Generate a sentence about the following topic: ",
            "Make a sentence using the concept of: ",
            "Here is a topic. Write one sentence: ",
            "A sentence that illustrates the idea of: ",
        ]
        prompt_text = random.choice(prompt_variants)
        tok = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.device)
        prompt_embeds = self.gpt2.transformer.wte(tok.input_ids)  # [1, L, d]

        # 2) 拼接软前缀
        full_embeds = torch.cat([prefix.float().unsqueeze(0), prompt_embeds], dim=1)

        # 3) 位置编码索引
        pos_ids = torch.arange(full_embeds.size(1), device=self.device).unsqueeze(0)

        # 4) 调用 GPT-2 generate
        gen_ids = self.gpt2.generate(
            inputs_embeds=full_embeds,
            position_ids=pos_ids,
            max_length=max_length,  # 总长（含 prefix+prompt）
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)

    @torch.no_grad()
    def generate_from_embedding(
        self,
        ext_embed: torch.Tensor,
        *,
        prompt_text: str = "",               # 可为空串
        max_length: int = 40,
        temperature: float = 1.0,
        top_p: float = 0.9,
        force_words_ids: Optional[List[List[int]]] = None,
        **generate_kwargs,
    ) -> str:
        """
        使用 *外部* embedding 作为软前缀生成文本。
        ----------
        参数
        ext_embed : torch.Tensor
            - 若形状 == (d_model,)  ：自动 broadcast 到 prefix_len×d
            - 若形状 == (k, d_model)：直接作为软前缀，k 可与 self.prefix_len 不同
        prompt_text : str
            追加在 embedding 后的明文 prompt，可以是 ""。
        force_words_ids : transformers.generate 的原生参数，
            可用于强制某些 token 出现（例如保证含 anchor 词）。
        generate_kwargs : 透传给 `self.gpt2.generate` 的其他参数
            (如 num_beams, repetition_penalty 等)。
        ----------
        返回
        str : 生成的文本（去掉 special tokens）
        """
        # ---- 1. 解析 ext_embed 形状 ----------------------------------------
        if ext_embed.dim() == 1:                       # [d] → broadcast
            ext_embed = ext_embed.unsqueeze(0).repeat(self.prefix.shape[0], 1)
        elif ext_embed.dim() == 2:
            pass  # (k,d) 直接用
        else:
            raise ValueError("ext_embed 必须是 (d,) 或 (k,d) 张量")

        # 把 ext_embed 移到同设备 + 同 dtype
        ext_embed = ext_embed.to(self.prefix.dtype).to(self.device)

        # ---- 2. prompt → embedding ----------------------------------------
        if prompt_text:
            tok = self.tokenizer(
                prompt_text, return_tensors="pt",
                add_special_tokens=False
            ).to(self.device)
            prompt_embeds = self.gpt2.transformer.wte(tok.input_ids)  # [1,L,d]
            full_embeds = torch.cat([ext_embed.unsqueeze(0), prompt_embeds], 1)
        else:
            full_embeds = ext_embed.unsqueeze(0)                      # [1,k,d]

        # ---- 3. 位置编码索引 ----------------------------------------------
        pos_ids = torch.arange(full_embeds.size(1),
                               device=self.device).unsqueeze(0)

        # ---- 4. 生成 ------------------------------------------------------
        gen_ids = self.gpt2.generate(
            inputs_embeds = full_embeds,
            position_ids  = pos_ids,
            max_length    = max_length,
            do_sample     = True,
            temperature   = temperature,
            top_p         = top_p,
            force_words_ids = force_words_ids,
            pad_token_id  = self.tokenizer.eos_token_id,
            **generate_kwargs
        )
        return self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)

    @torch.no_grad()
    def generate_from_text(self, text: str, max_length=50, top_p=0.9, temperature=1.0):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        output_ids = self.gpt2.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)


    # -------------------------------------------------
    #                 私 有 工 具
    # -------------------------------------------------
    def _encode_no_special(self, text: str) -> List[int]:
        """
        不加 BOS / EOS / padding，只做 BPE
        """
        return self.tokenizer.encode(text, add_special_tokens=False, add_prefix_space=True)

    # helper：禁用 BOS/EOS
    def _tokenize_no_special(self, text: str) -> List[int]:
        tok = GPT2Tokenizer.from_pretrained("gpt2")
        return tok(text, add_special_tokens=False).input_ids

def optimize_soft_prompt(
    target_label: int,
    black_box: Callable[[str], int],
    *,
    base_model: str = "gpt2",
    prefix_len: int = 10,
    prompt_text: str = " Please write an English sentence about: ",
    # 新增 ↓↓↓
    initial_words: Optional[List[str]] = None,
    init_embed:    Optional[np.ndarray] = None,
    # ---- 其余与旧版一致 ----
    pop_size: int = 8,
    sigma0: float = 0.5,
    iterations: int = 30,
    samples_per_eval: int = 50,
    temperature: float = 1.0,
    top_p: float = 0.9,
    seed: int = 42,
) -> Tuple[np.ndarray, List[str]]:
    """
    优化一段软前缀，使生成句子尽量被 black_box 分类为 target_label。

    参数说明（新增部分）
    ----------------
    initial_words : List[str] or None
        一组“概念词”，将取它们在 GPT-2 词向量表中的平均值作为软前缀初始化。
    init_embed    : np.ndarray or None
        如果已经用 WordNet 聚类好向量，可直接传进来；优先级高于 initial_words。
    """

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2Tokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    # ---------- 构造 SoftPromptGPT2 ----------
    model = SoftPromptGPT2(
        base_model   = base_model,
        prefix_len   = prefix_len,
        initial_words= initial_words,
        device       = device
    )

    # --- CMA-ES 参数向量化 ---
    z_flat0 = model.prefix.detach().cpu().numpy().reshape(-1)
    es = cma.CMAEvolutionStrategy(z_flat0, sigma0,
                                  {"popsize": pop_size, "seed": seed})

    def _fitness(vec_flat: np.ndarray) -> float:
        """CMA-ES 最小化 → 取负的命中率"""
        with torch.no_grad():
            model.prefix.copy_(
                torch.tensor(vec_flat.reshape(model.prefix.shape), device=device)
            )
            hits = 0
            for _ in range(samples_per_eval):
                txt = model.generate(temperature=temperature, top_p=top_p)
                hits += (black_box.predict(txt) == target_label)
            print(f"hits: {hits}")
            return -hits / samples_per_eval             # 越小越好

    # ---------- 优化循环 ----------
    for gen in range(iterations):
        candidates = es.ask()
        fitness    = [_fitness(c) for c in candidates]
        es.tell(candidates, fitness)
        best = -min(fitness)
        print(f"[Iter {gen+1:02d}] best hit-rate = {best:.3f}")

    best_vec = es.result.xbest
    model.prefix.copy_(torch.tensor(best_vec.reshape(model.prefix.shape), device=device))

    # ---------- 输出 ----------
    final_sents = [model.generate(temperature=temperature, top_p=top_p)
                   for _ in range(100)]

    return best_vec.reshape(model.prefix.shape), final_sents


def optimize_dimension(label, words, black_box, target_dimension=100):
    # 1. 初始化
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

    # 2. 选取能触发 label 的词
    valid_words, embeds = [], []
    for w in words:
        tokenized = tokenizer(w, add_special_tokens=False)
        if len(tokenized.input_ids) != 1:
            continue  # 跳过非单词项
        prompt = (
            f"Write one short English sentence using the word '{w}'. Only output the sentence.\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        sentence_gpt_ids =  model.generate(**inputs, max_length=64, temperature=0.8, top_p=0.95, do_sample=True, pad_token_id=tokenizer.eos_token_id)
        # 解码全部生成的文本
        full_output = tokenizer.decode(sentence_gpt_ids[0], skip_special_tokens=True)
        # 去掉前缀 prompt
        if prompt in full_output:
            sentence_gpt = full_output.replace(prompt, "").strip()
        else:
            sentence_gpt = full_output.strip()
        if black_box.predict(sentence_gpt) == label:
            emb = model.transformer.wte.weight[tokenized.input_ids[0]]
            valid_words.append(w)
            embeds.append(emb.detach().cpu().numpy())

    # 3. PCA 降维
    X = np.stack(embeds)  # [N, d]
    pca = PCA(n_components=target_dimension)
    Z = pca.fit_transform(X)  # [N, target_dimension]

    return Z, pca, valid_words  # 降维向量，PCA映射，选中词


# def optimize_dimension(label, words, black_box, target_dimension=100):
#     torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#         # --- 构造软前缀模型 ------------------------------------------------------
#     gpt_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
#     gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#     gpt_tokenizer.pad_token = gpt_tokenizer.eos_token

    ## 第一步，对words进行遍历，挑选embedding为1的word
    ## 第二步：对于word的embedding进行降维,要求即使降维后，用这个embedding生成的句子在black_box.predict()==label
    ## 第三步：输出降维后的embedding，用于之后cma算法


def optimize_soft_prompt_dynamic(
    target_label: int,
    black_box: Callable[[str], int],              # str → label
    *,
    base_model: str = "gpt2",
    prefix_len: int = 10,
    initial_words: Optional[List[str] | str] = None,
    init_embed: Optional[np.ndarray] = None,      # (k,d) or None
    pop_size: int = 8,
    sigma0: float = 0.5,
    iterations: int = 30,
    samples_per_eval: int = 50,
    temperature: float = 1.0,
    top_p: float = 0.9,
    mu: float = 0.05,                             # 距离正则权重
    seed: int = 42,
) -> Tuple[np.ndarray, List[str]]:
    """
    返回:
        best_prefix  : np.ndarray (k,d)  最优软前缀
        best_samples : List[str]         用最优前缀生成的 100 句
    """

    # --- reproducibility ----------------------------------------------------
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 构造软前缀模型 ------------------------------------------------------
    model = SoftPromptGPT2(
        base_model     = base_model,
        prefix_len     = prefix_len,
        initial_words  = initial_words,
        # init_embed     = None if init_embed is None else torch.tensor(init_embed),
        device         = device,
    )
    tokenizer = model.tokenizer

    # 保存初始前缀，用于距离正则
    z_init = model.prefix.detach().clone()

    # ---------- 预计算 wte.weight 归一化（余弦最近邻） ------------------------
    wte = model.gpt2.transformer.wte                    # [V, d]
    with torch.no_grad():
        wte_norm = wte.weight / wte.weight.norm(dim=1, keepdim=True)  # [V,d]

    def nearest_phrase(vec: torch.Tensor,
                    k: int = 3) -> str:
        """
        取 vec 在词表中的余弦最近 k 个 token，再拼接成短语
        e.g. ['comp', 'le', 'te'] → 'complete'
        """
        vec = vec.to(wte_norm.dtype) / vec.norm()
        sim, idxs = torch.topk(torch.matmul(wte_norm, vec), k=k)
        tokens = tokenizer.convert_ids_to_tokens(idxs.tolist(), skip_special_tokens=True)

        # 将 GPT-2 BPE token 拼接回字符串
        phrase = "".join([t.lstrip("Ġ") if i else t for i, t in enumerate(tokens)])
        return phrase.strip()

    # ---------- CMA-ES 设置 --------------------------------------------------
    z_flat0 = model.prefix.detach().cpu().numpy().reshape(-1)
    es = cma.CMAEvolutionStrategy(
        z_flat0,
        sigma0,
        {"popsize": pop_size, "seed": seed},
    )

    # ---------- 适应度函数 ----------------------------------------------------
    def fitness(vec_flat: np.ndarray) -> float:
        """CMA-ES 最小化 → 返回 (-reward)"""
        # 1) 更新 prefix
        prefix_tensor = torch.tensor(
            vec_flat.reshape(model.prefix.shape), device=device)

        model.prefix.copy_(prefix_tensor)
        
        # 3) 生成 & 评估
        hits = 0
        for _ in range(samples_per_eval):
            text = model.generate(
                prefix      = prefix_tensor,
                max_length  = 40,
                temperature = temperature,
                top_p       = top_p,
            )
            hits += (black_box.predict(text) == target_label)
            # print(f"text: {text}")
            # print(f"hits: {hits}")
        label_rate = hits / samples_per_eval            # 主 reward

        # 4) 距离正则
        dist_pen = ((prefix_tensor - z_init)**2).mean().item()

        reward = label_rate - mu * dist_pen             # 可加其它项
        return -reward                                  # CMA-ES → minimization

    # ---------- 主循环 -------------------------------------------------------
    for gen in range(1, iterations + 1):
        cands   = es.ask()
        scores  = [fitness(c) for c in cands]           # 越小越好
        es.tell(cands, scores)

        print(f"[{gen:02d}/{iterations}]  best_hit = {-es.best.f:.3f}")
        print(f"anchor = {nearest_phrase(torch.tensor(es.result.xbest.reshape(model.prefix.shape),device=device).mean(dim=0))}")

    # ---------- 结果输出 -----------------------------------------------------
    best_flat = es.result.xbest
    model.prefix.copy_(torch.tensor(best_flat.reshape(model.prefix.shape), device=device))

    final_samples = [
        model.generate(
            prefix      = torch.tensor(best_flat.reshape(model.prefix.shape), device=device),
            max_length  = 40,
            temperature = temperature,
            top_p       = top_p,
        ) for _ in range(100)
    ]

    return best_flat.reshape(model.prefix.shape), final_samples
