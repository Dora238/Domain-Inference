# t5_generator.py
from __future__ import annotations
import torch
from typing import List, Tuple, Dict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel


class T5Generator:
    """Prefix-tuned T5 文本生成（支持 step-by-step diverse-beam）。"""

    # --------------------------------------------------------------------- #
    # 初始化
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        model_name: str = "humarin/chatgpt_paraphraser_on_T5_base",
        peft_model_path: str | None = None,
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        base = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model = (
            PeftModel.from_pretrained(base, peft_model_path)
            if peft_model_path else base
        )
        self.model.to(self.device).eval()

        # 常用 ID
        self.pad_id   = self.tokenizer.pad_token_id
        self.eos_id   = self.tokenizer.eos_token_id
        self.start_id = (
            self.model.config.decoder_start_token_id
            if self.model.config.decoder_start_token_id is not None
            else (self.pad_id or self.eos_id)
        )

    # --------------------------------------------------------------------- #
    # 编码
    # --------------------------------------------------------------------- #
    @torch.inference_mode()
    def encode(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = self.tokenizer(
            text, return_tensors="pt",
            padding=True, truncation=True, max_length=512
        ).to(self.device)
        out = self.model.get_encoder()(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            return_dict=True,
        )
        return out.last_hidden_state, inputs.attention_mask

    # --------------------------------------------------------------------- #
    # 单步解码
    # --------------------------------------------------------------------- #
    @torch.inference_mode()
    def decode_step(
        self,
        encoder_hidden_states,
        encoder_attention_mask,
        decoder_input_ids,
        past_key_values=None,
    ):
        if decoder_input_ids.dim() == 1:
            decoder_input_ids = decoder_input_ids.unsqueeze(-1)

        dec_out = self.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        hidden = dec_out.last_hidden_state
        logits = self.model.get_output_embeddings()(hidden)
        return logits, dec_out.past_key_values

    # --------------------------------------------------------------------- #
    # 重复惩罚
    # --------------------------------------------------------------------- #
    @staticmethod
    def apply_repetition_penalty(logits: torch.Tensor,
                                 seqs: torch.Tensor,
                                 penalty: float):
        if penalty == 1.0:
            return
        for b, seq in enumerate(seqs):
            uniq = torch.unique(seq)
            neg_mask = logits[b, uniq] < 0
            logits[b, uniq[neg_mask]] *= penalty
            logits[b, uniq[~neg_mask]] /= penalty

    # --------------------------------------------------------------------- #
    # n-gram mask
    # --------------------------------------------------------------------- #
    @staticmethod
    def banned_ngram_mask(seqs: torch.Tensor,
                          vocab: int,
                          n: int) -> torch.BoolTensor:
        B, T = seqs.size()
        mask = torch.zeros(B, vocab, dtype=torch.bool, device=seqs.device)
        if n <= 0 or T + 1 < n:
            return mask

        for b in range(B):
            seq = seqs[b].tolist()
            hist: Dict[Tuple[int, ...], set[int]] = {}
            for i in range(len(seq) - n + 1):
                key = tuple(seq[i : i + n - 1])
                nxt = seq[i + n - 1]
                hist.setdefault(key, set()).add(nxt)
            key = tuple(seq[-(n - 1):])
            if key in hist:
                mask[b, list(hist[key])] = True
        return mask

    # --------------------------------------------------------------------- #
    # 重排 past
    # --------------------------------------------------------------------- #
    @staticmethod
    def reorder_past(past, beam_idx: torch.Tensor):
        beam_idx = beam_idx.to(past[0][0].device)
        return tuple(tuple(p.index_select(0, beam_idx) for p in layer) for layer in past)

    # --------------------------------------------------------------------- #
    # 主：分步 diverse-beam
    # --------------------------------------------------------------------- #
    def generate_step_by_step(
        self,
        text: str,
        *,
        max_length: int = 64,
        num_beams: int = 6,
        num_beam_groups: int = 3,
        num_return_sequences: int = 3,
        do_sample: bool = False,
        temperature: float = 1.0,
        repetition_penalty: float = 1.2,
        no_repeat_ngram_size: int = 2,
        diversity_penalty: float = 3.0,
        length_penalty: float = 1.0,
    ) -> List[str]:

        if num_return_sequences > num_beams:
            raise ValueError("num_return_sequences 不能超过 num_beams")
        if num_beams % num_beam_groups != 0:
            raise ValueError("num_beams 必须能被 num_beam_groups 整除")

        group_size = num_beams // num_beam_groups

        # 编码 enc_hid size [num_beams, seq_len, hidden_size]
        # enc_mask size [num_beams, seq_len]
        enc_hid, enc_mask = self.encode(text)
        enc_hid  = enc_hid.expand(num_beams, -1, -1).contiguous()
        enc_mask = enc_mask.expand(num_beams, -1).contiguous()

        # 初始 beam
        seqs   = torch.full((num_beams, 1), self.start_id,
                            dtype=torch.long, device=self.device)
        scores = torch.zeros(num_beams, device=self.device)
        finished = torch.zeros(num_beams, dtype=torch.bool, device=self.device)
        past_key_values = None

        # 解码循环
        for step in range(max_length):
            cur_in = seqs[:, -1].unsqueeze(-1) if past_key_values is not None else seqs
            logits, past_key_values = self.decode_step(
                enc_hid, enc_mask, cur_in, past_key_values
            )
            logits = logits[:, -1, :].clone()          # 克隆避免 inplace

            # 限制
            if repetition_penalty > 1.0:
                self.apply_repetition_penalty(logits, seqs, repetition_penalty)
            if no_repeat_ngram_size > 0:
                mask = self.banned_ngram_mask(seqs, logits.size(-1), no_repeat_ngram_size)
                logits = logits.masked_fill(mask, -float("inf"))

            # 采样 / 贪婪
            if do_sample:
                if temperature != 1.0:
                    logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token_scores = torch.log(probs + 1e-10)
            else:
                next_token_scores = torch.log_softmax(logits, dim=-1)

            next_scores = scores.unsqueeze(1) + next_token_scores
            group_view  = next_scores.view(num_beam_groups, group_size, -1)

            # Hamming diverse-beam
            beam_next_tokens, beam_next_scores, beam_next_indices = [], [], []
            token_freq = torch.zeros(         # 所有组共享
                self.model.config.vocab_size,
                device=self.device, dtype=torch.long
            )

            for g in range(num_beam_groups):
                group_scores = group_view[g]

                if diversity_penalty > 0.0 and step > 0:
                    penalty = diversity_penalty * token_freq.to(group_scores.dtype)
                    group_scores = group_scores - penalty

                flat = group_scores.view(-1)
                best_scores, best_idx = torch.topk(flat, k=group_size, dim=0)
                orig_beam = best_idx // logits.size(-1) + g * group_size
                next_tok  = best_idx %  logits.size(-1)

                beam_next_tokens.append(next_tok)
                beam_next_scores.append(best_scores)
                beam_next_indices.append(orig_beam)

                # 把已选 token 纳入频次，供后续组惩罚
                token_freq.scatter_add_(0, next_tok, torch.ones_like(next_tok, dtype=torch.long))

            next_tokens = torch.cat(beam_next_tokens,  dim=0)
            next_scores = torch.cat(beam_next_scores,  dim=0)
            old_indices = torch.cat(beam_next_indices, dim=0)

            # 更新
            seqs   = torch.cat([seqs[old_indices], next_tokens.unsqueeze(1)], dim=1)
            scores = next_scores
            finished = finished[old_indices] | next_tokens.eq(self.eos_id)

            enc_hid  = enc_hid.index_select(0, old_indices)
            enc_mask = enc_mask.index_select(0, old_indices)
            if past_key_values is not None:
                past_key_values = self.reorder_past(past_key_values, old_indices)

            if finished.all():
                break

        # 长度惩罚
        seq_lens = seqs.shape[1] * torch.ones_like(scores)
        if self.eos_id is not None:
            for i in range(num_beams):
                eos_pos = (seqs[i] == self.eos_id).nonzero(as_tuple=True)[0]
                if eos_pos.numel():
                    seq_lens[i] = eos_pos[0] + 1

        adj = scores / (((5.0 + seq_lens.float()) / 6.0) ** length_penalty)

        # 结果去重 + 补足
        uniq_text, uniq_idx = set(), []
        ranked = torch.topk(adj, k=num_beams).indices.tolist()
        for idx in ranked:
            txt = self.tokenizer.decode(
                seqs[idx, :int(seq_lens[idx])], skip_special_tokens=True
            )
            if txt not in uniq_text:
                uniq_text.add(txt)
                uniq_idx.append(idx)
            if len(uniq_idx) == num_return_sequences:
                break
        if len(uniq_idx) < num_return_sequences:
            extra = [i for i in ranked if i not in uniq_idx][: num_return_sequences - len(uniq_idx)]
            uniq_idx.extend(extra)
        result = [
            self.tokenizer.decode(
                seqs[i, :int(seq_lens[i])], skip_special_tokens=True
            ) for i in uniq_idx
        ]
        return result

    # --------------------------------------------------------------------- #
    # baseline：直接调用 HF generate
    # --------------------------------------------------------------------- #
    @torch.inference_mode()
    def generate(
        self,
        text: str,
        num_beams: int = 1,
        num_beam_groups: int = 1,
        diversity_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        max_length: int = 100,
        temperature: float = 1.0,
        num_return_sequences: int = 1,
        do_sample: bool = False,
    ) -> List[str]:
        inputs = self.tokenizer(
            text, return_tensors="pt",
            padding=True, truncation=True, max_length=512
        ).to(self.device)
        inputs = {k: v for k, v in inputs.items()
                  if k in {"input_ids", "attention_mask"}}

        outs = self.model.generate(
            **inputs,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            do_sample=do_sample,
            pad_token_id=self.pad_id,
            eos_token_id=self.eos_id,
            decoder_start_token_id=self.start_id,
        )
        return [self.tokenizer.decode(o, skip_special_tokens=True) for o in outs]


# ------------------------------------------------------------------------- #
# DEMO
# ------------------------------------------------------------------------- #
if __name__ == "__main__":
    cfg = {
        "model_name": "humarin/chatgpt_paraphraser_on_T5_base",
        "peft_model_path": "/home/dora/Domain-Inference/domain_discover/prefix_paraphraser/checkpoint-65500",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    gen = T5Generator(**cfg)

    text = "I am happy"

    print("=== HF generate() ===")
    original = gen.generate(
        text,
        max_length=64,
        num_beams=6, num_beam_groups=3,
        diversity_penalty=3.0,
        no_repeat_ngram_size=2,
        repetition_penalty=1.2,
        num_return_sequences=3,
        do_sample=False,
    )
    print(*original, sep="\n")

    print("\n=== step-by-step ===")
    step_by_step = gen.generate_step_by_step(
        text,
        max_length=50,
        num_beams=6, num_beam_groups=3,
        diversity_penalty=3.0,
        no_repeat_ngram_size=2,
        repetition_penalty=1.2,
        num_return_sequences=3,
    )
    print(*step_by_step, sep="\n")
