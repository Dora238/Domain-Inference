from __future__ import annotations

import torch
import random
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, PeftConfig

class T5Generator:
    """
    使用前缀微调的T5模型进行文本生成的类。
    该类提供了一个分步生成过程，明确分离编码器和解码器步骤。
    """
    
    def __init__(
        self,
        model_name: str = "humarin/chatgpt_paraphraser_on_T5_base",
        peft_model_path: str = None,
        device: str = None,
    ):
        """
        初始化T5Generator类
        
        参数:
            model_name: 基础T5模型名称或路径
            peft_model_path: 前缀微调模型路径
            device: 设备，默认为自动检测
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 加载基础模型
        self.base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        
        # 如果提供了前缀微调模型路径，则加载前缀微调模型
        if peft_model_path:
            self.model = PeftModel.from_pretrained(self.base_model, peft_model_path).to(self.device)
        else:
            self.model = self.base_model
            
        # 设置为评估模式
        self.model.eval()
        
    def encode(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用T5编码器对输入文本进行编码
        
        参数:
            text: 输入文本
            
        返回:
            encoder_outputs: 编码器输出
            attention_mask: 注意力掩码
        """
        # 对输入文本进行tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # 获取输入ID和注意力掩码
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        
        # 使用编码器进行编码
        with torch.no_grad():
            encoder_outputs = self.model.get_encoder()(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            
        return encoder_outputs.last_hidden_state, attention_mask

    def decode_step(
        self,
        encoder_hidden_states,
        encoder_attention_mask,
        decoder_input_ids,
        past_key_values=None,
    ):
        dec_out = self.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )

        hidden = dec_out.last_hidden_state   # [B, T, d]

        # 投影到词表
        lm_head = self.model.get_output_embeddings()   # 兼容 PEFT/T5
        logits  = lm_head(hidden)                      # [B, T, |V|]

        # 如果想继续兼容旧 bias，可以加上前面的 hasattr 判断
        return logits, dec_out.past_key_values



    
    def generate_step_by_step(
        self,
        text: str,
        max_length: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
        do_sample: bool = True,
    ) -> List[str]:

        # ---------- 编码 ----------
        enc_hid, enc_attn_mask = self.encode(text)        # [1, L, d], [1, L]
        enc_hid        = enc_hid.repeat_interleave(num_return_sequences, 0)
        enc_attn_mask  = enc_attn_mask.repeat_interleave(num_return_sequences, 0)

        # ---------- 解码器起始 ----------
        start_id = (
            self.model.config.decoder_start_token_id
            if self.model.config.decoder_start_token_id is not None
            else self.tokenizer.eos_token_id          # 退而求其次
        )
        decoder_input_ids = torch.full(
            (num_return_sequences, 1),
            start_id,
            dtype=torch.long,
            device=self.device,
        )
        generated_ids = decoder_input_ids.clone()

        past_key_values = None
        is_finished = torch.zeros(num_return_sequences, dtype=torch.bool, device=self.device)

        vocab_size = self.tokenizer.vocab_size
        top_k = min(top_k, vocab_size)

        for _ in range(max_length):
            # -------- ◆ 统一传递 decoder_input_ids --------
            logits, past_key_values = self.decode_step(
                encoder_hidden_states=enc_hid,
                encoder_attention_mask=enc_attn_mask,
                decoder_input_ids=decoder_input_ids,
                past_key_values=past_key_values,
            )
            logits = logits[:, -1, :]                   # 只取最后一个位置
            logits = logits / temperature

            # -------- Top-k --------
            if top_k > 0:
                kth_vals = torch.topk(logits, top_k)[0][..., -1, None]
                logits = torch.where(logits < kth_vals, torch.full_like(logits, -float("inf")), logits)

            # -------- Top-p --------
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cum_probs > top_p
                sorted_mask[..., 0] = False
                mask = sorted_mask.scatter(1, sorted_idx, sorted_mask)
                logits = torch.where(mask, torch.full_like(logits, -float("inf")), logits)

            # -------- 采样 / 贪婪 --------
            if do_sample:
                probs = torch.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probs, 1).squeeze(-1)
            else:
                next_tokens = torch.argmax(logits, dim=-1)

            # -------- ◆ 对已结束序列强制输出 EOS --------
            next_tokens = torch.where(is_finished, torch.full_like(next_tokens, self.tokenizer.eos_token_id), next_tokens)

            generated_ids = torch.cat([generated_ids, next_tokens.unsqueeze(-1)], dim=-1)
            is_finished |= next_tokens.eq(self.tokenizer.eos_token_id)

            if is_finished.all():
                break

            decoder_input_ids = next_tokens.unsqueeze(-1)   # ◆ 供下一步使用

        return [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]

    
    def generate(
        self,
        text: str,
        max_length: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
        do_sample: bool = True,
    ) -> List[str]:
        """
        使用标准的generate方法生成文本（用于比较）
        
        参数:
            text: 输入文本
            max_length: 最大生成长度
            temperature: 温度参数
            top_p: top-p采样参数
            top_k: top-k采样参数
            num_return_sequences: 返回序列数量
            do_sample: 是否进行采样
            
        返回:
            生成的文本列表
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        generation_config = dict(
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            decoder_start_token_id=self.model.config.decoder_start_token_id,
        )
        output_ids = self.model.generate(**inputs, **generation_config)
        results = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]

        return results


# 使用示例
if __name__ == "__main__":
    # 配置参数
    CONFIG = {
        "model_name": "humarin/chatgpt_paraphraser_on_T5_base",
        "peft_model_path": "/home/dora/Domain-Inference/domain_discover/prefix_paraphraser/checkpoint-65500",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    
    # 初始化生成器
    generator = T5Generator(
        model_name=CONFIG["model_name"],
        peft_model_path=CONFIG["peft_model_path"],
        device=CONFIG["device"]
    )
    
    # 测试输入
    test_text = "I am happy"
    
    # 使用分步生成
    print("=== 使用分步生成 ===")
    step_by_step_results = generator.generate_step_by_step(
        text=test_text,
        max_length=50,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=3,
        do_sample=True
    )
    for i, result in enumerate(step_by_step_results):
        print(f"结果 {i+1}: {result}")
    
    # 使用标准生成（用于比较）
    print("\n=== 使用标准生成 ===")
    standard_results = generator.generate(
        text=test_text,
        max_length=50,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=3,
        do_sample=True
    )
    for i, result in enumerate(standard_results):
        print(f"结果 {i+1}: {result}")