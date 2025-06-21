from __future__ import annotations

import argparse, random, torch
from pathlib import Path
from typing import Dict, List
import ast  # 用于安全地将字符串转换为列表
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from transformers import TrainerCallback
from peft import PrefixTuningConfig, get_peft_model, PeftModel, TaskType

# ────────────────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────────────────
CONFIG: Dict[str, object] = {
    "model_name": "humarin/chatgpt_paraphraser_on_T5_base",
    "dataset_name": "humarin/chatgpt-paraphrases",
    "num_virtual_tokens": 10,
    "max_src": 128,
    "max_tgt": 128,
    "lr": 1e-5,
    "batch": 64,
    "epochs": 2,
    "warmup": 1000,
    "out": "./prefix_paraphraser",
    "infer_model": "/home/dora/Domain-Inference/domain_discover/prefix_paraphraser/checkpoint-65500",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

class GenerateEveryN(TrainerCallback):
    """
    每 N 个 step 用固定的 probe 句子做一次生成，
    用来快速 eyeball 模型有没有在学习。
    """
    def __init__(self, tokenizer, every_n_steps=500, probes=None):
        self.tok = tokenizer
        self.every = every_n_steps
        # 默认放两条，也可以自己传
        self.probes = probes or [
            "I am happy",
            "Life is short.",
            "paraphrase: I am happy",
            "paraphrase: Life is short.",
        ]

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 0 or state.global_step % self.every:
            return

        model = kwargs["model"]
        device = model.device
        model.eval()
        with torch.no_grad():
            print(f"\n===== Probe @ step {state.global_step} =====")
            for src in self.probes:
                # 确保输入格式正确
                inputs = self.tok(src, return_tensors="pt").to(device)
                
                try:
                    # T5特定的生成参数
                    out_ids = model.generate(
                        **inputs,
                        max_length=50,  # 使用max_length而不是max_new_tokens
                        num_beams=4,
                        early_stopping=True,
                        do_sample=False,  # 对于调试，关闭采样
                    )
                    
                    # 打印调试信息
                    print(f"Output shape: {out_ids.shape}")
                    print(f"Output IDs: {out_ids[0][:10].tolist()}")
                    
                    # 解码
                    decoded = self.tok.decode(out_ids[0], skip_special_tokens=True)
                    print(f"src: {src}")
                    print(f"tgt: '{decoded}'")  # 用引号包围，便于查看空白字符
                    
                    # 如果解码结果为空，尝试不跳过特殊标记
                    if not decoded or decoded in ["[", "'"]:
                        print("Alternative decoding (with special tokens):")
                        print(f"'{self.tok.decode(out_ids[0], skip_special_tokens=False)}'")
                except Exception as e:
                    print(f"Error during generation: {e}")
        model.train()
tok = AutoTokenizer.from_pretrained(CONFIG["model_name"])
gen_cb = GenerateEveryN(tok, every_n_steps=500)   # 每 500 step 跑一次


# ────────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ────────────────────────────────────────────────────────────────────────────────


def prepare_dataset(tok, split="train"):
    ds = load_dataset(CONFIG["dataset_name"], split=split)

    def build_pairs_batch(batch):
        src_list, tgt_list = [], []
        for text, paras in zip(batch["text"], batch["paraphrases"]):
            paraphrases = paras
            # 数据集中有时将列表存成字符串，需要先解析
            if isinstance(paraphrases, str):
                try:
                    paraphrases = ast.literal_eval(paraphrases)
                except Exception:
                    paraphrases = []  # 解析失败则跳过该条

            for p in paraphrases:
                # src = f"paraphrase: {text}" if random.random() < 0.5 else text
                src = text
                src_list.append(src)
                tgt_list.append(p)
        return {"src": src_list, "tgt": tgt_list}

    ds = ds.map(build_pairs_batch,
                batched=True,
                remove_columns=ds.column_names)

    # ⚠️ 关键修正 1：不要自己把 <pad> 换成 -100，让 collator 处理
    def _tok(batch):
        model_inputs = tok(batch["src"],
                           truncation=True,
                           max_length=CONFIG["max_src"],
                           padding=False)

        # ⚠️ 关键修正 2：T5 必须用 as_target_tokenizer
        with tok.as_target_tokenizer():
            labels = tok(batch["tgt"],
                         truncation=True,
                         max_length=CONFIG["max_tgt"],
                         padding=False)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    ds = ds.map(_tok, batched=True, remove_columns=ds.column_names, num_proc=4)
    ds.set_format("torch")
    return ds
# ────────────────────────────────────────────────────────────────────────────────
# Model construction
# ────────────────────────────────────────────────────────────────────────────────

def build_model():
    base = AutoModelForSeq2SeqLM.from_pretrained(CONFIG["model_name"])
    print(f"Base model type: {type(base)}")
    cfg = PrefixTuningConfig(
        peft_type="PREFIX_TUNING",
        task_type=TaskType.SEQ_2_SEQ_LM,
        num_virtual_tokens=CONFIG["num_virtual_tokens"],
        encoder_hidden_size=base.config.d_model,
        prefix_projection=True,
    )
    model = get_peft_model(base, cfg)
    print(f"PEFT model type: {type(model)}")
    model.print_trainable_parameters()

    return model

# ────────────────────────────────────────────────────────────────────────────────
# Train
# ────────────────────────────────────────────────────────────────────────────────

def train():
    
    tok = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    ds = prepare_dataset(tok, split="train")

    model = build_model().to(CONFIG["device"])
    collator = DataCollatorForSeq2Seq(tok, model, label_pad_token_id=-100)

    model.print_trainable_parameters()

    # ② 自己建 AdamW，确保 Prefix 被 optimizer 捕获
    # optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])

    args = Seq2SeqTrainingArguments(

        output_dir=CONFIG["out"],
        per_device_train_batch_size=CONFIG["batch"],
        gradient_accumulation_steps=1,   # Effective batch ≈ 32
        num_train_epochs=CONFIG["epochs"],
        learning_rate=CONFIG["lr"],
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=CONFIG["warmup"],
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        save_strategy="epoch",
        report_to="none",
        max_grad_norm=1.0,
        # callbacks=[trainer_cb],
        # optimizer_type="adamw_torch",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=collator,
        callbacks=[gen_cb],
        # optimizers=(optimizer, None),
    )
    print("torch.cuda.is_available() ->", torch.cuda.is_available())
    print("trainer.args.device       ->", trainer.args.device)
    trainer.train()

    Path(CONFIG["out"]).mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(CONFIG["out"])
    tok.save_pretrained(CONFIG["out"])

# ────────────────────────────────────────────────────────────────────────────────
# Inference
# ────────────────────────────────────────────────────────────────────────────────

def infer(text: str, **gen_kw):
    tok = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    base = AutoModelForSeq2SeqLM.from_pretrained(CONFIG["model_name"])
    model = PeftModel.from_pretrained(base, CONFIG["infer_model"]).to(CONFIG["device"])
    model.eval()

    inputs = tok(text, return_tensors="pt").to(CONFIG["device"])

    # sane defaults
    params = dict(
        num_beams=10,
        num_beam_groups=10,
        max_length=128,
        repetition_penalty=10.0,
        no_repeat_ngram_size=2,
        num_return_sequences=10,
        # num_beam_groups=10,
        temperature=0.7,
        diversity_penalty=3.0,
    )
    params.update(gen_kw)

    ids = model.generate(**inputs, **params)
    res = tok.batch_decode(ids, skip_special_tokens=True)
    return res

# ────────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["train", "infer"], default="infer")
    ap.add_argument("--text", type=str, default="I am happy")
    args = ap.parse_args()

    if args.stage == "train":
        train()
    else:
        if not args.text:
            raise ValueError("--text is required for inference stage")
        print(infer(args.text))

if __name__ == "__main__":
    main()
