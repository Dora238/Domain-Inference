"""
基于 sample_seq2seq.py 的序列到序列采样模块，完全复制原始功能
同时利用 DiffusionTextGenerator 类的优点
"""

import argparse
import os
import json
import time
import sys
import numpy as np
import torch as th
import torch.distributed as dist
from transformers import set_seed
from functools import partial
from pathlib import Path

# 导入 domain_discover 模块
# 使用相对导入避免循环导入问题
from diffusion import DiffusionTextGenerator, _safe_setup_single_gpu_dist

# 确保 DiffuSeq 在路径中
def setup_diffuseq_path():
    """确保DiffuSeq路径正确添加到sys.path中"""
    # 尝试多种可能的路径
    possible_paths = [
        Path(__file__).parent.parent / 'DiffuSeq',
        Path(__file__).parent.parent / 'DiffuSeq' / 'DiffuSeq',
        Path('/home/dora/Domain-Inference/DiffuSeq'),
        Path('/home/dora/Domain-Inference/DiffuSeq/DiffuSeq')
    ]
    
    for path in possible_paths:
        if (path / 'diffuseq').exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
            return True
            
    return False

# 设置路径
setup_diffuseq_path()

# 导入 DiffuSeq 工具
from diffuseq.rounding import denoised_fn_round
from diffuseq.text_datasets import load_data_text
from diffuseq.utils import dist_util, logger
from basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    load_tokenizer
)

def create_argparser():
    """创建与原始 sample_seq2seq.py 相同的参数解析器"""
    defaults = dict(model_path='', step=0, out_dir='', top_p=0)
    decode_defaults = dict(split='valid', clamp_step=0, seed2=105, clip_denoised=False)
    defaults.update(load_defaults_config())
    defaults.update(decode_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

@th.no_grad()
def sample_seq2seq(args):
    """
    完全复制 sample_seq2seq.py 的功能，但使用 DiffusionTextGenerator 类
    
    Args:
        args: 命令行参数
    """
    # 设置分布式环境
    dist_util.setup_dist()
    logger.configure()

    world_size = dist.get_world_size() or 1
    rank = dist.get_rank() or 0

    # 创建 DiffusionTextGenerator 实例
    generator = DiffusionTextGenerator(
        model_path=args.model_path,
        device=dist_util.dev(),
        use_ddim=True if hasattr(args, 'step') and hasattr(args, 'diffusion_steps') and args.step < args.diffusion_steps else False,
        clip_denoised=args.clip_denoised if hasattr(args, 'clip_denoised') else False,
        ddim_steps=args.step if hasattr(args, 'step') else 200,
        top_p=args.top_p if hasattr(args, 'top_p') else 0.0,
        clamp_step=args.clamp_step if hasattr(args, 'clamp_step') else 0,
        batch_size=args.batch_size if hasattr(args, 'batch_size') else 4,
        seq_len=args.seq_len if hasattr(args, 'seq_len') else 128,
        split=args.split if hasattr(args, 'split') else 'valid'
    )
    
    # 设置随机种子
    set_seed(args.seed2)
    
    print(f"### Sampling...on {args.split}")
    
    # 加载数据 - 这是 sample_seq2seq.py 中的关键步骤
    data_valid = load_data_text(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        deterministic=True,
        data_args=args,
        split=args.split,
        loaded_vocab=generator.tokenizer,
        model_emb=generator.model_emb.cpu(),
        loop=False
    )
    
    start_t = time.time()
    
    # 设置输出路径
    model_base_name = os.path.basename(os.path.split(args.model_path)[0]) + f'.{os.path.split(args.model_path)[1]}'
    out_dir = os.path.join(args.out_dir, f"{model_base_name.split('.ema')[0]}")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"ema{model_base_name.split('.ema')[1]}.samples")
    if not os.path.isdir(out_path):
        os.makedirs(out_path, exist_ok=True)
    out_path = os.path.join(out_path, f"seed{args.seed2}_step{args.clamp_step}.json")
    
    # 收集测试数据
    all_test_data = []
    idx = 0
    
    try:
        while True:
            batch, cond = next(data_valid)
            if idx % world_size == rank:  # 按节点拆分数据
                all_test_data.append((batch, cond))
            idx += 1
    except StopIteration:
        print('### End of reading iteration...')
    
    # 处理余数情况
    if idx % world_size and rank >= idx % world_size:
        all_test_data.append(({}, {}))  # 为余数添加虚拟数据：用于 dist.barrier()
    
    # 设置迭代器
    if rank == 0:
        from tqdm import tqdm
        iterator = tqdm(all_test_data)
    else:
        iterator = iter(all_test_data)
    
    # 处理每个批次
    for batch, cond in iterator:
        if not cond:  # 余数的屏障
            for i in range(world_size):
                dist.barrier()
            continue
        
        # 提取输入 ID 和掩码
        input_ids_x = cond.pop('input_ids').to(dist_util.dev())
        x_start = generator.model.get_embeds(input_ids_x)
        input_ids_mask = cond.pop('input_mask')
        input_ids_mask_ori = input_ids_mask
        
        # 创建噪声和掩码
        noise = th.randn_like(x_start)
        input_ids_mask = th.broadcast_to(input_ids_mask.unsqueeze(dim=-1), x_start.shape).to(dist_util.dev())
        x_noised = th.where(input_ids_mask == 0, x_start, noise)
        
        # 设置模型参数
        model_kwargs = {}
        
        # 确定步长间隔
        diffusion_steps = generator.diffusion_steps
        step = args.step if hasattr(args, 'step') else 0
        
        if step == diffusion_steps:
            use_ddim = False
            step_gap = 1
        else:
            use_ddim = True
            step_gap = diffusion_steps // step if step > 0 else 1
        
        # 选择采样函数
        sample_fn = (
            generator.diffusion.p_sample_loop if not use_ddim else generator.diffusion.ddim_sample_loop
        )
        
        # 设置样本形状
        seq_len = args.seq_len if hasattr(args, 'seq_len') else generator.seq_len
        hidden_dim = generator.config.get("hidden_dim", 128)
        sample_shape = (x_start.shape[0], seq_len, hidden_dim)
        
        # 执行采样
        samples = sample_fn(
            generator.model,
            sample_shape,
            noise=x_noised,
            clip_denoised=args.clip_denoised,
            denoised_fn=partial(denoised_fn_round, argparse.Namespace(**generator.config), generator.model_emb),
            model_kwargs=model_kwargs,
            top_p=args.top_p,
            clamp_step=args.clamp_step,
            clamp_first=True,
            mask=input_ids_mask,
            x_start=x_start,
            gap=step_gap
        )
        
        # 获取最终样本
        sample = samples[-1]
        
        # 获取 logits 和候选项
        logits = generator.model.get_logits(sample)  # bsz, seqlen, vocab
        cands = th.topk(logits, k=1, dim=-1)
        
        # 准备结果列表
        word_lst_recover = []
        word_lst_ref = []
        word_lst_source = []
        
        # 解码生成的标记
        seq_len = args.seq_len if hasattr(args, 'seq_len') else generator.seq_len
        
        for seq, input_mask in zip(cands.indices, input_ids_mask_ori):
            len_x = seq_len - sum(input_mask).tolist()
            tokens = generator.tokenizer.decode_token(seq[len_x:])
            word_lst_recover.append(tokens)
        
        # 解码原始输入
        for seq, input_mask in zip(input_ids_x, input_ids_mask_ori):
            len_x = seq_len - sum(input_mask).tolist()
            word_lst_source.append(generator.tokenizer.decode_token(seq[:len_x]))
            word_lst_ref.append(generator.tokenizer.decode_token(seq[len_x:]))
        
        # 按顺序写入文件
        for i in range(world_size):
            if i == rank:
                with open(out_path, 'a') as fout:
                    for (recov, ref, src) in zip(word_lst_recover, word_lst_ref, word_lst_source):
                        print(json.dumps({"recover": recov, "reference": ref, "source": src}), file=fout)
            dist.barrier()
    
    print(f'### Total takes {time.time() - start_t:.2f}s .....')
    print(f'### Written the decoded output to {out_path}')
    
    return out_path

@th.no_grad()
def main():
    """主函数，解析参数并调用采样函数"""
    args = create_argparser().parse_args()
    output_path = sample_seq2seq(args)
    print(f"采样完成，结果保存在: {output_path}")

if __name__ == "__main__":
    main()
