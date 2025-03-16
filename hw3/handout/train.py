import argparse
import os
import sys
import time
import math
import json
from contextlib import nullcontext

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from transformers import AutoTokenizer

from datasets import load_dataset
from model import GPTConfig, GPT
import wandb
from lora import mark_only_lora_as_trainable
from dataloader import CustomDataLoader
from generate import ModelSampler

# Argument parser for configurations
def get_args():
    parser = argparse.ArgumentParser(description="Training Script with Configurable Parameters")

    # Directory Settings
    parser.add_argument('--out_dir', type=str, default='lora-gpt-default',help='Output directory.', required=True)
    parser.add_argument('--init_from', type=str, default='gpt2' , help='Initialization method.',  required=True)
    parser.add_argument('--wandb_project', type=str, default='HW3_lora_finetune_handout', help='WandB project name.')

    # LoRA settings
    parser.add_argument('--lora_rank', type=int, default=128, help='LoRA rank.')
    parser.add_argument('--lora_alpha', type=int, default=512, help='LoRA alpha.')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout.')

    # General

    parser.add_argument('--eval_interval', type=int, default=5, help='Interval for evaluation.')
    parser.add_argument('--log_interval', type=int, default=1, help='Interval for logging.')
    parser.add_argument('--eval_iters', type=int, default=20, help='Number of iterations for evaluation.')
    parser.add_argument('--eval_only', action='store_true', default=False, help='If true, only evaluate and then exit.')
    parser.add_argument('--always_save_checkpoint', action='store_true', default=False, help='Always save checkpoint after evaluation.')


    # Data
    parser.add_argument('--dataset', type=str, default='rotten_tomatoes', help='Dataset name.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=32, help='Gradient accumulation steps to simulate larger batch sizes.')
    parser.add_argument('--batch_size', type=int, default=4, help='Micro-batch size for gradient accumulation.')
    parser.add_argument('--block_size', type=int, default=1024, help='Block size.')

    # Model
    parser.add_argument('--n_layer', type=int, default=12, help='Number of layers in the model.')
    parser.add_argument('--n_head', type=int, default=12, help='Number of attention heads.')
    parser.add_argument('--n_embd', type=int, default=768, help='Size of embeddings.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
    parser.add_argument('--bias', action='store_true', default=False, help='Whether to use bias in LayerNorm and Linear layers.')

    # AdamW optimizer
    parser.add_argument('--learning_rate', type=float, default=2.5e-4, help='Maximum learning rate.')
    parser.add_argument('--max_iters', type=int, default=80, help='Total number of training iterations.')
    parser.add_argument('--weight_decay', type=float, default=1e-1, help='Weight decay for AdamW optimizer.')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 for AdamW optimizer.')
    parser.add_argument('--beta2', type=float, default=0.95, help='Beta2 for AdamW optimizer.')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value.')

    # Learning rate decay settings
    parser.add_argument('--decay_lr', action='store_true', default=False, help='Whether to decay learning rate.')
    parser.add_argument('--warmup_iters', type=int, default=2000, help='Number of warmup iterations.')
    parser.add_argument('--lr_decay_iters', type=int, default=600000, help='Iterations for learning rate decay.')
    parser.add_argument('--min_lr', type=float, default=5e-9, help='Minimum learning rate.')

    # DDP settings
    parser.add_argument('--backend', type=str, default='nccl', choices=['nccl', 'gloo'], help='Backend for distributed training.')

    # System
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (e.g., "cpu", "cuda").')
    parser.add_argument('--dtype', type=str, default='bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16',
                        choices=['float32', 'bfloat16', 'float16'], help='Data type to use.')
    parser.add_argument('--compile', action='store_true', default=False, help='Compile the model for faster training using PyTorch 2.0.')

    args = parser.parse_args()
    return args

@torch.no_grad()
def estimate_loss(model, eval_iters, ctx, train_batch_generator, val_batch_generator, device):
    """Evaluate the model and estimate the loss."""
    out = {}
    model.eval()
    for split in ['val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            if split == 'train':
                X, Y = next(train_batch_generator)
            else:
                X, Y = next(val_batch_generator)
            with ctx:
                _, loss = model(X.to(device), Y.to(device))
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it, args):
    if it < args.warmup_iters:
        return args.learning_rate * it / args.warmup_iters
    if it > args.lr_decay_iters:
        return args.min_lr
    decay_ratio = (it - args.warmup_iters) / (args.lr_decay_iters - args.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return args.min_lr + coeff * (args.learning_rate - args.min_lr)

def get_batch(loader, device):
    """Generate batches of data."""
    while True:
        for x, y in loader:
            input_ids = x.to(device)
            labels = y.to(device)
            yield input_ids, labels


def main():
    # Get arguments
    args = get_args()

    with open("wandb_api.json") as json_file:
        credentials = json.load(json_file)

    wandb_api = credentials['wandb_api_key']

    wandb.login(key=wandb_api)

    wandb_run_name = f"{args.out_dir}-r:{args.lora_rank}-alph:{args.lora_alpha}-lr:{args.learning_rate}-iter:{args.max_iters}"
    wandb.init(project=args.wandb_project, name=wandb_run_name)

    # DDP setup
    ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
    print("ddp", ddp)
    if ddp:
        init_process_group(backend=args.backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        args.device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(args.device)
        master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank  # each process gets a different seed
        assert args.gradient_accumulation_steps % ddp_world_size == 0
        args.gradient_accumulation_steps //= ddp_world_size
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1

    tokens_per_iter = args.gradient_accumulation_steps * ddp_world_size * args.batch_size * args.block_size
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    if master_process:
        os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if 'cuda' in args.device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Load dataset
    train_dataset = load_dataset(args.dataset, split='train').shuffle(seed=42).select([i for i in range(5000)])
    val_dataset = load_dataset(args.dataset, split='validation')

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Instantiate CustomDataLoader
    custom_loader_train = CustomDataLoader(train_dataset, tokenizer, args.batch_size)
    custom_loader_val = CustomDataLoader(val_dataset, tokenizer, args.batch_size)

    train_loader = custom_loader_train.get_loader(shuffle=True)
    val_loader = custom_loader_val.get_loader(shuffle=True)

    # get distribution of positive and negative labels from training data for logging
    pos = 0
    neg = 0
    for row in train_dataset:
        if row['label'] == 1:
            pos += 1
        else:
            neg += 1
    print(f"Positive: {pos}, Negative: {neg}")

    # Initialize model configuration
    model_args = dict(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, block_size=args.block_size,
                      bias=args.bias, vocab_size=None, dropout=args.dropout)

    # Model initialization
    if args.init_from == 'resume':
        ckpt_path = os.path.join(args.out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=args.device)
        model_args.update(checkpoint['model_args'])
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        model.load_state_dict(checkpoint['model'])
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
    elif args.init_from.startswith('gpt2'):
        if args.lora_rank>0:
            override_args = dict(dropout=args.dropout,lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,)
            wandb.log({
                "lora_rank":args.lora_rank,
                "lora_alpha":args.lora_alpha,
            })
            model = GPT.from_pretrained(args.init_from, override_args)

            for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'lora_rank', 'lora_alpha']:
                model_args[k] = getattr(model.config, k)
            ################################################## TODO: ##################################################
            #TODO: Mark the model parameters with LORA fine-tuneable.
            model = mark_only_lora_as_trainable(model)
        else:
            override_args = dict(dropout=args.dropout)

            model = GPT.from_pretrained(args.init_from, override_args)
            for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
                model_args[k] = getattr(model.config, k)
    else:
        print("Warning: Invalid Initialization. Returning..")
        sys.exit()

    if args.block_size < model.config.block_size:
        model.crop_block_size(args.block_size)
        model_args['block_size'] = args.block_size # so that the checkpoint will have the right value

    model.to(args.device)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))

    # Optimizer
    optimizer = model.configure_optimizers(args.weight_decay, args.learning_rate, (args.beta1, args.beta2), device_type)

    if args.init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None # free up memory

    if args.compile:
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model) # requires PyTorch 2.0
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # Training loop
    train_batch_generator = get_batch(train_loader, args.device)
    val_batch_generator = get_batch(val_loader, args.device)

    raw_model = model.module if ddp else model
    local_iter_num = 0
    best_val_loss = 1e9

    X, Y = next(train_batch_generator)
    t0 = time.time()
    running_mfu = -1.0
    for iter_num in range(1, args.max_iters + 1):
        if iter_num % args.eval_interval == 0:
            losses = estimate_loss(model, args.eval_iters, ctx, train_batch_generator, val_batch_generator, args.device)
            print(f"Evaluation at Iter {iter_num}: Val Loss {losses['val']:.4f}")
            wandb.log({"val/loss": losses['val']})

            if master_process and losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                print("Best val loss so far, saving Best Val checkpoint...")
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss
                }
                torch.save(checkpoint, os.path.join(args.out_dir, 'ckpt.pt'))

        if args.eval_only:
            break

        for micro_step in range(args.gradient_accumulation_steps):
            if ddp:
                model.require_backward_grad_sync = (micro_step == args.gradient_accumulation_steps - 1)
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / args.gradient_accumulation_steps
            X, Y = next(train_batch_generator)
            scaler.scale(loss).backward()

        if args.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % args.log_interval == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * args.gradient_accumulation_steps
            if local_iter_num >= 5: # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(args.batch_size * args.gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

            wandb.log({
                "iter": iter_num,
                "train loss": lossf,
                "lr": optimizer.param_groups[0]['lr'],            
            })
        local_iter_num += 1
    val_loss_last = losses['val']
    checkpoint_last = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': iter_num,
        'best_val_loss': val_loss_last
    }
    torch.save(checkpoint_last, os.path.join(args.out_dir, 'ckpt_last.pt'))

    if ddp:
        destroy_process_group()

    sampler = ModelSampler(out_dir=args.out_dir, init_from='resume', device=args.device, max_new_tokens=5, temperature=0.6, top_k=1)
    accuracy, pos_counter, neg_counter, counter = sampler.get_accuracy()
    print(f"Best Val Checkpoint || Accuracy: {accuracy}, Positive Predictions: {pos_counter}, Negative Predictions: {neg_counter}, Correct Predictions: {counter}")
    wandb.log({
        "best_val_accuracy": accuracy,
        "best_val_pos_counter": pos_counter,
        "best_val_neg_counter": neg_counter,
        "best_val_correct_counter": counter, })
    
    sampler = ModelSampler(out_dir=args.out_dir, init_from='resume', device=args.device, max_new_tokens=5, temperature=0.6, top_k=1, ckpt_last=True)
    accuracy, pos_counter, neg_counter, counter = sampler.get_accuracy()
    print(f"Last Iter Checkpoint || Accuracy: {accuracy}, Positive Predictions: {pos_counter}, Negative Predictions: {neg_counter}, Correct Predictions: {counter}")
    wandb.log({
        "last_iter_accuracy": accuracy,
        "last_iter_pos_counter": pos_counter,
        "last_iter_neg_counter": neg_counter,
        "last_iter_correct_counter": counter
    })


if __name__ == "__main__":
    main()