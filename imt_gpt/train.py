"""
train.py — Short training loop for GPT-2 small on WikiText-2.
Designed for init comparison: measure perplexity at checkpoints.

SAFETY: Gradient accumulation, MPS cache management, memory monitoring,
pre-flight checks, signal handling, process memory limits.
Tuned for M1 MacBook Pro 16GB.
"""
import gc
import time
import math
import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Config, GPT2TokenizerFast
from datasets import load_dataset

from imt_gpt.config import (
    TrainConfig, get_device, check_memory_safe,
    preflight_memory_check, set_process_memory_limit,
    install_signal_handlers, is_shutdown_requested,
)


def _clear_memory(device: str):
    """Aggressively free memory after heavy operations."""
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def load_data(config: TrainConfig, tokenizer):
    """Load and tokenize WikiText-2."""
    dataset = load_dataset(config.dataset, config.dataset_config)

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=False)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // config.max_length) * config.max_length
        result = {
            k: [t[i:i + config.max_length] for i in range(0, total_length, config.max_length)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_dataset = tokenized.map(group_texts, batched=True)
    lm_dataset.set_format("torch")
    return lm_dataset


def create_model(config: TrainConfig, init_fn=None):
    """Create a fresh GPT-2 small model, optionally with custom init."""
    gpt2_config = GPT2Config(
        vocab_size=50257,
        n_positions=config.max_length,
        n_embd=768,
        n_layer=12,
        n_head=12,
        activation_function="gelu_new",
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    model = GPT2LMHeadModel(gpt2_config)

    if init_fn is not None:
        init_fn(model)

    return model


def get_lr(step, config: TrainConfig):
    """Linear warmup + cosine decay schedule."""
    if step < config.warmup_steps:
        return config.lr * step / config.warmup_steps
    progress = (step - config.warmup_steps) / max(config.max_steps - config.warmup_steps, 1)
    return config.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def evaluate_perplexity(model, eval_dataloader, device, max_batches=20):
    """Compute validation perplexity."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for i, batch in enumerate(eval_dataloader):
            if i >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, labels=labels)
            total_loss += outputs.loss.item() * input_ids.numel()
            total_tokens += input_ids.numel()
            del input_ids, labels, outputs
    _clear_memory(device)
    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(min(avg_loss, 20))  # cap at exp(20) to avoid overflow


def _make_early_result(init_name, losses, checkpoints, model,
                       val_dataloader, device, n_params, start_time, reason):
    """Build result dict for early stopping, with cleanup."""
    _clear_memory(device)
    try:
        final_ppl = evaluate_perplexity(model, val_dataloader, device)
    except Exception:
        final_ppl = float("inf")
    elapsed = time.time() - start_time
    del model
    _clear_memory(device)
    return {
        "init_name": init_name,
        "losses": losses,
        "checkpoints": checkpoints,
        "final_ppl": final_ppl,
        "elapsed": elapsed,
        "n_params": n_params,
        "stopped_early": True,
        "stop_reason": reason,
    }


def train(config: TrainConfig, init_fn=None, init_name="default",
          verbose=True, return_checkpoints=True):
    """Run a short training loop and return convergence data.

    Uses gradient accumulation for memory safety on M1.
    effective_batch = config.batch_size * config.grad_accum_steps

    Safety features:
    - Pre-flight memory check before model creation
    - Periodic memory pressure checks during training
    - Clean Ctrl-C handling (finishes current step, frees memory)
    - Process memory limit via setrlimit
    - Explicit memory cleanup between phases
    """
    device = config.device or get_device()
    grad_accum = config.grad_accum_steps

    # Install safety infrastructure
    set_process_memory_limit(max_gb=12.0)
    prev_handler = install_signal_handlers()

    if verbose:
        eff_batch = config.batch_size * grad_accum
        print(f"\n{'='*60}")
        print(f"  Training: {init_name} | device={device}")
        print(f"  batch={config.batch_size} x accum={grad_accum} = eff_batch={eff_batch}")
        print(f"  seq_len={config.max_length} | steps={config.max_steps}")
        safe, level, detail = check_memory_safe(config.memory_pressure_threshold)
        print(f"  memory: {detail} {'OK' if safe else 'LOW'}")
        print(f"{'='*60}")

    # Pre-flight memory check
    preflight_memory_check(config, label=f"training:{init_name}")

    # Tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Data
    if verbose:
        print("Loading WikiText-2...")
    dataset = load_data(config, tokenizer)

    train_dataloader = DataLoader(
        dataset["train"],
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,  # SAFE: in-process, no fork memory doubling
    )
    val_dataloader = DataLoader(
        dataset["validation"],
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=0,
    )

    # Model
    if verbose:
        print("Creating model...")
    model = create_model(config, init_fn=init_fn)
    model = model.to(device).float()  # explicit float32, MPS float16 is unreliable
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"Parameters: {n_params:,}")

    # Memory check after model creation
    safe, level, detail = check_memory_safe(config.memory_pressure_threshold)
    if not safe:
        print(f"  WARNING: Memory low after model creation ({detail}). Aborting.")
        del model
        _clear_memory(device)
        raise MemoryError(f"Memory too low after model creation: {detail}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )

    # Training loop with gradient accumulation
    losses = []
    checkpoints = {}
    optimizer_step = 0  # counts actual optimizer steps
    micro_step = 0      # counts forward passes
    start_time = time.time()
    accum_loss = 0.0

    model.train()
    optimizer.zero_grad()

    try:
        while optimizer_step < config.max_steps:
            for batch in train_dataloader:
                if optimizer_step >= config.max_steps:
                    break

                # Check for shutdown request (Ctrl-C)
                if is_shutdown_requested():
                    if verbose:
                        print(f"\n  Clean shutdown at step {optimizer_step}")
                    return _make_early_result(
                        init_name, losses, checkpoints, model,
                        val_dataloader, device, n_params, start_time,
                        "user_interrupt")

                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                # Forward
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss / grad_accum  # scale loss for accumulation
                loss.backward()

                accum_loss += outputs.loss.item()
                micro_step += 1
                del input_ids, labels, outputs, loss

                # Optimizer step every grad_accum micro-batches
                if micro_step % grad_accum == 0:
                    # LR schedule
                    lr = get_lr(optimizer_step, config)
                    for pg in optimizer.param_groups:
                        pg["lr"] = lr

                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()

                    avg_loss = accum_loss / grad_accum
                    losses.append(avg_loss)

                    # Logging
                    if verbose and optimizer_step % config.log_every == 0:
                        elapsed = time.time() - start_time
                        ppl = math.exp(min(avg_loss, 20))
                        print(f"  step {optimizer_step:5d} | loss {avg_loss:.4f} | "
                              f"ppl {ppl:.1f} | lr {lr:.2e} | {elapsed:.0f}s")

                    # Checkpoints
                    if return_checkpoints and (optimizer_step + 1) in config.eval_steps:
                        _clear_memory(device)
                        ppl = evaluate_perplexity(model, val_dataloader, device)
                        checkpoints[optimizer_step + 1] = ppl
                        if verbose:
                            print(f"  >>> CHECKPOINT step {optimizer_step+1}: "
                                  f"val_ppl = {ppl:.2f}")
                        model.train()

                    # Memory safety check
                    check_interval = config.memory_check_interval
                    if optimizer_step % check_interval == 0 and optimizer_step > 0:
                        safe, level, detail = check_memory_safe(
                            config.memory_pressure_threshold)
                        if not safe:
                            print(f"\n  MEMORY ABORT at step {optimizer_step}: "
                                  f"{detail}")
                            return _make_early_result(
                                init_name, losses, checkpoints, model,
                                val_dataloader, device, n_params, start_time,
                                f"memory_pressure:{detail}")

                    accum_loss = 0.0
                    optimizer_step += 1

                    # Periodic memory cleanup on MPS
                    if device == "mps" and optimizer_step % 50 == 0:
                        _clear_memory(device)

    except MemoryError as e:
        print(f"\n  MEMORY ERROR at step {optimizer_step}: {e}")
        try:
            del model, optimizer
        except Exception:
            pass
        _clear_memory(device)
        return {
            "init_name": init_name,
            "losses": losses,
            "checkpoints": checkpoints,
            "final_ppl": float("inf"),
            "elapsed": time.time() - start_time,
            "n_params": n_params,
            "stopped_early": True,
            "stop_reason": f"oom:{e}",
        }

    # Final eval
    _clear_memory(device)
    final_ppl = evaluate_perplexity(model, val_dataloader, device)
    elapsed = time.time() - start_time

    if verbose:
        print(f"\nFinal val perplexity: {final_ppl:.2f} ({elapsed:.0f}s)")

    # Cleanup model to free memory for next run
    del model, optimizer
    _clear_memory(device)

    return {
        "init_name": init_name,
        "losses": losses,
        "checkpoints": checkpoints,
        "final_ppl": final_ppl,
        "elapsed": elapsed,
        "n_params": n_params,
    }
