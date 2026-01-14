import os
import json
import argparse
import logging
import pdb

import yaml
import random
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from collections import deque
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from models.modeling_qwen2_audio import Qwen2AudioForConditionalGeneration
from models.processing_qwen2_audio import Qwen2AudioProcessor
from models.configuration_qwen2_audio import Qwen2AudioConfig
# from dataset.dataset import AudioSpeechDataset
from dataset.dataset_mix import AudioSpeechDataset
import shutil
from transformers import AutoModelForCausalLM

# --- LORA IMPORTS ---
from peft import LoraConfig, get_peft_model, TaskType

# --------------------

# Fix seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Color codes for logging
RED = '\033[91m'
RESET = '\033[0m'
BLUE = '\033[94m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
CYAN = '\033[96m'


# Setup logging to console and file
def setup_logging(log_file_path, rank):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        if rank == 0:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        fh = logging.FileHandler(log_file_path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


# Parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Speech Conversation.")
    parser.add_argument('--config_model', type=str, required=True, help='Path to the model config directory.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with smaller datasets.')
    return parser.parse_args()


# Save model and tokenizer
def save_model(model, processor, path2save, logger, args, config):
    # Khi dùng PEFT, model.module (trong DDP) chính là PeftModel
    # Nó sẽ chỉ lưu adapter weights (file nhỏ) thay vì toàn bộ model
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(path2save)
    processor.save_pretrained(path2save)

    config_file_path = os.path.join(config["training"]["output_dir"], "config.json")
    destination_path = os.path.join(path2save, "config.json")

    # Chỉ copy config nếu đường dẫn khác nhau
    if os.path.abspath(config_file_path) != os.path.abspath(destination_path):
        # Kiểm tra file tồn tại trước khi copy để tránh lỗi
        if os.path.exists(config_file_path):
            shutil.copy(config_file_path, path2save)
            logger.info(f"Config file saved to {path2save}")
    else:
        logger.info(f"Source and destination for config.json are the same. Skipping copy.")

    logger.info(f"Model (LoRA adapters) saved to {path2save}")


# Load config from YAML
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    args = parse_arguments()

    # Initialize Distributed Process Group
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if world_size > 1:
        dist.init_process_group(backend='nccl', init_method='env://')

    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    # Load configuration
    configs_training = os.path.join(args.config_model, 'configs_training.yaml')
    config = load_config(configs_training)
    path2save = config['training']['output_dir']
    if rank == 0:
        os.makedirs(path2save, exist_ok=True)

    # Setup logging
    log_file = os.path.join(path2save, 'training.log')
    logger = setup_logging(log_file_path=log_file, rank=rank)
    if rank == 0:
        logger.info("Logging is set up. Logs will be saved to both console and file.")

    # Initialize TensorBoard (rank 0 only)
    if rank == 0:
        tb_log_dir = os.path.join(path2save, 'tensorboard_logs')
        writer = SummaryWriter(log_dir=tb_log_dir)
        logger.info(f"TensorBoard logging is set up. Logs will be saved to {tb_log_dir}")
    else:
        writer = None

    # Save config (rank 0 only)
    if rank == 0:
        with open(os.path.join(path2save, "configs_training.yaml"), 'w') as f:
            yaml.dump(config, f)
        logger.info("Configuration file saved.")
        # Kiểm tra file tồn tại trước khi copy
        src_config_json = os.path.join(args.config_model, "config.json")
        if os.path.exists(src_config_json):
            shutil.copy(src_config_json, path2save)

    # Initialize processor and model
    processor = Qwen2AudioProcessor.from_pretrained(args.config_model)

    # Load Base Model
    logger.info(f"Loading base model from {args.config_model}...")
    # model = Qwen2AudioForConditionalGeneration.from_pretrained(
    #     args.config_model,
    #     torch_dtype=torch.float16,  # Khuyến nghị dùng float16/bfloat16 khi train LoRA
    #     device_map={"": device}  # Map trực tiếp vào device hiện tại để tránh lỗi DDP
    # )

    ################### train from scatch ###################
    print(f"{BLUE}Train from scratch with config: {configs_training}{RESET}")
    config_model_file = os.path.join(args.config_model, "config.json")
    with open(config_model_file, 'r') as f:
        config_model_data = json.load(f)
    model = Qwen2AudioForConditionalGeneration(Qwen2AudioConfig(**config_model_data))
    ################### train from scatch ###################
    ################### LOAD PRETRAINED AUDIO TOWER ###################
    print(f"{BLUE}Loading pretrained Audio Tower from:{RESET}")
    original_model = Qwen2AudioForConditionalGeneration.from_pretrained(
        "configs/ConversationV1",
        torch_dtype="auto",
        device_map={"": device}
    )

    audio_tower_state_dict = original_model.audio_tower.state_dict()
    missing, unexpected = model.audio_tower.load_state_dict(audio_tower_state_dict, strict=True)
    print(f"{GREEN}Audio Tower weights loaded successfully.{RESET}")
    if len(missing) > 0:
        print(f"{RED}Missing keys in Audio Tower: {missing}{RESET}")

    del original_model
    torch.cuda.empty_cache()
    ################### LOAD PRETRAINED AUDIO TOWER ###################
    ################### LOAD PRETRAINED TEXT MODEL (Qwen2.5-3B) ###################
    text_model_name = "Qwen/Qwen2.5-0.5B"
    print(f"{BLUE}Loading pretrained Text Model from: {text_model_name}{RESET}")

    text_model_src = AutoModelForCausalLM.from_pretrained(
        text_model_name,
        device_map="cpu",
        torch_dtype="auto"
    )
    # 2. Xử lý lệch Vocab Size (QUAN TRỌNG)
    target_vocab_size = model.language_model.config.vocab_size
    src_vocab_size = text_model_src.config.vocab_size

    if target_vocab_size != src_vocab_size:
        print(f"{YELLOW}Resizing vocab from {src_vocab_size} to {target_vocab_size}...{RESET}")
        text_model_src.resize_token_embeddings(target_vocab_size)

    missing_keys, unexpected_keys = model.language_model.load_state_dict(
        text_model_src.state_dict(),
        strict=True
    )

    print(f"{GREEN}Text Model weights loaded successfully.{RESET}")
    if len(missing_keys) > 0:
        print(f"{RED}Missing keys: {missing_keys}{RESET}")

    del text_model_src
    torch.cuda.empty_cache()
    ################### LOAD PRETRAINED TEXT MODEL (Qwen2.5-3B) ###################

    ################### FREEZE AUDIO TOWER ###################
    for param in model.audio_tower.parameters():
        param.requires_grad = False
    model.audio_tower.eval()
    print(f"{YELLOW}Audio Tower has been frozen (requires_grad=False).{RESET}")
    sample_param = next(model.audio_tower.parameters())
    print(f"Check Audio Tower grad: {sample_param.requires_grad}")  # Should be False
    ################### FREEZE AUDIO TOWER ###################
    # ================= FIX OOM & WARNINGS MỚI =================
    if rank == 0:
        logger.info(f"{YELLOW}Applying Gradient Checkpointing and Audio-Gradient Patch...{RESET}")

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.multi_modal_projector.register_forward_hook(lambda m, i, o: o.requires_grad_(True))
    # ==========================================================

    model = model.to(device) # Đã map device ở from_pretrained rồi, nhưng giữ lại cũng không sao

    # Wrap model with DDP
    if world_size > 1:
        # Lưu ý: find_unused_parameters=True đôi khi cần thiết cho models phức tạp hoặc PEFT
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )
    print(model)
    # Load datasets
    json_files = config['data']['json_files']  # List of JSON files from config
    train_dataset = AudioSpeechDataset(
        json_files=json_files['train'],
        processor=processor,
        max_length=config['data'].get('max_length', 16000 * 20),
        sampling_rate=config['data'].get('sampling_rate', 16000)
    )
    dev_dataset = AudioSpeechDataset(
        json_files=json_files['dev'],
        processor=processor,
        max_length=config['data'].get('max_length', 16000 * 10),
        sampling_rate=config['data'].get('sampling_rate', 16000)
    )

    if rank == 0:
        logger.info("Created PyTorch datasets.")

    # Create Distributed Samplers
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        dev_sampler = DistributedSampler(dev_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        dev_sampler = None

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config['training']['per_device_train_batch_size']),
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        pin_memory=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=int(config['training']['num_workers'])
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=int(config['training']['per_device_eval_batch_size']),
        sampler=dev_sampler,
        shuffle=False,
        pin_memory=True,
        collate_fn=dev_dataset.collate_fn,
        num_workers=int(config['training']['num_workers'])
    )
    if rank == 0:
        logger.info(f"Model and processor initialized. Device: {device}")
        # PEFT model structure is different, params counting handled by print_trainable_parameters
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        pytorch_total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"{GREEN}Total parameters: {pytorch_total_params}{RESET}")
        logger.info(f"{GREEN}Trainable parameters: {pytorch_total_params_trainable}{RESET}")

        logger.info("Created DataLoaders for training and validation sets.")
        logger.info(f"{RED}Warning: num_workers is set to {config['training']['num_workers']}{RESET}")
        logger.info(f"{RED}Warning: learning_rate is set to {config['training']['learning_rate']}{RESET}")

    # Define optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=float(config['training']['learning_rate']),
        betas=(float(config['training'].get('adam_beta1', 0.9)), float(config['training'].get('adam_beta2', 0.999))),
        eps=float(config['training'].get('adam_epsilon', 1e-8)),
        weight_decay=float(config['training']['weight_decay'])
    )
    num_epochs = int(config['training']['num_train_epochs'])
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config['training'].get('warmup_steps', 0)),
        num_training_steps=total_steps
    )

    # Early Stopping parameters
    early_stopping_patience = int(config['training'].get('early_stopping_patience', 3))
    best_val_loss = float('inf')
    epochs_no_improve = 0
    checkpoint_queue = deque()

    # Training loop
    global_step = 0
    primary_loss_total = 0.0
    save_iteration = config['debug']['save_iterations'] if args.debug else config['training']['save_iterations']

    while global_step < total_steps:
        if world_size > 1 and train_sampler is not None:
            train_sampler.set_epoch(global_step // len(train_sampler))

        model.train()
        progress_bar = tqdm(train_loader, desc="Training", leave=False, disable=(rank != 0))
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            # Forward pass
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                input_features=batch.get('input_features', None),
                feature_attention_mask=batch.get('feature_attention_mask', None),
                labels=batch['labels']
            )
            loss = outputs.loss
            if loss.dim() > 0:
                loss = loss.mean()

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            primary_loss_total += loss.item()
            global_step += 1
            progress_bar.set_description(f"Training {global_step}: | Loss: {loss.item():.4f}")

            # Evaluate and save
            if global_step % save_iteration == 0:
                if rank == 0:
                    avg_primary_loss = primary_loss_total / save_iteration
                    logger.info(f"Iteration {global_step}: Train Average Loss: {avg_primary_loss:.4f}")
                    if writer:
                        writer.add_scalar('Loss/Train', avg_primary_loss, global_step)
                    primary_loss_total = 0.0

                # Validation
                model.eval()
                val_loss_total = 0.0
                with torch.no_grad():
                    for val_batch in tqdm(dev_loader, desc="Validation", leave=False, disable=(rank != 0)):
                        val_batch = {k: v.to(device, non_blocking=True) for k, v in val_batch.items()}
                        outputs = model(
                            input_ids=val_batch['input_ids'],
                            attention_mask=val_batch['attention_mask'],
                            input_features=val_batch['input_features'],
                            feature_attention_mask=val_batch['feature_attention_mask'],
                            labels=val_batch['labels']
                        )
                        val_loss = outputs.loss
                        if val_loss.dim() > 0:
                            val_loss = val_loss.mean()
                        val_loss_total += val_loss.item()
                        if args.debug: break

                if rank == 0:
                    avg_val_loss = val_loss_total / len(dev_loader)
                    logger.info(f"Iteration {global_step}: Validation Average Loss: {avg_val_loss:.4f}")
                    if writer:
                        writer.add_scalar('Loss/Validation', avg_val_loss, global_step)

                    # Checkpointing
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        epochs_no_improve = 0
                        checkpoint_path = os.path.join(path2save, f"checkpoint_iter_{global_step}")
                        save_model(model, processor, checkpoint_path, logger, args, config)
                        checkpoint_queue.append(checkpoint_path)
                        logger.info(f"{GREEN}Checkpoint saved at iteration {global_step}{RESET}")
                        if len(checkpoint_queue) > config['training'].get('keep_last_n_checkpoints', 3):
                            oldest_checkpoint = checkpoint_queue.popleft()
                            if os.path.isdir(oldest_checkpoint):
                                shutil.rmtree(oldest_checkpoint)
                                logger.info(f"{CYAN}Removed oldest checkpoint: {oldest_checkpoint}{RESET}")
                    else:
                        epochs_no_improve += 1
                        logger.info(
                            f"{YELLOW}No improvement in validation loss for {epochs_no_improve} iteration(s).{RESET}")
                        if epochs_no_improve >= early_stopping_patience:
                            logger.info(f"{RED}Early stopping triggered at iteration {global_step}.{RESET}")

                model.train()
                if world_size > 1:
                    dist.barrier()

    if rank == 0 and writer:
        writer.close()
        logger.info("TensorBoard writer closed.")
        logger.info(f"{RED}Training completed at iteration {global_step}.{RESET}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()