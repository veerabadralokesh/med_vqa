import argparse 
import torch 
import os 
from tqdm import tqdm 
from torch import optim 
from torch.utils.data import DataLoader
import torch.utils.tensorboard as tensorboard
from torch.cuda.amp import GradScaler

from llama import LlamaTokenizer, LlamaForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

from utils import world_info_from_env, init_distributed_device, ImageTextDataSet, is_master, get_autocast
from model import MultimodalLlama 


special_tokens_dict = {'additional_special_tokens': ['[boi]','[eoi]']}


################################################################################
# QLoRA parameters
################################################################################

# LoRA attention dimension
lora_r = 64

# Alpha parameter for LoRA scaling
lora_alpha = 16

# Dropout probability for LoRA layers
lora_dropout = 0.1

################################################################################
# bitsandbytes parameters
################################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

################################################################################
# TrainingArguments parameters
################################################################################

# Output directory where the model predictions and checkpoints will be stored
output_dir = "./results"

# Number of training epochs
num_train_epochs = 1

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = False

# Batch size per GPU for training
per_device_train_batch_size = 4

# Batch size per GPU for evaluation
per_device_eval_batch_size = 4

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 1

# Enable gradient checkpointing
gradient_checkpointing = True

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3

# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001

# Optimizer to use
# optim = "paged_adamw_32bit"

# Learning rate schedule
lr_scheduler_type = "cosine"

# Number of training steps (overrides num_train_epochs)
max_steps = -1

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True

# Save checkpoint every X updates steps
save_steps = 0

# Log every X updates steps
logging_steps = 25

################################################################################
# SFT parameters
################################################################################

# Maximum sequence length to use
max_seq_length = None

# Pack multiple short examples in the same input sequence to increase efficiency
packing = False

# Load the entire model on the GPU 0
device_map = {"": 0}

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a llama model on a causal language modeling task")
    parser.add_argument(
        "--train_file", type=str, default='train.pkl', help="A pkl file containing the training data."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default='NousResearch/Llama-2-7b-chat-hf',
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument("--output_dir", type=str, default='out', help="Where to store the final model.")
    parser.add_argument(
        "--tensorboard_path", type=str, default="./tensorboard",
    )
    parser.add_argument(
        "--image_length",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=4e-3,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--norm_gradient_clip", type=float, default=1.0, help="Gradient clip."
    )
    parser.add_argument("--beta1", type=float, default=0.98, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=1e-6, help="Adam epsilon.")
    parser.add_argument("--weight_decay", type=float, default=0.2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank.")
    
    parser.add_argument(
        "--precision",
        choices=["amp", "amp_bfloat16", "fp16", "fp32"],
        default="fp16",
        help="Floating point precision."
    )
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training."
    )
    parser.add_argument(
        "--debug",
        default=False,
        help="if in debug mode",
    )
    args = parser.parse_args()
    return args 


def main():
    args = parse_args()
    print(args) 
    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    
    # discover initial world args early so we can log properly
    args.distributed = False
    args.local_rank, args.rank, args.world_size = world_info_from_env()

    # fully initialize distributed device environment
    device = init_distributed_device(args)
    
    if is_master(args):
        if not os.path.exists(args.tensorboard_path): 
            os.makedirs(args.tensorboard_path)
        writer = tensorboard.SummaryWriter(args.tensorboard_path)
    else:
        writer = None

    # Load base model

    # llama_model = LlamaForCausalLM.from_pretrained("../llama/llama-2-7b")
    # tokenizer = LlamaTokenizer.from_pretrained("../llama/llama-2-7b")
    
        
    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    # Check GPU compatibility with bfloat16
    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)

    llama_model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-2-7b-hf',
        quantization_config=bnb_config,
        device_map=device_map)
    llama_model.config.use_cache = False
    llama_model.config.pretraining_tp = 1
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
    
    num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    print(num_added_tokens)
    token_ids = tokenizer.convert_tokens_to_ids(['[boi]', '[eoi]']) 
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training
    print(token_ids)

    # llama_model = LlamaForCausalLM.from_pretrained(args.model_name_or_path) 
    # llama_model = LlamaForCausalLM.from_pretrained("/Users/arushisharma/Documents/projects/llama/llama-2-7b/ggml-model-f32_q4_0.gguf")
    
    llama_model.resize_token_embeddings(len(tokenizer)) 
    
    model = MultimodalLlama(image_length=args.image_length, llama=llama_model,)
    # model = 
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps,)
    scaler = GradScaler() if args.precision == "amp" else None

    train_dataset = ImageTextDataSet(args.train_file, tokenizer=tokenizer, image_length=args.image_length)
    train_loader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size) 

    # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
        ],
    )

    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim="paged_adamw_32bit",
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        report_to="tensorboard"
    )
    print("STARTING")
    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=packing,
    )

    # Train model
    trainer.train()

    # Save trained model
    new_model = "llama-2-7b-multimodal-med-vqa-arushi"
    trainer.model.save_pretrained(new_model)

    # for epoch in range(args.num_train_epochs): 
    #     model.train()
    #     device = torch.device(args.device)
    #     autocast = get_autocast(args.precision) 

    #     num_batches_per_epoch = len(train_loader)
    #     loss_cum = .0
    #     progress = tqdm(total=len(train_loader), desc='llama fine-tuning') 

    #     accumulation_steps = 4

    #     for i, batch in enumerate(train_loader):  
    #         step = num_batches_per_epoch * epoch + i
    #         image_embedding, tokens, mask = batch 
    #         image_embedding, tokens, mask = image_embedding.to(device), tokens.to(device), mask.to(device)
            
    #         if args.precision == 'fp16':
    #             with torch.autocast(device_type=device, dtype=torch.float16):
    #                 loss = model(tokens=tokens, labels=tokens, image_embedding=image_embedding, mask=mask).loss / accumulation_steps
    #         else:
    #             with autocast(): 
    #                 loss = model(tokens=tokens, labels=tokens, image_embedding=image_embedding, mask=mask).loss / accumulation_steps

    #         if scaler is not None: 
    #             scaler.scale(loss).backward()
     
    #             if args.norm_gradient_clip is not None:
    #                 scaler.unscale_(optimizer)
    #                 torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
                
    #             # Zero out the gradients for all token embeddings except the newly added embeddings
    #             grads = model.llm.get_input_embeddings().weight.grad  

    #             # Get the index for tokens that we want to zero the grads for 
    #             index_grads_to_zero = torch.arange(len(tokenizer)) != token_ids[0]
    #             index_grads_to_zero *= torch.arange(len(tokenizer)) != token_ids[1] 
    #             grads.data[index_grads_to_zero, :] = grads.data[index_grads_to_zero, :].fill_(0)

    #             scaler.step(optimizer)
    #             scaler.update()
    #             optimizer.zero_grad() 
    #         else: 
    #             loss.backward() 
    #             if (i + 1) % accumulation_steps == 0:
    #                 grads = model.llm.get_input_embeddings().weight.grad  
    #                 index_grads_to_zero = torch.arange(len(tokenizer)) != token_ids[0]
    #                 index_grads_to_zero *= torch.arange(len(tokenizer)) != token_ids[1] 
    #                 grads.data[index_grads_to_zero, :] = grads.data[index_grads_to_zero, :].fill_(0)
    #                 optimizer.step() 
    #                 optimizer.zero_grad() 
            
    #         loss_cum += loss.item()
    #         progress.set_postfix({"loss": loss_cum / (i + 1)})
    #         progress.update() 
    #         if is_master(args) and  i % 10 == 0: 
    #             writer.add_scalar("train/loss", loss.item(), step)
            
    #         if args.debug == True: 
    #             break 
    #     if args.debug == True:
    #         break 

    #     if is_master(args):
    #         print('save modeling')
    #         torch.save(model.state_dict(), args.output_dir + str(epoch) + '.pt') 
    #         torch.cuda.synchronize()


if __name__ == "__main__":
    main()


