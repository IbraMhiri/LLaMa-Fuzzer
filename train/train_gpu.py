import torch
from transformers import get_linear_schedule_with_warmup, DataCollatorForLanguageModeling, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset, concatenate_datasets
import deepspeed
import os
from transformers.deepspeed import HfDeepSpeedConfig
import logging
from torch.utils.data import DataLoader
from os.path import join
from helpers import get_model_dir
from tqdm.auto import tqdm

max_length = 2000
lr = 3e-2
num_epochs = 1
batch_size = 4
logging.basicConfig(format='[%(asctime)s] [%(levelname)-8s] %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

#ds config
ds_config = {
    "fp16": {
        "enabled": False
    },
    "bf16": {
        "enabled": "auto"
    },
    "optimizer": {
        "type": "AdamW"
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 5e-05,
            "warmup_num_steps": 0
        }
    },
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "cpu"
        },
        "offload_optimizer": {
            "device": "cpu"
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": False
    },
    "gradient_accumulation_steps": 1,
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": 'auto',
    "train_micro_batch_size_per_gpu": 'auto',
    "wall_clock_breakdown": False
}

def fine_tune(model, tokenizer, model_name):
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed()


    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    #prepare data
    dataset_normal = load_dataset('csv', data_files='train/normal.csv')
    dataset_malicious = load_dataset('csv', data_files='train/malicious.csv')

    dataset_normal = dataset_normal['train'].shuffle().select(range(1))

    #dataset = concatenate_datasets([dataset_normal, dataset_malicious['train']])
    dataset = dataset_normal

    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(examples['xml'], return_tensors='pt', truncation=True, padding='max_length', max_length=max_length)

    processed_train_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=45,
        remove_columns=dataset.column_names,
        desc="Running tokenizer on dataset",
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, return_tensors="pt")

    train_dataloader = DataLoader(processed_train_dataset, shuffle=True, collate_fn=data_collator, batch_size=batch_size, pin_memory=True, num_workers=40)

    #set optemizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    logging.info("Starting training....")
    training_args = TrainingArguments(output_dir="test_trainer", deepspeed=ds_config)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_train_dataset,
        data_collator=data_collator,
    )
    trainer.train()

    model.save_pretrained(f"{join(get_model_dir, model_name)}_fine_tuned")
    tokenizer.save_pretrained(f"{join(get_model_dir, model_name)}_fine_tuned")