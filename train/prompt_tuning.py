from transformers import get_linear_schedule_with_warmup, DataCollatorForLanguageModeling
from peft import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType
import torch
from datasets import Dataset, load_dataset, concatenate_datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from helpers import get_model_dir
from os.path import join, exists, isdir
import shutil
import re

logging.basicConfig(format='[%(asctime)s] [%(levelname)-8s] %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

MAX_LENGTH = 2048
LR = 1e-3
EPOCHS = 1
BATCH_SIZE = 1

peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        peft_type=PeftType.PROMPT_TUNING,
        prompt_tuning_init=PromptTuningInit.TEXT,
        num_virtual_tokens=100,
        prompt_tuning_init_text='<?xml version="1.0" encoding="UTF-8">')

def prompt_tune(model, tokenizer, model_name):
    #Setup
    peft_config.tokenizer_name_or_path=join(get_model_dir(), model_name)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    
    #prepare data
    dataset_normal = load_dataset('csv', data_files='train/normal.csv')
    dataset_malicious = load_dataset('csv', data_files='train/malicious.csv')
    dataset_normal = dataset_normal['train'].shuffle().select(range(100))
    dataset = concatenate_datasets([dataset_normal, dataset_malicious['train']])

    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        return tokenizer(examples['xml'], padding='max_length', return_tensors='pt', truncation=True, max_length=MAX_LENGTH)

    processed_train_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=45,
        remove_columns=dataset.column_names,
        desc="Running tokenizer on dataset",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,  mlm=False, return_tensors="pt")
    train_dataloader = DataLoader(processed_train_dataset, shuffle=True, collate_fn=data_collator, batch_size=BATCH_SIZE, pin_memory=True, num_workers=45)

    #set optemizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * EPOCHS),
    )

    logging.info("Starting training...")
    model.train()
    for epoch in range(EPOCHS):
        for step, batch in enumerate(tqdm(train_dataloader)):
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
    logging.info("Training Finished")
    
    dirpath = join(get_model_dir(), f"{model_name}_prompttuned_{MAX_LENGTH}")
    if exists(dirpath) and isdir(dirpath):
        shutil.rmtree(dirpath)
    
    model.save_pretrained(dirpath)
    model.base_model.save_pretrained(dirpath)
    tokenizer.save_pretrained(dirpath)
   
def feedback_learn(model, tokenizer, model_name, sampels):
    #Setup
    model = model.to(torch.device('cpu'))
    peft_config.tokenizer_name_or_path=join(get_model_dir(), model_name)
    model = get_peft_model(model, peft_config)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    
    max_length = int(re.findall(r'\d+', model_name)[0])
    
    dataset = Dataset.from_dict(sampels)
    def tokenize_function(examples):
        return tokenizer(examples['xml'], padding='max_length', return_tensors='pt', truncation=True, max_length=max_length)

    processed_train_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=45,
        remove_columns=dataset.column_names,
        desc="Running tokenizer on dataset",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,  mlm=False, return_tensors="pt")
    train_dataloader = DataLoader(processed_train_dataset, shuffle=True, collate_fn=data_collator, batch_size=BATCH_SIZE, pin_memory=True, num_workers=45)
    
    #set optemizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * EPOCHS),
    )
    
    logging.info("Starting feedback learning...")
    model.train()
    for step, batch in enumerate(tqdm(train_dataloader)):
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
    logging.info("Feedback learning finished")
    
    # Perform manually copy here
    model_name = model_name.replace("prompttuned",  "feedbackloop")
    
    dirpath = join(get_model_dir(), model_name)
        
    logging.info(f"Saved model {dirpath}")
    model.save_pretrained(dirpath)
    model.base_model.save_pretrained(dirpath)
    tokenizer.save_pretrained(dirpath)
    
    return model_name