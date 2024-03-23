from tqdm.auto import tqdm
from transformers import  get_linear_schedule_with_warmup, DataCollatorForLanguageModeling
import torch
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from os.path import join
from helpers import get_model_dir
import traceback

max_length = 2048
lr = 3e-2
num_epochs = 5
batch_size = 4

logging.basicConfig(format='[%(asctime)s] [%(levelname)-8s] %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

def fine_tune(model, tokenizer, model_name):
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    #prepare data
    dataset_normal = load_dataset('csv', data_files='train/normal.csv')
    dataset_malicious = load_dataset('csv', data_files='train/malicious.csv')
    dataset_normal = dataset_normal['train'].shuffle().select(range(100))
    dataset = concatenate_datasets([dataset_normal, dataset_malicious['train']])

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
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    progress_bar = tqdm(range(len(train_dataloader) * num_epochs))

    logging.info("Starting training...")
    model.train()
    try:
        for epoch in range(num_epochs):
            for batch in train_dataloader:
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
            logging.info("Finished epoch")
        logging.info("Finished training")
    except Exception:
        print(traceback.format_exc)
            
    model.save_pretrained(f"{join(get_model_dir(), model_name)}_finetuned")
    tokenizer.save_pretrained(f"{join(get_model_dir(), model_name)}_finetuned")