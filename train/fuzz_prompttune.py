from transformers import get_linear_schedule_with_warmup, DataCollatorForLanguageModeling
from peft import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

max_length = 2000
lr = 3e-2
num_epochs = 1
batch_size = 4
logging.basicConfig(format='[%(asctime)s] [%(levelname)-8s] %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

def prompt_tune(model, tokenizer, samples):
    #Setup
    peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.TEXT,
        tokenizer_name_or_path=model,
    )

    model = get_peft_model(model, peft_config)

    #prepare data
    dataset = Dataset.from_list({'xml': samples})

    def tokenize_function(examples):
        return tokenizer(examples['XML'], padding='max_length', return_tensors='pt', truncation=True, max_length=max_length)

    processed_train_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=40,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )


    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,  mlm=False, return_tensors="pt"
    )

    train_dataloader = DataLoader(processed_train_dataset, shuffle=True, collate_fn=data_collator, batch_size=batch_size, pin_memory=True, num_workers=45)

    #set optemizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    logging.info("Starting training....")

    for epoch in range(num_epochs):
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader)):
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()