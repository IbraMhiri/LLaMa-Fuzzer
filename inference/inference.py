import os
import torch
from statistics import mean 
from accelerate import Accelerator
import logging
import deepspeed
import time
from transformers.deepspeed import HfDeepSpeedConfig
import deepspeed
import traceback

logging.basicConfig(format='[%(asctime)s] [%(levelname)-8s] %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
DEFAULT_GEN_CONFIG = {
    "max_new_tokens": 2000,  
    "top_k": 25,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "pad_token_id": 0,
    #"top_p": 0.85,
    "temperature": 0.6,
    "do_sample": True,
    "no_repeat_ngram_size": 2,
    #"early_stopping": True
}

def cpu_generate_n_samples(model, tokenizer, num_samles: int, gen_config=DEFAULT_GEN_CONFIG):
    #Prepare input and generate
    input = tokenizer('<?xml version="1.0" encoding="UTF-8">',  return_tensors="pt", return_token_type_ids=False)

    Accelerator()
    durations = []
    samples = []
    #counter = 0
    for i in range(num_samles):
        logging.info(f"Generating XML...")
        
        start= time.time()
        try:
            with torch.no_grad():
                output = model.generate(**input,  **gen_config)
        except RuntimeError:
            logging.error(traceback.format_exc())
            return
        logging.info(f"End of generating XML")

        durations.append(time.time() - start)
        xml = tokenizer.decode(output[0], skip_special_tokens=True)
        samples.append(xml)

    logging.info(f"Avg Time: {mean(durations)/60} Mins")
    return samples