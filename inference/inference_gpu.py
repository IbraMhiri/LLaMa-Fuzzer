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

#logger
logging.basicConfig(format='[%(asctime)s] [%(levelname)-4s] %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

DEFAULT_GEN_CONFIG = {
    "max_new_tokens": 4096,  
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

def init_model_for_inference(model, config):
    # Set up deepspeed
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed()

    model_hidden_size = config.hidden_size

    #ds config
    ds_config = {
        "fp16": {
            "enabled": False
        },
        "bf16": {
            "enabled": False
        },
        "zero_optimization": {
            # "stage": 3,
            # "offload_param": {
            #     "device": "none",
            #     "pin_memory": True
            # },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": model_hidden_size * model_hidden_size,
            "stage3_prefetch_bucket_size": 0.9 * model_hidden_size * model_hidden_size,
            "stage3_param_persistence_threshold": 10 * model_hidden_size
        },
        "steps_per_print": 2000,
        "train_batch_size": 1,
        "train_micro_batch_size_per_gpu": 1,
        "wall_clock_breakdown": False
    }
    dschf = HfDeepSpeedConfig(ds_config)

    #Load and init Model
    logging.info("Loading Model...")

    ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
    ds_engine.module.eval()

    logging.info("Model Loaded.")
    return ds_engine

def generate_n_samples(ds_engine, tokenizer, num_samles: int, gen_config=DEFAULT_GEN_CONFIG):
    #Prepare input and generate
    input = tokenizer('<?xml version="1.0" encoding="UTF-8">',  return_tensors="pt", return_token_type_ids=False)
    input = input.to("cuda")

    Accelerator()
    durations = []

    samples = []
    #counter = 0
    for i in range(num_samles):
        #logging.info(f"Generating XML {i}...")
        
        start= time.time()
        try:
            with torch.no_grad():
                output = ds_engine.module.generate(**input,  **gen_config)
                
        except RuntimeError:
            logging.error(traceback.format_exc())
            return
        #logging.info(f"End of generating XML")

        durations.append(time.time() - start)
        xml = tokenizer.decode(output[0], skip_special_tokens=True)
        samples.append(xml)

    #logging.info(f"Avg Time: {mean(durations)/60} Mins")
    return samples