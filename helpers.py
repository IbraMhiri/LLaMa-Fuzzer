from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from deepspeed.runtime.zero.stage_1_and_2 import estimate_zero2_model_states_mem_needs_all_live
from os.path import join, exists
from os import listdir, remove
import logging
from peft import AutoPeftModelForCausalLM
from peft import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType
import random

MODELS_DIR = "/home/ibrahim/workspace/models"

peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        peft_type=PeftType.PROMPT_TUNING,
        prompt_tuning_init=PromptTuningInit.TEXT,
        num_virtual_tokens=100,
        prompt_tuning_init_text='<?xml version="1.0" encoding="UTF-8">')

logging.basicConfig(format='[%(asctime)s] [%(levelname)-4s] %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

def load_model(model_name: str):
    logging.info(f"Loading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(join(MODELS_DIR, model_name), trust_remote_code=True)
    logging.info("Model loaded...")
    return model

def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(join(MODELS_DIR, model_name))
    return tokenizer

def load_config(model_name: str):
    config = AutoConfig.from_pretrained(join(MODELS_DIR, model_name), trust_remote_code=True)
    return config

def load_generation_config(model_name: str):
    config = GenerationConfig.from_pretrained(join(MODELS_DIR, model_name), trust_remote_code=True)
    return config

def load_peft_model(model_name: str, config=None):
    logging.info(f"Loading model {join(MODELS_DIR, model_name)}")
    model = AutoPeftModelForCausalLM.from_pretrained(join(MODELS_DIR, model_name), config=config)
    logging.info("Model loaded...")
    return model

def estimate_memory_reqirements(model):
    estimate_zero2_model_states_mem_needs_all_live(model, num_gpus_per_node=1, num_nodes=1)

def prepare_artifacts(model_name: str):
    if '_prompttuned' in model_name or '_feedbackloop' in model_name:
        return load_peft_model(model_name), load_tokenizer(model_name), load_config(model_name)
    else:
        return load_model(model_name), load_tokenizer(model_name), load_config(model_name)

def prepare_artifacts_no_config(model_name: str):
    if model_name.endswith('_prompttuned') or model_name.endswith('_feedbackloop'):
        return load_peft_model(model_name), load_tokenizer(model_name)
    else:
        return load_model(model_name), load_tokenizer(model_name)

def get_model_dir() ->str:
    return MODELS_DIR

def print_trainable_params(model_name):
    peft_config.tokenizer_name_or_path=join(get_model_dir(), model_name)
    model = load_model(model_name)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

def check_feedback(current_fuzzing_dir: str):
    logging.info("Checking for fuzzer feedback...")
    feedback_dir = join(current_fuzzing_dir, "out", "new_cov")
    xmls = []
    ctr = 0
    try:
        samples = listdir(feedback_dir)
    except Exception:
        logging.error("Feedback folder not found %s", feedback_dir)
        return {"xml": []}
    random.shuffle(samples)
    for sample in samples:
        sample_path = join(feedback_dir, sample)
        if ctr <= 10:
            try:
                with open(sample_path, 'r') as fp:
                    xml = fp.read()
                    xmls.append(xml)
                    ctr += 1
            
            #some samples are not readable due to afl mutation
            except Exception:
                remove(sample_path)
                continue
            remove(sample_path)
    
    logging.info(f"{len(xmls)} xmls found")
    result = {"xml": xmls}
    return result

def cleanup():
    pass

