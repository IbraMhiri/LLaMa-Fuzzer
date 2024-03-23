import argparse
from inference.inference_gpu import generate_n_samples, init_model_for_inference
from helpers import prepare_artifacts, estimate_memory_reqirements, check_feedback, prepare_artifacts_no_config
from random import randint
import subprocess
import os
from train.train import fine_tune
from train.prompt_tuning import prompt_tune, feedback_learn
from datetime import datetime, timedelta
import logging

logging.basicConfig(format='[%(asctime)s] [%(levelname)-8s] %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

MODELS = ['llama2', 'llama2_finetuned', 'llama2_prompttuned_2048', 'llama2_prompttuned_3072', 'llama2_prompttuned_4096']
TARGETS= ['libxml2', 'tinyxml2', 'rapidxml']
OPERATIONS = ['inference', 'train', 'fuzz', 'learn_loop', 'prompttune', 'estimate']
FUZZING_DIR = '$Fuzzing_dir'
COUNT = 20
FUZZING_PERIOD = timedelta(days=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='ProgramName', description='Initiates the grammar model for inference, fuzzing or finetuning', epilog='Valid operations: train/inference')
    parser.add_argument('-c', '--count')
    parser.add_argument('-t', '--target', choices=TARGETS)
    parser.add_argument('-m', '--model', required=True, choices=MODELS)
    parser.add_argument('operation', choices=OPERATIONS)
    
    #deepspeed args
    parser.add_argument('--local_rank')

    args = parser.parse_args()
    model_name = args.model
    target = args.target

    if(args.operation == 'train'):
        model, tokenizer, config = prepare_artifacts(model_name) 
        fine_tune(model, tokenizer, model_name)


    elif(args.operation == 'inference'):
        if args.count == None or args.count == "":
            count = COUNT
        else:
            count = int(args.count)
        model, tokenizer, config = prepare_artifacts(model_name)
        ds_engine = init_model_for_inference(model, config)
        
        counter = 0
        samples = generate_n_samples(ds_engine, tokenizer, count)
        for xml in samples:
            with open(f"./inference/result/{model_name}_{str(counter)}", "w") as file:   
                file.write(xml)
                counter += 1

    elif(args.operation == 'fuzz'):
        if args.target == None or args.target == "":
            target = "libxml2"
        
        model, tokenizer, config = prepare_artifacts(model_name) 
        ds_engine = init_model_for_inference(model, config)
        if "finetuned" in model_name:
            fuzz_type = 'finetuned_fuzz'
        elif "prompttuned" in model_name:
            fuzz_type = 'prompttuned_fuzz'
        else:
            fuzz_type = 'LLM_fuzz'
        
        start = datetime.now()
        counter = 0
        target_path = os.path.join(FUZZING_DIR, target, model_name)
        
        # Check if folder exists and are not empty
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        if os.path.exists(target_path + "/in"):
            if os.listdir(target_path + "/in"):
                print("[ERROR]: " + target_path + "/in Folder is not empty")
        else:
            os.makedirs(target_path + "/in")
                                                                            
        if not os.path.isfile(target_path + "/fuzz.sh"):
            print("[ERROR]: " + target_path + "/fuzz.sh File does not exist! Are you sure you have a fuzz.sh script and a target binary in the right folder?")
        os.chdir(target_path)
        while(datetime.now() < start + FUZZING_PERIOD):
            samples = generate_n_samples(ds_engine, tokenizer, COUNT)
            for xml in samples:
                with open(os.path.join(target_path + "/in", model_name + "_" + str(counter)), "w") as file:
                    file.write(xml)
                counter += 1

            if counter == COUNT:
                #child process lauches fuzz after first batch of samples is generated
                print("Starting Fuzzing")
                subprocess.run(f"TMUX='' tmux new-session -d ./fuzz.sh", shell=True)

        logging.info("Fuzzinng ended")
        logging.info(f"{counter} samples generated")

    elif(args.operation == 'prompttune'):
        model, tokenizer, config = prepare_artifacts(model_name) 
        prompt_tune(model, tokenizer, model_name)


    elif(args.operation == 'learn_loop'):
        if args.target == None or args.target == "":
            target = "libxml2"
        
        model, tokenizer, config = prepare_artifacts(model_name)
        ds_engine = init_model_for_inference(model, config)

        fuzz_type = 'fuzz_learn'
        start = datetime.now()
        feed_back_samples = 0
        counter = 0
        current_fuzzing_dir = os.path.join(FUZZING_DIR, target, fuzz_type)
        os.chdir(current_fuzzing_dir)
        while(datetime.now() < start + FUZZING_PERIOD):
            samples = generate_n_samples(ds_engine, tokenizer, COUNT)
            for xml in samples:
                with open(os.path.join('in', model_name + "_" + str(counter)), "w") as file:
                    file.write(xml)
                counter += 1
            if counter == COUNT:
                #child process lauches fuzz after first batch of samples is generated
                print("Starting Fuzzing")
                subprocess.run(f"TMUX='' tmux new-session -d ./fuzz.sh", shell=True)
            
            #check for good xmls for prompttuneing
            if (counter % 100) == 0:
                learn_samples = check_feedback(current_fuzzing_dir)
                feed_back_samples += len(learn_samples["xml"])
                if len(learn_samples["xml"]) > 0:
                    model_name = feedback_learn(model, tokenizer, model_name, learn_samples)
                    model, tokenizer = prepare_artifacts_no_config(model_name)
                    ds_engine = init_model_for_inference(model, config)
                    
        logging.info("Fuzzinng ended")
        logging.info(f"{counter} samples generated")
        logging.info(f"{feed_back_samples} samples were used ford evolutionary learning")

    elif(args.operation == 'estimate'):
        model, tokenizer, config = prepare_artifacts(model_name) 
        estimate_memory_reqirements(model)
    
    else:
        raise ValueError("Invalid operation")
