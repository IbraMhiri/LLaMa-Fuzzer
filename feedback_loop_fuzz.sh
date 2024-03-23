#!/bin/bash

source ./venv/bin/activate
echo 'Please enter a model to be used for inference:'
model="llama2_prompttuned_3072"
echo 'Please enter a fuzzing target:'
read target
DS_ACCELERATOR='cuda' deepspeed main.py learn_loop -m $model -t $target