#!/bin/bash

source ./venv/bin/activate
echo 'Please enter a model to be used for inference:'
read model
echo 'Please enter a fuzzing target:'
read target
DS_ACCELERATOR='cuda' deepspeed main.py fuzz -m $model -t $target