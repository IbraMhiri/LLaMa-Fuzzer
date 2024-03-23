#!/bin/bash

source ./venv/bin/activate
echo 'Please enter a model to be used for train'
read model
TOKENIZERS_PARALLELISM=false DS_ACCELERATOR='cpu' deepspeed --bind_cores_to_rank main.py train -m $model