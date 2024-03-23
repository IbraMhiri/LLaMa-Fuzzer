#!/bin/bash

source ./venv/bin/activate
echo 'Please enter a model to be used for inference'
read model
DS_ACCELERATOR='cuda' deepspeed main.py inference -m $model
