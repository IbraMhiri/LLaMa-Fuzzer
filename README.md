# LLaMa-Fuzzer

Using a fuzzer (currently AFL) and large language Models, this tool is dedicted to enhance the fuzzing process and make it more efficient.

## Rquirements

All requirements can be installed using `pip install -r requirements.txt`

### CPU multiprocessing

To use the CPU multiprocessing to enhance the fine-tuning the following steps are required:

```
python -m pip install oneccl_bind_pt==2.0 -f https://developer.intel.com/ipex-whl-stable-cpu
git clone https://github.com/oneapi-src/oneCCL
cd oneCCL
mkdir build
cd build
cmake ..
make
make install
```

## Usage

The program can easily used by the command `deepspeed [operation] -m model`
Currently support operations are: inference/train/fuzz
Currently support operations are: Vicuna/GPT2/Llama2/OpenAssistant

### Fine-tuning

When using the program to fine-tune a model, using `deepspeed --bind_cores_to_rank main.py train -m model` will enable CPU multiprocessing and make the program more performant
