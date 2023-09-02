This package will create flask endpoints of a given model in the configuration file

## Setting Up
### Install Conda and Setup CUDA
```bash
conda create -n benchmark python=3.11 anaconda
conda activate benchmark
# conda install -c nvidia cuda-python
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit #-- REPLACE THIS WITH YOUR CUDA VERSION --#
python -m pip install cuda-python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/\

python -m pip install tqdm
python -m pip install torch
python -m pip install tiktoken
python -m pip install -U bitsandbytes
python -m pip install -U git+https://github.com/huggingface/transformers.git
python -m pip install -U git+https://github.com/huggingface/peft.git
python -m pip install -U git+https://github.com/huggingface/accelerate.git


```

### Install Dependancies
`python -m pip install torch torchvision torchaudio`
`python -m pip install vllm`

### VastAI setup

```bash
pip3 install tqdm torch tiktoken transformers peft accelerate torchvision torchaudio vllm auto-gptq optimum
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install
git lfs clone
```
