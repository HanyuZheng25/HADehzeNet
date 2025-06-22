# <p align=center> :Codebook Prior-Guided Hybrid Attention Dehazing Network</p>

![Python 3.8](https://img.shields.io/badge/python-3.8-g) ![pytorch 1.12.0](https://img.shields.io/badge/pytorch-1.12.0-blue.svg)

This is the official PyTorch codes for the paper.  

## Dependencies and Installation
- Ubuntu >= 18.04
- CUDA >= 11.0
- Other required packages in `requirements.txt`

# create new anaconda env
conda create -n HADehzeNet python=3.8
conda activate HADehzeNet

# install python dependencies
pip install -r requirements.txt

### Train your model
CUDA_VISIBLE_DEVICES=X python basicsr/train.py -opt options/Train_your_model

