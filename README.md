# <p align=center> Codebook Prior-Guided Hybrid Attention Dehazing Network</p>

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
CUDA_VISIBLE_DEVICES=X python basicsr/train.py -opt options/Train_your_model.yml

