# DynaDA3

## File Struct
checkpoint
dataset
dataset_dynada3_train
log
output
DynaDA3
DynaDA3-SLAM

## trainning
unzip DynaDA3-main.zip 

## dataset
### SAM3 for dataset:
conda create -n sam3 python=3.12
conda deactivate
conda activate sam3
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -e ../sam3

conda activate sam3
python SAM3_eval.py

### dataset wildgs-slam
git clone https://huggingface.co/datasets/gradient-spaces/Wild-SLAM

