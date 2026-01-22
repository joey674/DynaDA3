
# SAM3 for dataset:
conda create -n sam3 python=3.12
conda deactivate
conda activate sam3
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
cd /home/zhouyi/repo/sam3
pip install -e .

conda activate sam3
python SAM3_eval.py

# dataset wildgs-slam
git clone https://huggingface.co/datasets/gradient-spaces/Wild-SLAM

