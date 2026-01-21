# DA3
cd /home/zhouyi/repo/SegDA3
pip install xformers torch\>=2 torchvision
pip install -e . 

# SegDA3
conda activate da3-slam
python SegDA3_eval.py

# SAM3 for dataset:
conda activate sam3
python SAM3_eval.py

# dataset wildgs-slam
git clone https://huggingface.co/datasets/gradient-spaces/Wild-SLAM

