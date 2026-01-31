# DynaDA3

## File Struct
checkpoint
dataset
dataset_dynada3_train
log
output
DynaDA3
DynaDA3-SLAM

## deploy
Torch: 2.0.0+cu117
CUDA: 11.7
xFormers 
numpy<2
torch>=2

### DynaDA3
```bash
### conda
conda create -n dynada3 python=3.11
conda activate dynada3
### torch torchvision cuda numpy
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu117 \
  torch==2.0.0+cu117 torchvision==0.15.0+cu117
python -m pip install xformers --no-deps
python -m pip install "numpy<2"
### da3 dependency
pip install -e .
```

### MMSeg
```bash
# python -m pip install --extra-index-url https://download.pytorch.org/whl/cu117 torch==2.0.0+cu117 torchvision==0.15.0+cu117
python -m pip install --no-cache-dir --no-deps --no-index \
  -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html \
  mmcv-full==1.7.2
python -m pip install mmsegmentation==0.30.0
# python -m pip install --force-reinstall "numpy<2"
python -m pip install addict pyyaml yapf packaging opencv-python
```

### validate
```bash
python - <<'PY'
import numpy, torch, torchvision, mmcv, mmseg
print("numpy", numpy.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("torchvision", torchvision.__version__)
print("mmcv", mmcv.__version__)
print("mmseg", mmseg.__version__)
PY
python semantic/semantic_segmentation.py
python DynaDA3_eval.py
```


## trainning in AutoDL
```bash
unzip DynaDA3-main.zip 

python DynaDA3_train.py
python DynaDA3_eval.py
python ./semantic/semantic_segmentation.py
```

## model version
当前uncertainty_head.pth是代码会调用的ckpt;
这个uncertainty_head.pth是最新训练的ckpt版本的复制(比如是uncertainty_headv3_stable.pth)作为当前调用的对象
当训练新的版本时, 请把model的版本和ckpt的版本对应上保存;
模型就保存在utils里
```bash
git checkout -b V3-Depth-Conf
git fetch
git branch --set-upstream-to=origin/V3-Depth-Conf V3-Depth-Conf
```

## dataset
### SAM3 for dataset:
```bash
conda create -n sam3 python=3.12
conda deactivate
conda activate sam3
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -e ../sam3

conda activate sam3
python SAM3_eval.py

### dataset wildgs-slam
git clone https://huggingface.co/datasets/gradient-spaces/Wild-SLAM
```

