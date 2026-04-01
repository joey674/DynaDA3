#
conda deactivate 
conda remove -n seg -y --all
conda create -n seg python=3.10 -y
conda activate seg
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu117 \
  torch==2.0.0+cu117 torchvision==0.15.0+cu117
python -m pip install addict pyyaml yapf packaging opencv-python
python -m pip install --no-cache-dir --no-deps --no-index \
  -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html \
  mmcv-full==1.7.2
python -m pip install mmsegmentation==0.30.0
python -m pip install --force-reinstall "numpy<2"


#
python - <<'PY'
import numpy, torch, torchvision, mmcv, mmseg
print("numpy", numpy.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("torchvision", torchvision.__version__)
print("mmcv", mmcv.__version__)
print("mmseg", mmseg.__version__)
PY

conda activate seg
python semantic/semantic_segmentation.py 
python semantic/semantic_dinov2.py 