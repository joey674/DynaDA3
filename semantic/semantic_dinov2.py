import sys
import os

# Add src to sys.path to find dinov2_seg package
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.append(src_path)

import math
import itertools
from functools import partial
import urllib.request

import torch
import torch.nn.functional as F
from mmseg.apis import init_segmentor, inference_segmentor

import dinov2_seg
import dinov2_seg.models

#######################################
# 由于输出的语义分割图大小需要是patch size(14)的整数倍，因此需要在输入图像上进行中心填充
class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output

# 接口转换：通过 partial 改写 model.backbone.forward，使其调用 DINOv2 的 get_intermediate_layers 方法。这允许分割头（Segmentation Head）获取 Transformer 内部层的特征图。
def create_segmenter(cfg, backbone_model):
    model = init_segmentor(cfg)
    model.backbone.forward = partial(
        backbone_model.get_intermediate_layers,
        n=cfg.model.backbone.out_indices,
        reshape=True,
    )
    if hasattr(backbone_model, "patch_size"):
        model.backbone.register_forward_pre_hook(lambda _, x: CenterPadding(backbone_model.patch_size)(x[0]))
    model.init_weights()
    return model

#######################################
# Load pretrained backbone
BACKBONE_SIZE = "small" # in ("small", "base", "large" or "giant")

backbone_archs = {
    "small": "vits14",
    "base": "vitb14",
    "large": "vitl14",
    "giant": "vitg14",
}
backbone_arch = backbone_archs[BACKBONE_SIZE]
backbone_name = f"dinov2_{backbone_arch}"

# Keep using torch.hub to load the backbone as requested
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoint")

backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name, pretrained=False)
backbone_checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{backbone_name}_pretrain.pth")
backbone_model.load_state_dict(torch.load(backbone_checkpoint_path))
backbone_model.eval()
backbone_model.cuda()

########################################
# Load pretrained segmentation head
import mmcv
from mmcv.runner import load_checkpoint

def load_config_from_url(url: str) -> str:
    with urllib.request.urlopen(url) as f:
        return f.read().decode()

HEAD_SCALE_COUNT = 3 # more scales: slower but better results, in (1,2,3,4,5)
HEAD_DATASET = "voc2012" # in ("ade20k", "voc2012")
HEAD_TYPE = "ms" # in ("ms, "linear")

head_config_path = os.path.join(CHECKPOINT_DIR, f"{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_config.py")
head_checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_head.pth")

cfg = mmcv.Config.fromfile(head_config_path)
if HEAD_TYPE == "ms":
    cfg.data.test.pipeline[1]["img_ratios"] = cfg.data.test.pipeline[1]["img_ratios"][:HEAD_SCALE_COUNT]
    print("scales:", cfg.data.test.pipeline[1]["img_ratios"])

model = create_segmenter(cfg, backbone_model=backbone_model)
load_checkpoint(model, head_checkpoint_path, map_location="cpu")
model.cuda()
model.eval()

########################################
# load sample image
from PIL import Image
image_path = "semantic/checkpoint/example.jpg"
image = Image.open(image_path).convert("RGB")

# ########################################
# run segmentation
import numpy as np
import time
# Import colormaps from extracted source
import dinov2_seg.utils.colormaps as colormaps

DATASET_COLORMAPS = {
    "ade20k": colormaps.ADE20K_COLORMAP,
    "voc2012": colormaps.VOC2012_COLORMAP,
}

def render_segmentation(segmentation_logits, dataset):
    colormap = DATASET_COLORMAPS[dataset]
    colormap_array = np.array(colormap, dtype=np.uint8)
    segmentation_values = colormap_array[segmentation_logits + 1]
    return Image.fromarray(segmentation_values)

start_time = time.time()
array = np.array(image)[:, :, ::-1] # BGR
segmentation_logits = inference_segmentor(model, array)[0]
segmented_image = render_segmentation(segmentation_logits, HEAD_DATASET)
end_time = time.time()
print(f"Segmentation took {end_time - start_time:.2f} seconds")
# 这里名字是日期加时间: MMDD_HHMM
file_name = f"semantic/output/{os.path.basename(image_path).rsplit('.', 1)[0]}_{time.strftime('%m%d_%H%M%S', time.localtime())}.png"
segmented_image.save(file_name)
print(f"saved {file_name}")