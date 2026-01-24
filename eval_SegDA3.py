import os
import torch
import matplotlib.pyplot as plt
import numpy as np

from SegDA3_model import SegDA3
from depth_anything_3.utils.visualize import visualize_depth


# ================= config =================
# UKA
# IMG_PATHS = [
#     "../dataset/UKA/UKA1/Case1Part1_1cropped/cropped_000956.jpg",
#     "../dataset/UKA/UKA1/Case1Part1_1cropped/cropped_000957.jpg",
#     "../dataset/UKA/UKA1/Case1Part1_1cropped/cropped_000958.jpg",
#     "../dataset/UKA/UKA1/Case1Part1_1cropped/cropped_000959.jpg",
#     "../dataset/UKA/UKA1/Case1Part1_1cropped/cropped_000960.jpg",
# ]

# 2077scene1
# IMG_PATHS = [ 
#     "../dataset/2077/2077_scene1/000005.jpg",
#     "../dataset/2077/2077_scene1/000006.jpg",
#     "../dataset/2077/2077_scene1/000007.jpg",
#     "../dataset/2077/2077_scene1/000008.jpg",
#     "../dataset/2077/2077_scene1/000009.jpg", 
#     "../dataset/2077/2077_scene1/000010.jpg", 
#     "../dataset/2077/2077_scene1/000011.jpg", 
#     "../dataset/2077/2077_scene1/000012.jpg", 
#     "../dataset/2077/2077_scene1/000013.jpg", 
# ]

# wildgs-slam
IMG_PATHS = [ 
    "../dataset/2077/2077_scene1/000005.jpg",
    "../dataset/2077/2077_scene1/000006.jpg",
    "../dataset/2077/2077_scene1/000007.jpg",
    "../dataset/2077/2077_scene1/000008.jpg",
    "../dataset/2077/2077_scene1/000009.jpg", 
    "../dataset/2077/2077_scene1/000010.jpg", 
    "../dataset/2077/2077_scene1/000011.jpg", 
    "../dataset/2077/2077_scene1/000012.jpg", 
    "../dataset/2077/2077_scene1/000013.jpg", 
]


SAVE_PATH = "../output"
ckpt_path = "../checkpoint/SegDA3-LARGE-1.1/motion_head.pth"
# ===========================================


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载模型
    print("Loading model...")
    model = SegDA3(
        model_name='vitl', 
        motion_head_ckpt_path=ckpt_path
    ).to(device)

    # 推理
    print("Running inference...")
    pred = model.inference(image=IMG_PATHS)

    # 确认 DA3 输出是否正常
    print("== DA3 OUTPUT CHECK ==")
    print("processed_images:",
          type(pred.processed_images),
          getattr(pred.processed_images, "shape", None),
          getattr(pred.processed_images, "dtype", None))
    print("depth:",
          type(pred.depth),
          getattr(pred.depth, "shape", None),
          getattr(pred.depth, "dtype", None))
    print("conf:",
          type(pred.conf),
          getattr(pred.conf, "shape", None),
          getattr(pred.conf, "dtype", None))
    print("aux keys:",
          list(pred.aux.keys()) if hasattr(pred, "aux") and isinstance(pred.aux, dict) else None)
    print("======================")

    # 准备数据
    images = pred.processed_images  # numpy uint8, [N,3,H,W]
    depths = pred.depth  # numpy [N,H,W]
    confs = pred.conf  # numpy [N,H,W]
    masks = pred.motion_seg_mask.detach().cpu().numpy()  # [N,H,W]

    # 绘图
    print("Plotting...")
    num_imgs = len(IMG_PATHS)
    fig, axes = plt.subplots(4, num_imgs, figsize=(3 * num_imgs, 12))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    for i in range(num_imgs):
        axes[0, i].imshow(images[i])
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Input RGB", fontsize=12, loc="left")

        depth_vis = visualize_depth(depths[i], cmap="Spectral")
        axes[1, i].imshow(depth_vis)
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("Depth Prediction (DA3)", fontsize=12, loc="left")

        conf_map = confs[i] 
        axes[2, i].imshow(conf_map, cmap="inferno", vmin=0.0, vmax=15) 
        axes[2, i].axis("off")
        if i == 0:
            axes[2, i].set_title("Confidence Map (DA3)", fontsize=12)

        axes[3, i].imshow(masks[i], cmap="jet", interpolation="nearest", vmin=0, vmax=1)
        axes[3, i].axis("off")
        if i == 0:
            axes[3, i].set_title("Motion Mask (motion Head)", fontsize=12, loc="left")

    # 保存
    save_filename = os.path.join(SAVE_PATH, "SegDA3_eval_" + os.path.dirname(IMG_PATHS[0]).split("/")[-1] + ".png")
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    plt.savefig(save_filename, bbox_inches="tight", dpi=150)
    print(f"Result saved to {save_filename}")

if __name__ == "__main__":
    main()
