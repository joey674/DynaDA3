import os
import sam3
import torch
import time
from sam3.model_builder import build_sam3_video_predictor

import glob
import numpy as np
from sam3.visualization_utils import (
    prepare_masks_for_visualization,
)
from tqdm import tqdm
from PIL import Image  # 核心：全流程使用 PIL

# ==============================================================================
# static settings
# ==============================================================================
video_path = "../dataset/wildgs-slam/wildgs_racket1"
text_prompt = "person . racket"

output_dir = "../DynaDA3/inputs"
# ==============================================================================
# setup
# ==============================================================================
sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
gpus_to_use = range(torch.cuda.device_count())

predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)

start_time = time.time()

video_path_name = os.path.basename(video_path)
output_dir = os.path.join(output_dir, video_path_name)
os.makedirs(output_dir, exist_ok=True)
output_image_dir = os.path.join(output_dir, "images")
os.makedirs(output_image_dir, exist_ok=True)
output_mask_dir = os.path.join(output_dir, "masks")
os.makedirs(output_mask_dir, exist_ok=True)

# ==============================================================================
# helper functions
# ==============================================================================
def propagate_in_video(predictor, session_id):
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]
    return outputs_per_frame

# ==============================================================================
# load video/images
# ==============================================================================
# 注意：这里我们只收集路径，不读取图片
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.PNG']
video_frames_for_vis = []
for ext in image_extensions:
    video_frames_for_vis.extend(glob.glob(os.path.join(video_path, ext)))

try:
    video_frames_for_vis.sort(
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
    )
except ValueError:
    print("Sorting by filename failed, using lexicographic sort.")
    video_frames_for_vis.sort()

if not video_frames_for_vis:
    raise ValueError(f"No images found in {video_path}")

# ==============================================================================
# Inference Session
# ==============================================================================
response = predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=video_path,
    )
)
session_id = response["session_id"]

# ==============================================================================
# Segmentation
# ==============================================================================
_ = predictor.handle_request(
    request=dict(
        type="reset_session",
        session_id=session_id,
    )
)

frame_idx = 0 
response = predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=frame_idx,
        text=text_prompt,
    )
)

outputs_per_frame = propagate_in_video(predictor, session_id)
outputs_per_frame = prepare_masks_for_visualization(outputs_per_frame)

# ==============================================================================
# save outputs (Pure PIL Version)
# ==============================================================================
sorted_frame_indices = sorted(outputs_per_frame.keys())
print("Saving results using PIL...")

for frame_idx in tqdm(sorted_frame_indices):
    if frame_idx >= len(video_frames_for_vis):
        continue
        
    frame_data = video_frames_for_vis[frame_idx]
    
    # 1. 安全读取图像 (PIL vs Numpy)
    frame_rgb = None
    
    try:
        if isinstance(frame_data, str):
            # 关键修改：使用 PIL 打开图片，完全避开 cv2.imread
            with Image.open(frame_data) as img:
                frame_rgb = np.array(img.convert("RGB"))
        elif isinstance(frame_data, np.ndarray):
            # 如果是视频流过来的已经是 RGB 数组
            frame_rgb = frame_data
    except Exception as e:
        print(f"Error reading frame {frame_data}: {e}")
        continue
        
    if frame_rgb is None:
        continue

    h, w = frame_rgb.shape[:2]

    # 2. 处理 Mask (保持 Numpy 计算)
    masks_dict = outputs_per_frame.get(frame_idx, {})
    combined_mask = np.zeros((h, w), dtype=bool)
    
    for obj_id, mask in masks_dict.items():
        if hasattr(mask, "cpu"):
            mask = mask.cpu().numpy()
        if mask.ndim > 2:
            mask = mask.squeeze()
        combined_mask = np.logical_or(combined_mask, mask > 0.5)

    final_mask_arr = (combined_mask.astype(np.uint8)) * 255
    
    # 3. 保存 (使用 PIL)
    filename_base = f"{frame_idx:05d}"
    
    try:
        # 保存原图
        Image.fromarray(frame_rgb).save(os.path.join(output_image_dir, f"{filename_base}.png"))

        # 保存 Mask (L模式=灰度)
        Image.fromarray(final_mask_arr, mode='L').save(os.path.join(output_mask_dir, f"{filename_base}.png"))
        
    except Exception as e:
        print(f"Error saving frame {frame_idx}: {e}")

print("Processing completed. Outputs saved to:", output_dir)

# ==============================================================================
# time
# ==============================================================================
end_time = time.time()
elapsed_time = end_time - start_time
print(f"used time: {elapsed_time:.2f} s")