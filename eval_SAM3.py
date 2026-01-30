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
from PIL import Image  

# ==============================================================================
# static settings
# ==============================================================================
# wildgs_racket1
# video_path = "../dataset/wildgs-slam/wildgs_racket1"
# text_prompts = ["person", "racket"] 
# wildgs_racket2
# video_path = "../dataset/wildgs-slam/wildgs_racket2"
# text_prompts = ["person", "racket"] 
# # wildgs_racket3
# video_path = "../dataset/wildgs-slam/wildgs_racket3"
# text_prompts = ["person", "racket"] 
# # wildgs_racket4
# video_path = "../dataset/wildgs-slam/wildgs_racket4"
# text_prompts = ["person", "racket"] 
# # wildgs_ANYmala1
# video_path = "../dataset/wildgs-slam/wildgs_ANYmal1"
# text_prompts = ["red robot with four legs"] 
# # wildgs_ANYmala2
# video_path = "../dataset/wildgs-slam/wildgs_ANYmal2"
# text_prompts = ["red robot with four legs"] 
# wildgs_ANYmala3
# video_path = "../dataset/wildgs-slam/wildgs_ANYmal3"
# text_prompts = ["red robot with four legs"] 
video_path = "../dataset/wildgs-slam/wildgs_ANYmal3"
text_prompts = ["red robot with four legs"] 


output_dir = "../dataset_dynada3_train"

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
# load images
# ==============================================================================
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
# Inference Session Initialization
# ==============================================================================
response = predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=video_path,
    )
)
session_id = response["session_id"]

# ==============================================================================
# Multi-Prompt Segmentation Loop
# ==============================================================================
# 用于存储每一帧的最终合并 Mask
# 结构: { frame_idx: np.ndarray(H, W) bool }
global_mask_storage = {}

print(f"Processing prompts sequentially: {text_prompts}")

for prompt in text_prompts:
    print("="*20)
    print(f"Running inference for prompt: '{prompt}'")
    
    # 重置 Session
    _ = predictor.handle_request(
        request=dict(
            type="reset_session",
            session_id=session_id,
        )
    )

    # 添加当前 Prompt
    frame_idx = 0 
    _ = predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=frame_idx,
            text=prompt,
        )
    )

    # Propagate
    outputs_per_frame = propagate_in_video(predictor, session_id)
    
    # 格式化数据
    outputs_per_frame = prepare_masks_for_visualization(outputs_per_frame)
    
    # 将当前 Prompt 的结果合并到全局存储中
    for f_idx, masks_dict in outputs_per_frame.items():
        if not masks_dict:
            continue
            
        # 提取当前 Prompt 在该帧下的所有物体 Mask，并将它们合并成一个 layer
        current_prompt_mask = None
        
        for obj_id, mask in masks_dict.items():
            if hasattr(mask, "cpu"):
                mask = mask.cpu().numpy()
            if mask.ndim > 2:
                mask = mask.squeeze()
            
            mask_bool = mask > 0.5
            
            if current_prompt_mask is None:
                current_prompt_mask = mask_bool
            else:
                current_prompt_mask = np.logical_or(current_prompt_mask, mask_bool)
        
        if current_prompt_mask is None:
            continue

        # 将当前 Prompt 的结果合并到全局存储 (Global Storage)
        if f_idx not in global_mask_storage:
            global_mask_storage[f_idx] = current_prompt_mask
        else:
            # 逻辑或：如果之前已经有 Mask，保留之前的，并加上新的
            global_mask_storage[f_idx] = np.logical_or(global_mask_storage[f_idx], current_prompt_mask)

print("All prompts processed. Saving results...")

# ==============================================================================
# save outputs (Pure PIL Version)
# ==============================================================================
total_frames = len(video_frames_for_vis)

for frame_idx in tqdm(range(total_frames)):
    
    frame_data = video_frames_for_vis[frame_idx]
    
    # 1. 安全读取图像 (PIL)
    frame_rgb = None
    try:
        if isinstance(frame_data, str):
            with Image.open(frame_data) as img:
                frame_rgb = np.array(img.convert("RGB"))
        elif isinstance(frame_data, np.ndarray):
            frame_rgb = frame_data
    except Exception as e:
        print(f"Error reading frame {frame_data}: {e}")
        continue
        
    if frame_rgb is None:
        continue

    h, w = frame_rgb.shape[:2]

    # 2. 获取最终 Mask
    if frame_idx in global_mask_storage:
        combined_mask = global_mask_storage[frame_idx]
    else:
        # 如果这一帧没有任何 prompt 检测到东西，生成全黑 Mask
        combined_mask = np.zeros((h, w), dtype=bool)

    # 3. 尺寸校验 (防止 Mask 尺寸和原图不一致)
    if combined_mask.shape != (h, w):
        # 极少情况，Resize mask
        combined_mask = cv2.resize(combined_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

    final_mask_arr = (combined_mask.astype(np.uint8)) * 255
    
    # 4. 保存 (使用 PIL)
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