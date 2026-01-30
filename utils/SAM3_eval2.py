import os
import sam3
import torch
import time
from sam3.model_builder import build_sam3_video_predictor

import glob
import cv2  # 仅用于处理视频流
import numpy as np
from sam3.visualization_utils import (
    prepare_masks_for_visualization,
)
from tqdm import tqdm
from PIL import Image  # 全程使用 PIL

# ==============================================================================
# 1. 设置 Settings
# ==============================================================================
video_path = "../dataset/wildgs-slam/wildgs_racket1"
text_prompts = ["person", "racket"] 

output_dir = "../dataset_dynada3_train"
# ==============================================================================
# 2. 初始化 Setup
# ==============================================================================
sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
gpus_to_use = range(torch.cuda.device_count())

# 初始化模型
print("Building SAM3 Predictor...")
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
# 3. 辅助函数 Helpers
# ==============================================================================
def propagate_in_video(predictor, session_id):
    outputs_per_frame = {}
    # 这里的 stream request 必须依赖于一个干净的 session_id
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]
    return outputs_per_frame

# ==============================================================================
# 4. 预加载图片路径 Load Images Info
# ==============================================================================
# 我们需要先知道有哪些图片，用于最后的保存循环
if isinstance(video_path, str) and video_path.endswith(".mp4"):
    cap = cv2.VideoCapture(video_path)
    video_frames_for_vis = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        video_frames_for_vis.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
else:
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

total_frames = len(video_frames_for_vis)
print(f"Total frames to process: {total_frames}")

# ==============================================================================
# 5. 推理阶段 Inference Loop (修正版：Strict Lifecycle)
# ==============================================================================
# 结构: [ dict_result_prompt_1, dict_result_prompt_2, ... ]
all_prompts_results_list = []

for i, prompt in enumerate(text_prompts):
    print(f"\n========================================================")
    print(f"[Prompt {i+1}/{len(text_prompts)}] Processing: '{prompt}'")
    print(f"========================================================")
    
    # --- 关键修正 1: Start Session 在循环内 ---
    # 这会强制模型重新加载视频帧的 Embedding，确保从第0帧开始
    # 虽然会有 "loading images" 的开销，但解决了 0it 问题
    print(" -> Starting new session (reloading video context)...")
    response = predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=video_path,
        )
    )
    session_id = response["session_id"]
    
    # --- 关键修正 2: 显式 Reset (保险起见，参考 notebook) ---
    _ = predictor.handle_request(
        request=dict(
            type="reset_session",
            session_id=session_id,
        )
    )

    # 添加 Prompt (Frame 0)
    print(f" -> Adding prompt '{prompt}' to frame 0...")
    _ = predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=0,
            text=prompt,
        )
    )

    # 推理 (Propagate)
    print(" -> Propagating prompt through video...")
    outputs_per_frame = propagate_in_video(predictor, session_id)
    
    # 检查是否成功
    frames_detected = len(outputs_per_frame)
    if frames_detected == 0:
        print(f"WARNING: No objects detected for prompt '{prompt}'! (0 frames)")
    else:
        print(f" -> Success! Detected objects in {frames_detected} frames.")

    # 格式化并暂存结果
    formatted_output = prepare_masks_for_visualization(outputs_per_frame)
    all_prompts_results_list.append(formatted_output)
    
    # --- 关键修正 3: Close Session 在循环内 ---
    # 释放显存，销毁当前 Session 的所有状态
    print(" -> Closing session to clear state.")
    _ = predictor.handle_request(
        request=dict(
            type="close_session",
            session_id=session_id,
        )
    )

print("\nAll prompts processed. Starting Merge & Save process...")

# ==============================================================================
# 6. 合并与保存 Merge & Save (逻辑不变，纯 PIL)
# ==============================================================================
for frame_idx in tqdm(range(total_frames), desc="Merging & Saving"):
    
    frame_data = video_frames_for_vis[frame_idx]
    
    # --- 6.1 读取原图 ---
    frame_rgb = None
    try:
        if isinstance(frame_data, str):
            with Image.open(frame_data) as img:
                frame_rgb = np.array(img.convert("RGB"))
        elif isinstance(frame_data, np.ndarray):
            frame_rgb = frame_data
    except Exception as e:
        print(f"Error reading frame {frame_idx}: {e}")
        continue
        
    if frame_rgb is None:
        continue

    h, w = frame_rgb.shape[:2]

    # --- 6.2 核心合并逻辑 ---
    # 初始化底图 (False = 黑色)
    combined_mask_bool = np.zeros((h, w), dtype=bool)
    
    # 遍历列表，把 "person" 和 "racket" 的结果依次叠上去
    for prompt_result_dict in all_prompts_results_list:
        
        if frame_idx in prompt_result_dict:
            masks_dict = prompt_result_dict[frame_idx]
            
            for obj_id, mask_tensor in masks_dict.items():
                if hasattr(mask_tensor, "cpu"):
                    mask_arr = mask_tensor.cpu().numpy()
                else:
                    mask_arr = mask_tensor
                
                if mask_arr.ndim > 2:
                    mask_arr = mask_arr.squeeze()
                
                # Logical OR 合并: 只要任意一个 prompt 说是前景，就是前景
                combined_mask_bool = np.logical_or(combined_mask_bool, mask_arr > 0.5)

    # --- 6.3 保存 ---
    final_mask_uint8 = (combined_mask_bool.astype(np.uint8)) * 255
    
    filename_base = f"{frame_idx:05d}"
    
    try:
        # 保存原图
        Image.fromarray(frame_rgb).save(os.path.join(output_image_dir, f"{filename_base}.png"))

        # 保存 Mask
        Image.fromarray(final_mask_uint8, mode='L').save(os.path.join(output_mask_dir, f"{filename_base}.png"))
        
    except Exception as e:
        print(f"Error saving frame {frame_idx}: {e}")

# --- 关键修正 4: 彻底关闭 Predictor (参考 notebook 的 Clean-up) ---
try:
    print("Shutting down predictor...")
    predictor.shutdown()
except Exception as e:
    print(f"Shutdown warning: {e}")

print("Processing completed. Outputs saved to:", output_dir)

# ==============================================================================
# Time
# ==============================================================================
end_time = time.time()
elapsed_time = end_time - start_time
print(f"used time: {elapsed_time:.2f} s")