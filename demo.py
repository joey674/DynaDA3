from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.visualize import visualize_depth
import numpy as np
import time
import torch
import os
import matplotlib.pyplot as plt

# 创建保存结果的文件夹
output_folder = "/home/zhouyi/repo/model_DepthAnythingV3/output"
os.makedirs(output_folder, exist_ok=True)

visualization_path = os.path.join(output_folder, "input_and_depth_visualization.png")
if os.path.exists(visualization_path):
    os.remove(visualization_path)
    print(f"Removed existing file: {visualization_path}")

model_path = "/home/zhouyi/repo/model_DepthAnythingV3/checkpoints/DA3-LARGE-1.1"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = DepthAnything3.from_pretrained(model_path).to(device)
   
image1 = "/home/zhouyi/repo/dataset/2077/scene1/000005.jpg"
image2 = "/home/zhouyi/repo/dataset/2077/scene1/000006.jpg"
image3 = "/home/zhouyi/repo/dataset/2077/scene1/000007.jpg"
image4 = "/home/zhouyi/repo/dataset/2077/scene1/000008.jpg"
image5 = "/home/zhouyi/repo/dataset/2077/scene1/000009.jpg"
prediction = model.inference(
    image=[image1, image2, image3, image4, image5], 
)

print(f"Depth shape: {prediction.depth.shape}")
print(f"Extrinsics: {prediction.extrinsics.shape if prediction.extrinsics is not None else 'None'}")
print(f"Intrinsics: {prediction.intrinsics.shape if prediction.intrinsics is not None else 'None'}")

n_images = prediction.depth.shape[0]
fig, axes = plt.subplots(2, n_images, figsize=(12, 6))

if n_images == 1:
    axes = axes.reshape(2, 1)

for i in range(n_images):
    # Show original image
    if prediction.processed_images is not None:
        axes[0, i].imshow(prediction.processed_images[i])
    axes[0, i].set_title(f"Input {i+1}")
    axes[0, i].axis('off')
    
    # Show depth map
    depth_vis = visualize_depth(prediction.depth[i], cmap="Spectral")
    axes[1, i].imshow(depth_vis)
    axes[1, i].set_title(f"Depth {i+1}")
    axes[1, i].axis('off')

plt.tight_layout()

# 保存可视化结果
plt.savefig(visualization_path, dpi=150, bbox_inches='tight')