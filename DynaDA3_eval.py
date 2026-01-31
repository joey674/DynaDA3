import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from DynaDA3_model import DynaDA3
from depth_anything_3.utils.visualize import visualize_depth


# ================= config =================
DATASETS = {
    "UKA": [
        "../dataset/UKA/UKA_Case1Part1_cropped/000319.jpg",
        "../dataset/UKA/UKA_Case1Part1_cropped/000320.jpg",
        "../dataset/UKA/UKA_Case1Part1_cropped/000321.jpg",
        "../dataset/UKA/UKA_Case1Part1_cropped/000322.jpg",
        "../dataset/UKA/UKA_Case1Part1_cropped/000323.jpg",
        "../dataset/UKA/UKA_Case1Part1_cropped/000324.jpg",
        "../dataset/UKA/UKA_Case1Part1_cropped/000325.jpg",
        "../dataset/UKA/UKA_Case1Part1_cropped/000326.jpg",
    ],
    "2077_scene1": [
        "../dataset/2077/2077_scene1/000005.jpg",
        "../dataset/2077/2077_scene1/000006.jpg",
        "../dataset/2077/2077_scene1/000007.jpg",
        "../dataset/2077/2077_scene1/000008.jpg",
        "../dataset/2077/2077_scene1/000009.jpg", 
        "../dataset/2077/2077_scene1/000010.jpg", 
        "../dataset/2077/2077_scene1/000011.jpg", 
        "../dataset/2077/2077_scene1/000012.jpg", 
        "../dataset/2077/2077_scene1/000013.jpg", 
    ],
    "wildgs_anymal": [ 
        "../dataset/wildgs-slam/wildgs_ANYmal_test/frame_00601.png",
        "../dataset/wildgs-slam/wildgs_ANYmal_test/frame_00606.png",
        "../dataset/wildgs-slam/wildgs_ANYmal_test/frame_00611.png",
        "../dataset/wildgs-slam/wildgs_ANYmal_test/frame_00616.png",
        "../dataset/wildgs-slam/wildgs_ANYmal_test/frame_00621.png",
        "../dataset/wildgs-slam/wildgs_ANYmal_test/frame_00626.png",
        "../dataset/wildgs-slam/wildgs_ANYmal_test/frame_00631.png",
        "../dataset/wildgs-slam/wildgs_ANYmal_test/frame_00636.png",
        "../dataset/wildgs-slam/wildgs_ANYmal_test/frame_00641.png",
    ],
    "wildgs_racket": [ 
        "../dataset/wildgs-slam/wildgs_racket_test/frame_00830.png",
        "../dataset/wildgs-slam/wildgs_racket_test/frame_00840.png",
        "../dataset/wildgs-slam/wildgs_racket_test/frame_00850.png",
        "../dataset/wildgs-slam/wildgs_racket_test/frame_00860.png",
        "../dataset/wildgs-slam/wildgs_racket_test/frame_00870.png",
        "../dataset/wildgs-slam/wildgs_racket_test/frame_00880.png",
        "../dataset/wildgs-slam/wildgs_racket_test/frame_00890.png",
    ],
    "wildgs_tower": [ 
        "../dataset/wildgs-slam/wildgs_tower_test/frame_01000.png",
        "../dataset/wildgs-slam/wildgs_tower_test/frame_01010.png",
        "../dataset/wildgs-slam/wildgs_tower_test/frame_01020.png",
        "../dataset/wildgs-slam/wildgs_tower_test/frame_01030.png",
        "../dataset/wildgs-slam/wildgs_tower_test/frame_01040.png",
        "../dataset/wildgs-slam/wildgs_tower_test/frame_01050.png",
        "../dataset/wildgs-slam/wildgs_tower_test/frame_01060.png",
        "../dataset/wildgs-slam/wildgs_tower_test/frame_01070.png",
        "../dataset/wildgs-slam/wildgs_tower_test/frame_01080.png",
        "../dataset/wildgs-slam/wildgs_tower_test/frame_01090.png",
    ]
}

SAVE_PATH = "../output"
ckpt_path = "../checkpoint/DynaDA3-LARGE-1.1/uncertainty_head.pth"
# ===========================================

def evaluate_single_dataset(model, dataset_name, img_paths, device):
    print(f"\nProcessing dataset: {dataset_name} ({len(img_paths)} images)")
    
    # 推理
    print("Running inference...")
    try:
        pred = model.inference(image=img_paths)
    except Exception as e:
        print(f"Error during inference on {dataset_name}: {e}")
        return

    # 准备数据
    images = pred.processed_images  # numpy uint8, [N,3,H,W]
    depths = pred.depth  # numpy [N,H,W]
    confs = pred.conf  # numpy [N,H,W]
    masks = pred.uncertainty_seg_mask.detach().cpu().numpy()  # [N,H,W]

    # ==== conf statistics ====
    confs_arr = np.array(confs)
    N, H, W = confs_arr.shape
    total_points = confs_arr.size
    
    # histogram with 5 bins
    num_bins = 5
    hist, bin_edges = np.histogram(confs_arr, bins=num_bins)

    # Determine display range
    display_vmin = 1.0
    tail_count = int(hist[-1])
    tail_pct = tail_count / total_points if total_points > 0 else 0.0
    negligible_tail_thresh = 1e-5
    
    if tail_pct < negligible_tail_thresh:
         display_vmax = float(bin_edges[-2])
    else:
        display_vmax = float(np.nanpercentile(confs_arr, 99.9))
        
    if display_vmax <= display_vmin:
        display_vmax = display_vmin + 1.0

    print(f"== {dataset_name} STATS: N={N}, vmin={display_vmin:.2f}, vmax={display_vmax:.2f} ==")

    # 绘图
    print("Plotting...")
    num_imgs = len(img_paths)
    
    fig, axes = plt.subplots(4, num_imgs, figsize=(3 * num_imgs, 12))
    # Handle single image case where axes might be significantly different shape
    if num_imgs == 1:
        axes = axes.reshape(4, 1)

    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    for i in range(num_imgs):
        # 1. RGB
        axes[0, i].imshow(images[i])
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title(f"RGB ({dataset_name})", fontsize=12, loc="left")

        # 2. Depth
        depth_vis = visualize_depth(depths[i], cmap="Spectral")
        axes[1, i].imshow(depth_vis)
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("Depth Prediction (DA3)", fontsize=12, loc="left")

        # 3. Confidence
        conf_map = confs[i]
        frame_min = float(np.nanmin(conf_map))
        frame_max = float(np.nanmax(conf_map))
        im = axes[2, i].imshow(conf_map, cmap="inferno", vmin=display_vmin, vmax=display_vmax)
        axes[2, i].axis("off")
        if i == 0:
            axes[2, i].set_title(f"Confidence\nmin {frame_min:.1f}, max {frame_max:.1f}", fontsize=12)
        else:
            axes[2, i].set_title(f"min {frame_min:.1f}, max {frame_max:.1f}", fontsize=9)

        # 4. Uncertainty Mask
        mask_bw = (masks[i] > 0).astype(np.uint8)
        axes[3, i].imshow(mask_bw, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
        axes[3, i].axis("off")
        if i == 0:
            axes[3, i].set_title("Uncertainty Mask (white=uncertain)", fontsize=12, loc="left")

    # 保存
    dt_str = datetime.now().strftime("%m%d_%H%M")
    save_filename = os.path.join(SAVE_PATH, f"DynaDA3_eval_{dataset_name}_{dt_str}.png")
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    # Colorbar
    try:
        from matplotlib import cm
        sm = cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(vmin=display_vmin, vmax=display_vmax))
        sm.set_array([])
        fig.colorbar(sm, ax=axes[2, :], orientation='vertical', fraction=0.02, label='confidence')
    except Exception:
        pass

    plt.savefig(save_filename, bbox_inches="tight", dpi=150)
    plt.close(fig) # Close plot to free memory
    print(f"Result saved to {save_filename}")
    return save_filename


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载模型
    print("Loading model...")
    model = DynaDA3(
        model_name='vitl', 
        # uncertainty_head_ckpt_path=ckpt_path
    ).to(device)
    
    # 遍历数据集
    saved_files = []
    for name, paths in DATASETS.items():
        try:
             outfile = evaluate_single_dataset(model, name, paths, device)
             if outfile:
                 saved_files.append(outfile)
        except Exception as e:
            print(f"Failed to process dataset {name}: {e}")

    print("\n" + "="*40)
    print("Evaluation Summary: All files saved to:")
    for f in saved_files:
        print(f" -> {f}")
    print("="*40)

if __name__ == "__main__":
    main()
