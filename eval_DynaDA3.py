import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from DynaDA3_model import DynaDA3
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

# wildgs-slam AnYmal test
# IMG_PATHS = [ 
#     "../dataset/wildgs-slam/wildgs_ANYmal_test/frame_00601.png",
#     "../dataset/wildgs-slam/wildgs_ANYmal_test/frame_00606.png",
#     "../dataset/wildgs-slam/wildgs_ANYmal_test/frame_00611.png",
#     "../dataset/wildgs-slam/wildgs_ANYmal_test/frame_00616.png",
#     "../dataset/wildgs-slam/wildgs_ANYmal_test/frame_00621.png",
#     "../dataset/wildgs-slam/wildgs_ANYmal_test/frame_00626.png",
#     "../dataset/wildgs-slam/wildgs_ANYmal_test/frame_00631.png",
#     "../dataset/wildgs-slam/wildgs_ANYmal_test/frame_00636.png",
#     "../dataset/wildgs-slam/wildgs_ANYmal_test/frame_00641.png",
# ]

# # wildgs-slam racket test
# IMG_PATHS = [ 
#     "../dataset/wildgs-slam/wildgs_racket_test/frame_00830.png",
#     "../dataset/wildgs-slam/wildgs_racket_test/frame_00840.png",
#     "../dataset/wildgs-slam/wildgs_racket_test/frame_00850.png",
#     "../dataset/wildgs-slam/wildgs_racket_test/frame_00860.png",
#     "../dataset/wildgs-slam/wildgs_racket_test/frame_00870.png",
#     "../dataset/wildgs-slam/wildgs_racket_test/frame_00880.png",
#     "../dataset/wildgs-slam/wildgs_racket_test/frame_00890.png",
# ]

# wildgs-slam tower test
# IMG_PATHS = [ 
#     "../dataset/wildgs-slam/wildgs_tower_test/frame_01000.png",
#     "../dataset/wildgs-slam/wildgs_tower_test/frame_01010.png",
#     "../dataset/wildgs-slam/wildgs_tower_test/frame_01020.png",
#     "../dataset/wildgs-slam/wildgs_tower_test/frame_01030.png",
#     "../dataset/wildgs-slam/wildgs_tower_test/frame_01040.png",
#     "../dataset/wildgs-slam/wildgs_tower_test/frame_01050.png",
#     "../dataset/wildgs-slam/wildgs_tower_test/frame_01060.png",
#     "../dataset/wildgs-slam/wildgs_tower_test/frame_01070.png",
#     "../dataset/wildgs-slam/wildgs_tower_test/frame_01080.png",
#     "../dataset/wildgs-slam/wildgs_tower_test/frame_01090.png",
# ]


SAVE_PATH = "../output"
ckpt_path = "../checkpoint/DynaDA3-LARGE-1.1/motion_head.pth"
# ===========================================


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载模型
    print("Loading model...")
    model = DynaDA3(
        model_name='vitl', 
        # uncertainty_head_ckpt_path=ckpt_path
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
    masks = pred.uncertainty_seg_mask.detach().cpu().numpy()  # [N,H,W]

    # ==== conf statistics ====
    confs_arr = np.array(confs)
    if confs_arr.ndim != 3:
        raise ValueError(f"Unexpected confs shape: {confs_arr.shape}, expected [N,H,W]")
    N, H, W = confs_arr.shape
    total_points = confs_arr.size
    per_frame_points = H * W
    global_min = float(np.nanmin(confs_arr))
    global_max = float(np.nanmax(confs_arr))
    global_mean = float(np.nanmean(confs_arr))
    global_std = float(np.nanstd(confs_arr))

    # histogram with 5 bins (quartile-like ranges)
    num_bins = 5
    hist, bin_edges = np.histogram(confs_arr, bins=num_bins)

    # Determine display range: start at 1, choose adaptive vmax to avoid extreme outliers
    display_vmin = 1.0
    # if the highest histogram bin contains negligible points, cap at previous bin edge
    tail_count = int(hist[-1])
    tail_pct = tail_count / total_points if total_points > 0 else 0.0
    # threshold for negligible tail (example: 0.001% = 1e-5)
    negligible_tail_thresh = 1e-5
    if tail_pct < negligible_tail_thresh:
        # cap at upper edge of the second-to-last bin
        display_vmax = float(bin_edges[-2])
        cap_reason = f"tail bin negligible ({tail_count} pts, {tail_pct:.6f} fraction) -> cap at bin_edges[-2]"
    else:
        # otherwise use a high percentile to exclude extreme outliers
        display_vmax = float(np.nanpercentile(confs_arr, 99.9))
        cap_reason = "use 99.9th percentile"
    # ensure vmax > vmin
    if display_vmax <= display_vmin:
        display_vmax = display_vmin + 1.0

    # count how many points are above the display_vmax
    above_cap = int(np.sum(confs_arr > display_vmax))
    above_cap_pct = 100.0 * above_cap / total_points if total_points > 0 else 0.0

    print("== CONF STATISTICS ==")
    print(f"frames (N): {N}")
    print(f"points per frame: {per_frame_points}")
    print(f"total points: {total_points}")
    print(f"global min: {global_min:.6f}, global max: {global_max:.6f}")
    print(f"global mean: {global_mean:.6f}, global std: {global_std:.6f}")
    print(f"display vmin: {display_vmin:.6f}, display vmax: {display_vmax:.6f} ({cap_reason})")
    print(f"points above display_vmax: {above_cap} ({above_cap_pct:.6f}%)")
    print("Histogram (counts) and ranges:")
    for i in range(num_bins):
        lo = bin_edges[i]
        hi = bin_edges[i+1]
        cnt = hist[i]
        pct = 100.0 * cnt / total_points if total_points > 0 else 0.0
        print(f"  bin {i+1}: [{lo:.6f}, {hi:.6f}) -> {cnt} points ({pct:.2f}%)")
    print("======================")

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
        # per-frame min/max
        frame_min = float(np.nanmin(conf_map))
        frame_max = float(np.nanmax(conf_map))
        im = axes[2, i].imshow(conf_map, cmap="inferno", vmin=display_vmin, vmax=display_vmax)
        axes[2, i].axis("off")
        if i == 0:
            axes[2, i].set_title(f"Confidence Map (DA3)\nmin {frame_min:.3f}, max {frame_max:.3f}", fontsize=12)
        else:
            axes[2, i].set_title(f"min {frame_min:.3f}, max {frame_max:.3f}", fontsize=10)

        axes[3, i].imshow(masks[i], cmap="jet", interpolation="nearest", vmin=0, vmax=1)
        axes[3, i].axis("off")
        if i == 0:
            axes[3, i].set_title("Motion Mask (motion Head)", fontsize=12, loc="left")

    # 保存（文件名加入日期时间，例如 0129_1139）
    dt_str = datetime.now().strftime("%m%d_%H%M")
    scene_name = os.path.dirname(IMG_PATHS[0]).split("/")[-1]
    save_filename = os.path.join(SAVE_PATH, f"DynaDA3_eval_{scene_name}_{dt_str}.png")
    os.makedirs(SAVE_PATH, exist_ok=True)
    # add a single colorbar for the confidence row using the global range
    try:
        from matplotlib import cm
        from matplotlib import colors
        sm = cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(vmin=display_vmin, vmax=display_vmax))
        sm.set_array([])
        fig.colorbar(sm, ax=axes[2, :], orientation='vertical', fraction=0.02, label='confidence')
    except Exception:
        pass

    plt.savefig(save_filename, bbox_inches="tight", dpi=150)
    print(f"Result saved to {save_filename}")

if __name__ == "__main__":
    main()
