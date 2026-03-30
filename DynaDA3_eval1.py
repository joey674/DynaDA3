#################
# DynaDA3 Evaluation Script
# DynaDA3_eval1.py 是“逐项导出”脚本。它不画 confidence，而是把每帧的 rgb/ depth/ uncertainty_mask 分别存到子目录里，并且如果模型输出了 extrinsics，还会额外画一张 3D 相机轨迹图
#################

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import gc

from DynaDA3_model import DynaDA3
from depth_anything_3.utils.visualize import visualize_depth


# ================= config =================
DATASETS = {
    #     "2077_scene1": [
    #     "../dataset/2077/2077_scene1/000005.jpg",
    #     "../dataset/2077/2077_scene1/000006.jpg",
    #     "../dataset/2077/2077_scene1/000007.jpg",
    #     "../dataset/2077/2077_scene1/000008.jpg",
    #     "../dataset/2077/2077_scene1/000009.jpg", 
    #     "../dataset/2077/2077_scene1/000010.jpg", 
    #     "../dataset/2077/2077_scene1/000011.jpg", 
    #     "../dataset/2077/2077_scene1/000012.jpg", 
    #     "../dataset/2077/2077_scene1/000013.jpg", 
    # ],
    # "wildgs_anymal": [ 
    #     "../dataset/wildgs-slam/wildgs_ANYmal_test/frame_00601.png",
    #     "../dataset/wildgs-slam/wildgs_ANYmal_test/frame_00606.png",
    #     "../dataset/wildgs-slam/wildgs_ANYmal_test/frame_00611.png",
    #     "../dataset/wildgs-slam/wildgs_ANYmal_test/frame_00616.png",
    #     "../dataset/wildgs-slam/wildgs_ANYmal_test/frame_00621.png",
    #     "../dataset/wildgs-slam/wildgs_ANYmal_test/frame_00626.png",
    #     "../dataset/wildgs-slam/wildgs_ANYmal_test/frame_00631.png",
    #     "../dataset/wildgs-slam/wildgs_ANYmal_test/frame_00636.png",
    #     "../dataset/wildgs-slam/wildgs_ANYmal_test/frame_00641.png",
    # ],
    # "wildgs_racket": [ 
    #     "../dataset/wildgs-slam/wildgs_racket_test/frame_00830.png",
    #     "../dataset/wildgs-slam/wildgs_racket_test/frame_00840.png",
    #     "../dataset/wildgs-slam/wildgs_racket_test/frame_00850.png",
    #     "../dataset/wildgs-slam/wildgs_racket_test/frame_00860.png",
    #     "../dataset/wildgs-slam/wildgs_racket_test/frame_00870.png",
    #     "../dataset/wildgs-slam/wildgs_racket_test/frame_00880.png",
    #     "../dataset/wildgs-slam/wildgs_racket_test/frame_00890.png",
    # ],
    # "wildgs_tower": [ 
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
    # ],
        "UKA": [
        "../dataset/UKA/UKA_Case1Part1_cropped/000319.jpg",
        # "../dataset/UKA/UKA_Case1Part1_cropped/000320.jpg",
        "../dataset/UKA/UKA_Case1Part1_cropped/000321.jpg",
        # "../dataset/UKA/UKA_Case1Part1_cropped/000322.jpg",
        "../dataset/UKA/UKA_Case1Part1_cropped/000323.jpg",
        # "../dataset/UKA/UKA_Case1Part1_cropped/000324.jpg",
        "../dataset/UKA/UKA_Case1Part1_cropped/000325.jpg",
        # "../dataset/UKA/UKA_Case1Part1_cropped/000326.jpg",
    ],
}

SAVE_PATH = "/outputs"
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
    images = pred.processed_images
    depths = pred.depth
    masks = pred.uncertainty_seg_mask.detach().cpu().numpy()
    extrinsics = getattr(pred, "extrinsics", None)

    def to_hwc_uint8(image):
        arr = np.asarray(image)
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        if np.issubdtype(arr.dtype, np.floating):
            max_val = float(np.nanmax(arr)) if arr.size > 0 else 0.0
            if max_val <= 1.0:
                arr = arr * 255.0
        return np.clip(arr, 0, 255).astype(np.uint8)

    # 分开保存 RGB / depth / uncertainty mask
    print("Saving images...")
    dt_str = datetime.now().strftime("%m%d_%H%M")
    save_root = os.path.join(SAVE_PATH, f"DynaDA3_eval_{dataset_name}_{dt_str}")
    rgb_dir = os.path.join(save_root, "rgb")
    depth_dir = os.path.join(save_root, "depth")
    mask_dir = os.path.join(save_root, "uncertainty_mask")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    saved_files = []
    num_imgs = len(img_paths)
    for i in range(num_imgs):
        img_name = os.path.splitext(os.path.basename(img_paths[i]))[0]
        stem = f"{i:03d}_{img_name}"

        rgb_path = os.path.join(rgb_dir, f"{stem}.png")
        depth_path = os.path.join(depth_dir, f"{stem}.png")
        mask_path = os.path.join(mask_dir, f"{stem}.png")

        rgb_img = to_hwc_uint8(images[i])
        depth_vis = to_hwc_uint8(visualize_depth(depths[i], cmap="Spectral"))
        mask_bw = ((masks[i] > 0).astype(np.uint8) * 255)

        plt.imsave(rgb_path, rgb_img)
        plt.imsave(depth_path, depth_vis)
        plt.imsave(mask_path, mask_bw, cmap="gray", vmin=0, vmax=255)

        saved_files.extend([rgb_path, depth_path, mask_path])

    # 保存相机位姿可视化（若有 extrinsics）
    if extrinsics is not None:
        try:
            exts = np.asarray(extrinsics)
            if exts.ndim != 3:
                raise ValueError(f"Invalid extrinsics shape: {exts.shape}")

            if exts.shape[1:] == (3, 4):
                exts_h = np.tile(np.eye(4, dtype=np.float64), (exts.shape[0], 1, 1))
                exts_h[:, :3, :] = exts
            elif exts.shape[1:] == (4, 4):
                exts_h = exts.astype(np.float64, copy=False)
            else:
                raise ValueError(f"Unsupported extrinsics shape: {exts.shape}")

            c2w = np.linalg.inv(exts_h)
            cam_centers = c2w[:, :3, 3]
            cam_forward = c2w[:, :3, 2]

            pose_path = os.path.join(save_root, "camera_pose_3d.png")
            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(111, projection="3d")

            ax.plot(
                cam_centers[:, 0],
                cam_centers[:, 1],
                cam_centers[:, 2],
                "-o",
                markersize=4,
                linewidth=1.5,
                color="tab:blue",
                label="Camera centers",
            )

            span_x = float(np.ptp(cam_centers[:, 0])) if cam_centers.shape[0] > 0 else 0.0
            span_y = float(np.ptp(cam_centers[:, 1])) if cam_centers.shape[0] > 0 else 0.0
            span_z = float(np.ptp(cam_centers[:, 2])) if cam_centers.shape[0] > 0 else 0.0
            arrow_len = max(span_x, span_y, span_z, 1e-3) * 0.08
            ax.quiver(
                cam_centers[:, 0],
                cam_centers[:, 1],
                cam_centers[:, 2],
                cam_forward[:, 0],
                cam_forward[:, 1],
                cam_forward[:, 2],
                length=arrow_len,
                normalize=True,
                color="tab:red",
                linewidth=1.0,
                label="Forward",
            )

            if cam_centers.shape[0] > 0:
                center = cam_centers.mean(axis=0)
                radius = max(span_x, span_y, span_z, 1e-3) * 0.6
                ax.set_xlim(center[0] - radius, center[0] + radius)
                ax.set_ylim(center[1] - radius, center[1] + radius)
                ax.set_zlim(center[2] - radius, center[2] + radius)

            ax.set_title(f"Camera Poses ({dataset_name})")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.legend(loc="upper right")
            plt.tight_layout()
            plt.savefig(pose_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            saved_files.append(pose_path)
            print(f"Saved camera pose visualization: {pose_path}")
        except Exception as e:
            print(f"Failed to save camera pose visualization for {dataset_name}: {e}")
    else:
        print(f"No extrinsics available for {dataset_name}, skip camera pose visualization.")

    print(f"Results saved to directory: {save_root}")
    print(f"Saved {num_imgs} RGB, {num_imgs} depth, {num_imgs} uncertainty-mask images.")

    # Clean up memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return saved_files


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载模型
    print("Loading model...")
    
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at {ckpt_path}, loading with None.")
        uncertainty_ckpt = None
    else:
        uncertainty_ckpt = ckpt_path

    model = DynaDA3(
        model_name='vitl', 
        uncertainty_head_ckpt_path=uncertainty_ckpt
    ).to(device)
    
    # 遍历数据集
    saved_files = []
    for name, paths in DATASETS.items():
        try:
            outfiles = evaluate_single_dataset(model, name, paths, device)
            if outfiles:
                saved_files.extend(outfiles)
        except Exception as e:
            print(f"Failed to process dataset {name}: {e}")

    print("\n" + "="*40)
    print("Evaluation Summary: All files saved to:")
    for f in saved_files:
        print(f" -> {f}")
    print("="*40)

if __name__ == "__main__":
    main()
