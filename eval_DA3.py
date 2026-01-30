from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.visualize import visualize_depth

import numpy as np
import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt


# ================= 配置区域 =================
output_folder = "../output"
os.makedirs(output_folder, exist_ok=True)

model_path = "../checkpoint/DA3-GIANT-1.1"

image_paths = [
    "../dataset/2077/2077_scene1/000005.jpg",
    "../dataset/2077/2077_scene1/000006.jpg",
    "../dataset/2077/2077_scene1/000007.jpg",
    "../dataset/2077/2077_scene1/000008.jpg",
    "../dataset/2077/2077_scene1/000009.jpg",
]

export_feat_layers = [3, 7, 11, 15, 19, 23]

save_path = os.path.join(output_folder, "DA3_eval.png")
if os.path.exists(save_path):
    os.remove(save_path)
    print(f"Removed existing file: {save_path}")
# ===========================================

def feat_layer_to_heatmaps(feat_nhwc_np: np.ndarray, H: int, W: int, reduce="l2") -> np.ndarray:
    """
    写死按照你当前 API inference 的输出：
      prediction.aux["feat_layer_x"] 是 numpy.ndarray，形状 [N, h, w, C]
      例如 (5, 28, 36, 1024)

    返回：
      heatmaps: [N, H, W]，范围 [0,1]
    """
    assert isinstance(feat_nhwc_np, np.ndarray), f"Expected numpy, got {type(feat_nhwc_np)}"
    assert feat_nhwc_np.ndim == 4, f"Expected [N,h,w,C], got {feat_nhwc_np.shape}"

    N, h, w, C = feat_nhwc_np.shape

    feat = torch.from_numpy(feat_nhwc_np).float()        # [N,h,w,C]
    feat = feat.permute(0, 3, 1, 2).contiguous()         # [N,C,h,w]

    if reduce == "mean":
        hm = feat.mean(dim=1, keepdim=True)              # [N,1,h,w]
    elif reduce == "l2":
        hm = torch.sqrt((feat ** 2).sum(dim=1, keepdim=True) + 1e-6)
    else:
        raise ValueError("reduce must be 'mean' or 'l2'")

    # normalize per image
    hm_flat = hm.view(N, -1)
    mn = hm_flat.min(dim=1)[0].view(N, 1, 1, 1)
    mx = hm_flat.max(dim=1)[0].view(N, 1, 1, 1)
    hm = (hm - mn) / (mx - mn + 1e-6)

    hm = F.interpolate(hm, size=(H, W), mode="bilinear", align_corners=False)  # [N,1,H,W]
    return hm[:, 0].numpy()  # [N,H,W]

def overlay(rgb_uint8: np.ndarray, heat_01: np.ndarray, alpha=0.45, cmap="jet") -> np.ndarray:
    """
    写死按照 processed_images 输出：
      rgb_uint8: [H,W,3] uint8
      heat_01:   [H,W] float in [0,1]
    """
    rgb = rgb_uint8.astype(np.float32) / 255.0
    cm = plt.get_cmap(cmap)
    hm_rgb = cm(heat_01)[..., :3].astype(np.float32)
    out = (1 - alpha) * rgb + alpha * hm_rgb
    return np.clip(out, 0, 1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    print("Loading model...")
    model = DepthAnything3.from_pretrained(model_path).to(device)

    print("Running inference...")
    prediction = model.inference(
        image=image_paths,                
        export_feat_layers=export_feat_layers,
    )

    print(f"Depth shape: {prediction.depth.shape}")
    print(f"Extrinsics: {prediction.extrinsics.shape if prediction.extrinsics is not None else 'None'}")
    print(f"Intrinsics: {prediction.intrinsics.shape if prediction.intrinsics is not None else 'None'}")
    print("Prediction keys:", prediction.__dict__.keys())

    # 写死：processed_images numpy uint8，[N,H,W,3]
    images = prediction.processed_images
    assert isinstance(images, np.ndarray) and images.ndim == 4 and images.shape[-1] == 3, \
        f"processed_images bad shape/type: {type(images)} {getattr(images,'shape',None)}"

    # 写死：depth numpy，[N,H,W]
    depths = prediction.depth
    assert isinstance(depths, np.ndarray) and depths.ndim == 3, \
        f"depth bad shape/type: {type(depths)} {getattr(depths,'shape',None)}"

    N = depths.shape[0]
    H, W = depths.shape[-2], depths.shape[-1]

    # 写死：aux dict，内部 feat_layer_x 都是 numpy [B,N,h,w,C]
    assert hasattr(prediction, "aux") and isinstance(prediction.aux, dict), "prediction.aux missing"
    feat_keys = [k for k in prediction.aux.keys() if k.startswith("feat_layer_")]
    feat_keys = sorted(feat_keys, key=lambda k: int(k.split("_")[-1]))
    print("Exported feat keys:", feat_keys)
    assert len(feat_keys) > 0, "No exported features found. Did you set export_feat_layers?"

    # plot: 2 + len(feat_keys) rows, N cols
    rows = 2 + len(feat_keys)
    fig, axes = plt.subplots(rows, N, figsize=(3.0 * N, 3.0 * rows))
    if N == 1:
        axes = axes.reshape(rows, 1)

    # Row 0: input
    for i in range(N):
        axes[0, i].imshow(images[i])
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Input", loc="left")

    # Row 1: depth
    for i in range(N):
        depth_vis = visualize_depth(depths[i], cmap="Spectral")
        axes[1, i].imshow(depth_vis)
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("Depth", loc="left")

    # Feature rows (overlay heatmap)
    for r, k in enumerate(feat_keys):
        feat_np = prediction.aux[k]  # numpy [B,N,h,w,C]
        assert isinstance(feat_np, np.ndarray), f"{k} is not numpy: {type(feat_np)}"
        heatmaps = feat_layer_to_heatmaps(feat_np, H, W, reduce="l2")  # [N,H,W]
        for i in range(N):
            axes[2 + r, i].imshow(overlay(images[i], heatmaps[i], alpha=0.45, cmap="jet"))
            axes[2 + r, i].axis("off")
            if i == 0:
                axes[2 + r, i].set_title(k, loc="left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print("Saved visualization to:", save_path)


if __name__ == "__main__":
    main()
