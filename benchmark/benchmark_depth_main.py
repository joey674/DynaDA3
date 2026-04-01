from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError("benchmark_depth_main.py requires opencv-python (cv2).") from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from DynaDA3_model import DynaDA3
from depth_anything_3.utils.visualize import visualize_depth


DEFAULT_DATASET_ROOT = REPO_ROOT / "inputs" / "c3vd"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "benchmark"
DEFAULT_PROCESS_RES = 320
DEFAULT_PROCESS_RES_METHOD = "upper_bound_resize"
DEFAULT_POSE_WINDOW = 5
DEFAULT_SPEED_WARMUP = 1
DEFAULT_SPEED_REPEATS = 3
DEPTH_CLAMP_MM = 100.0


@dataclass
class C3VDDepthClip:
    dataset_name: str
    indices: list[int]
    color_paths: list[str]
    depth_paths: list[str]
    gt_depths_mm: np.ndarray

    @property
    def num_frames(self) -> int:
        return len(self.indices)


def timestamp_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def timestamp_now_minute() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_default_uncertainty_ckpt() -> str | None:
    ckpt = REPO_ROOT.parent / "checkpoint" / "DynaDA3-LARGE-1.1" / "uncertainty_head.pth"
    return str(ckpt) if ckpt.exists() else None


def create_output_dir(output_root: str | Path, dataset_name: str) -> Path:
    output_root = ensure_dir(output_root)
    base_name = f"depth_metrics_{dataset_name}_{timestamp_now_minute()}"
    candidate = output_root / base_name
    if not candidate.exists():
        return ensure_dir(candidate)

    suffix = 1
    while True:
        candidate = output_root / f"{base_name}_{suffix:02d}"
        if not candidate.exists():
            return ensure_dir(candidate)
        suffix += 1


def list_c3vd_datasets(dataset_root: str | Path = DEFAULT_DATASET_ROOT) -> list[str]:
    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        return []
    return sorted(
        path.name
        for path in dataset_root.iterdir()
        if path.is_dir() and list(path.glob("*_color.png"))
    )


def validate_dataset(dataset_name: str, dataset_root: str | Path = DEFAULT_DATASET_ROOT) -> Path:
    dataset_dir = Path(dataset_root) / dataset_name
    if not dataset_dir.exists():
        available = ", ".join(list_c3vd_datasets(dataset_root))
        raise FileNotFoundError(
            f"C3VD dataset `{dataset_name}` not found under {dataset_root}. Available: {available}"
        )
    return dataset_dir


def discover_frame_indices(dataset_dir: str | Path) -> list[int]:
    dataset_dir = Path(dataset_dir)
    indices: list[int] = []
    for path in dataset_dir.glob("*_color.png"):
        stem = path.name.split("_")[0]
        if stem.isdigit():
            indices.append(int(stem))
    if not indices:
        raise FileNotFoundError(f"No C3VD frames found in {dataset_dir}")
    return sorted(indices)


def sample_uniform_indices(frame_indices: Sequence[int], num_frames: int) -> list[int]:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")
    frame_indices = np.asarray(frame_indices, dtype=np.int32)
    if num_frames >= len(frame_indices):
        return frame_indices.tolist()
    picked = np.linspace(0, len(frame_indices) - 1, num_frames)
    picked = np.unique(np.round(picked).astype(np.int32))
    while len(picked) < num_frames:
        for idx in range(len(frame_indices)):
            if idx not in picked:
                picked = np.append(picked, idx)
                if len(picked) == num_frames:
                    break
    picked = np.sort(picked[:num_frames])
    return frame_indices[picked].tolist()


def sample_consecutive_indices(frame_indices: Sequence[int], num_frames: int) -> list[int]:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")
    frame_indices = list(frame_indices)
    if num_frames >= len(frame_indices):
        return frame_indices
    start = max(0, (len(frame_indices) - num_frames) // 2)
    return frame_indices[start : start + num_frames]


def build_frame_paths(dataset_dir: str | Path, indices: Sequence[int]) -> tuple[list[str], list[str]]:
    dataset_dir = Path(dataset_dir)
    color_paths: list[str] = []
    depth_paths: list[str] = []
    for idx in indices:
        color_path = dataset_dir / f"{idx:04d}_color.png"
        depth_path = dataset_dir / f"{idx:04d}_depth.tiff"
        if not color_path.exists():
            raise FileNotFoundError(f"Missing color frame: {color_path}")
        if not depth_path.exists():
            raise FileNotFoundError(f"Missing depth frame: {depth_path}")
        color_paths.append(str(color_path))
        depth_paths.append(str(depth_path))
    return color_paths, depth_paths


def load_c3vd_depth_mm(depth_path: str | Path) -> np.ndarray:
    raw = np.asarray(Image.open(depth_path), dtype=np.float32)
    return raw / 65535.0 * DEPTH_CLAMP_MM


def load_c3vd_depths_mm(depth_paths: Sequence[str]) -> np.ndarray:
    return np.stack([load_c3vd_depth_mm(path) for path in depth_paths], axis=0)


def build_depth_clip(
    dataset_name: str,
    num_frames: int,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    sample_mode: str = "uniform",
) -> C3VDDepthClip:
    dataset_dir = validate_dataset(dataset_name, dataset_root)
    frame_indices = discover_frame_indices(dataset_dir)

    if sample_mode == "uniform":
        selected = sample_uniform_indices(frame_indices, num_frames)
    elif sample_mode == "consecutive":
        selected = sample_consecutive_indices(frame_indices, num_frames)
    else:
        raise ValueError(f"Unsupported sample_mode `{sample_mode}`. Use `uniform` or `consecutive`.")

    color_paths, depth_paths = build_frame_paths(dataset_dir, selected)
    gt_depths_mm = load_c3vd_depths_mm(depth_paths)
    return C3VDDepthClip(
        dataset_name=dataset_name,
        indices=list(selected),
        color_paths=color_paths,
        depth_paths=depth_paths,
        gt_depths_mm=gt_depths_mm,
    )


def maybe_cuda_sync(device: str | torch.device) -> None:
    device = str(device)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def load_dynada3_model(
    model_name: str = "vitl",
    device: str | None = None,
    uncertainty_head_ckpt_path: str | None = None,
) -> tuple[DynaDA3, str]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if uncertainty_head_ckpt_path is None:
        uncertainty_head_ckpt_path = get_default_uncertainty_ckpt()

    model = DynaDA3(
        model_name=model_name,
        uncertainty_head_ckpt_path=uncertainty_head_ckpt_path,
    ).to(device)
    model.eval()
    return model, device


def run_dynada3_inference(
    model: DynaDA3,
    color_paths: Sequence[str],
    device: str,
    process_res: int = DEFAULT_PROCESS_RES,
    process_res_method: str = DEFAULT_PROCESS_RES_METHOD,
):
    maybe_cuda_sync(device)
    t0 = time.perf_counter()
    prediction = model.inference(
        image=list(color_paths),
        process_res=process_res,
        process_res_method=process_res_method,
    )
    maybe_cuda_sync(device)
    elapsed = time.perf_counter() - t0
    return prediction, elapsed


def resize_prediction_depths_to_gt(pred_depths: np.ndarray, gt_depths: np.ndarray) -> np.ndarray:
    pred_depths = np.asarray(pred_depths, dtype=np.float32)
    gt_depths = np.asarray(gt_depths, dtype=np.float32)
    resized = []
    for pred, gt in zip(pred_depths, gt_depths, strict=False):
        h, w = gt.shape
        resized_pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
        resized.append(resized_pred.astype(np.float32))
    return np.stack(resized, axis=0)


def compute_sequence_scale_lstsq(gt_mm: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    gt_vals = np.asarray(gt_mm[mask], dtype=np.float64)
    pred_vals = np.asarray(pred[mask], dtype=np.float64)
    valid = np.isfinite(gt_vals) & np.isfinite(pred_vals) & (gt_vals > 1e-6)
    gt_vals = gt_vals[valid]
    pred_vals = pred_vals[valid]
    denom = float(np.dot(pred_vals, pred_vals))
    if gt_vals.size == 0 or pred_vals.size == 0 or denom <= 1e-12:
        return 1.0
    return float(np.dot(pred_vals, gt_vals) / denom)


def compute_depth_metrics(
    gt_mm: np.ndarray,
    pred_mm: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    gt = np.asarray(gt_mm, dtype=np.float64)
    pred = np.asarray(pred_mm, dtype=np.float64)
    if mask is None:
        mask = gt > 0
    mask = mask & np.isfinite(gt) & np.isfinite(pred) & (gt > 1e-6) & (pred > 1e-6)
    if not np.any(mask):
        raise ValueError("No valid pixels available for depth evaluation.")

    gt = gt[mask]
    pred = pred[mask]
    diff = pred - gt
    ratio = np.maximum(gt / pred, pred / gt)

    return {
        "abs_rel": float(np.mean(np.abs(diff) / gt)),
        "sq_rel": float(np.mean((diff**2) / gt)),
        "rmse_mm": float(np.sqrt(np.mean(diff**2))),
        "rmse_log": float(np.sqrt(np.mean((np.log(pred) - np.log(gt)) ** 2))),
        "delta_1.25": float(np.mean(ratio < 1.25)),
    }


def _depth_range_mm(depth_mm: np.ndarray) -> tuple[float | None, float | None]:
    depth_mm = np.asarray(depth_mm, dtype=np.float32)
    valid = np.isfinite(depth_mm) & (depth_mm > 1e-6)
    if not np.any(valid):
        return None, None
    return float(depth_mm[valid].min()), float(depth_mm[valid].max())


def _plot_depth_metrics(metrics: dict[str, float], output_path: str | Path) -> None:
    labels = ["abs_rel", "sq_rel", "rmse_mm", "rmse_log", "delta_1.25"]
    values = [metrics[label] for label in labels]
    colors = ["#304ffe", "#00acc1", "#00897b", "#f57c00", "#7cb342"]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    bars = ax.bar(labels, values, color=colors)
    ax.set_title("DynaDA3 on C3VD: Depth Metrics")
    ax.set_ylabel("Metric Value")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for bar, value in zip(bars, values, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    info = (
        f"fps={metrics['fps']:.2f} | seconds={metrics['seconds']:.3f} | "
        f"global_scale={metrics['global_scale_lstsq']:.4f}"
    )
    ax.text(0.99, 0.98, info, transform=ax.transAxes, ha="right", va="top", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_depth_comparison(
    clip: C3VDDepthClip,
    pred_depths_aligned_mm: np.ndarray,
    output_path: str | Path,
) -> list[dict[str, Any]]:
    num_frames = clip.num_frames
    fig, axes = plt.subplots(num_frames, 4, figsize=(14.0, 3.5 * num_frames))
    if num_frames == 1:
        axes = np.asarray([axes])

    frame_records: list[dict[str, Any]] = []
    for row, (frame_idx, color_path, gt, pred) in enumerate(
        zip(clip.indices, clip.color_paths, clip.gt_depths_mm, pred_depths_aligned_mm, strict=False)
    ):
        rgb = plt.imread(color_path)
        gt_min, gt_max = _depth_range_mm(gt)
        pred_min, pred_max = _depth_range_mm(pred)
        gt_vis = visualize_depth(np.asarray(gt, dtype=np.float32), cmap="Spectral")
        pred_vis = visualize_depth(np.asarray(pred, dtype=np.float32), cmap="Spectral")
        diff = np.asarray(gt, dtype=np.float32) - np.asarray(pred, dtype=np.float32)
        valid_diff = np.isfinite(diff) & np.isfinite(gt) & np.isfinite(pred) & (gt > 1e-6) & (pred > 1e-6)
        diff_vis = diff.copy()
        diff_vis[~valid_diff] = np.nan
        if np.any(valid_diff):
            diff_abs_limit = float(np.percentile(np.abs(diff[valid_diff]), 95))
            if diff_abs_limit <= 1e-6:
                diff_abs_limit = float(np.max(np.abs(diff[valid_diff])))
            if diff_abs_limit <= 1e-6:
                diff_abs_limit = 1.0
            diff_min = float(np.min(diff[valid_diff]))
            diff_max = float(np.max(diff[valid_diff]))
        else:
            diff_abs_limit = 1.0
            diff_min = None
            diff_max = None

        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title(f"RGB #{frame_idx:04d}")
        axes[row, 1].imshow(gt_vis)
        axes[row, 1].set_title(
            "GT Depth\n"
            f"{gt_min:.2f} to {gt_max:.2f} mm" if gt_min is not None and gt_max is not None else "GT Depth\nn/a"
        )
        axes[row, 2].imshow(pred_vis)
        axes[row, 2].set_title(
            "Pred Depth (Scale-Aligned)\n"
            f"{pred_min:.2f} to {pred_max:.2f} mm"
            if pred_min is not None and pred_max is not None
            else "Pred Depth (Scale-Aligned)\nn/a"
        )
        axes[row, 3].imshow(diff_vis, cmap="coolwarm", vmin=-diff_abs_limit, vmax=diff_abs_limit)
        axes[row, 3].set_title(
            "GT - Pred (mm)\n"
            f"{diff_min:.2f} to {diff_max:.2f} mm | vis +/- {diff_abs_limit:.2f}"
            if diff_min is not None and diff_max is not None
            else "GT - Pred (mm)\nn/a"
        )

        for col in range(4):
            axes[row, col].axis("off")

        frame_records.append(
            {
                "frame_index": int(frame_idx),
                "source_color_path": str(color_path),
                "gt_min_mm": gt_min,
                "gt_max_mm": gt_max,
                "pred_min_mm": pred_min,
                "pred_max_mm": pred_max,
                "gt_minus_pred_min_mm": diff_min,
                "gt_minus_pred_max_mm": diff_max,
                "gt_minus_pred_vis_abs_limit_mm": diff_abs_limit,
            }
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return frame_records


def benchmark_depth_main(
    task: str = "depth",
    dataset: str = "desc_t4_a",
    num_frames: int = 16,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    process_res: int = DEFAULT_PROCESS_RES,
    process_res_method: str = DEFAULT_PROCESS_RES_METHOD,
    sample_mode: str | None = None,
    pose_window: int = DEFAULT_POSE_WINDOW,
    speed_warmup: int = DEFAULT_SPEED_WARMUP,
    speed_repeats: int = DEFAULT_SPEED_REPEATS,
    model_name: str = "vitl",
    device: str | None = None,
    uncertainty_head_ckpt_path: str | None = None,
) -> dict[str, Any]:
    """
    Self-contained depth benchmark for DynaDA3 on C3VD.

    Only depth evaluation is supported. Compatibility arguments unrelated to depth
    are accepted for API parity but ignored.
    """
    task = task.lower()
    if task not in {"depth", "depth_infer", "infer_depth"}:
        raise ValueError(f"benchmark_depth_main only supports depth evaluation, got task={task!r}.")

    model, resolved_device = load_dynada3_model(
        model_name=model_name,
        device=device,
        uncertainty_head_ckpt_path=uncertainty_head_ckpt_path,
    )

    clip = build_depth_clip(
        dataset_name=dataset,
        num_frames=num_frames,
        dataset_root=dataset_root,
        sample_mode=sample_mode or "uniform",
    )
    prediction, elapsed = run_dynada3_inference(
        model=model,
        color_paths=clip.color_paths,
        device=resolved_device,
        process_res=process_res,
        process_res_method=process_res_method,
    )

    pred_depths = resize_prediction_depths_to_gt(np.asarray(prediction.depth, dtype=np.float32), clip.gt_depths_mm)
    valid_mask = clip.gt_depths_mm > 0
    scale = compute_sequence_scale_lstsq(clip.gt_depths_mm, pred_depths, valid_mask)
    pred_depths_aligned = pred_depths * scale

    metrics = compute_depth_metrics(clip.gt_depths_mm, pred_depths_aligned, valid_mask)
    metrics["fps"] = float(clip.num_frames / max(elapsed, 1e-8))
    metrics["seconds"] = float(elapsed)
    metrics["global_scale_lstsq"] = float(scale)

    output_dir = create_output_dir(output_root, dataset)
    metrics_png = output_dir / "parameter.png"
    comparison_png = output_dir / "result.png"

    _plot_depth_metrics(metrics, metrics_png)
    frame_records = _plot_depth_comparison(clip, pred_depths_aligned, comparison_png)

    return {
        "task": "depth",
        "dataset_name": dataset,
        "config": {
            "task": task,
            "num_frames": int(num_frames),
            "dataset_root": str(dataset_root),
            "output_root": str(output_root),
            "process_res": int(process_res),
            "process_res_method": process_res_method,
            "sample_mode": sample_mode or "uniform",
            "model_name": model_name,
            "device": resolved_device,
            "uncertainty_head_ckpt_path": uncertainty_head_ckpt_path,
            "ignored_compatibility_args": {
                "pose_window": pose_window,
                "speed_warmup": speed_warmup,
                "speed_repeats": speed_repeats,
            },
        },
        "metrics": metrics,
        "artifacts": {
            "output_dir": str(output_dir),
            "metrics_png": str(metrics_png),
            "comparison_png": str(comparison_png),
        },
        "extra": {
            "clip": {
                "dataset_name": clip.dataset_name,
                "indices": clip.indices,
                "num_frames": clip.num_frames,
                "color_paths": clip.color_paths,
                "depth_paths": clip.depth_paths,
            },
            "prediction_depth_shape": list(prediction.depth.shape),
            "frame_depth_ranges_mm": frame_records,
        },
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Depth-only benchmark for DynaDA3 on C3VD.")
    parser.add_argument(
        "--task",
        default="depth",
        help="Compatibility arg. Only depth/depth_infer/infer_depth is supported.",
    )
    parser.add_argument("--dataset", default="desc_t4_a", help="Dataset folder under inputs/c3vd")
    parser.add_argument("--num-frames", type=int, default=16, help="Number of frames to sample")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--process-res", type=int, default=DEFAULT_PROCESS_RES)
    parser.add_argument("--process-res-method", default=DEFAULT_PROCESS_RES_METHOD)
    parser.add_argument("--sample-mode", default="uniform", choices=["uniform", "consecutive"])
    parser.add_argument("--pose-window", type=int, default=DEFAULT_POSE_WINDOW)
    parser.add_argument("--speed-warmup", type=int, default=DEFAULT_SPEED_WARMUP)
    parser.add_argument("--speed-repeats", type=int, default=DEFAULT_SPEED_REPEATS)
    parser.add_argument("--model-name", default="vitl", choices=["vitl", "vitg"])
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--uncertainty-head-ckpt-path",
        default=get_default_uncertainty_ckpt(),
        help="Path to uncertainty head checkpoint. By default uses ../checkpoint/DynaDA3-LARGE-1.1/uncertainty_head.pth when present.",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="Print datasets discovered under dataset_root and exit.",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    if args.list_datasets:
        for dataset in list_c3vd_datasets(args.dataset_root):
            print(dataset)
        return

    result = benchmark_depth_main(
        task=args.task,
        dataset=args.dataset,
        num_frames=args.num_frames,
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        process_res=args.process_res,
        process_res_method=args.process_res_method,
        sample_mode=args.sample_mode,
        pose_window=args.pose_window,
        speed_warmup=args.speed_warmup,
        speed_repeats=args.speed_repeats,
        model_name=args.model_name,
        device=args.device,
        uncertainty_head_ckpt_path=args.uncertainty_head_ckpt_path,
    )
    print(result["artifacts"]["output_dir"])


if __name__ == "__main__":
    main()
