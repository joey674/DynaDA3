import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from depth_anything_3.api import DepthAnything3


# ==========================================
# 1) DPT 分割头
#    输入：list[Tensor]，每个 Tensor: [N, C, h, w]
#    输出：logits [N, K, H, W]
# ==========================================
class SimpleDPTHead(nn.Module):
    def __init__(self, in_channels=1024, embed_dim=256, num_classes=2, readout_indices=(0, 1, 2, 3)):
        super().__init__()
        self.readout_indices = list(readout_indices)

        self.projects = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, embed_dim, kernel_size=1),
                    nn.ReLU(inplace=True),
                )
                for _ in range(len(self.readout_indices))
            ]
        )

        self.output_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, num_classes, kernel_size=1),
        )

    def forward(self, features: list[torch.Tensor], H: int, W: int) -> torch.Tensor:
        # features: list of [N, C, h, w]
        selected = [features[i] for i in self.readout_indices]

        # 对齐到同一分辨率（以第一个为基准）
        target_h, target_w = selected[0].shape[-2:]
        proj = []
        for i, feat in enumerate(selected):
            x = self.projects[i](feat)
            if x.shape[-2:] != (target_h, target_w):
                x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
            proj.append(x)

        fused = 0
        for x in proj:
            fused = fused + x

        logits = self.output_head(fused)  # [N, K, target_h, target_w]
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)  # [N,K,H,W]
        return logits


# ==========================================
# 2) SegDA3：完全走 API inference
#    - 主体输出：DepthAnything3.inference() 的 Prediction 原样
#    - 额外输出：motion_seg_logits / motion_seg_mask
# ==========================================
class SegDA3(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        embed_dim: int = 256,
        in_channels: int = 1024,
        export_feat_layers=(3, 7, 11, 15, 19, 23),
    ):
        super().__init__()
        model_path = "/home/zhouyi/repo/model_DepthAnythingV3/checkpoints/DA3-LARGE-1.1"
        print(f"Loading DA3 from local path: {model_path}...")
        self.da3 = DepthAnything3.from_pretrained(model_path)

        # 冻结 DA3，只训练 seg_head
        for p in self.da3.parameters():
            p.requires_grad = False
        self.da3.eval()

        self.export_feat_layers = list(export_feat_layers)

        self.seg_head = SimpleDPTHead(
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_classes=num_classes,
            readout_indices=(0, 1, 2, 3),
        )

    @staticmethod
    def _aux_feat_to_nchw(feat: np.ndarray, device: torch.device) -> torch.Tensor:
        """
        写死按照你当前 API inference 的实际输出：
          feat shape: [N, h, w, C]  (例如 5,28,36,1024)
        返回：
          torch.Tensor [N, C, h, w] on device
        """
        assert isinstance(feat, np.ndarray), f"aux feat must be numpy, got {type(feat)}"
        assert feat.ndim == 4, f"aux feat must be [N,h,w,C], got {feat.shape}"
        t = torch.from_numpy(feat).to(device=device, dtype=torch.float32)     # [N,h,w,C]
        t = t.permute(0, 3, 1, 2).contiguous()                               # [N,C,h,w]
        return t

    def _run_motion_head(self, prediction, device: torch.device):
        """
        从 prediction.aux 取特征，跑 seg_head，写回 prediction.motion_seg_*
        """
        assert hasattr(prediction, "aux") and isinstance(prediction.aux, dict), "prediction.aux missing"

        # 用 processed_images 的分辨率作为最终输出 H,W（和你 demo 对齐）
        assert prediction.processed_images is not None
        H, W = prediction.processed_images.shape[1], prediction.processed_images.shape[2]

        # 按 export_feat_layers 的顺序组织 feature list
        feats = []
        for layer in self.export_feat_layers:
            k = f"feat_layer_{layer}"
            assert k in prediction.aux, f"Missing {k} in prediction.aux"
            feats.append(self._aux_feat_to_nchw(prediction.aux[k], device))

        logits = self.seg_head(feats, H, W)                # [N,K,H,W]
        mask = torch.argmax(logits, dim=1)                 # [N,H,W]

        prediction.motion_seg_logits = logits
        prediction.motion_seg_mask = mask

    @torch.no_grad()
    def inference(self, image, **kwargs):
        """
        完全走官方 API inference（主体输出不改），只额外挂 motion seg。
        你原来的 demo 用 image=，这里也用 image=。
        """
        device = next(self.seg_head.parameters()).device

        pred = self.da3.inference(
            image=image,
            export_feat_layers=self.export_feat_layers,
        )

        # 额外 head（只有 head 在 device 上跑）
        self._run_motion_head(pred, device)

        return pred

    def forward(self, *args, **kwargs):
        """
        训练建议别走这里（因为 api.inference 通常是 no_grad + numpy 输出）。
        如果你后面要训练 motion head，建议你用：
          - 先离线跑 inference 导出 aux（numpy）
          - 或者改为：直接调用 da3.model.forward 得到 torch feat（会更快且可控）
        但你当前需求是“主体必须用 api output”，所以这里不提供训练 forward。
        """
        raise RuntimeError("Use .inference(...) for this wrapper (API-driven).")
