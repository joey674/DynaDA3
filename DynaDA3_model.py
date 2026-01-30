import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.logger import logger

DA3_VITG_CHANNELS = 1536 
DA3_VITL_CHANNELS = 1024
DA3_VITG_FEAT_LAYERS=(21, 27, 33, 39)
DA3_VITL_FEAT_LAYERS=(11, 15, 19, 23)
DA3_VITG_CKPT_PATH = "../checkpoint/DA3-GIANT-1.1"
DA3_VITL_CKPT_PATH = "../checkpoint/DA3-LARGE-1.1"
uncertaintyDPT_EMBED_DIM = 256

MODEL_CONFIGS = {
    'vitl': {
        'channels': DA3_VITL_CHANNELS,
        'feat_layers': DA3_VITL_FEAT_LAYERS,
        'ckpt_path': DA3_VITL_CKPT_PATH
    },
    'vitg': {
        'channels': DA3_VITG_CHANNELS,
        'feat_layers': DA3_VITG_FEAT_LAYERS,
        'ckpt_path': DA3_VITG_CKPT_PATH
    }
}

class UncertaintyDPT(nn.Module):
    """
    uncertainty DPT 
    Args:
        B: Batch Size=1
        N: Frame Sequence Length
        H, W: 原始输入图像分辨率 (例如 518 * 518)
        h, w: 特征图分辨率; 对于 ViT 架构，通常 patch size 为 14, 所以 h = H/14, w = W/14 (518/14=37)
        c_in: DINO输入通道数 (DA3_CHANNELS = 1536)
        c_embed: 嵌入维度 (uncertaintyDPT_EMBED_DIM)
        feat_layers: 从 DA3 提取的指定特征层索引列表
        features: DA3 backbone 指定层输出特征列表, 每个元素 shape: [N, c_in, h, w]
        conf: 置信度图, shape: [N, 1, H, W]
        depth: 深度图, shape: [N, 1, H, W]
    """
    def __init__(self, c_in, feat_layers, c_embed=uncertaintyDPT_EMBED_DIM):
        super().__init__()
        self.feat_layers = list(feat_layers)# 选择哪些 DINO 层的特征用于分割头
        self.c_embed = c_embed

        # 将每层特征映射到统一的 embedding 维度
        self.feat_projs = nn.ModuleList(
            [nn.Conv2d(c_in, c_embed, kernel_size=1) for _ in self.feat_layers]
        )

        # 融合 conf/depth 与特征（在高分辨率 H,W 上）
        self.fuse_layer = nn.Sequential(
            nn.Conv2d(c_embed + 2, c_embed, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_embed),
            nn.ReLU(inplace=True),
        )


        num_classes = 2  # 运动分割类别数：移动 / 静止
        self.output_layer = nn.Sequential(
            nn.Conv2d(c_embed, c_embed // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_embed // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_embed // 2, num_classes, kernel_size=1),
        )

    def forward(
        self,
        features: list[torch.Tensor],
        H: int,
        W: int,
        conf: torch.Tensor,
        depth: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: List[[N, c_in, h, w]]
            H, W: 原始输入图像分辨率
            conf: 置信度图 Tensor, shape: [N, 1, H, W]
            depth: 深度图 Tensor, shape: [N, 1, H, W]
        Returns:
            logits: 运动分割 logits, shape: [N, num_classes, H, W]
        """
        assert len(features) == len(self.feat_projs), (
            f"features length {len(features)} != feat_projs length {len(self.feat_projs)}"
        )

        # 1) 投影并融合 DA3 特征 (低分辨率 h,w)
        proj_feats = []
        for feat, proj in zip(features, self.feat_projs):
            proj_feats.append(proj(feat))
        fused_feat = torch.stack(proj_feats, dim=0).mean(dim=0)  # [N, c_embed, h, w]

        # 2) 上采样到 H,W 后与 conf/depth 融合
        fused_feat = F.interpolate(fused_feat, size=(H, W), mode="bilinear", align_corners=False)
        fused_feat = torch.cat([fused_feat, conf, depth], dim=1)  # [N, c_embed+2, H, W]
        fused_feat = self.fuse_layer(fused_feat)  # [N, c_embed, H, W]

        # 3) 输出 logits
        logits = self.output_layer(fused_feat)  # [N, num_classes, H, W]
        return logits


class DynaDA3(nn.Module):
    """
    DynaDA3
    """
    def __init__(
        self,
        model_name: str = 'vitl', # 'vitl' or 'vitg'
        uncertainty_head_ckpt_path: str = None, # 训练好的 uncertainty head 权重路径; 注意只有在训练时才可以不输入该参数
    ):
        super().__init__()

        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"model_name must be one of {list(MODEL_CONFIGS.keys())}")
        
        config = MODEL_CONFIGS[model_name]
        ckpt_path = config['ckpt_path']
        channels = config['channels']
        self.export_feat_layers = list(config['feat_layers'])

        print(f"Loading DA3 ({model_name}) from local path: {ckpt_path}...")
        self.da3 = DepthAnything3.from_pretrained(ckpt_path)

        # 冻结 DA3
        for p in self.da3.parameters():
            p.requires_grad = False
        self.da3.eval()

        # 初始化 UncertaintyDPT 
        self.uncertainty_head = UncertaintyDPT(
            c_in=channels,
            feat_layers=range(len(self.export_feat_layers)),
        )

        if uncertainty_head_ckpt_path:
            print(f"Loading uncertainty head from {uncertainty_head_ckpt_path}...")
            state_dict = torch.load(uncertainty_head_ckpt_path, map_location='cpu')
            missing_keys, _ = self.uncertainty_head.load_state_dict(state_dict, strict=True)
            if len(missing_keys) > 0:
                raise ValueError(f"Failed to load uncertainty_head weights. Missing: {missing_keys}")

        
    @staticmethod
    def _nhwc_to_nchw(feat: np.ndarray, device: torch.device) -> torch.Tensor:
        """
          feat shape: [N, h, w, C]  
        Returns:
          torch.Tensor [N, C, h, w] on device
        """
        assert isinstance(feat, np.ndarray), f"feat must be numpy, got {type(feat)}"
        assert feat.ndim == 4, f"feat must be [N,h,w,C], got {feat.shape}"
        
        t = torch.from_numpy(feat).to(device=device, dtype=torch.float32)     # [N,h,w,C]
        t = t.permute(0, 3, 1, 2).contiguous()                               # [N,C,h,w]
        return t

    def _run_uncertainty_head(self, prediction, device: torch.device):
        """
        从 prediction.aux 取特征，外加 conf/depth 跑 uncertainty_head
        并写回 prediction.uncertainty_seg_logits/uncertainty_seg_mask
        """
        # 用 processed_images 的分辨率作为最终输出 H,W
        H, W = prediction.processed_images.shape[1], prediction.processed_images.shape[2]

        # 按 export_feat_layers 的顺序组织 feature list
        feats = []
        for layer in self.export_feat_layers:
            k = f"feat_layer_{layer}"
            feats.append(self._nhwc_to_nchw(prediction.aux[k], device))

        # 处理 conf, prediction.conf [N, H, W]
        assert hasattr(prediction, "conf"), "prediction.conf missing"
        conf_np = prediction.conf # [N, H, W]
        conf_tensor = torch.from_numpy(conf_np).to(device=device, dtype=torch.float32).unsqueeze(1) # [N, H, W] -> [N, 1, H, W]

        # depth: prediction.depth [N, H, W]
        depth_np = prediction.depth
        depth_tensor = torch.from_numpy(depth_np).to(device=device, dtype=torch.float32).unsqueeze(1) # [N, H, W] -> [N, 1, H, W]

        logits = self.uncertainty_head(  # [N,K,H,W]
            feats, H, W, conf=conf_tensor, depth=depth_tensor
        )
        mask = torch.argmax(logits, dim=1) # [N,H,W]

        prediction.uncertainty_seg_logits = logits
        prediction.uncertainty_seg_mask = mask

    @torch.no_grad()
    def inference(self, image, **kwargs):
        """
        在推理过程(需要所有其他头的输出)使用inferrence;
        由于只训练 uncertainty_head,所以这里inferrence需要返回uncertainty_head的输出
        Args:
            images: [B, N, 3, H, W] Tensor, normalized (ImageNet mean/std)
        Returns:
            prediction: DepthAnything3.Prediction  包含 uncertainty_seg_logits / uncertainty_seg_mask
        """
        device = next(self.uncertainty_head.parameters()).device

        output = self.da3.inference(
            image=image,
            export_feat_layers=self.export_feat_layers,
        )

        self._run_uncertainty_head(output, device)

        return output

    def forward(self, image):
        """
        在训练过程(只训练特定任务头)中使用forward, 由于训练 uncertainty_head,所以这里forward需要返回uncertainty_head的logits; 
        在推理过程(需要所有其他头的输出)使用inferrence;
        Args:
            images: [B, N, 3, H, W] Tensor, normalized (ImageNet mean/std)
        Returns:
            logits: [B, num_classes, H, W]
                num_classes: uncertainty segmentation classes = 2 (moving / static)

        """
        # [B, N, 3, H, W]
        if image.ndim != 5: 
            logger.error(f"Input image must be 5-D Tensor [B, N, 3, H, W], got {image.shape}")
            assert False

        B, N, _, H, W = image.shape

        # 冻结 DA3 并提取特征
        with torch.no_grad():
            out = self.da3.model(
                image, 
                export_feat_layers=self.export_feat_layers,
            )

        # feat_layers特征处理
        feats = [] 
        for layer in self.export_feat_layers:
            feat = out['aux'][f"feat_layer_{layer}"]  # 原始 Shape: [B, N, h, w, C]
            _, _, h, w, C = feat.shape
            feat = feat.view(B * N, h, w, C) # 合并B,N: [B, N, h, w, C] -> [B*N, h, w, C]
            feat = feat.permute(0, 3, 1, 2).contiguous()# 调整顺序 channel first: [B*N, h, w, C] -> [B*N, C, h, w] 
            feats.append(feat)

        # conf 处理 (注意 da3的forward不输出conf, 只输出depth_conf)
        conf = out['depth_conf'] # [B, N, H, W] 
        conf = conf.view(B * N, 1, H, W) # [B, N, H, W] -> [B*N, 1, H, W]

        # depth 处理 (DA3 main head 输出 depth)
        depth = out['depth'] # [B, N, H, W]
        depth = depth.view(B * N, 1, H, W) # [B, N, H, W] -> [B*N, 1, H, W]
        

        # feats: List[[B*N, C, h, w]];  conf: [B*N, 1, H, W] 
        logits = self.uncertainty_head(
            feats, H, W, conf=conf, depth=depth
        ) #  logits: [B*N, 2, H, W]
        
        return logits
