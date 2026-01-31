from pyexpat import features
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.logger import logger

DA3_VITG_CHANNELS = 1536 
DA3_VITL_CHANNELS = 1024
DA3_VITG_FEAT_LAYERS=(21, 27, 33, 39)
DA3_VITL_FEAT_LAYERS=(11, 15, 19, 23)
DA3_VITG_CKPT_PATH = "../checkpoint/DA3-GIANT-1.1"
DA3_VITL_CKPT_PATH = "../checkpoint/DA3-LARGE-1.1"
DPT_EMBED_DIM = 256

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

#######################################################
# UncertaintyDPT V3.1
# https://gemini.google.com/share/af6a5517a0e8
#######################################################
class UncertaintyDPT(nn.Module):
    def __init__(self, c_in, feat_layers, c_embed=256):
        super().__init__()
        self.feat_layers = list(feat_layers)
        
        # 1x1 卷积投影
        self.projects = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c_in, c_embed, kernel_size=1),
                nn.ReLU(inplace=True),
            ) for _ in range(len(self.feat_layers))
        ])

        # 几何分支改在低分辨率运行
        self.geo_head = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, c_embed, kernel_size=1),
        )

        self.fusion_layer = nn.Sequential(
            nn.Conv2d(c_embed * 2, c_embed, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_embed),
            nn.ReLU(inplace=True),
        )

        self.output_layer = nn.Sequential(
            nn.Conv2d(c_embed, c_embed // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_embed // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_embed // 2, 2, kernel_size=1),
        )

    def _normalize(self, x, stats_size=128):
        """
        加速版归一化：在低分辨率下计算分位数
        """
        N, C, H, W = x.shape
        # 降采样统计
        x_small = F.interpolate(x, size=(stats_size, stats_size), mode='area')
        x_flat = x_small.view(N, -1)
        
        q_low = torch.quantile(x_flat, 0.05, dim=1, keepdim=True).view(N, 1, 1, 1)
        q_high = torch.quantile(x_flat, 0.95, dim=1, keepdim=True).view(N, 1, 1, 1)
        
        denom = q_high - q_low
        return (x - q_low) / denom.clamp(min=1e-6)

    def _get_depth_grad(self, x):
        """
        使用卷积代替切片计算梯度，减少访存
        """
        # 利用 F.grad 计算简单的 x,y 差分
        kernel_x = torch.tensor([[[[0, 0, 0], [-1, 1, 0], [0, 0, 0]]]], device=x.device, dtype=x.dtype)
        kernel_y = torch.tensor([[[[0, -1, 0], [0, 1, 0], [0, 0, 0]]]], device=x.device, dtype=x.dtype)
        
        grad_x = F.conv2d(x, kernel_x, padding=1)
        grad_y = F.conv2d(x, kernel_y, padding=1)
        return (grad_x.abs() + grad_y.abs())

    def forward(self, feats, H, W, conf, depth):
        # 1. 在原图尺度处理几何先验，但立即降采样到特征图尺度
        # 获取特征图的目标分辨率 (h, w)
        target_h, target_w = feats[0].shape[-2:]
        
        conf_norm = self._normalize(conf)
        depth_norm = self._normalize(depth)
        
        # 构造先验
        u_prior = (1.0 - conf_norm) * (1.0 - depth_norm)
        d_grad = self._get_depth_grad(depth_norm)
        
        geo_raw = torch.cat([
            depth_norm, conf_norm, d_grad, u_prior, (1.0 - conf_norm)
        ], dim=1) # [N, 5, H, W]
        
        # 立即降采样到特征图尺度进行后续卷积
        geo_small = F.interpolate(geo_raw, size=(target_h, target_w), mode='area')
        geo_feat = self.geo_head(geo_small)

        # 2. 图像特征投影 (已经在 target_h, target_w)
        img_feat_sum = sum(proj(f) for proj, f in zip(self.projects, feats))

        # 3. 低分辨率融合
        fused = torch.cat([geo_feat, img_feat_sum], dim=1)
        fused = self.fusion_layer(fused)

        # 4. 在低分辨率完成输出转换，最后只做一次大尺寸上采样
        logits_small = self.output_layer(fused)
        logits = F.interpolate(logits_small, size=(H, W), mode="bilinear", align_corners=False)
            
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
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()

        device = next(self.uncertainty_head.parameters()).device

        output = self.da3.inference(
            image=image,
            export_feat_layers=self.export_feat_layers,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.time()

        self._run_uncertainty_head(output, device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t2 = time.time()

        logger.info(f"DynaDA3 Inference | Total: {t2-t0:.3f}s | Backbone: {t1-t0:.3f}s | Uncertainty Head: {t2-t1:.3f}s")

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
