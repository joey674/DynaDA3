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
DA3_VITG_CKPT_PATH = "/home/zhouyi/repo/checkpoint/DA3-GIANT-1.1"
DA3_VITL_CKPT_PATH = "/home/zhouyi/repo/checkpoint/DA3-LARGE-1.1"
MOTIONDPT_EMBED_DIM = 256

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

class MotionDPT(nn.Module):
    """
    Motion DPT 
    Args:
        B: Batch Size=1
        N: Frame Sequence Length
        H, W: 原始输入图像分辨率 (例如 518 * 518)
        h, w: 特征图分辨率; 对于 ViT 架构，通常 patch size 为 14, 所以 h = H/14, w = W/14 (518/14=37)
        c_in: 输入通道数 (DA3_CHANNELS = 1536)
        c_embed: 嵌入维度 (MOTIONDPT_EMBED_DIM)
        feat_layers: 从 DA3 提取的特征层索引列表 (例如 [3, 9, 15, 21, 27, 33, 39])
    """
    # [修改默认参数: 去除默认的 DA3_VITL_CHANNELS, 避免混淆, 由外部传入]
    def __init__(self, c_in, c_embed=MOTIONDPT_EMBED_DIM, feat_layers=()):
        super().__init__()
        self.feat_layers = list(feat_layers)# 选择哪些 Transformer 层的特征用于分割头

        self.projects_layer = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(c_in, c_embed, kernel_size=1),# 1x1 卷积: 用于将原始高维特征(如 1024 维)降维到统一的 embed_dim（如 256 维），减少计算量。
                    nn.ReLU(inplace=True),
                )
                for _ in range(len(self.feat_layers))# 这个投影层有并行的N个卷积模块, 每个对应处理不同特征层出来的特征
            ]
        )

        num_classes = 2  # 运动分割类别数：移动 / 静止
        self.output_layer = nn.Sequential(
            nn.Conv2d(c_embed, c_embed // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_embed // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_embed // 2, num_classes, kernel_size=1),
        )

    def forward(self, features: list[torch.Tensor], H: int, W: int) -> torch.Tensor:
        # features:  [N, c_in, h, w]
        selected_feat_layers = [features[i] for i in self.feat_layers]

        target_h, target_w = selected_feat_layers[0].shape[-2:]
        proj = []
        for i, feat in enumerate(selected_feat_layers):
            # 特征投影 [N, c_in, h, w] => [N, c_embed, h, w]
            # 使用projects_layer对每个选定的特征块进行投影
            x = self.projects_layer[i](feat)

            # 对齐与插值 [N, c_embed, h, w] =>[N, c_embed, h, w]
            # F.interpolate作用: 确保所有层特征分辨率一致（以第一层特征的 h, w 为准）
            # 在标准 ViT 中通常尺寸一致，但此步骤保证了鲁棒性
            if x.shape[-2:] != (target_h, target_w):
                x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
            proj.append(x)

        # 特征融合 [N, c_embed, h, w] =>[N, c_embed, h, w]
        # 这是一种 Simple Summation 的融合方式 将浅层（纹理细节丰富）和深层（语义信息丰富）的特征直接叠加
        fused = 0
        for x in proj:
            fused = fused + x

        # 输出层 [N, c_embed, h, w] => [N, c_embed/2, h, w]
        # 融合后的特征通过一个小型卷积网络进行最后的预测
        # 3x3 卷积 + BN + ReLU: 用于平滑特征，消除由于插值（Interpolation）产生的伪影，并进一步提取局部特征
        # 1x1 卷积: 最终投影到 2 通道，得到每个类别的置信度分数（Logits）
        logits = self.output_layer(fused) 
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False) 
        return logits


class SegDA3(nn.Module):
    """
    SegDA3
    """
    def __init__(
        self,
        model_name: str = 'vitl', # 'vitl' or 'vitg'
        motion_head_ckpt_path: str = None, # 训练好的 motion head 权重路径; 注意只有在训练时才可以不输入该参数
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

        self.motion_head = MotionDPT(
            c_in=channels,
            feat_layers=range(len(self.export_feat_layers)),
        )

        if motion_head_ckpt_path:
            print(f"Loading motion head from {motion_head_ckpt_path}...")
            # 加载权重到内存
            state_dict = torch.load(motion_head_ckpt_path, map_location='cpu')
            missing_keys, _ = self.motion_head.load_state_dict(state_dict, strict=True)
            if len(missing_keys) > 0:
                raise ValueError(f"Failed to load motion_head weights. Missing: {missing_keys}")

        
    @staticmethod
    def _aux_feat_to_nchw(feat: np.ndarray, device: torch.device) -> torch.Tensor:
        """
          feat shape: [N, h, w, C]  (例如 5,28,36,1024)
        Returns:
          torch.Tensor [N, C, h, w] on device
        """
        assert isinstance(feat, np.ndarray), f"aux feat must be numpy, got {type(feat)}"
        assert feat.ndim == 4, f"aux feat must be [N,h,w,C], got {feat.shape}"
        t = torch.from_numpy(feat).to(device=device, dtype=torch.float32)     # [N,h,w,C]
        t = t.permute(0, 3, 1, 2).contiguous()                               # [N,C,h,w]
        return t

    def _run_motion_head(self, prediction, device: torch.device):
        """
        从 prediction.aux 取特征，跑 motion_head 写回 prediction.motion_seg_logits/motion_seg_mask
        """
        assert hasattr(prediction, "aux") and isinstance(prediction.aux, dict), "prediction.aux missing"

        # 用 processed_images 的分辨率作为最终输出 H,W
        assert prediction.processed_images is not None
        H, W = prediction.processed_images.shape[1], prediction.processed_images.shape[2]

        # 按 export_feat_layers 的顺序组织 feature list
        feats = []
        for layer in self.export_feat_layers:
            k = f"feat_layer_{layer}"
            assert k in prediction.aux, f"Missing {k} in prediction.aux"
            feats.append(self._aux_feat_to_nchw(prediction.aux[k], device))

        logits = self.motion_head(feats, H, W)                # [N,K,H,W]
        mask = torch.argmax(logits, dim=1)                 # [N,H,W]

        prediction.motion_seg_logits = logits
        prediction.motion_seg_mask = mask

    @torch.no_grad()
    def inference(self, image, **kwargs):
        """
        在推理过程(需要所有其他头的输出)使用inferrence;
        由于只训练 motion_head,所以这里inferrence需要返回motion_head的输出
        Args:
            images: [B, N, 3, H, W] Tensor, normalized (ImageNet mean/std)
        Returns:
            prediction: DepthAnything3.Prediction  包含 motion_seg_logits / motion_seg_mask
        """
        device = next(self.motion_head.parameters()).device

        output = self.da3.inference(
            image=image,
            export_feat_layers=self.export_feat_layers,
        )

        self._run_motion_head(output, device)

        return output

    def forward(self, image):
        """
        在训练过程(只训练特定任务头)中使用forward, 由于训练 motion_head,所以这里forward需要返回motion_head的logits; 
        在推理过程(需要所有其他头的输出)使用inferrence;
        Args:
            images: [B, N, 3, H, W] Tensor, normalized (ImageNet mean/std)
        Returns:
            logits: [B, num_classes, H, W]
                num_classes: motion segmentation classes = 2 (moving / static)

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
                export_feat_layers=self.export_feat_layers
            )

        # 特征处理
        feats = [] 
        for layer in self.export_feat_layers:
            feat = out['aux'][f"feat_layer_{layer}"]  # 原始 Shape: [B, N, h, w, Feat_Dim]
            
            # 不管 N 是多少，直接把 B 和 N 合并; 因为分割头（DPTHead）是基于 2D 卷积的，它不认识序列维度 N
            _, _, h, w, Feat_Dim = feat.shape
            
            # [B, N, h, w, Feat_Dim] -> [B*N, h, w, Feat_Dim]
            feat = feat.view(B * N, h, w, Feat_Dim)

            # [B*N, h, w, Feat_Dim] -> [B*N, Feat_Dim, h, w] (Channel-First 适配卷积层)
            feat = feat.permute(0, 3, 1, 2).contiguous()
            feats.append(feat)

        # 运行分割头
        # 此时 feats 里的每个 Tensor 都是 [Batch_total, Feat_Dim, h, w]
        logits = self.motion_head(feats, H, W) #  [B*N, 2, H, W]
        
        return logits