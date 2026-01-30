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

class GatedRefinementUnit(nn.Module):
    """
    门控精炼单元 (Gated Refinement Unit)
    用于融合特征与置信度图, 通过门控机制增强重要特征
    """
    def __init__(self, in_channels):
        super().__init__()
        
        conf_patch_size=14
        self.conf_channels = conf_patch_size * conf_patch_size

        # [特征变换分支]
        self.feature_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        # 门控生成分支: 输入 (Feature + Conf)
        # 修改: 输入通道数变为 in_channels + self.conf_channels
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_channels + self.conf_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid() # 输出 0~1 门控信号
        )
        # 最终融合
        self.final_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, feat, conf):
        """
        feat: [B, c_embed, h, w], 
        conf: [B, 1, H, W]
        """
        
        #  转换conf格式使其与feat对应: [B, 1, H, W] => [B, 196, h, w] 
        B, C, h, w = feat.shape
        _, _, H, W = conf.shape
        patch_h, patch_w = H // h, W // w
        conf_reshaped = conf.view(B, 1, h, patch_h, w, patch_w)
        conf_permuted = conf_reshaped.permute(0, 1, 3, 5, 2, 4)
        conf = conf_permuted.reshape(B, -1, h, w)

        # 将feat特征解耦
        feat_transformed = self.feature_conv(feat)
        
        # 拼接feat与conf生成新向量, 输入gate层 同时学feat和conf两种语义; gate最终会输出0-1, 融合两种语义进行统一输出
        gate_input = torch.cat([feat, conf], dim=1)
        gate = self.gate_conv(gate_input)
        
        # 将feat与门控相乘, 使得门控生效(gate=0由于乘法的缘故就会直接抑制; gate=1就不影响)
        gated_feat = feat_transformed * gate

        # 残差链接 (提升稳定性)
        output = feat + self.final_conv(gated_feat)
        return output

class GateDPT(nn.Module):
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
    def __init__(self, c_in, feat_layers, c_embed=MOTIONDPT_EMBED_DIM):
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

        # 初始化门控模块
        self.gated_refine = GatedRefinementUnit(c_embed)

        num_classes = 2  # 运动分割类别数：移动 / 静止
        self.output_layer = nn.Sequential(
            nn.Conv2d(c_embed, c_embed // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_embed // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_embed // 2, num_classes, kernel_size=1),
        )

    def forward(self, features: list[torch.Tensor], H: int, W: int, conf: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: List[[N, c_in, h, w]]
            H, W: 原始输入图像分辨率
            conf: 置信度图 Tensor, shape: [N, 1, H, W]
        Returns:
            logits: 运动分割 logits, shape: [N, num_classes, H, W]
        """
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

        # 门控 fused: [N, c_embed, h, w] conf: [N, c_embed, H, W] (conf将在内部展开)
        fused = self.gated_refine(fused, conf)

        # 输出层 [N, c_embed, h, w] => [N, c_embed/2, h, w]
        # 融合后的特征通过一个小型卷积网络进行最后的预测
        # 3x3 卷积 + BN + ReLU: 用于平滑特征，消除由于插值（Interpolation）产生的伪影，并进一步提取局部特征
        # 1x1 卷积: 最终投影到 2 通道，得到每个类别的置信度分数（Logits）
        logits = self.output_layer(fused) 
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False) 
        return logits


class DynaDA3(nn.Module):
    """
    DynaDA3
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

        # 初始化 GateDPT 
        self.motion_head = GateDPT(
            c_in=channels,
            feat_layers=range(len(self.export_feat_layers)),
        )

        if motion_head_ckpt_path:
            print(f"Loading motion head from {motion_head_ckpt_path}...")
            state_dict = torch.load(motion_head_ckpt_path, map_location='cpu')
            missing_keys, _ = self.motion_head.load_state_dict(state_dict, strict=True)
            if len(missing_keys) > 0:
                raise ValueError(f"Failed to load motion_head weights. Missing: {missing_keys}")

        
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

    def _run_motion_head(self, prediction, device: torch.device):
        """
        从 prediction.aux 取特征，跑 motion_head 写回 prediction.motion_seg_logits/motion_seg_mask
        """
        # 用 processed_images 的分辨率作为最终输出 H,W
        H, W = prediction.processed_images.shape[1], prediction.processed_images.shape[2]

        # 按 export_feat_layers 的顺序组织 feature list
        feats = []
        for layer in self.export_feat_layers:
            k = f"feat_layer_{layer}"
            feats.append(self._nhwc_to_nchw(prediction.aux[k], device))

        # 处理 conf, prediction.conf [N, H, W]]
        assert hasattr(prediction, "conf"), "prediction.conf missing"
        conf_np = prediction.conf # [N, H, W]
        conf_tensor = torch.from_numpy(conf_np).to(device=device, dtype=torch.float32).unsqueeze(1) # [N, H, W] -> [N, 1, H, W]

        logits = self.motion_head(feats, H, W, conf=conf_tensor)   # [N,K,H,W]
        mask = torch.argmax(logits, dim=1) # [N,H,W]

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
        

        # feats: List[[B*N, C, h, w]];  conf: [B*N, 1, H, W] 
        logits = self.motion_head(feats, H, W, conf=conf) #  logits: [B*N, 2, H, W]
        
        return logits