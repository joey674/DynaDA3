import torch
import torch.nn as nn
# 假设 api.py 定义的类叫 DepthAnything3，你需要根据实际 import
from depth_anything_3.api import DepthAnything3 

class SegDA3(nn.Module):
    def __init__(self, repo_id="LiheYoung/depth-anything-3-giant", num_classes=2):
        super().__init__()
        
        # 1. 直接使用官方 API 加载“满血”模型
        # 这一步会自动处理 config 加载、权重下载和填充
        print(f"Loading DA3 from {repo_id}...")
        self.da3_full = DepthAnything3.from_pretrained(repo_id)
        
        # 2. 找到 Backbone
        # 根据 yaml，Backbone 被命名为 'net'。
        # 在 PyTorch Module 中，通常可以通过属性直接访问
        # 如果 api.py 也是包了一层，可能需要 self.da3_full.model.net
        # 我们可以通过 print(self.da3_full) 来确认路径，这里假设是直接属性
        self.backbone = self.da3_full.net 
        
        # 3. 冻结整个 DA3 (包括 backbone 和 原来的 head)
        for param in self.da3_full.parameters():
            param.requires_grad = False
        self.da3_full.eval()
        
        # 4. 定义你的分割头 (使用 DPT 结构)
        # Giant 版本的维度是 1536，Base 是 768
        embed_dim = 768 
        self.seg_head = SimpleDPTHead(
            in_channels=embed_dim, 
            embed_dim=256, 
            num_classes=num_classes
        )
        
        # 5. 确保分割头是可训练的
        for param in self.seg_head.parameters():
            param.requires_grad = True

    def forward(self, x):
        # x: [B, N, 3, H, W]
        B, N, C, H, W = x.shape
        
        # === 核心：借用 DA3 的 Backbone ===
        with torch.no_grad():
            # 这里直接调用 backbone 的 forward
            # 注意：DINOv2 / DA3 Backbone 通常需要 input 也是 (B, N, ...)
            # 返回的 feats 通常是一个 list
            feats = self.backbone(x) 
            
            # 调试技巧：如果你不知道 feats 是啥，可以在这里 print(len(feats), feats[0].shape)
            
        # === 后面流程不变 ===
        
        # 1. 维度调整 (B, N) -> (B*N)
        flat_feats = []
        for f in feats:
             # 假设 f 是 [B, N, C, Hp, Wp]
            if len(f.shape) == 5:
                _B, _N, _C, _Hp, _Wp = f.shape
                f = f.reshape(_B * _N, _C, _Hp, _Wp)
            flat_feats.append(f)
            
        # 2. 分割头预测
        logits = self.seg_head(flat_feats, H, W)
        
        # 3. 恢复维度
        logits = logits.view(B, N, -1, H, W)
        
        return logits