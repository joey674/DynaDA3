import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler  # 混合精度训练，节省显存

# 导入你的模型
from SegDA3_model import SegDA3

# ================= 配置区域 =================
CONFIG = {
    # 路径配置
    "train_img_dir": "/home/zhouyi/repo/model_DepthAnythingV3/outputs/0001/images",
    "train_mask_dir": "/home/zhouyi/repo/model_DepthAnythingV3/outputs/0001/masks",
    "save_dir": "/home/zhouyi/repo/model_DepthAnythingV3/checkpoints/SegDA3",
    
    # 训练超参
    "num_classes": 2,          # 0:Static, 1:Motion
    "batch_size": 4,           # 根据显存调整，DA3较大，显存小建议设为 1 或 2
    "lr": 1e-4,                # 学习率
    "epochs": 5,
    "input_size": (518, 518),  # 必须是14的倍数，DA3标准输入
    "num_workers": 4,
}
# ===========================================

class MotionSegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, input_size=(518, 518)):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*")))
        self.input_size = input_size
        
        # 严格检查
        assert len(self.img_paths) == len(self.mask_paths), "图片和Mask数量不一致！"
        if len(self.img_paths) == 0:
            raise RuntimeError(f"未在 {img_dir} 找到数据")

        # 图像预处理：转Tensor + ImageNet归一化
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 1. 读取
        img = Image.open(self.img_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]) # 保持原始模式，通常是 'L' 或 '1'

        # 2. Resize (核心关键点)
        # 图片用双线性插值
        img = img.resize(self.input_size, resample=Image.BILINEAR)
        # Mask必须用最近邻插值，防止出现小数类别
        mask = mask.resize(self.input_size, resample=Image.NEAREST)

        # 3. 转换
        img_tensor = self.img_transform(img)
        
        # 处理 Mask 值域: [H, W]
        mask_array = np.array(mask)
        # 如果 Mask 是 0/255 的格式，归一化为 0/1
        if mask_array.max() > 1:
            mask_array = (mask_array > 128).astype(int)
        
        mask_tensor = torch.from_numpy(mask_array).long()

        return img_tensor, mask_tensor

def calculate_iou(pred, label, num_classes):
    """简单的 IoU 计算"""
    pred = torch.argmax(pred, dim=1)
    iou_list = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (label == cls)
        intersection = (pred_inds[target_inds]).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        if union == 0:
            iou_list.append(float('nan')) # 该类别未出现
        else:
            iou_list.append(float(intersection) / float(max(union, 1)))
    return np.nanmean(iou_list)

def train():
    # 0. 准备环境
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 数据集
    #  Image Tensor [B, 3, 518, 518] 
    #  Mask Tensor [B, 518, 518]
    dataset = MotionSegDataset(CONFIG["train_img_dir"], CONFIG["train_mask_dir"], CONFIG["input_size"])
    dataloader = DataLoader(
        dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=True, 
        num_workers=CONFIG["num_workers"],
        pin_memory=True
    )


    # 2. 模型
    print("Loading SegDA3 Model...")
    model = SegDA3(num_classes=CONFIG["num_classes"]).to(device)
    model.train() # 开启 BatchNorm 等

    # 3. 优化器 (只优化 head，因为 da3 被冻结了)
    optimizer = optim.AdamW(model.seg_head.parameters(), lr=CONFIG["lr"], weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # 混合精度 Scaler
    scaler = GradScaler()

    # 4. 训练循环
    print("Start Training...")
    for epoch in range(CONFIG["epochs"]):
        epoch_loss = 0
        epoch_iou = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")

        for imgs, masks in pbar:
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            # 混合精度前向
            with autocast():
                logits = model(imgs) 
                loss = criterion(logits, masks)

            # 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 记录指标
            loss_val = loss.item()
            epoch_loss += loss_val
            
            # 计算简单的 batch IoU 用于显示
            with torch.no_grad():
                batch_iou = calculate_iou(logits, masks, CONFIG["num_classes"])
                epoch_iou += batch_iou

            pbar.set_postfix({"Loss": f"{loss_val:.4f}", "mIoU": f"{batch_iou:.4f}"})

        # Epoch 结束统计
        avg_loss = epoch_loss / len(dataloader)
        avg_iou = epoch_iou / len(dataloader)
        print(f"Epoch {epoch+1} Done. Avg Loss: {avg_loss:.4f}, Avg mIoU: {avg_iou:.4f}")

        # 保存模型
        save_path = os.path.join(CONFIG["save_dir"], "model.pth")
        torch.save(model.state_dict(), save_path)

    print(f"Training finished. Model saved to {CONFIG['save_dir']}")

if __name__ == "__main__":
    train()