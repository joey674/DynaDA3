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
from torch.cuda.amp import autocast, GradScaler

from SegDA3_model import SegDA3

# ================= 配置区域 =================
CONFIG = {
    "train_img_dir": "/home/zhouyi/repo/model_DepthAnythingV3/outputs/dancer/images",
    "train_mask_dir": "/home/zhouyi/repo/model_DepthAnythingV3/outputs/dancer/masks",
    "save_dir": "/home/zhouyi/repo/model_DepthAnythingV3/checkpoints/SegDA3",
    
    "num_classes": 2,
    "seq_len": 3,               # 每个训练单位包含的图片张数 (N)
    "batch_size": 2,            # B (注意：显存占用 = B * N)
    "lr": 1e-4,
    "epochs": 5,
    "input_size": (518, 518),
    "num_workers": 4,
}
# ===========================================

class MotionSegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, input_size=(518, 518), seq_len=1):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*")))
        self.input_size = input_size
        self.seq_len = seq_len
        
        assert len(self.img_paths) == len(self.mask_paths)
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_paths) - self.seq_len + 1

    def __getitem__(self, idx):
        clip_imgs, clip_masks = [], []
        # 读取连续的 seq_len 张图
        for i in range(idx, idx + self.seq_len):
            img = Image.open(self.img_paths[i]).convert('RGB').resize(self.input_size, Image.BILINEAR)
            mask = Image.open(self.mask_paths[i]).resize(self.input_size, Image.NEAREST)
            
            mask_arr = np.array(mask)
            if mask_arr.max() > 1: mask_arr = (mask_arr > 128).astype(int)
            
            clip_imgs.append(self.img_transform(img))
            clip_masks.append(torch.from_numpy(mask_arr).long())

        # 返回 [N, 3, H, W] 和 [N, H, W]
        return torch.stack(clip_imgs), torch.stack(clip_masks)

def calculate_iou(pred, label, num_classes):
    pred = torch.argmax(pred, dim=1)
    iou_list = []
    for cls in range(num_classes):
        intersection = ((pred == cls) & (label == cls)).sum().item()
        union = ((pred == cls) | (label == cls)).sum().item()
        if union == 0: iou_list.append(float('nan'))
        else: iou_list.append(float(intersection) / float(union))
    return np.nanmean(iou_list)

def train():
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MotionSegDataset(CONFIG["train_img_dir"], CONFIG["train_mask_dir"], 
                               CONFIG["input_size"], CONFIG["seq_len"])
    dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True, 
                            num_workers=CONFIG["num_workers"], pin_memory=True)

    model = SegDA3(num_classes=CONFIG["num_classes"]).to(device)
    model.train()

    optimizer = optim.AdamW(model.seg_head.parameters(), lr=CONFIG["lr"], weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    for epoch in range(CONFIG["epochs"]):
        epoch_loss, epoch_iou = 0, 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")

        for imgs, masks in pbar:
            # imgs: [B, N, 3, H, W], masks: [B, N, H, W]
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()

            with autocast():
                # model forward 返回 [B*N, 2, H, W]
                logits = model(imgs) 
                # 将 masks 也展平为 [B*N, H, W] 以对齐
                masks_flatten = masks.view(-1, masks.shape[-2], masks.shape[-1])
                loss = criterion(logits, masks_flatten)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            with torch.no_grad():
                batch_iou = calculate_iou(logits, masks_flatten, CONFIG["num_classes"])
                epoch_iou += batch_iou
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "mIoU": f"{batch_iou:.4f}"})

        print(f"Epoch {epoch+1} Avg Loss: {epoch_loss/len(dataloader):.4f}, mIoU: {epoch_iou/len(dataloader):.4f}")
        torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], "model.pth"))

if __name__ == "__main__":
    train()