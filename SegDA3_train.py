import os
import glob
import random
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

# ================= config =================
CONFIG = {
    # Data Configuration
    "video_dirs": [
        "/home/zhouyi/repo/dataset_segda3_train/dancer",
        # "/home/zhouyi/repo/dataset_segda3_train/wildgs_ANYmal1",
        # "/home/zhouyi/repo/dataset_segda3_train/wildgs_racket1",
        # "/home/zhouyi/repo/dataset_segda3_train/wildgs_ANYmal2",
        # "/home/zhouyi/repo/dataset_segda3_train/wildgs_racket2",
        # "/home/zhouyi/repo/dataset_segda3_train/wildgs_ANYmal3",
        # "/home/zhouyi/repo/dataset_segda3_train/wildgs_racket3",
        # "/home/zhouyi/repo/dataset_segda3_train/wildgs_racket4",
    ],
    "save_dir": "/home/zhouyi/repo/checkpoint/SegDA3",
    "seq_range": (2, 20),
    # Training Hyperparameters
    "learning_rate": 1e-4, # 决定了模型更新权重的步长,1e-4 是 Transformer 类模型常用的经验值
    "epochs": 1,
    "batch_size": 1, # 每次迭代epoch训练中使用的样本数量, 限定就为1, 因为每个样本已经包含了一个视频序列
    # System Configuration
    "num_workers": 4, # DataLoader 的子进程数量,决定了有多少个 CPU 子进程在后台同时帮你读取和预处理图片
}

# ================= dataset =================
class MultiVideoDataset(Dataset):
    def __init__(self, video_dirs, input_size=(518, 518), seq_range=(2, 5)):
        self.input_size = input_size
        self.seq_min, self.seq_max = seq_range
        self.samples = []

        for v_dir in video_dirs:
            img_dir = os.path.join(v_dir, "images")
            mask_dir = os.path.join(v_dir, "masks")
            
            if not (os.path.exists(img_dir) and os.path.exists(mask_dir)):
                print(f"Skipping: {v_dir} (missing folders)")
                continue

            v_imgs = sorted(glob.glob(os.path.join(img_dir, "*")))
            v_masks = sorted(glob.glob(os.path.join(mask_dir, "*")))

            if len(v_imgs) < self.seq_max:
                continue

            for i in range(len(v_imgs) - self.seq_max + 1):
                self.samples.append({
                    "imgs": v_imgs,
                    "masks": v_masks,
                    "start": i
                })

        print(f"Dataset initialized: {len(self.samples)} samples.")

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        start = s["start"]
        n = random.randint(self.seq_min, self.seq_max)
        
        clip_imgs, clip_masks = [], []
        for i in range(start, start + n):
            img = Image.open(s["imgs"][i]).convert('RGB').resize(self.input_size, Image.BILINEAR)
            mask = Image.open(s["masks"][i]).resize(self.input_size, Image.NEAREST)
            
            mask_arr = np.array(mask)
            if mask_arr.max() > 1: mask_arr = (mask_arr > 128).astype(int)
            
            clip_imgs.append(self.img_transform(img))
            clip_masks.append(torch.from_numpy(mask_arr).long())

        # Dataset 返回的是单个样本，不需要自己伪造 Batch 维度
        # imgs_tensor shape: [N, 3, H, W]
        # masks_tensor shape: [N, H, W]
        imgs_tensor = torch.stack(clip_imgs)   
        masks_tensor = torch.stack(clip_masks) 
        
        return imgs_tensor, masks_tensor


# ================= train =================
def train():
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MultiVideoDataset(video_dirs=CONFIG["video_dirs"],seq_range= CONFIG["seq_range"])
    
    # DataLoader 会自动把多个样本堆叠成 Batch [1, N, 3, H, W]
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True)

    model = SegDA3().to(device)
    model.train()

    optimizer = optim.AdamW(model.motion_head.parameters(), lr=CONFIG["learning_rate"], weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    for epoch in range(CONFIG["epochs"]):
        epoch_loss = 0 
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")

        for imgs, masks in pbar:
            # DataLoader 出来的 imgs 已经是 [B, N, 3, H, W]，此处 B=1
            imgs = imgs.to(device)   
            masks = masks.to(device) 
            
            optimizer.zero_grad()

            with autocast():
                # model 接收 [B, N, 3, H, W]
                logits = model(imgs) # 返回 [B*N, 2, H, W]
                
                # Masks: [B, N, H, W] -> [B*N, H, W] 以匹配 Logits
                masks_flatten = masks.view(-1, masks.shape[-2], masks.shape[-1])
                
                loss = criterion(logits, masks_flatten)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        print(f"Epoch {epoch+1} Avg Loss: {epoch_loss/len(dataloader):.4f}")
        torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], "motion_head.pth"))

if __name__ == "__main__":
    train()