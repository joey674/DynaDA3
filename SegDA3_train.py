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

# [新增: 引入日志和时间相关库]
import logging
import time
import json
from datetime import datetime

# ================= config =================
CONFIG = {
    # Data Configuration
    "video_dirs": [
        "/home/zhouyi/repo/dataset_segda3_train/dancer",
    ],
    "model_name": 'vitl', # 'vitl' or 'vitg'
    "save_dir": "/home/zhouyi/repo/checkpoint/SegDA3-LARGE-1.1",
    "log_dir": "/home/zhouyi/repo/log", 
    "seq_range": (2, 20),
    # Training Hyperparameters
    "learning_rate": 1e-4, 
    "epochs": 1,
    "batch_size": 1, 
    # System Configuration
    "num_workers": 4,
}

# ================= logger =================
def get_logger(log_dir, model_name):
    """
    初始化日志记录器
    文件名格式: model_name + 时间戳 (例如: vitl_20240123_123000.log)
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成带时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{model_name}_{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)
    
    logger = logging.getLogger(model_name)
    logger.setLevel(logging.INFO)
    
    # 防止重复添加 handler (在 notebook 或多次调用时有用)
    if not logger.handlers:
        # File Handler: 写入文件
        fh = logging.FileHandler(log_path, mode='w')
        fh.setLevel(logging.INFO)
        # 定义日志格式: 时间 - 级别 - 消息
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        # Stream Handler: 输出到控制台 (可选，因为 tqdm 已经有进度条了，这里主要用于报错或重要信息)
        sh = logging.StreamHandler()
        sh.setLevel(logging.WARNING) # 控制台只打印警告以上，避免干扰 tqdm
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        
    return logger, log_path

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
        
        # [修改: 这里用 print 即可，或者也可以传 logger 进来记录]
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

        imgs_tensor = torch.stack(clip_imgs)   
        masks_tensor = torch.stack(clip_masks) 
        
        return imgs_tensor, masks_tensor

# ================= train =================
def train():
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    
    # [新增: 初始化 Logger]
    logger, log_file_path = get_logger(CONFIG["log_dir"], CONFIG["model_name"])
    print(f"Training log will be saved to: {log_file_path}")
    
    # [新增: 记录本次训练的配置信息]
    logger.info("================ Training Start ================")
    logger.info(f"Configuration:\n{json.dumps(CONFIG, indent=4, ensure_ascii=False)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    dataset = MultiVideoDataset(video_dirs=CONFIG["video_dirs"],seq_range= CONFIG["seq_range"])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True)

    logger.info(f"Dataset loaded. Total batches per epoch: {len(dataloader)}")

    # [新增: 记录模型加载开始]
    logger.info(f"Loading model: {CONFIG['model_name']}...")
    model = SegDA3(model_name=CONFIG["model_name"]).to(device)
    model.train()
    logger.info("Model loaded successfully.")

    optimizer = optim.AdamW(model.motion_head.parameters(), lr=CONFIG["learning_rate"], weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    for epoch in range(CONFIG["epochs"]):
        epoch_loss = 0 
        
        # [新增: 记录 Epoch 开始时间]
        epoch_start_time = time.time()
        logger.info(f"Epoch {epoch+1}/{CONFIG['epochs']} started.")
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")

        # [修改: enumerate 方便获取 step 索引]
        for step, (imgs, masks) in enumerate(pbar):
            # [新增: 记录 Step 开始时间]
            step_start_time = time.time()
            
            imgs = imgs.to(device)   
            masks = masks.to(device) 
            
            optimizer.zero_grad()

            with autocast():
                logits = model(imgs) 
                masks_flatten = masks.view(-1, masks.shape[-2], masks.shape[-1])
                loss = criterion(logits, masks_flatten)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # [新增: 计算 Step 耗时]
            step_end_time = time.time()
            step_duration = step_end_time - step_start_time
            
            current_loss = loss.item()
            epoch_loss += current_loss
            
            # 更新进度条显示
            pbar.set_postfix({"Loss": f"{current_loss:.4f}"})
            
            # [新增: 将每一步的详细信息写入日志文件]
            # 格式: Epoch [1/1] Step [5/100] Loss: 0.1234 Time: 0.5s
            logger.info(f"Epoch [{epoch+1}/{CONFIG['epochs']}] Step [{step+1}/{len(dataloader)}] "
                        f"Loss: {current_loss:.6f} Time: {step_duration:.4f}s")

        # [新增: 记录 Epoch 统计信息]
        epoch_duration = time.time() - epoch_start_time
        avg_loss = epoch_loss/len(dataloader)
        
        log_msg = (f"Epoch {epoch+1} Completed. "
                   f"Avg Loss: {avg_loss:.4f} "
                   f"Total Time: {epoch_duration:.2f}s "
                   f"Avg Time/Step: {epoch_duration/len(dataloader):.4f}s")
        
        print(log_msg) # 控制台打印
        logger.info(log_msg) # 日志记录

        # 保存模型
        save_path = os.path.join(CONFIG["save_dir"], "motion_head.pth")
        torch.save(model.motion_head.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")

    logger.info("================ Training Finished ================")

if __name__ == "__main__":
    train()