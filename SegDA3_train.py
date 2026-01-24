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
import logging
import time
import json
from datetime import datetime

# ================= config =================
CONFIG = {
    # Data Configuration
    "video_dirs": [
        "/home/zhouyi/repo/dataset_segda3_train/dancer",
        "/home/zhouyi/repo/dataset_segda3_train/wildgs_ANYmal1",
        "/home/zhouyi/repo/dataset_segda3_train/wildgs_ANYmal2",
        "/home/zhouyi/repo/dataset_segda3_train/wildgs_ANYmal3",
        "/home/zhouyi/repo/dataset_segda3_train/wildgs_racket1",
        "/home/zhouyi/repo/dataset_segda3_train/wildgs_racket2",
        "/home/zhouyi/repo/dataset_segda3_train/wildgs_racket3",
        "/home/zhouyi/repo/dataset_segda3_train/wildgs_racket4",
    ],
    "model_name": 'vitl', # 'vitl' or 'vitg'
    "save_dir": "/home/zhouyi/repo/checkpoint/SegDA3-LARGE-1.1",
    "log_dir": "/home/zhouyi/repo/log", 
    "seq_range": (2, 20),
    # Training Hyperparameters
    "learning_rate": 1e-4, 
    "epochs": 50,
    "batch_size": 1, # 固定为1
    "samples_per_epoch": 10000, # 由于帧长度随机采样, 每个 epoch 包含多少个样本可以自定义
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
    def __init__(self, video_dirs, samples_per_epoch, seq_range, input_size=(518, 518)):
        """
        Args:
            video_dirs: 包含视频/场景子文件夹的路径列表
            samples_per_epoch: 虚拟的数据集长度，决定了一个 Epoch 训练多少个 step
            input_size: 图片大小
            seq_range: (min, max) 每次随机抽取的图片数量范围
        """
        self.input_size = input_size
        self.seq_min, self.seq_max = seq_range
        self.samples_per_epoch = samples_per_epoch
        self.video_pools = [] # 存储每个文件夹的图片列表

        print("Scanning video directories...")
        for v_dir in video_dirs:
            img_dir = os.path.join(v_dir, "images")
            mask_dir = os.path.join(v_dir, "masks")
            
            if not (os.path.exists(img_dir) and os.path.exists(mask_dir)):
                continue

            v_imgs = sorted(glob.glob(os.path.join(img_dir, "*")))
            v_masks = sorted(glob.glob(os.path.join(mask_dir, "*")))

            # 过滤掉图片太少的文件夹
            if len(v_imgs) < self.seq_min:
                print(f"Skipping {v_dir}: not enough images ({len(v_imgs)} < {self.seq_min})")
                continue
            
            # 确保存储的是对应关系正确的列表
            self.video_pools.append({
                "imgs": v_imgs,
                "masks": v_masks,
                "count": len(v_imgs)
            })
        
        if len(self.video_pools) == 0:
            raise ValueError("No valid video folders found!")

        print(f"Dataset initialized. Found {len(self.video_pools)} video folders.")
        print(f"Virtual Epoch Length: {self.samples_per_epoch}")

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        # 欺骗 DataLoader，告诉它我们有这么多数据
        # 这样 tqdm 进度条就会显示 samples_per_epoch 的长度
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # idx 在这里没有实际意义，因为我们是纯随机采样
        # 但为了保证 randomness 的多样性，我们在内部做随机

        # 1. 随机选一个视频文件夹 (场景)
        # 也可以根据 weights=video['count'] 来加权，让图片多的文件夹被选中的概率大一点
        video_data = random.choice(self.video_pools)
        
        total_imgs = video_data['count']
        
        # 2. 随机确定这次要抽几张图 (N)
        # 也就是 min(用户上限, 该文件夹实际图片数)
        current_seq_max = min(self.seq_max, total_imgs)
        n = random.randint(self.seq_min, current_seq_max)

        # 3. 从该文件夹的所有索引中，无放回地随机抽取 n 个索引
        # random.sample 能保证取出的索引不重复
        indices = random.sample(range(total_imgs), n)
        
        # [可选] 如果你还是希望保留时间顺序(从小到大)，可以 uncomment 下面这行：
        # indices.sort() 
        # 虽然你说不需要顺序，但对于 Motion 任务，通常按时间排序更符合物理规律，
        # 不过既然你的模型是 Summation 融合，乱序确实没影响。

        clip_imgs, clip_masks = [], []
        for i in indices:
            img_path = video_data["imgs"][i]
            mask_path = video_data["masks"][i]

            img = Image.open(img_path).convert('RGB').resize(self.input_size, Image.BILINEAR)
            mask = Image.open(mask_path).resize(self.input_size, Image.NEAREST)
            
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
    
    # 初始化 Logger
    logger, log_file_path = get_logger(CONFIG["log_dir"], CONFIG["model_name"])
    print(f"Training log will be saved to: {log_file_path}")
    logger.info("================ Training Start ================")
    logger.info(f"Configuration:\n{json.dumps(CONFIG, indent=4, ensure_ascii=False)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    dataset = MultiVideoDataset(
        video_dirs=CONFIG["video_dirs"],
        samples_per_epoch=CONFIG["samples_per_epoch"], 
        seq_range=CONFIG["seq_range"]
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True)

    logger.info(f"Dataset loaded. Total batches per epoch: {len(dataloader)}")
    logger.info(f"Loading model: {CONFIG['model_name']}...")
    model = SegDA3(model_name=CONFIG["model_name"]).to(device)
    model.train()
    logger.info("Model loaded successfully.")

    optimizer = optim.AdamW(model.motion_head.parameters(), lr=CONFIG["learning_rate"], weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    for epoch in range(CONFIG["epochs"]):
        epoch_loss = 0 
        
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