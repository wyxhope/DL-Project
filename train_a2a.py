import torch
import math
import os
from torch import nn, optim
from torch.utils.data import DataLoader
from model_and_data import MaestroA2ADataset, create_a2a_model
from utils import TrainingLogger

# ================= 路径配置 =================
DATA_DISK = "/root/autodl-tmp"
MODEL_DIR = os.path.join(DATA_DISK, "stable_audio_model")
DATA_ROOT = os.path.join(DATA_DISK, "processed")
MANIFEST = os.path.join(DATA_ROOT, "train_manifest.json") # 使用 train 集清单

SAVE_DIR = os.path.join(DATA_DISK, "maestro_a2a_ckpts_pro")
os.makedirs(SAVE_DIR, exist_ok=True)

# ================= 终极超参数 =================
BATCH_SIZE = 16          # 96GB 显存完美适配
EPOCHS = 100             # 跑到 100 轮磨出极限音质
LR_GATEWAY = 1e-4        # Gateway 旁路学习率：较快，负责提取合成器特征
LR_BACKBONE = 3e-6       # DiT 骨干学习率：极低，进行“纳米级”音质修复，防止杂音
WEIGHT_DECAY = 1e-2      # 增加权重衰减，防止后期过拟合产生高频噪声
EMA_DECAY = 0.999
CROP_LEN = 1024
# ============================================

def train():
    os.environ["OMP_NUM_THREADS"] = "1"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. 初始化模型与网关
    print(">>> 正在初始化 A2A 声学重塑模型...")
    model, gateway = create_a2a_model(
        os.path.join(MODEL_DIR, "model_config.json"), 
        os.path.join(MODEL_DIR, "model.ckpt"), 
        device
    )
    
    # 2. 建立 EMA (指数移动平均)
    ema_model = optim.swa_utils.AveragedModel(
        model.model, 
        multi_avg_fn=optim.swa_utils.get_ema_multi_avg_fn(EMA_DECAY)
    )

    # 3. 参数分层与优化器
    # gateway_p 包含 cond_bridge, learnable_scale
    gateway_p = [p for n, p in model.model.named_parameters() if "cond_bridge" in n or "learnable_scale" in n]
    backbone_p = [p for n, p in model.model.named_parameters() if not any(x in n for x in ["cond_bridge", "learnable_scale"])]
    
    optimizer = optim.AdamW([
        {'params': gateway_p, 'lr': LR_GATEWAY},
        {'params': backbone_p, 'lr': LR_BACKBONE}
    ], weight_decay=WEIGHT_DECAY)

    # 4. 数据加载
    dataset = MaestroA2ADataset(MANIFEST, DATA_ROOT, crop_len=CROP_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)
    logger = TrainingLogger(log_dir=os.path.join(DATA_DISK, "results_a2a_pro"))

    # 5. 学习率调度器
    total_steps = EPOCHS * len(loader)
    warmup_steps = int(total_steps * 0.05) # 5% Warmup
    scheduler_w = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    scheduler_c = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(total_steps - warmup_steps), eta_min=1e-6)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler_w, scheduler_c], milestones=[warmup_steps])

    print(f"🔥 全量 MAESTRO 物理音效极致重塑启动！总步数: {total_steps}")
    global_step = 0
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for i, batch in enumerate(loader):
            optimizer.zero_grad()

            # --- A. 准备文本条件 ---
            # 15% 概率 Text Dropout：迫使模型在无文本时也能发挥滤镜作用
            cond_list = []
            for j in range(BATCH_SIZE):
                prompt = "" if torch.rand(1).item() < 0.15 else batch["text"][j]
                cond_list.append({
                    "prompt": prompt, 
                    "seconds_start": batch["seconds_start"][j].item(), 
                    "seconds_total": batch["seconds_total"][j].item()
                })

            with torch.amp.autocast('cuda', dtype=torch.float16):
                with torch.no_grad():
                    raw_cond = model.conditioner(cond_list, device)
                    raw_cond_dict = model.get_conditioning_inputs(raw_cond)
                    valid_keys = {"cross_attn_cond", "cross_attn_mask", "global_cond", "prepend_embeds", "prepend_mask"}
                    cond_dict = {k: v for k, v in raw_cond_dict.items() if k in valid_keys}

                # --- B. 处理合成器音频条件 (Condition) ---
                cond_latent = batch["cond"].to(device)
                
                # 关键：特征标准化，保证输入处于安全区间
                cond_latent = cond_latent / (cond_latent.std(dim=(1,2), keepdim=True) + 1e-6) * 0.5
                
                # 15% 概率完全丢弃音频条件，维持模型原生的自由扩散能力
                if torch.rand(1).item() < 0.15: 
                    cond_latent *= 0.0 
                
                # 注入网关
                gateway.current_cond = cond_latent

                # --- C. Logit-Normal 加噪 (v-prediction) ---
                z_target = batch["target"].to(device)
                
                # 让模型多学中间难度，少学两端极端难度
                u = torch.randn((z_target.shape[0],), device=device)
                t = torch.sigmoid(u * 1.2) 
                
                noise = torch.randn_like(z_target)
                theta = t * (math.pi / 2)
                c_skip, c_out = torch.cos(theta).view(-1,1,1), torch.sin(theta).view(-1,1,1)
                
                z_noisy = c_skip * z_target + c_out * noise
                target_v = c_skip * noise - c_out * z_target

                # --- D. 前向传播 ---
                # 注意：这里直接输入 64 维的 z_noisy，因为 Gateway 会在内部加 bias
                pred_v = model.model(z_noisy, t * 1000.0, **cond_dict)

                # --- E. Min-SNR-Gamma 损失加权 ---
                snr = (c_skip / c_out) ** 2
                weight = (torch.clamp(snr, max=5.0) / snr).view(-1)
                raw_mse = nn.functional.mse_loss(pred_v, target_v, reduction='none').mean(dim=(1, 2))
                loss = (raw_mse * weight).mean()

            # --- F. 反向传播与优化 ---
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # 更新 EMA
            ema_model.update_parameters(model.model)

            # --- G. 日志记录 ---
            logger.log(epoch, global_step, loss.item())
            
            if global_step % 20 == 0:
                print(f"Ep {epoch} | Step {global_step} | Loss: {loss.item():.4f} | Scale: {gateway.learnable_scale.item():.4f} | LR_B: {optimizer.param_groups[1]['lr']:.2e}")
            
            global_step += 1

        # 每个 Epoch 结束更新曲线
        logger.save_plot()

        # 每 5 轮保存一次 Checkpoint
        if epoch % 5 == 0:
            torch.save(ema_model.module.state_dict(), os.path.join(SAVE_DIR, f"ema_ep{epoch}.pt"))
            print(f"💾 Checkpoint 已保存至: ema_ep{epoch}.pt")

if __name__ == "__main__":
    train()