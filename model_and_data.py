import torch
from torch import nn
from torch.utils.data import Dataset
import json
import os
import random
from stable_audio_tools.models.factory import create_model_from_config

class Audio2AudioGateway(nn.Module):
    def __init__(self, old_project_in):
        super().__init__()
        self.old_project_in = old_project_in
        hidden_dim = old_project_in.out_features
        
        # 旁路桥接器：处理合成音频 Latent
        self.cond_bridge = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128), nn.SiLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.GroupNorm(8, 256), nn.SiLU(),
            nn.Conv1d(256, hidden_dim, kernel_size=1)
        )
        # 可学习缩放因子
        self.learnable_scale = nn.Parameter(torch.ones(1) * 1.0)
        
        with torch.no_grad():
            self.cond_bridge[-1].weight.zero_()
            self.cond_bridge[-1].bias.zero_()
            
        self.current_cond = None

    def forward(self, x):
        base_out = self.old_project_in(x)
        if self.current_cond is not None:
            # 适配推理时的 CFG 批量复制
            c = self.current_cond
            if base_out.shape[0] == c.shape[0] * 2:
                c = torch.cat([c, torch.zeros_like(c)], dim=0)
            
            # 【核心优化】：使用 Tanh 限制增益范围在 -1.2 到 1.2 之间，防止过载噪音
            safe_scale = torch.tanh(self.learnable_scale) * 1.2
            bias = self.cond_bridge(c).transpose(1, 2)
            base_out = base_out + bias * safe_scale
            
        return base_out

def create_a2a_model(config_path, ckpt_path, device="cuda"):
    with open(config_path, 'r') as f: config = json.load(f)
    model = create_model_from_config(config)
    print(">>> 正在加载原厂预训练权重...")
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu").get("state_dict", {}), strict=False)
    
    # 动态挂载
    if hasattr(model.model, "transformer"): transformer = model.model.transformer
    else: transformer = model.model.model.transformer
    transformer.project_in = Audio2AudioGateway(transformer.project_in)
    
    model.to(device)
    model.pretransform.requires_grad_(False)
    model.conditioner.requires_grad_(False)
    model.model.requires_grad_(True)
    return model, transformer.project_in

class MaestroA2ADataset(Dataset):
    def __init__(self, manifest_path, data_root, crop_len=1024):
        with open(manifest_path, 'r') as f: self.data = json.load(f)
        self.data_root = data_root
        self.crop_len = crop_len
        self.fps = 44100 / 2048 

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        t_all = torch.load(os.path.join(self.data_root, item['target_latent']))
        c_all = torch.load(os.path.join(self.data_root, item['synth_latent']))
        
        L = t_all.shape[1]
        if L > self.crop_len:
            start = random.randint(0, L - self.crop_len)
            target = t_all[:, start : start + self.crop_len]
            cond = c_all[:, start : start + self.crop_len]
            sec_start = start / self.fps
        else:
            target = torch.nn.functional.pad(t_all, (0, self.crop_len - L))
            cond = torch.nn.functional.pad(c_all, (0, self.crop_len - L))
            sec_start = 0.0
            
        return {"target": target, "cond": cond, "text": item['text'],
                "seconds_start": sec_start, "seconds_total": self.crop_len / self.fps}