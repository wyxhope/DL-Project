import os
import torch
import torchaudio
import json
import librosa
import numpy as np
import pandas as pd
import math
import random
from tqdm import tqdm
from torch import nn
from midi2audio import FluidSynth
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.inference.generation import generate_diffusion_cond
from model_and_data import create_a2a_model # 确保 model_and_data.py 在同目录

# ================= 路径配置 =================
DATA_DISK = "/root/autodl-tmp"
MAESTRO_MIDI_ROOT = os.path.join(DATA_DISK, "maestro_original_midi") 
CSV_PATH = "maestro-v3.0.0.csv"

MODEL_DIR = os.path.join(DATA_DISK, "stable_audio_model")
CONFIG_PATH = os.path.join(MODEL_DIR, "model_config.json")
BASE_CKPT_PATH = os.path.join(MODEL_DIR, "model.ckpt")

# 【核心】指向你训练好的 A2A 极致音质权重
FINETUNED_EMA_PATH = os.path.join(DATA_DISK, "maestro_a2a_ckpts_pro/ema_ep100.pt")

# 音色库搜索路径
SF2_SEARCH_PATHS = [
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/root/autodl-tmp/FluidR3_GM.sf2"
]

# 输出保存文件夹
OUTPUT_DIR = "./midi_results_clear"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= 极致音质调优参数 =================
CFG_SCALE = 4.5          
STEPS = 300              
SAMPLER_TYPE = "k-heun"  
SIGMA_MIN = 0.3          
OVERRIDE_SCALE = 0.6     

POSITIVE_PROMPT = (
    "A pristine, high-fidelity concert grand piano recording. "
    "Warm acoustic timbre, rich wooden resonance, natural hall reverberation, "
    "extremely clear articulation, 44.1kHz masterpiece."
)
NEGATIVE_PROMPT = "hiss, noise, static, electronic, synthesizer, thin sound, muffled, mono, distorted, low quality."

# 滑动窗口配置 (OLA)
WINDOW_LEN = 1024        
OVERLAP_LEN = 512        
STEP_LEN = WINDOW_LEN - OVERLAP_LEN 
# ===================================================

class Audio2AudioGateway(nn.Module):
    def __init__(self, old_project_in):
        super().__init__()
        self.old_project_in = old_project_in
        hidden_dim = old_project_in.out_features
        self.cond_bridge = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.GroupNorm(8, 128), nn.SiLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1), nn.GroupNorm(8, 256), nn.SiLU(),
            nn.Conv1d(256, hidden_dim, kernel_size=1)
        )
        self.learnable_scale = nn.Parameter(torch.ones(1) * 1.0)
        self.current_cond = None

    def forward(self, x):
        base_out = self.old_project_in(x)
        if self.current_cond is not None:
            c = self.current_cond
            if base_out.shape[0] == c.shape[0] * 2:
                c = torch.cat([c, torch.zeros_like(c)], dim=0)
            safe_scale = torch.tanh(self.learnable_scale) * 1.2
            bias = self.cond_bridge(c).transpose(1, 2)
            base_out = base_out + bias * safe_scale
        return base_out

@torch.no_grad()
def encode_wav_safe(model, wav_path, device, chunk_sec=30):
    wave, sr = torchaudio.load(wav_path, backend="soundfile")
    if sr != 44100: wave = torchaudio.functional.resample(wave, sr, 44100)
    if wave.shape[0] == 1: wave = wave.repeat(2, 1)
    
    chunk_size = chunk_sec * 44100
    total_samples = wave.shape[-1]
    latents = []
    for start in range(0, total_samples, chunk_size):
        end = min(start + chunk_size, total_samples)
        chunk = wave[:, start:end].unsqueeze(0).to(device)
        res = model.pretransform.encode(chunk)
        c_lat = res if isinstance(res, torch.Tensor) else res.mean()
        latents.append(c_lat.cpu())
        del chunk
        torch.cuda.empty_cache()
    return torch.cat(latents, dim=-1).to(device)

@torch.no_grad()
def decode_latent_safe(model, latent, chunk_size=512):
    total_l = latent.shape[-1]
    chunks = []
    device = next(model.parameters()).device
    
    for start in range(0, total_l, chunk_size):
        end = min(start + chunk_size, total_l)
        chunk = latent[:, :, start:end].to(device)
        decoded = model.pretransform.decode(chunk)
        chunks.append(decoded.squeeze(0).cpu())
        torch.cuda.empty_cache()
    return torch.cat(chunks, dim=-1)

def get_cross_fade_mask(W, overlap, is_first, is_last, device):
    mask = torch.ones(W, device=device)
    if not is_first:
        mask[:overlap] = torch.linspace(0.0, 1.0, overlap, device=device)
    if not is_last:
        mask[-overlap:] = torch.linspace(1.0, 0.0, overlap, device=device)
    return mask.view(1, 1, -1)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. 扫描 MAESTRO 并抽取测试曲目
    df = pd.read_csv(CSV_PATH)
    test_df = df[df['split'] == 'test'] 
    selected_row = test_df.sample(n=1).iloc[0]
    
    file_id = os.path.splitext(os.path.basename(selected_row['audio_filename']))[0]
    midi_filename = os.path.basename(selected_row['midi_filename'])
    
    print(f"\n🎯 随机选中测试集曲目: {selected_row['canonical_title']}")
    print(f"👉 作曲家: {selected_row['canonical_composer']}")

    midi_path = os.path.join(MAESTRO_MIDI_ROOT, midi_filename)
    if not os.path.exists(midi_path):
        print(f"❌ 错误：找不到对应的 MIDI 文件: {midi_path}")
        return

    active_sf2 = None
    for p in SF2_SEARCH_PATHS:
        if os.path.exists(p):
            active_sf2 = p
            break
    if active_sf2 is None:
        print("❌ 错误：找不到 FluidR3_GM.sf2 音色库")
        return
        
    print("\n>>> [1/4] 正在使用 FluidSynth 渲染完整的 MIDI 文件...")
    fs = FluidSynth(active_sf2, sample_rate=44100)
    temp_synth_wav = os.path.join(OUTPUT_DIR, f"temp_synth_{file_id}.wav")
    fs.midi_to_audio(midi_path, temp_synth_wav)
    
    synth_wave, sr = torchaudio.load(temp_synth_wav, backend="soundfile")
    if sr != 44100: synth_wave = torchaudio.functional.resample(synth_wave, sr, 44100)
    if synth_wave.shape[0] == 1: synth_wave = synth_wave.repeat(2, 1)
    
    torchaudio.save(temp_synth_wav, synth_wave, 44100)
    print("✅ 原始合成器音频已对齐并保存。")

    print("\n>>> [2/4] 正在加载扩散重塑引擎...")
    model, gateway = create_a2a_model(CONFIG_PATH, BASE_CKPT_PATH, device)
    model.model.load_state_dict(torch.load(FINETUNED_EMA_PATH, map_location=device))
    
    if OVERRIDE_SCALE is not None:
        gateway.learnable_scale.data.fill_(OVERRIDE_SCALE)
    print(f"当前生效的条件注入 Scale: {gateway.learnable_scale.item():.4f}")
    model.eval()

    print("\n>>> [3/4] 正在提取合成音频特征...")
    c_latent_all = encode_wav_safe(model, temp_synth_wav, device)
    total_l = c_latent_all.shape[-1]
    print(f"✅ 特征提取完成！全长共: {total_l} 帧 (约 {total_l / 21.5:.1f} 秒)")
    
    out_latent = torch.zeros((1, 64, total_l), device=device)
    weight_sum = torch.zeros((1, 1, total_l), device=device)

    print("\n>>> [4/4] 开始执行大视野滑动窗口去噪...")
    
    window_starts = list(range(0, total_l, STEP_LEN))
    if window_starts[-1] + WINDOW_LEN > total_l:
        if total_l > WINDOW_LEN:
            window_starts[-1] = total_l - WINDOW_LEN
        else:
            window_starts = [0]

    for idx, start in enumerate(window_starts):
        end = start + WINDOW_LEN
        is_first = (idx == 0)
        is_last = (idx == len(window_starts) - 1)
        
        c_chunk = c_latent_all[:, :, start:end]
        if c_chunk.shape[-1] < WINDOW_LEN:
            pad_len = WINDOW_LEN - c_chunk.shape[-1]
            c_chunk = torch.nn.functional.pad(c_chunk, (0, pad_len))
        
        c_chunk_norm = c_chunk / (c_chunk.std() + 1e-6) * 0.5
        gateway.current_cond = c_chunk_norm

        print(f"📦 正在去噪窗口 {idx+1}/{len(window_starts)} (帧: {start} - {end})")
        
        with torch.amp.autocast('cuda', dtype=torch.float16):
            # 💡 这里直接传列表，内部会自动处理文本特征，安全无痛！
            output_latent = generate_diffusion_cond(
                model, steps=STEPS, cfg_scale=CFG_SCALE,
                conditioning=[{"prompt": POSITIVE_PROMPT, "seconds_start": 0, "seconds_total": 47.5}],
                negative_conditioning=[{"prompt": NEGATIVE_PROMPT, "seconds_start": 0, "seconds_total": 47.5}],
                sample_size=WINDOW_LEN * 2048, sigma_min=SIGMA_MIN, sigma_max=500,
                sampler_type=SAMPLER_TYPE, device=device,
                return_latents=True 
            )

        blend_mask = get_cross_fade_mask(WINDOW_LEN, OVERLAP_LEN, is_first, is_last, device)
        actual_slice_len = min(total_l - start, WINDOW_LEN) 
        
        out_latent[:, :, start:start+actual_slice_len] += output_latent.float()[:, :, :actual_slice_len] * blend_mask[:, :, :actual_slice_len]
        weight_sum[:, :, start:start+actual_slice_len] += blend_mask[:, :, :actual_slice_len]

        torch.cuda.empty_cache()

    out_latent = out_latent / (weight_sum + 1e-8)
    
    print(">>> 去噪完成，正在分段还原全长高保真音频波形...")
    audio_full = decode_latent_safe(model, out_latent)

    max_val = torch.max(torch.abs(audio_full))
    if max_val > 0:
        audio_full = (audio_full / max_val) * 0.95

    before_wav = os.path.join(OUTPUT_DIR, f"{file_id}_BEFORE_synth.wav")
    after_wav = os.path.join(OUTPUT_DIR, f"{file_id}_AFTER_enhanced.wav")
    
    os.rename(temp_synth_wav, before_wav)
    torchaudio.save(after_wav, audio_full, 44100)
    
    print(f"\n🎉 完美全长重制圆满成功！")
    print(f"👉 原始合成音 (Before): {before_wav}")
    print(f"✨ AI 大师演奏 (After):  {after_wav}")

if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        main()
    except Exception as e:
        print(f"\n❌ 流程中断: {e}")
        import traceback
        traceback.print_exc()