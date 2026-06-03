import os
import torch
import torchaudio
import json
import librosa
import numpy as np
import math
from tqdm import tqdm
from torch import nn
from piano_transcription_inference import PianoTranscription, sample_rate as pt_sr
from midi2audio import FluidSynth
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.inference.generation import generate_diffusion_cond
from model_and_data import create_a2a_model # 确保 model_and_data.py 在同目录

# ================= 路径配置 =================
DATA_DISK = "/root/autodl-tmp"

# 1. 扒谱模型本地路径 (已适配你下载的文件名)
PT_CHECKPOINT_PATH = "/root/piano_transcription_inference_data/CRNN_note_F13D0.9186.pth"

# 2. Stable Audio 基础模型路径
MODEL_DIR = os.path.join(DATA_DISK, "stable_audio_model")
CONFIG_PATH = os.path.join(MODEL_DIR, "model_config.json")
BASE_CKPT_PATH = os.path.join(MODEL_DIR, "model.ckpt")

# 3. 你的微调权重路径 (使用效果最好的 ep100)
FINETUNED_EMA_PATH = os.path.join(DATA_DISK, "maestro_a2a_ckpts_pro/ema_ep100.pt")

# 4. 音色库搜索路径
SF2_SEARCH_PATHS = [
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/root/autodl-tmp/FluidR3_GM.sf2",
    "/root/autodl-tmp/maestro-v3.0.0/FluidR3_GM.sf2"
]

# 5. 输入输出定义
USER_INPUT_AUDIO = "my_piano_recording.wav" # 输入你需要重制的任意长度旧录音WAV
FINAL_HIFI_AUDIO = "remastered_long_masterpiece.wav" # 最终的高清输出成品

# ================= 极致音质调优参数 =================
CFG_SCALE = 4.5          
STEPS = 300              # 300 步高精去噪
SAMPLER_TYPE = "k-heun"  # 确定性采样器，背景绝无噪音
SIGMA_MIN = 0.3          # 稳定末端去噪
OVERRIDE_SCALE = 0.6     # 0.6 黄金对齐引导强度

# 极致声学提示词
POSITIVE_PROMPT = (
    "A pristine, high-fidelity concert grand piano recording. "
    "Warm acoustic timbre, rich wooden resonance, natural hall reverberation, "
    "extremely clear articulation, 44.1kHz masterpiece."
)
NEGATIVE_PROMPT = "hiss, noise, static, electronic, synthesizer, thin sound, muffled, mono, distorted, low quality."

# 滑动窗口配置 (OLA)
WINDOW_LEN = 1024        # 1024 帧 约 47.5 秒 (单次去噪最大窗口)
OVERLAP_LEN = 512        # 512 帧 约 23.7 秒 (重叠渐变长度)
STEP_LEN = WINDOW_LEN - OVERLAP_LEN 
# ===================================================

# --- 130维 Gateway 架构定义 (保持不变) ---
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

def robust_load_audio(path, sr=16000):
    """使用 librosa 稳健加载任意长度音频"""
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio

@torch.no_grad()
def encode_wav_safe(model, wav_path, device, chunk_sec=30):
    """分段 VAE 编码，防止处理几分钟的音频时爆显存"""
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
    """分段 VAE 解码，防止还原超长音频时爆显存"""
    total_l = latent.shape[-1]
    chunks = []
    # 动态获取设备，避免 AttributeError
    device = next(model.parameters()).device
    
    for start in range(0, total_l, chunk_size):
        end = min(start + chunk_size, total_l)
        chunk = latent[:, :, start:end].to(device)
        decoded = model.pretransform.decode(chunk)
        chunks.append(decoded.squeeze(0).cpu())
        torch.cuda.empty_cache()
    return torch.cat(chunks, dim=-1)

def get_cross_fade_mask(W, overlap, is_first, is_last, device):
    """梯形交叉淡化窗口，确保重叠区域相加权重恒等于 1.0"""
    mask = torch.ones(W, device=device)
    if not is_first:
        mask[:overlap] = torch.linspace(0.0, 1.0, overlap, device=device)
    if not is_last:
        mask[-overlap:] = torch.linspace(1.0, 0.0, overlap, device=device)
    return mask.view(1, 1, -1)

def remaster_pipeline():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if not os.path.exists(USER_INPUT_AUDIO):
        print(f"❌ 错误：找不到输入音频: {USER_INPUT_AUDIO}")
        return

    # ====================================================
    # 步骤 1：使用 AI 扒谱 (Audio -> MIDI)
    # ====================================================
    print("\n>>> [1/5] 启动 AI 扒谱引擎...")
    transcriptor = PianoTranscription(device=device, checkpoint_path=PT_CHECKPOINT_PATH)
    audio_data = robust_load_audio(USER_INPUT_AUDIO, sr=pt_sr)
    temp_midi = "temp_remaster_long.mid"
    transcriptor.transcribe(audio_data, temp_midi)
    print("✅ 扒谱提取完成。")

    # ====================================================
    # 步骤 2：MIDI 渲染 (MIDI -> Synth WAV)
    # ====================================================
    print("\n>>> [2/5] 寻找音色库并进行基础合成音渲染...")
    active_sf2 = None
    for p in SF2_SEARCH_PATHS:
        if os.path.exists(p):
            active_sf2 = p
            break
            
    if active_sf2 is None:
        print("❌ 错误：找不到 FluidR3_GM.sf2 音色库！")
        return
        
    print(f">>> 使用音色库: {active_sf2}")
    fs = FluidSynth(active_sf2, sample_rate=44100)
    temp_synth_wav = "temp_synth_long.wav"
    fs.midi_to_audio(temp_midi, temp_synth_wav)
    print("✅ 渲染基础合成音频完成。")

    # ====================================================
    # 步骤 3：加载 A2A 扩散重塑模型
    # ====================================================
    print("\n>>> [3/5] 正在加载微调后的扩散重塑引擎...")
    model, gateway = create_a2a_model(CONFIG_PATH, BASE_CKPT_PATH, device)
    
    # 加载你的微调权重
    print(f">>> 注入权重: {FINETUNED_EMA_PATH}")
    model.model.load_state_dict(torch.load(FINETUNED_EMA_PATH, map_location=device))
    
    if OVERRIDE_SCALE is not None:
        gateway.learnable_scale.data.fill_(OVERRIDE_SCALE)
    print(f"当前生效的条件注入 Scale: {gateway.learnable_scale.item():.4f}")
    
    model.eval()

    # ====================================================
    # 步骤 4：提取完整合成特征 (VAE 编码)
    # ====================================================
    print("\n>>> [4/5] 正在提取完整旋律的潜空间特征...")
    c_latent_all = encode_wav_safe(model, temp_synth_wav, device)
    total_l = c_latent_all.shape[-1]
    print(f"✅ 提取成功！全曲特征长度: {total_l} 帧 (约 {total_l / (44100/2048):.1f} 秒)")

    # 初始化重合相加的张量
    out_latent = torch.zeros((1, 64, total_l), device=device)
    weight_sum = torch.zeros((1, 1, total_l), device=device)

    # ====================================================
    # 步骤 5：滑动窗口重叠相加去噪 (OLA)
    # ====================================================
    print("\n>>> [5/5] 开始执行大视野滑动窗口去噪与极致声学重塑...")
    
    window_starts = list(range(0, total_l, STEP_LEN))
    if window_starts[-1] + WINDOW_LEN > total_l:
        if total_l > WINDOW_LEN:
            window_starts[-1] = total_l - WINDOW_LEN
        else:
            window_starts = [0]

    global current_chunk_norm

    for idx, start in enumerate(window_starts):
        end = start + WINDOW_LEN
        is_first = (idx == 0)
        is_last = (idx == len(window_starts) - 1)
        
        c_chunk = c_latent_all[:, :, start:end]
        if c_chunk.shape[-1] < WINDOW_LEN:
            pad_len = WINDOW_LEN - c_chunk.shape[-1]
            c_chunk = torch.nn.functional.pad(c_chunk, (0, pad_len))
        
        # 归一化特征
        c_chunk_norm = c_chunk / (c_chunk.std() + 1e-6) * 0.5
        # 直接赋给网关变量，原生 forward 会自动处理 CFG 对齐
        gateway.current_cond = c_chunk_norm

        print(f"📦 正在去噪窗口 {idx+1}/{len(window_starts)} (帧: {start} - {end})")
        
        with torch.amp.autocast('cuda', dtype=torch.float16):
            # 💡 【核心修复】：直接传原始列表，让官方采样器处理内部 get_conditioning_inputs
            # 完全避免了中间的 'prompt' 键丢失导致崩溃的问题
            output_latent = generate_diffusion_cond(
                model, steps=STEPS, cfg_scale=CFG_SCALE,
                conditioning=[{"prompt": POSITIVE_PROMPT, "seconds_start": 0, "seconds_total": 47.5}],
                negative_conditioning=[{"prompt": NEGATIVE_PROMPT, "seconds_start": 0, "seconds_total": 47.5}],
                sample_size=WINDOW_LEN * 2048, sigma_min=SIGMA_MIN, sigma_max=500,
                sampler_type=SAMPLER_TYPE, device=device,
                return_latents=True 
            )

        blend_mask = get_cross_fade_mask(WINDOW_LEN, OVERLAP_LEN, is_first, is_last, device)
        actual_slice_len = min(total_l - start, WINDOW_LEN) # 边界安全裁剪
        
        out_latent[:, :, start:start+actual_slice_len] += output_latent.float()[:, :, :actual_slice_len] * blend_mask[:, :, :actual_slice_len]
        weight_sum[:, :, start:start+actual_slice_len] += blend_mask[:, :, :actual_slice_len]

        torch.cuda.empty_cache()

    # 归一化消除重叠增益
    out_latent = out_latent / (weight_sum + 1e-8)
    
    # 分段安全解码还原全曲波形
    print(">>> 去噪完成，正在分段还原高保真音频波形...")
    audio_full = decode_latent_safe(model, out_latent)

    # 最终母带处理：真峰值归一化限制
    max_val = torch.max(torch.abs(audio_full))
    if max_val > 0:
        audio_full = (audio_full / max_val) * 0.95

    # 保存
    torchaudio.save(FINAL_HIFI_AUDIO, audio_full, 44100)
    
    # 清理临时文件 (这里安全删除了 temp_synth_long.wav 并只在 temp_midi 存在时进行清理)
    if os.path.exists(temp_midi): os.remove(temp_midi)
    if os.path.exists(temp_synth_wav): os.remove(temp_synth_wav)
    
    print(f"\n🎉 完美全长重制圆满成功！")
    print(f"👉 原始旧录音: {USER_INPUT_AUDIO}")
    print(f"✨ 高保真成品: {FINAL_HIFI_AUDIO}")

if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        remaster_pipeline()
    except Exception as e:
        print(f"\n❌ 流程中断: {e}")
        import traceback
        traceback.print_exc()