import torch
import torchaudio
import json
import os
import librosa
import numpy as np
from torch import nn
from piano_transcription_inference import PianoTranscription, sample_rate as pt_sr
from midi2audio import FluidSynth
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.inference.generation import generate_diffusion_cond
from model_and_data import create_a2a_model, Audio2AudioGateway # 确保 model_and_data.py 在同目录

# ================= 路径配置 =================
DATA_DISK = "/root/autodl-tmp"

# 1. 扒谱模型路径
PT_CHECKPOINT_PATH = "/root/piano_transcription_inference_data/CRNN_note_F13D0.9186.pth"

# 2. Stable Audio 基础模型路径
MODEL_DIR = os.path.join(DATA_DISK, "stable_audio_model")
CONFIG_PATH = os.path.join(MODEL_DIR, "model_config.json")
BASE_CKPT_PATH = os.path.join(MODEL_DIR, "model.ckpt")

# 3. 你的微调权重路径 (使用效果最好的 ep100)
FINETUNED_EMA_PATH = os.path.join(DATA_DISK, "maestro_a2a_ckpts_pro/ema_ep100.pt")

# 4. 音色库路径
SF2_PATH = "/usr/share/sounds/sf2/FluidR3_GM.sf2"
if not os.path.exists(SF2_PATH):
    SF2_PATH = "/root/autodl-tmp/FluidR3_GM.sf2"

# 5. 输入输出
USER_INPUT_AUDIO = "my_piano_recording.wav" # 你上传的原始音频
FINAL_OUTPUT_WAV = "remastered_final_masterpiece.wav"

# ================= 极致音质调优参数 (已同步测试版) =================
CFG_SCALE = 4.5          
STEPS = 400              
SAMPLER_TYPE = "k-heun"  # 确定性采样器，背景最干净
SIGMA_MIN = 0.3          
OVERRIDE_SCALE = 0.6     # 0.6 引导强度，兼顾旋律服从与音色真实

POSITIVE_PROMPT = (
    "A pristine, high-fidelity concert grand piano recording. "
    "Warm acoustic timbre, rich wooden resonance, natural hall reverberation, "
    "extremely clear articulation, 44.1kHz masterpiece."
)

NEGATIVE_PROMPT = "hiss, noise, static, electronic, synthesizer, thin sound, muffled, mono, distorted, low quality."
# ===================================================

def robust_load_audio(path, sr=16000):
    """使用 librosa 稳健加载音频"""
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio

def remaster_pipeline():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if not os.path.exists(USER_INPUT_AUDIO):
        print(f"❌ 错误：找不到输入音频文件 {USER_INPUT_AUDIO}")
        return

    # ----------------------------------------------------
    # 步骤 1: AI 扒谱 (Acoustic -> MIDI)
    # ----------------------------------------------------
    print("\n>>> [1/4] 启动 AI 扒谱引擎...")
    transcriptor = PianoTranscription(device=device, checkpoint_path=PT_CHECKPOINT_PATH)
    audio_data = robust_load_audio(USER_INPUT_AUDIO, sr=pt_sr)
    temp_midi = "temp_remaster.mid"
    transcriptor.transcribe(audio_data, temp_midi)
    print("✅ 扒谱提取完成。")

    # ----------------------------------------------------
    # 步骤 2: MIDI 渲染 (MIDI -> Synth WAV)
    # ----------------------------------------------------
    print("\n>>> [2/4] 渲染基础合成音频...")
    fs = FluidSynth(SF2_PATH, sample_rate=44100)
    temp_synth_wav = "temp_synth.wav"
    
    # 在波形层面填充到 47.5 秒，确保 VAE 编码后的特征与模型对齐
    fs.midi_to_audio(temp_midi, temp_synth_wav)
    
    # 加载并对齐长度
    synth_wave, sr = torchaudio.load(temp_synth_wav, backend="soundfile")
    if sr != 44100: synth_wave = torchaudio.functional.resample(synth_wave, sr, 44100)
    if synth_wave.shape[0] == 1: synth_wave = synth_wave.repeat(2, 1)
    
    MAX_SAMPLES = 1024 * 2048 # 47.55秒
    if synth_wave.shape[1] < MAX_SAMPLES:
        pad = torch.zeros((2, MAX_SAMPLES - synth_wave.shape[1]))
        synth_wave = torch.cat([synth_wave, pad], dim=1)
    else:
        synth_wave = synth_wave[:, :MAX_SAMPLES]
    
    # 重新存为对齐后的临时文件
    torchaudio.save(temp_synth_wav, synth_wave, 44100)
    print(f"✅ 合成器基准音频已对齐并保存。")

    # ----------------------------------------------------
    # 步骤 3: 加载 A2A 扩散重塑模型
    # ----------------------------------------------------
    print("\n>>> [3/4] 正在加载微调后的扩散重塑引擎...")
    model, gateway = create_a2a_model(CONFIG_PATH, BASE_CKPT_PATH, device)
    
    # 加载你的微调权重
    print(f">>> 注入权重: {FINETUNED_EMA_PATH}")
    model.model.load_state_dict(torch.load(FINETUNED_EMA_PATH, map_location=device))
    
    # 应用最优 Scale
    if OVERRIDE_SCALE is not None:
        gateway.learnable_scale.data.fill_(OVERRIDE_SCALE)
    
    model.eval()

    # ----------------------------------------------------
    # 步骤 4: 执行声学重塑 (Synth -> HiFi Acoustic)
    # ----------------------------------------------------
    print("\n>>> [4/4] 正在执行极致音质重制...")
    
    # VAE 提取合成音频特征
    input_wave = synth_wave.unsqueeze(0).to(device)
    c_latent = model.pretransform.encode(input_wave)
    c_latent = c_latent if isinstance(c_latent, torch.Tensor) else c_latent.mean()
    
    # 确保 Latent 长度严格为 1024
    if c_latent.shape[-1] > 1024: c_latent = c_latent[:, :, :1024]
    
    # 标准化 (0.5 是训练时的关键基准)
    c_latent_norm = c_latent / (c_latent.std() + 1e-6) * 0.5
    gateway.current_cond = c_latent_norm

    print(f">>> 扩散采样中 (Sampler: {SAMPLER_TYPE}, Steps: {STEPS})...")
    with torch.amp.autocast('cuda', dtype=torch.float16):
        output = generate_diffusion_cond(
            model, 
            steps=STEPS, 
            cfg_scale=CFG_SCALE,
            conditioning=[{"prompt": POSITIVE_PROMPT, "seconds_start": 0, "seconds_total": 47.5}],
            negative_conditioning=[{"prompt": NEGATIVE_PROMPT, "seconds_start": 0, "seconds_total": 47.5}],
            sample_size=MAX_SAMPLES, 
            sigma_min=SIGMA_MIN, 
            sigma_max=500,
            sampler_type=SAMPLER_TYPE, 
            device=device
        )

    # 最终处理：归一化
    print(">>> 正在进行后期处理...")
    audio_out = output.squeeze(0).cpu()
    
    # 峰值归一化防止斩波噪音
    max_val = torch.max(torch.abs(audio_out))
    if max_val > 0:
        audio_out = (audio_out / max_val) * 0.95 
    
    torchaudio.save(FINAL_OUTPUT_WAV, audio_out, 44100)
    
    # 清理
    if os.path.exists(temp_midi): os.remove(temp_midi)
    if os.path.exists(temp_synth_wav): os.remove(temp_synth_wav)
    
    print(f"\n✨ 恭喜！端到端重制完成。")
    print(f"👉 原始录音: {USER_INPUT_AUDIO}")
    print(f"👉 最终成品: {FINAL_OUTPUT_WAV}")

if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        remaster_pipeline()
    except Exception as e:
        print(f"\n❌ 流程中断: {e}")
        import traceback
        traceback.print_exc()