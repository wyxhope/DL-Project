import torch
import torchaudio
import json
import os
import random
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.inference.generation import generate_diffusion_cond
from model_and_data import create_a2a_model  # 直接导入我们在 model_and_data 里写好的干净创建函数

# ================= 路径配置 =================
DATA_DISK = "/root/autodl-tmp"
MODEL_DIR = os.path.join(DATA_DISK, "stable_audio_model")
CONFIG_PATH = os.path.join(MODEL_DIR, "model_config.json")
BASE_CKPT_PATH = os.path.join(MODEL_DIR, "model.ckpt")

# 配合最新的训练脚本路径 maestro_a2a_ckpts_pro 和权重文件名
FINETUNED_EMA_PATH = os.path.join(DATA_DISK, "maestro_a2a_ckpts_pro/ema_ep100.pt")
TEST_MANIFEST = os.path.join(DATA_DISK, "processed/test_manifest.json")

OUTPUT_DIR = "./final_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= 极致音质调优参数 =================
CFG_SCALE = 6.0         # 解决杂音的关键
STEPS = 300              # 增加步数，打磨高频细节
SAMPLER_TYPE = "k-heun" 
SIGMA_MIN = 0.3        # 调小此值以捕获微弱的空气感 (默认 0.3)
OVERRIDE_SCALE = 0.6    # 手动微调合成器信号强度 (0.6 - 0.9)

# 增强提示词
POSITIVE_PROMPT = (
    "A pristine, high-fidelity concert grand piano recording. "
    "Warm acoustic timbre, rich wooden resonance, natural hall reverberation, "
    "extremely clear articulation, 44.1kHz masterpiece."
)
# 负面提示词：过滤数码味
NEGATIVE_PROMPT = "hiss, noise, static, electronic, synthesizer, thin sound, muffled, mono, distorted, low quality."
# ===================================================

@torch.no_grad()
def generate_pro():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. 采用 model_and_data 内置的助手函数，完美完成 Gateway 挂载和冻结
    print(">>> 正在初始化模型架构与基础权重...")
    model, gateway = create_a2a_model(CONFIG_PATH, BASE_CKPT_PATH, device)
    
    # 2. 覆盖加载微调后的 EMA 权重
    print(f">>> 正在加载微调后的 EMA 权重: {FINETUNED_EMA_PATH}")
    model.model.load_state_dict(torch.load(FINETUNED_EMA_PATH, map_location=device))
    
    # 应用手动 Scale 覆盖
    if OVERRIDE_SCALE is not None:
        gateway.learnable_scale.data.fill_(OVERRIDE_SCALE)
    print(f"当前生效的条件注入 Scale: {gateway.learnable_scale.item():.4f}")
    
    model.eval()

    # 3. 随机抽取测试数据
    with open(TEST_MANIFEST, 'r') as f: 
        test_data = json.load(f)
    item = random.choice(test_data)
    file_id = item['id']
    print(f"\n>>> 正在处理测试曲目: {file_id}")
    
    # 🚨 【核心修复】：读取 "synth_latent" 而不是 "cond_latent"
    c_latent_path = os.path.join(DATA_DISK, "processed", item['synth_latent'])
    t_latent_path = os.path.join(DATA_DISK, "processed", item['target_latent'])
    
    c_latent = torch.load(c_latent_path).to(device)
    t_latent = torch.load(t_latent_path).to(device)

    # 长度对齐 (1024 帧 约 47.5 秒)
    def align(l):
        if l.shape[-1] > 1024: return l[:, :1024]
        return torch.nn.functional.pad(l, (0, 1024 - l.shape[-1]))
    
    c_batch = align(c_latent).unsqueeze(0)
    t_batch = align(t_latent).unsqueeze(0)

    # 4. 保存对比音频 (Before 和 Ground Truth)
    print(">>> 正在解码原始合成器音频与目标录音...")
    audio_synth = model.pretransform.decode(c_batch).squeeze(0).cpu()
    audio_gt = model.pretransform.decode(t_batch).squeeze(0).cpu()
    torchaudio.save(os.path.join(OUTPUT_DIR, f"{file_id}_0_SYNTH.wav"), audio_synth, 44100)
    torchaudio.save(os.path.join(OUTPUT_DIR, f"{file_id}_1_GROUND_TRUTH.wav"), audio_gt, 44100)

    # 5. 执行 AI 声学重塑
    # 标准化（必须与训练一致）
    c_latent_norm = c_batch / (c_batch.std() + 1e-6) * 0.5
    gateway.current_cond = c_latent_norm

    print(f">>> 正在通过扩散模型生成声学细节 (Sampler: {SAMPLER_TYPE}, Steps: {STEPS})...")
    with torch.amp.autocast('cuda', dtype=torch.float16):
        output = generate_diffusion_cond(
            model, 
            steps=STEPS, 
            cfg_scale=CFG_SCALE,
            conditioning=[{"prompt": POSITIVE_PROMPT, "seconds_start": 0, "seconds_total": 47.5}],
            negative_conditioning=[{"prompt": NEGATIVE_PROMPT, "seconds_start": 0, "seconds_total": 47.5}],
            sample_size=1024 * 2048, 
            sigma_min=SIGMA_MIN, 
            sigma_max=500,
            sampler_type=SAMPLER_TYPE, 
            device=device
        )

    # 6. 保存处理后音频 (After)
    print(">>> 正在保存增强后的音频...")
    audio_ai = output.squeeze(0).cpu()
    audio_ai = audio_ai / (torch.max(torch.abs(audio_ai)) + 1e-8)
    torchaudio.save(os.path.join(OUTPUT_DIR, f"{file_id}_2_AI_ENHANCED.wav"), audio_ai, 44100)
    print(f"\n✨ 生成成功！请检查 {OUTPUT_DIR} 目录。")

if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    generate_pro()