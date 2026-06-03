import os
import torch
import torchaudio
import json
import random
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cdist
from transformers import ClapModel, ClapProcessor
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.inference.generation import generate_diffusion_cond
from model_and_data import create_a2a_model 

# ================= 1. 路径与参数配置 =================
DATA_DISK = "/root/autodl-tmp"
MODEL_DIR = os.path.join(DATA_DISK, "stable_audio_model")
CONFIG_PATH = os.path.join(MODEL_DIR, "model_config.json")
BASE_CKPT_PATH = os.path.join(MODEL_DIR, "model.ckpt")

# 权重路径
FINETUNED_EMA_PATH = os.path.join(DATA_DISK, "maestro_a2a_ckpts_pro/ema_ep100.pt")
TEST_MANIFEST = os.path.join(DATA_DISK, "processed/test_manifest.json")
CLAP_PATH = os.path.join(DATA_DISK, "clap_larger")

# 输出目录
EVAL_DIR = "./robust_evaluation_results"
os.makedirs(EVAL_DIR, exist_ok=True)

# 极致推理参数 (完全采用你的成功参数组合)
NUM_SAMPLES = 50         
CFG_SCALE = 6.0          
STEPS = 300
SIGMA_MIN = 0.3
SAMPLER_TYPE = "k-heun" 
OVERRIDE_SCALE = 0.6  

POSITIVE_PROMPT = (
    "A pristine, high-fidelity concert grand piano recording. "
    "Warm acoustic timbre, rich wooden resonance, natural hall reverberation, "
    "extremely clear articulation, 44.1kHz masterpiece."
)
NEGATIVE_PROMPT = "hiss, noise, static, electronic, synthesizer, thin sound, muffled, mono, distorted, low quality."

# ================= 2. 核心鲁棒性算法 =================

def get_robust_chroma_score(audio_ai, audio_ref, sr=44100):
    """
    使用动态时间规整(DTW)计算旋律忠实度。
    """
    try:
        hop_len = 512
        chroma_ai = librosa.feature.chroma_cqt(y=audio_ai, sr=sr, hop_length=hop_len)
        chroma_ref = librosa.feature.chroma_cqt(y=audio_ref, sr=sr, hop_length=hop_len)
        
        import scipy.ndimage
        chroma_ai = scipy.ndimage.median_filter(chroma_ai, size=(1, 5))
        chroma_ref = scipy.ndimage.median_filter(chroma_ref, size=(1, 5))

        D, wp = librosa.sequence.dtw(X=chroma_ai, Y=chroma_ref, metric='cosine')
        
        aligned_ai = chroma_ai[:, wp[:, 0]]
        aligned_ref = chroma_ref[:, wp[:, 1]]
        
        corrs = []
        for k in range(aligned_ai.shape[1]):
            c = np.corrcoef(aligned_ai[:, k], aligned_ref[:, k])[0, 1]
            if not np.isnan(c):
                corrs.append(c)
                
        score = np.mean(corrs) if len(corrs) > 0 else 0.0
        return max(0.0, score)
    except Exception as e:
        print(f"Chroma 计算出错: {e}")
        return 0.0

@torch.no_grad()
def run_robust_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(">>> 正在加载生成模型...")
    model, gateway = create_a2a_model(CONFIG_PATH, BASE_CKPT_PATH, device)
    model.model.load_state_dict(torch.load(FINETUNED_EMA_PATH, map_location=device))
    if OVERRIDE_SCALE is not None:
        gateway.learnable_scale.data.fill_(OVERRIDE_SCALE)
    model.eval()

    print(f">>> 正在加载评估裁判 (Larger CLAP)...")
    clap_model = ClapModel.from_pretrained(CLAP_PATH).to(device)
    clap_processor = ClapProcessor.from_pretrained(CLAP_PATH)
    clap_model.eval()

    with open(TEST_MANIFEST, 'r') as f:
        test_data = json.load(f)
    samples = random.sample(test_data, min(NUM_SAMPLES, len(test_data)))

    report_list = []
    all_embeds = []

    print(f">>> 启动鲁棒性评估流水线 (n={len(samples)})...")
    
    for item in tqdm(samples):
        file_id = item['id']
        c_path = os.path.join(DATA_DISK, "processed", item['synth_latent'])
        t_path = os.path.join(DATA_DISK, "processed", item['target_latent'])
        
        c_latent = torch.load(c_path).to(device)
        t_latent = torch.load(t_path).to(device)

        def align(l):
            if l.shape[-1] > 1024: return l[:, :1024]
            return torch.nn.functional.pad(l, (0, 1024 - l.shape[-1]))
        
        c_batch = align(c_latent).unsqueeze(0)
        t_batch = align(t_latent).unsqueeze(0)

        # 解码参考音频 (VAE 解码 Ground Truth)
        audio_ref = model.pretransform.decode(t_batch).squeeze(0).cpu().float()
        audio_ref_mono = audio_ref[0].numpy()

        # 解码原始合成器音频 (用于计算 Before CLAP)
        audio_synth = model.pretransform.decode(c_batch).squeeze(0).cpu().float()
        audio_synth_mono = audio_synth[0].numpy()

        # 执行 AI 增强
        gateway.current_cond = c_batch / (c_batch.std() + 1e-6) * 0.5
        
        with torch.amp.autocast('cuda', dtype=torch.float16):
            output = generate_diffusion_cond(
                model, steps=STEPS, cfg_scale=CFG_SCALE,
                conditioning=[{"prompt": POSITIVE_PROMPT, "seconds_start": 0, "seconds_total": 47.5}],
                negative_conditioning=[{"prompt": NEGATIVE_PROMPT, "seconds_start": 0, "seconds_total": 47.5}],
                sample_size=1024 * 2048, sigma_min=SIGMA_MIN, sigma_max=500,
                sampler_type=SAMPLER_TYPE, device=device
            )

        audio_ai = output.squeeze(0).cpu().float()
        audio_ai = audio_ai / (torch.max(torch.abs(audio_ai)) + 1e-8) # 峰值归一化

        # --- 计算鲁棒性指标 ---
        # 1. DTW-Chroma Fidelity (旋律忠实度)
        fidelity = get_robust_chroma_score(audio_ai[0].numpy(), audio_ref_mono)
        
        # 2. 计算 AI 增强音频的 CLAP Score (After)
        audio_ai_48k = librosa.resample(audio_ai.numpy(), orig_sr=44100, target_sr=48000)
        audio_clap = np.mean(audio_ai_48k, axis=0)
        inputs_ai = clap_processor(text=[POSITIVE_PROMPT], audio=[audio_clap], 
                                    return_tensors="pt", padding=True, sampling_rate=48000).to(device)
        outputs_ai = clap_model(**inputs_ai)
        clap_score_ai = torch.nn.functional.cosine_similarity(outputs_ai.audio_embeds, outputs_ai.text_embeds).item()
        all_embeds.append(outputs_ai.audio_embeds.cpu().numpy())

        # 3. 【核心新增】：计算原始合成器音频的 CLAP Score (Before)
        audio_synth_48k = librosa.resample(audio_synth.numpy(), orig_sr=44100, target_sr=48000)
        audio_synth_clap = np.mean(audio_synth_48k, axis=0)
        inputs_synth = clap_processor(text=[POSITIVE_PROMPT], audio=[audio_synth_clap], 
                                       return_tensors="pt", padding=True, sampling_rate=48000).to(device)
        outputs_synth = clap_model(**inputs_synth)
        clap_score_synth = torch.nn.functional.cosine_similarity(outputs_synth.audio_embeds, outputs_synth.text_embeds).item()

        # 保存生成结果供听感复核
        torchaudio.save(os.path.join(EVAL_DIR, f"{file_id}_ENHANCED.wav"), audio_ai, 44100)
        
        report_list.append({
            "id": file_id,
            "robust_chroma": fidelity,
            "clap_score_synth": clap_score_synth, # BEFORE
            "clap_score_ai": clap_score_ai,       # AFTER
            "clap_delta": clap_score_ai - clap_score_synth # 音质提升净值
        })
        
        torch.cuda.empty_cache()

    # --- 最终统计 ---
    df = pd.DataFrame(report_list)
    avg_chroma = df["robust_chroma"].mean()
    avg_clap_synth = df["clap_score_synth"].mean()
    avg_clap_ai = df["clap_score_ai"].mean()
    avg_delta = df["clap_delta"].mean()
    
    all_embeds = np.vstack(all_embeds)
    diversity = np.mean(cdist(all_embeds, all_embeds, 'cosine'))

    print("\n" + "="*60)
    print(f"📊 钢琴声学重塑 A/B 深度对比报告")
    print("-" * 60)
    print(f"✨ 鲁棒旋律忠实度 (DTW-Chroma): {avg_chroma:.4f} (理想 > 0.8)")
    print(f"✨ 原始合成器音色 (CLAP Synth):  {avg_clap_synth:.4f} (生硬的电子声基准)")
    print(f"✨ 顶级声学重塑音色 (CLAP AI):     {avg_clap_ai:.4f} (极致钢琴质感)")
    print(f"✨ 真实感净提升 (CLAP Delta):    {avg_delta:+.4f} (显著正值证明声学重构有效)")
    print(f"✨ 音色表现多样性 (Cos Distance):  {diversity:.4f}")
    print("="*60)
    
    df.to_csv(os.path.join(EVAL_DIR, "final_compare_report.csv"), index=False)
    print(f"📁 详细 A/B 报告已写入: {EVAL_DIR}/final_compare_report.csv")

if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    run_robust_pipeline()