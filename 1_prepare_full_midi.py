import os
import torch
import torchaudio
import pandas as pd
import json
import math
from concurrent.futures import ProcessPoolExecutor
from midi2audio import FluidSynth
from stable_audio_tools.models.factory import create_model_from_config
import time
import gc

# ================= 路径配置 =================
DATA_DISK = "/root/autodl-tmp"
MAESTRO_MIDI_ROOT = os.path.join(DATA_DISK, "maestro_original_midi") 
CSV_PATH = "maestro-v3.0.0.csv"
MODEL_DIR = os.path.join(DATA_DISK, "stable_audio_model")
CONFIG_PATH = os.path.join(MODEL_DIR, "model_config.json")
CKPT_PATH = os.path.join(MODEL_DIR, "model.ckpt")
PROCESSED_DIR = os.path.join(DATA_DISK, "processed")
SF2_PATH = "/usr/share/sounds/sf2/FluidR3_GM.sf2"

# --- 关键：将临时目录设在空间大的数据盘 ---
TEMP_DIR = os.path.join(DATA_DISK, "render_temp")
os.makedirs(TEMP_DIR, exist_ok=True)

# 性能配置
NUM_CPU_WORKERS = 8 
BATCH_CHUNK_SIZE = 50  # 每组处理 50 首，防止堆积临时文件
# ============================================

def render_one_midi(midi_path, wav_path):
    try:
        fs = FluidSynth(SF2_PATH, sample_rate=44100)
        fs.midi_to_audio(midi_path, wav_path)
        return True if os.path.exists(wav_path) else False
    except:
        return False

class RobustPreprocessor:
    def __init__(self, device="cuda"):
        self.device = device
        self.sample_rate = 44100
        print(">>> 正在初始化 VAE...")
        with open(CONFIG_PATH, 'r') as f: config = json.load(f)
        ae_config = {"model_type": "autoencoder", "sample_rate": 44100, "model": config["model"]["pretransform"]["config"]}
        self.vae = create_model_from_config(ae_config)
        ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)["state_dict"]
        vae_state_dict = {k[len("pretransform.model."):]: v for k, v in ckpt.items() if k.startswith("pretransform.model.")}
        self.vae.load_state_dict(vae_state_dict)
        self.vae.to(device).eval()

    @torch.no_grad()
    def encode_wav(self, wav_path):
        wave, sr = torchaudio.load(wav_path, backend="soundfile")
        if sr != self.sample_rate: wave = torchaudio.functional.resample(wave, sr, self.sample_rate)
        if wave.shape[0] == 1: wave = wave.repeat(2, 1)
        chunk_size = 30 * self.sample_rate
        latents = []
        for start in range(0, wave.shape[-1], chunk_size):
            end = min(start + chunk_size, wave.shape[-1])
            chunk = wave[:, start:end]
            if chunk.shape[-1] < 44100: chunk = torch.nn.functional.pad(chunk, (0, 44100 - chunk.shape[-1]))
            res = self.vae.encode(chunk.unsqueeze(0).to(self.device))
            c_lat = res if isinstance(res, torch.Tensor) else res.mean()
            latents.append(c_lat.cpu())
        return torch.cat(latents, dim=-1).squeeze(0)

    def run(self):
        df = pd.read_csv(CSV_PATH)
        synth_dir = os.path.join(PROCESSED_DIR, "synth_latents")
        target_dir = os.path.join(PROCESSED_DIR, "target_latents")
        os.makedirs(synth_dir, exist_ok=True)
        
        # 1. 扫描待处理任务
        all_tasks = []
        for _, row in df.iterrows():
            file_id = os.path.splitext(os.path.basename(row['audio_filename']))[0]
            target_pt = os.path.join(target_dir, f"{file_id}.pt")
            synth_pt = os.path.join(synth_dir, f"{file_id}.pt")
            
            if os.path.exists(synth_pt) or not os.path.exists(target_pt):
                continue

            base_name = os.path.splitext(os.path.basename(row['midi_filename']))[0]
            midi_path = None
            for ext in ['.midi', '.mid', '.MIDI', '.MID']:
                test_path = os.path.join(MAESTRO_MIDI_ROOT, base_name + ext)
                if os.path.exists(test_path):
                    midi_path = test_path
                    break
            
            if midi_path:
                all_tasks.append((file_id, midi_path, target_pt, synth_pt))

        print(f"📋 剩余待处理任务: {len(all_tasks)}")
        if not all_tasks: return

        # 2. 分批处理逻辑
        for i in range(0, len(all_tasks), BATCH_CHUNK_SIZE):
            chunk = all_tasks[i : i + BATCH_CHUNK_SIZE]
            print(f"\n📦 正在处理批次 {i//BATCH_CHUNK_SIZE + 1} ({len(chunk)}首)...")
            
            # CPU 并行渲染
            with ProcessPoolExecutor(max_workers=NUM_CPU_WORKERS) as executor:
                futures = []
                for fid, m_path, t_pt, s_pt in chunk:
                    temp_wav = os.path.join(TEMP_DIR, f"temp_{fid}.wav")
                    futures.append(executor.submit(render_one_midi, m_path, temp_wav))
                
                # 等待本批次渲染全部完成
                results = [f.result() for f in futures]

            # GPU 串行编码（本批次）
            for (fid, m_path, t_pt, s_pt), success in zip(chunk, results):
                temp_wav = os.path.join(TEMP_DIR, f"temp_{fid}.wav")
                if success and os.path.exists(temp_wav):
                    try:
                        c_lat = self.encode_wav(temp_wav)
                        t_lat = torch.load(t_pt)
                        min_l = min(t_lat.shape[1], c_lat.shape[1])
                        torch.save(t_lat[:, :min_l], t_pt)
                        torch.save(c_lat[:, :min_l], s_pt)
                    except Exception as e:
                        print(f"❌ {fid} 编码失败: {e}")
                    finally:
                        if os.path.exists(temp_wav): os.remove(temp_wav)
                else:
                    print(f"⚠️ {fid} 渲染失败或文件丢失")

            # 批次间清理
            gc.collect()
            torch.cuda.empty_cache()
            print(f"✅ 批次进度: {min(i + BATCH_CHUNK_SIZE, len(all_tasks))}/{len(all_tasks)}")

        print("\n🎉 所有的任务处理完成！正在刷新清单...")
        self.refresh_manifests(df)

    def refresh_manifests(self, df):
        """
        扫描处理好的文件，根据 CSV 划分 train/validation/test 并生成包含 text 的 JSON
        """
        synth_dir = os.path.join(PROCESSED_DIR, "synth_latents")
        
        # 准备三个集合
        manifests = {'train': [], 'validation': [], 'test': []}

        print(">>> 正在扫描磁盘并构建最终清单...")
        for _, row in df.iterrows():
            file_id = os.path.splitext(os.path.basename(row['audio_filename']))[0]
            split = row['split'] # 从 CSV 获取划分信息
            
            # 检查这个文件是否真的处理成功并存在了
            synth_pt_path = os.path.join(synth_dir, f"{file_id}.pt")
            
            if os.path.exists(synth_pt_path):
                # ========================================================
                # 💡 这里就是生成 "text" 的地方！
                # ========================================================
                prompt = (
                    f"A high-quality acoustic piano performance of {row['canonical_title']} "
                    f"composed by {row['canonical_composer']}, recorded in a professional studio."
                )
                
                manifests[split].append({
                    "id": file_id,
                    "target_latent": f"target_latents/{file_id}.pt",
                    "synth_latent": f"synth_latents/{file_id}.pt",
                    "text": prompt # <--- 存入 JSON
                })

        # 写入三个文件
        for split_name, data_list in manifests.items():
            save_path = os.path.join(PROCESSED_DIR, f"{split_name}_manifest.json")
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(data_list, f, indent=4, ensure_ascii=False)
            print(f"✅ {split_name} 集清单已更新: {len(data_list)} 条记录")

if __name__ == "__main__":
    RobustPreprocessor().run()