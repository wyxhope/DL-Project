import os
import torch
import torchaudio
import json
import mido
import numpy as np
import math
import pandas as pd
from midi2audio import FluidSynth
from stable_audio_tools.models.factory import create_model_from_config

class MelodyProcessor:
    def __init__(self, model_config_path, ckpt_path, soundfont_path, csv_path, maestro_root, device="cuda"):
        self.device = device
        self.sample_rate = 44100
        self.maestro_root = maestro_root
        
        # 1. 初始化 FluidSynth
        if not os.path.exists(soundfont_path):
            raise FileNotFoundError(f"找不到 SoundFont: {soundfont_path}")
        self.fs = FluidSynth(soundfont_path, sample_rate=self.sample_rate)
        
        # 2. 加载 CSV 建立 ID -> 原始 MIDI 路径的映射
        print(f"正在读取元数据索引: {csv_path}")
        df = pd.read_csv(csv_path)
        self.midi_path_map = {}
        for _, row in df.iterrows():
            file_id = os.path.splitext(os.path.basename(row['audio_filename']))[0]
            self.midi_path_map[file_id] = row['midi_filename']

        # 3. 构建精简版 VAE
        with open(model_config_path, 'r') as f:
            model_config = json.load(f)
        ae_config = {
            "model_type": "autoencoder",
            "sample_rate": 44100,
            "model": model_config["model"]["pretransform"]["config"],
        }
        self.vae = create_model_from_config(ae_config)

        # 4. 加载权重
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        vae_state_dict = {k[len("pretransform.model."):]: v for k, v in state_dict.items() if k.startswith("pretransform.model.")}
        self.vae.load_state_dict(vae_state_dict)
        self.vae.to(device).eval()
        for param in self.vae.parameters():
            param.requires_grad = False
        print("VAE 加载成功。")

    def _calculate_averages(self, file_id):
        """从原始 MAESTRO MIDI 中提取平均踏板和力度"""
        midi_rel_path = self.midi_path_map.get(file_id)
        if not midi_rel_path:
            return 0.5, 0.6
        
        full_path = os.path.join(self.maestro_root, midi_rel_path)
        try:
            mid = mido.MidiFile(full_path)
            p_vals, v_vals = [], []
            for track in mid.tracks:
                for msg in track:
                    if msg.type == 'control_change' and msg.control == 64:
                        p_vals.append(msg.value / 127.0)
                    if msg.type == 'note_on' and msg.velocity > 0:
                        v_vals.append(msg.velocity / 127.0)
            avg_p = sum(p_vals) / len(p_vals) if p_vals else 0.5
            avg_v = sum(v_vals) / len(v_vals) if v_vals else 0.6
            return float(avg_p), float(avg_v)
        except:
            return 0.5, 0.6

    @torch.no_grad()
    def _encode_chunked_safe(self, wav_path, chunk_sec=30):
        """分段编码并处理短片段 Padding"""
        waveform, sr = torchaudio.load(wav_path, backend="soundfile")
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        if waveform.shape[0] == 1: waveform = waveform.repeat(2, 1)
        
        chunk_size = chunk_sec * self.sample_rate
        total_samples = waveform.shape[-1]
        latents = []
        min_req = 44100 # 最小 1 秒

        for start in range(0, total_samples, chunk_size):
            end = min(start + chunk_size, total_samples)
            actual_len = end - start
            chunk = waveform[:, start:end]
            
            is_padded = False
            if actual_len < min_req:
                chunk = torch.nn.functional.pad(chunk, (0, min_req - actual_len))
                is_padded = True
            
            gpu_chunk = chunk.unsqueeze(0).to(self.device)
            res = self.vae.encode(gpu_chunk)
            # 修正逻辑：如果是张量直接用，不是则取 mean
            c_lat = res if isinstance(res, torch.Tensor) else res.mean()
            c_lat = c_lat.cpu()
            
            if is_padded:
                valid_len = math.ceil(actual_len / 2048)
                c_lat = c_lat[:, :, :valid_len]
            
            latents.append(c_lat)
            del gpu_chunk
            torch.cuda.empty_cache()
        return torch.cat(latents, dim=-1).squeeze(0)

    def process(self, melody_midi_dir, target_latent_dir, output_dir, descriptions_json):
        melody_out_dir = os.path.join(output_dir, "melody_latents")
        os.makedirs(melody_out_dir, exist_ok=True)

        with open(descriptions_json, 'r', encoding='utf-8') as f:
            descriptions_data = json.load(f)
            
        manifest = []
        midi_files = [f for f in os.listdir(melody_midi_dir) if f.endswith(('.mid', '.midi'))]
        print(f"开始渲染并提取 {len(midi_files)} 个旋律文件...")

        for i, f_name in enumerate(midi_files):
            file_id = os.path.splitext(f_name)[0]
            target_pt = os.path.join(target_latent_dir, f"{file_id}.pt")
            if not os.path.exists(target_pt):
                continue

            print(f"[{i+1}/{len(midi_files)}] 处理并对齐: {file_id}")
            temp_wav = f"temp_m_{file_id}.wav"
            try:
                # 1. 渲染旋律
                self.fs.midi_to_audio(os.path.join(melody_midi_dir, f_name), temp_wav)
                m_latent = self._encode_chunked_safe(temp_wav)
                
                # 2. 帧对齐
                t_latent = torch.load(target_pt)
                min_len = min(t_latent.shape[1], m_latent.shape[1])
                
                torch.save(t_latent[:, :min_len], target_pt)
                torch.save(m_latent[:, :min_len], os.path.join(melody_out_dir, f"{file_id}.pt"))
                
                # 3. 提取物理参数
                avg_p, avg_v = self._calculate_averages(file_id)
                
                # 4. 获取文本描述 (支持嵌套格式)
                desc_entry = descriptions_data.get(file_id, {})
                text_content = desc_entry.get('description', "Professional piano performance.") if isinstance(desc_entry, dict) else "Professional piano performance."

                manifest.append({
                    "id": file_id,
                    "target_latent": f"target_latents/{file_id}.pt",
                    "melody_latent": f"melody_latents/{file_id}.pt",
                    "avg_pedal": float(avg_p),
                    "avg_velocity": float(avg_v),
                    "text": str(text_content),
                    "length": int(min_len)
                })
            except Exception as e:
                print(f"  [Skip] {file_id} 失败: {e}")
            finally:
                if os.path.exists(temp_wav): os.remove(temp_wav)

        # 保存最终清单
        with open(os.path.join(output_dir, "train_manifest.json"), 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=4, ensure_ascii=False)
        print("所有数据处理完成！")

if __name__ == "__main__":
    # --- 路径配置 (请根据 AutoDL 实际路径修改) ---
    MODEL_DIR = "/root/autodl-tmp/stable_audio_model"
    MAESTRO_ROOT = "/root/autodl-tmp/maestro-v3.0.0"
    CSV_PATH = "/root/autodl-tmp/maestro-v3.0.0/maestro-v3.0.0.csv"
    SF2_PATH = "/usr/share/sounds/sf2/FluidR3_GM.sf2"
    
    processor = MelodyProcessor(
        model_config_path=os.path.join(MODEL_DIR, "model_config.json"), 
        ckpt_path=os.path.join(MODEL_DIR, "model.ckpt"), 
        soundfont_path=SF2_PATH,
        csv_path=CSV_PATH,
        maestro_root=MAESTRO_ROOT
    )
    
    processor.process(
        melody_midi_dir="./maestro_melodies_only/allmidi", # 你提取出的旋律 MIDI
        target_latent_dir="./processed/target_latents",     # 脚本 1 生成的目录
        output_dir="./processed",
        descriptions_json="maestro_descriptions.json"
    )
