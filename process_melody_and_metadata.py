import os
import torch
import torchaudio
import json
import mido
import numpy as np
import math
from midi2audio import FluidSynth
from stable_audio_tools.models.factory import create_model_from_config

class MelodyProcessor:
    def __init__(self, model_config_path, ckpt_path, soundfont_path, device="cuda"):
        self.device = device
        self.sample_rate = 44100
        self.fs = FluidSynth(soundfont_path, sample_rate=44100)
        
        with open(model_config_path, 'r') as f:
            model_config = json.load(f)
        ae_config = {
            "model_type": "autoencoder", "sample_rate": 44100,
            "model": model_config["model"]["pretransform"]["config"],
        }
        self.vae = create_model_from_config(ae_config)

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)["state_dict"]
        vae_state_dict = {k[len("pretransform.model."):]: v for k, v in checkpoint.items() if k.startswith("pretransform.model.")}
        self.vae.load_state_dict(vae_state_dict)
        self.vae.to(device).eval()
        for param in self.vae.parameters(): param.requires_grad = False

    def _calculate_averages(self, midi_path):
        try:
            mid = mido.MidiFile(midi_path)
            p, v = [], []
            for msg in mid:
                if msg.type == 'control_change' and msg.control == 64: p.append(msg.value/127.0)
                if msg.type == 'note_on' and msg.velocity > 0: v.append(msg.velocity/127.0)
            return (sum(p)/len(p) if p else 0.5), (sum(v)/len(v) if vels else 0.6)
        except: return 0.5, 0.6

    @torch.no_grad()
    def _encode_chunked_safe(self, wav_path, chunk_sec=30):
        waveform, sr = torchaudio.load(wav_path, backend="soundfile")
        if sr != self.sample_rate: waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        if waveform.shape[0] == 1: waveform = waveform.repeat(2, 1)
        
        chunk_size = chunk_sec * self.sample_rate
        total_samples = waveform.shape[-1]
        latents = []
        min_required = 44100

        for start in range(0, total_samples, chunk_size):
            end = min(start + chunk_size, total_samples)
            actual_len = end - start
            chunk = waveform[:, start:end]
            
            is_padded = False
            if actual_len < min_required:
                padding_size = min_required - actual_len
                chunk = torch.nn.functional.pad(chunk, (0, padding_size))
                is_padded = True
            
            chunk_gpu = chunk.unsqueeze(0).to(self.device)
            
            # --- 修正后的逻辑 ---
            res = self.vae.encode(chunk_gpu)
            if isinstance(res, torch.Tensor):
                c_lat = res
            else:
                c_lat = res.mean()
            
            c_lat = c_lat.cpu()
            
            if is_padded:
                valid_len = math.ceil(actual_len / 2048)
                c_lat = c_lat[:, :, :valid_len]
            
            latents.append(c_lat)
            del chunk_gpu
            torch.cuda.empty_cache()
        return torch.cat(latents, dim=-1).squeeze(0)

    def process(self, melody_midi_dir, original_midi_dir, target_latent_dir, output_dir, descriptions_json):
        melody_out_dir = os.path.join(output_dir, "melody_latents")
        os.makedirs(melody_out_dir, exist_ok=True)
        with open(descriptions_json, 'r', encoding='utf-8') as f:
            descriptions = json.load(f)
            
        manifest = []
        midi_files = [f for f in os.listdir(melody_midi_dir) if f.endswith(('.mid', '.midi'))]

        for i, f_name in enumerate(midi_files):
            file_id = os.path.splitext(f_name)[0]
            target_pt = os.path.join(target_latent_dir, f"{file_id}.pt")
            if not os.path.exists(target_pt): continue

            print(f"[{i+1}/{len(midi_files)}] 处理并对齐: {file_id}")
            temp_wav = f"temp_m_{file_id}.wav"
            try:
                self.fs.midi_to_audio(os.path.join(melody_midi_dir, f_name), temp_wav)
                m_latent = self._encode_chunked_safe(temp_wav)
                
                t_latent = torch.load(target_pt)
                min_len = min(t_latent.shape[1], m_latent.shape[1])
                
                torch.save(t_latent[:, :min_len], target_pt)
                torch.save(m_latent[:, :min_len], os.path.join(melody_out_dir, f"{file_id}.pt"))
                
                p_avg, v_avg = self._calculate_averages(os.path.join(original_midi_dir, f_name))
                desc_entry = descriptions.get(file_id, {})
                desc = desc_entry.get('description', "Professional piano performance.") if isinstance(desc_entry, dict) else "Professional piano performance."

                manifest.append({
                    "id": file_id, "target_latent": f"target_latents/{file_id}.pt",
                    "melody_latent": f"melody_latents/{file_id}.pt",
                    "avg_pedal": float(p_avg), "avg_velocity": float(v_avg),
                    "text": str(desc), "length": int(min_len)
                })
            except Exception as e: print(f"  [Skip] {file_id}: {e}")
            finally: 
                if os.path.exists(temp_wav): os.remove(temp_wav)

        with open(os.path.join(output_dir, "train_manifest.json"), 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    MODEL_DIR = "/root/autodl-tmp/stable_audio_model"
    SF2_PATH = "/usr/share/sounds/sf2/FluidR3_GM.sf2"
    
    processor = MelodyProcessor(
        os.path.join(MODEL_DIR, "model_config.json"), 
        os.path.join(MODEL_DIR, "model.ckpt"), 
        SF2_PATH
    )
    processor.process(
        melody_midi_dir="./maestro_melodies_only/allmidi", 
        original_midi_dir="/root/autodl-tmp/maestro_original_midi",
        target_latent_dir="./processed/target_latents",
        output_dir="./processed",
        descriptions_json="maestro_descriptions.json"
    )
