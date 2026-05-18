import torch
import torchaudio
import json
import os
import math
from stable_audio_tools import create_model_from_config

class TargetLatentExtractor:
    def __init__(self, model_config_path, ckpt_path, device="cuda"):
        self.device = device
        self.sample_rate = 44100
        
        with open(model_config_path, 'r') as f:
            model_config = json.load(f)
        ae_config = {
            "model_type": "autoencoder",
            "sample_rate": 44100,
            "model": model_config["model"]["pretransform"]["config"],
        }
        self.vae = create_model_from_config(ae_config)

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)["state_dict"]
        vae_state_dict = {k[len("pretransform.model."):]: v for k, v in checkpoint.items() if k.startswith("pretransform.model.")}
        self.vae.load_state_dict(vae_state_dict)
        self.vae.to(device).eval()
        
        for param in self.vae.parameters():
            param.requires_grad = False
        print("Target VAE 加载成功。")

    @torch.no_grad()
    def process(self, input_dir, output_dir, chunk_sec=30):
        os.makedirs(output_dir, exist_ok=True)
        files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]
        chunk_size = chunk_sec * self.sample_rate
        min_required = 44100 

        for i, f in enumerate(files):
            file_id = os.path.splitext(f)[0]
            save_path = os.path.join(output_dir, f"{file_id}.pt")
            if os.path.exists(save_path): continue

            print(f"[{i+1}/{len(files)}] 提取 Target: {file_id}")
            
            waveform, sr = torchaudio.load(os.path.join(input_dir, f), backend="soundfile")
            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
            if waveform.shape[0] == 1: waveform = waveform.repeat(2, 1)
            elif waveform.shape[0] > 2: waveform = waveform[:2, :]

            total_samples = waveform.shape[-1]
            latents_list = []

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
                
                res = self.vae.encode(chunk_gpu)
                if isinstance(res, torch.Tensor):
                    chunk_latent = res # 如果是张量，直接用
                else:
                    chunk_latent = res.mean() # 只有是分布对象时才调用 .mean()
                
                chunk_latent = chunk_latent.cpu()
                
                if is_padded:
                    valid_len = math.ceil(actual_len / 2048)
                    chunk_latent = chunk_latent[:, :, :valid_len]
                
                latents_list.append(chunk_latent)
                del chunk_gpu
                torch.cuda.empty_cache()

            full_latent = torch.cat(latents_list, dim=-1).squeeze(0)
            torch.save(full_latent, save_path)

if __name__ == "__main__":
    MODEL_DIR = "/root/autodl-tmp/stable_audio_model"
    extractor = TargetLatentExtractor(
        os.path.join(MODEL_DIR, "model_config.json"), 
        os.path.join(MODEL_DIR, "model.ckpt")
    )
    extractor.process("/root/autodl-tmp/maestro_full_audio", "/root/autodl-tmp/processed/target_latents")
