import torch
import torchaudio
import json
import os
from stable_audio_tools import create_model_from_config

class StableAudioVAEDecoder:
    def __init__(self, model_config_path, ckpt_path, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        
        # 1. 加载配置并构建精简版 Autoencoder
        with open(model_config_path, 'r') as f:
            model_config = json.load(f)

        ae_config = {
            "model_type": "autoencoder",
            "sample_rate": model_config["sample_rate"],
            "model": model_config["model"]["pretransform"]["config"],
        }
        self.vae = create_model_from_config(ae_config)

        # 2. 剥离前缀并加载权重
        print(f"正在加载权重: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)["state_dict"]
        vae_state_dict = {}
        prefix = "pretransform.model."
        for key, value in checkpoint.items():
            if key.startswith(prefix):
                vae_state_dict[key[len(prefix):]] = value

        self.vae.load_state_dict(vae_state_dict)
        self.vae.to(device).eval()
        self.sample_rate = model_config.get("sample_rate", 44100)
        print("VAE 解码器准备就绪。")

    @torch.no_grad()
    def decode_pt(self, pt_path, output_wav_path):
        """将 .pt 文件还原为音频"""
        # 加载 Latent: [64, L]
        latent = torch.load(pt_path)
        
        # 增加 Batch 维度: [1, 64, L]
        if latent.dim() == 2:
            latent = latent.unsqueeze(0)
            
        latent = latent.to(self.device)
        
        print(f"正在解码形状为 {latent.shape} 的 Latent...")
        
        # VAE 解码
        # 注意：如果 Latent 非常长（比如 5 分钟），解码也可能占用大量显存
        # 如果报错 OOM，请将 self.device 改为 "cpu"
        decoded_audio = self.vae.decode(latent)
        
        # 移除 Batch 维度并转回 CPU
        decoded_audio = decoded_audio.squeeze(0).cpu()
        
        # 保存音频
        torchaudio.save(output_wav_path, decoded_audio, self.sample_rate)
        print(f"还原成功！保存至: {output_wav_path}")

# --- 使用示例 ---
if __name__ == "__main__":
    MODEL_DIR = "/root/autodl-tmp/stable_audio_model"
    CONFIG_PATH = os.path.join(MODEL_DIR, "model_config.json")
    CKPT_PATH = os.path.join(MODEL_DIR, "model.ckpt")
    
    # 填入你想要测试的 .pt 文件路径
    TEST_PT = "./processed/melody_latents/MIDI-Unprocessed_06_R3_2011_MID--AUDIO_R3-D3_05_Track05_wav.pt" 
    OUTPUT_WAV = "verify_reconstruction.wav"

    if not os.path.exists(TEST_PT):
        print(f"错误: 找不到文件 {TEST_PT}")
    else:
        # 如果显存不够，请将 device 设为 "cpu"
        decoder = StableAudioVAEDecoder(CONFIG_PATH, CKPT_PATH, device="cuda")
        decoder.decode_pt(TEST_PT, OUTPUT_WAV)
