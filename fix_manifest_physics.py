import json
import os
import mido
import pandas as pd

# --- 配置路径 ---
MANIFEST_PATH = "./processed/train_manifest.json"
MAESTRO_CSV_PATH = "/root/autodl-tmp/maestro-v3.0.0/maestro-v3.0.0.csv" 
MAESTRO_ROOT = "/root/autodl-tmp/maestro-v3.0.0" 

def extract_midi_physics(midi_path):
    """精准提取平均踏板和力度"""
    try:
        mid = mido.MidiFile(midi_path)
        p_values = []
        v_values = []
        
        for track in mid.tracks:
            for msg in track:
                # 提取踏板 (CC64)
                if msg.type == 'control_change' and msg.control == 64:
                    p_values.append(msg.value / 127.0)
                # 提取击键力度 (Note On)
                if msg.type == 'note_on' and msg.velocity > 0:
                    v_values.append(msg.velocity / 127.0)
        
        # 计算平均值
        avg_p = sum(p_values) / len(p_values) if p_values else 0.5
        avg_v = sum(v_values) / len(v_values) if v_values else 0.6
        return float(avg_p), float(avg_v)
    except Exception as e:
        print(f"  [Error] 无法解析 {midi_path}: {e}")
        return 0.5, 0.6

def fix():
    # 1. 加载现有的清单
    with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    
    # 2. 加载 CSV 建立路径映射 (因为 MIDI 文件分布在不同年份文件夹)
    df = pd.read_csv(MAESTRO_CSV_PATH)
    # 建立 ID -> MIDI相对路径 的映射
    path_map = {}
    for _, row in df.iterrows():
        file_id = os.path.splitext(os.path.basename(row['audio_filename']))[0]
        path_map[file_id] = row['midi_filename']

    print(f"开始修复 {len(manifest)} 条数据的物理参数...")

    fixed_count = 0
    for item in manifest:
        file_id = item['id']
        midi_rel_path = path_map.get(file_id)
        
        if midi_rel_path:
            full_midi_path = os.path.join(MAESTRO_ROOT, midi_rel_path)
            if os.path.exists(full_midi_path):
                avg_p, avg_v = extract_midi_physics(full_midi_path)
                
                # 更新数值
                item['avg_pedal'] = avg_p
                item['avg_velocity'] = avg_v
                fixed_count += 1
                
                if fixed_count % 100 == 0:
                    print(f"已修复 {fixed_count} 条...")
            else:
                print(f"  [Missing] 找不到文件: {full_midi_path}")
        else:
            print(f"  [Unknown ID] 清单中的 ID {file_id} 不在 CSV 中")

    # 3. 保存修复后的清单
    with open(MANIFEST_PATH, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=4, ensure_ascii=False)
    
    print(f"\n修复完成！共更新 {fixed_count} 条数据。")

if __name__ == "__main__":
    fix()
