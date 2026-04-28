import pretty_midi
import os
import glob
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def extract_romantic_melody(input_midi_path, output_midi_path, velocity_weight=0.75, pitch_weight=0.25, time_window=0.05):
    """
    针对浪漫派/古典派钢琴曲的主旋律提取算法 (力度+音高双重加权)
    
    参数:
    - input_midi_path: 原始完整钢琴曲 MIDI 路径 (如 MAESTRO 数据集文件)
    - output_midi_path: 提取后只包含主旋律的 MIDI 保存路径
    - velocity_weight: 力度权重 (默认 0.75，浪漫派音乐中演奏力度是判断旋律的最核心指标)
    - pitch_weight: 音高权重 (默认 0.25，辅助判断，倾向于高音区)
    - time_window: 时间容差窗口 (默认 0.05秒/50毫秒，用于判定哪些音符是“同时按下”的或正在发声的)
    """
    try:
        # 1. 加载 MIDI 数据
        midi_data = pretty_midi.PrettyMIDI(input_midi_path)
    except Exception as e:
        print(f"读取 MIDI 失败 {input_midi_path}: {e}")
        return False

    all_notes =[]
    # 提取所有非打击乐轨道的音符
    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            all_notes.extend(instrument.notes)
            
    if not all_notes:
        print(f"跳过 {input_midi_path}：未检测到有效音符。")
        return False
        
    # 2. 按起始时间严格排序
    all_notes.sort(key=lambda x: x.start)
    
    # 3. 统计全局最大/最小值，用于归一化 (Normalization)
    min_pitch = min(n.pitch for n in all_notes)
    max_pitch = max(n.pitch for n in all_notes)
    min_vel = min(n.velocity for n in all_notes)
    max_vel = max(n.velocity for n in all_notes)
    
    # 防止分母为 0 的异常情况 (整首曲子音高/力度全一样)
    pitch_range = max(1, max_pitch - min_pitch)
    vel_range = max(1, max_vel - min_vel)

    melody_notes =[]
    
    # 4. 核心逻辑：遍历每一个音符，进行上下文打分
    for note in all_notes:
        # 【关键点】寻找当前音符的“竞争者”：
        # 竞争者包括：1. 正在持续发声的音符；2. 与当前音符几乎同时按下的音符 (50ms容差)
        concurrent_notes =[
            n for n in all_notes 
            if (n.start <= note.start < n.end) or (abs(n.start - note.start) < time_window)
        ]
        
        # 如果当前时刻只有这一个音符在发声，那它毫无疑问就是旋律音
        if len(concurrent_notes) <= 1:
            melody_notes.append(note)
            continue
            
        # 在竞争者中寻找得分最高的“王者”
        best_note = None
        highest_score = -1.0
        
        for c_note in concurrent_notes:
            # 归一化力度 (0.0 ~ 1.0)
            norm_vel = (c_note.velocity - min_vel) / vel_range
            # 归一化音高 (0.0 ~ 1.0)
            norm_pitch = (c_note.pitch - min_pitch) / pitch_range
            
            # 计算综合得分
            score = (velocity_weight * norm_vel) + (pitch_weight * norm_pitch)
            
            if score > highest_score:
                highest_score = score
                best_note = c_note
                
        # 【过滤与采纳】
        # 如果当前遍历的音符正好是当前时刻得分最高的音符，说明它是旋律，采纳它！
        if best_note == note:
            # 去重逻辑：防止和弦中出现完全相同的音高被重复添加
            if not melody_notes or not (
                abs(note.start - melody_notes[-1].start) < 0.01 and 
                note.pitch == melody_notes[-1].pitch
            ):
                melody_notes.append(note)

    # 5. 导出为新的 MIDI 文件
    melody_midi = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    melody_instrument = pretty_midi.Instrument(program=piano_program)
    
    # 写入提取出的旋律音符
    melody_instrument.notes = melody_notes
    melody_midi.instruments.append(melody_instrument)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_midi_path), exist_ok=True)
    melody_midi.write(output_midi_path)
    
    return True

# ==========================================
# 批量处理脚本 (用于构建 Diffusion 训练集)
# ==========================================
def process_single_file(file_info):
    """
    单文件处理的包装函数，方便多进程调用
    file_info 格式: (input_path, output_path)
    """
    input_path, output_path = file_info
    try:
        # 针对 MAESTRO，保持黄金比例权重
        success = extract_romantic_melody(
            input_midi_path=input_path,
            output_midi_path=output_path,
            velocity_weight=0.75, 
            pitch_weight=0.25
        )
        return success
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False

def batch_process_parallel(input_dir, output_dir, max_workers=None):
    """
    并行批量处理
    max_workers: 使用的 CPU 核心数，None 表示使用全部核心
    """
    # 1. 搜集所有待处理任务
    midi_files = glob.glob(os.path.join(input_dir, '**', '*.mid'), recursive=True)
    midi_files += glob.glob(os.path.join(input_dir, '**', '*.midi'), recursive=True)
    
    tasks = []
    for file_path in midi_files:
        rel_path = os.path.relpath(file_path, input_dir)
        out_path = os.path.join(output_dir, rel_path)
        tasks.append((file_path, out_path))

    print(f"找到 {len(tasks)} 个文件，准备使用 {os.cpu_count() if max_workers is None else max_workers} 个核心并行处理...")

    # 2. 使用进程池进行处理
    success_count = 0
    # 使用 tqdm 显示实时进度条
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务
        futures = {executor.submit(process_single_file, task): task for task in tasks}
        
        for future in tqdm(as_completed(futures), total=len(tasks), desc="提取进度"):
            if future.result():
                success_count += 1

    print(f"处理完成！成功: {success_count}/{len(tasks)}")

if __name__ == "__main__":
    MAESTRO_DIR = "./maestro-v3.0.0/allmidi" 
    MELODY_OUT_DIR = "./maestro_melodies_only/allmidi"
    
    # 测试单曲
    # extract_romantic_melody("test_chopin.mid", "test_chopin_melody.mid")
    
    # 批量处理
    batch_process_parallel(MAESTRO_DIR, MELODY_OUT_DIR, max_workers=8)
