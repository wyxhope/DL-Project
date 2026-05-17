import pandas as pd
import json
import os
import time
from openai import OpenAI

# --- 配置 ---
API_KEY = "sk-woFm37LWUHIjGIkazE4cPA"
BASE_URL = "https://llmapi.paratera.com"
MODEL_NAME = "DeepSeek-V3.2"
CSV_PATH = "maestro-v3.0.0/maestro-v3.0.0.csv"
OUTPUT_JSON = "maestro_descriptions.json"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def generate_description(composer, title):
    prompt = (
        f"As a musicologist, write a concise (10-20 words) audio description for the piano piece '{title}' by {composer}. "
        "You should explicitly include the music style in your output (e.g., Baroque, Clasical, Romantic, Impressionist, Modern)."
        "Do NOT use the composer or title in your output. Output ONLY the English description."
    )
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [Error] {title}: {e}")
        return None

def main():
    df = pd.read_csv(CSV_PATH)
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
            descriptions = json.load(f)
    else:
        descriptions = {}

    print(f"Generating description， {len(df)} pieces of data in total...")

    for i, row in df.iterrows():
        file_id = os.path.splitext(os.path.basename(row['audio_filename']))[0]
        
        if file_id in descriptions:
            continue # 跳过已存在的

        print(f"[{i+1}/{len(df)}] Generationg description for {row['canonical_title']}...")
        desc = generate_description(row['canonical_composer'], row['canonical_title'])
        
        if desc:
            descriptions[file_id] = {
                "composer": row['canonical_composer'],
                "title": row['canonical_title'],
                "description": desc
            }
        
        # 每 20 条保存一次
        if i % 20 == 0:
            with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                json.dump(descriptions, f, indent=4, ensure_ascii=False)
        
        time.sleep(0.2)

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(descriptions, f, indent=4, ensure_ascii=False)
    print("Description generation completed！")

if __name__ == "__main__":
    main()
