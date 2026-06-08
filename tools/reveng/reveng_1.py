import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import json
import google.generativeai as genai
from pathlib import Path
from config import *
import time

GEMINI_API_KEY = gemini_key
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash")

INPUT_DIR = Path("./dataset/PROCESS_TXT")
OUTPUT_DIR = Path("./dataset/FINEJSONL")

REVERSE_PROMPT = """你的任務是：根據一段已知的 assistant 回答內容，逆向推敲出「當時的 system 指令」與「當時的 user 問題」，並以可泛用於所有領域的方式執行。

你將扮演一位「逆向提示詞工程師（Reverse Prompt Engineer）」，具備第一性原理分析、語意反推、角色建構、專業語氣重建能力。目標是：給定任意一段助手回覆，產生高可信度、具可重複泛化能力的「system prompt」及「user prompt」。

工作流程：

第一階段：語意解構
1. 深度解析 assistant 回答，辨識內容類型（技術、法律、醫學、學術、敘事、解釋、程式碼、推論、建議等）
2. 分析隱含的專業領域、術語、句式風格、推論方式，推測 system 角色設定
3. 從回答反推使用者可能的請求形式（定義、比較、說明、生成範例、解釋流程、分析、摘要、設計等）
4. 透過第一性原理拆解資訊結構，推測哪部分是直接回覆、哪部分是角色義務延伸

第二階段：重建 system prompt
1. 構造中立且高泛化的「角色與任務描述」
2. 明確敘述專業領域、語氣風格、輸出格式需求
3. 確保 system prompt 可適用於同領域任何問題，不依賴特定案例

第三階段：重建 user prompt
1. 以最小充分性原則重建使用者問題，使 assistant 回答可被自然解釋為直接回應
2. 保持自然務實語氣，不應看起來像逆向推敲產物
3. 若回答包含結構化內容、步驟、範例，需在 user prompt 明確加入促使此類輸出的需求

輸出格式（僅輸出以下三個欄位，不要有其他內容）：

```json
{
  "system": "重建的完整 system prompt",
  "user": "重建的完整 user prompt",
  "rationale": "反推思考過程摘要（高層次描述推理邏輯，不含 chain-of-thought）"
}
```

風格規範：
- 嚴禁模稜兩可、不具體、無條件描述
- 嚴禁空洞結論（如「希望能幫助你」）
- 使用自然語句，不使用生硬框架或標題式語氣
- 嚴格限制指示代名詞，盡量使用明確名詞
- 句子長度需有節奏變化

請針對以下 assistant 回答進行逆向推理："""

def read_txt_files(input_dir):
    txt_files = []
    if not input_dir.exists():
        print(f"錯誤：目錄不存在 {input_dir}")
        return txt_files
    for txt_file in input_dir.glob("*.txt"):
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    txt_files.append({
                        "filename": txt_file.name,
                        "content": content
                    })
                    print(f"✓ 讀取: {txt_file.name} ({len(content)} 字元)")
                else:
                    print(f"⚠ 跳過空檔案: {txt_file.name}")
        except Exception as e:
            print(f"✗ 讀取失敗 {txt_file.name}: {e}")
    return txt_files

def extract_json_from_response(text):
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        json_str = text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        json_str = text[start:end].strip()
    else:
        json_str = text.strip()
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"JSON 解析錯誤: {e}")
        print(f"原始回應: {text[:200]}...")
        return None

def generate_training_data(txt_files, output_filename="reverse_engineered_dataset.jsonl", delay=1.0):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / output_filename
    results = []
    failed_files = []
    print(f"\n{'=' * 60}")
    print(f"開始處理 {len(txt_files)} 個 txt 檔案")
    print(f"{'=' * 60}\n")
    for idx, file_info in enumerate(txt_files, 1):
        filename = file_info["filename"]
        answer = file_info["content"]
        print(f"[{idx}/{len(txt_files)}] 處理: {filename}")
        try:
            full_prompt = f"{REVERSE_PROMPT}\n\n{answer}"
            response = model.generate_content(full_prompt)
            parsed = extract_json_from_response(response.text)
            if parsed and all(k in parsed for k in ["system", "user"]):
                entry = {
                    "messages": [
                        {"role": "system", "content": parsed["system"]},
                        {"role": "user", "content": parsed["user"]},
                        {"role": "assistant", "content": answer}
                    ]
                }
                results.append(entry)
                print(f"  ✓ 成功生成")
                print(f"    System: {parsed['system'][:60]}...")
                print(f"    User: {parsed['user'][:60]}...")
            else:
                print(f"  ✗ 格式不完整，跳過")
                failed_files.append(filename)
            if idx < len(txt_files):
                time.sleep(delay)
        except Exception as e:
            print(f"  ✗ 處理失敗: {e}")
            failed_files.append(filename)
            continue
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"\n{'=' * 60}")
    print(f"處理完成！")
    print(f"  總檔案數: {len(txt_files)}")
    print(f"  成功生成: {len(results)} 筆")
    print(f"  失敗數量: {len(failed_files)} 筆")
    if failed_files:
        print(f"  失敗檔案: {', '.join(failed_files)}")
    print(f"  輸出位置: {output_path}")
    print(f"{'=' * 60}")
    return results

if __name__ == "__main__":
    print("開始讀取 txt 檔案...")
    txt_files = read_txt_files(INPUT_DIR)
    if not txt_files:
        print("未找到任何 txt 檔案，程式結束")
    else:
        print(f"\n共找到 {len(txt_files)} 個 txt 檔案\n")
        generate_training_data(
            txt_files,
            output_filename="reverse_engineered_dataset.jsonl",
            delay=1.0
        )