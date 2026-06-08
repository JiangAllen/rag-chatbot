import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import *
import json
from openai import AzureOpenAI
from pathlib import Path
import time

# Azure OpenAI 設定
AZURE_ENDPOINT = "https://zeronetim.openai.azure.com/"
AZURE_API_KEY = openai_key
AZURE_API_VERSION = api_version
AZURE_DEPLOYMENT_NAME = "gpt-4.1"

# 初始化 Azure OpenAI 客戶端
client = AzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION
)

# 路徑設定
INPUT_DIR = Path("./dataset/PROCESS_TXT")
OUTPUT_DIR = Path("./dataset/FINEJSONL")

# 系統 prompt
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
    """
    讀取指定目錄下所有 txt 檔案內容

    Args:
        input_dir: 輸入目錄路徑

    Returns:
        list of dict: [{"filename": "xxx.txt", "content": "..."}]
    """
    txt_files = []

    if not input_dir.exists():
        print(f"錯誤：目錄不存在 {input_dir}")
        return txt_files

    for txt_file in input_dir.glob("*.txt"):
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:  # 只處理非空檔案
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
    """從 GPT 回應中提取 JSON 內容"""
    # 嘗試尋找 JSON 程式碼區塊
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        json_str = text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        json_str = text[start:end].strip()
    else:
        # 嘗試直接解析整個回應
        json_str = text.strip()

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"  JSON 解析錯誤: {e}")
        print(f"  原始回應: {text[:200]}...")
        return None


def call_azure_openai(prompt):
    """
    呼叫 Azure OpenAI API

    Args:
        prompt: 完整的提示詞

    Returns:
        str: API 回應內容
    """
    try:
        response = client.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000,
            response_format={"type": "json_object"}  # 強制 JSON 輸出
        )

        return response.choices[0].message.content

    except Exception as e:
        raise Exception(f"Azure OpenAI API 呼叫失敗: {e}")


def generate_training_data_per_file(txt_files, delay=1.0):
    """
    為每個 txt 檔案生成對應的 jsonl 檔案

    Args:
        txt_files: list of dict，每個元素包含 filename 和 content
        delay: API 呼叫間隔秒數（避免超過 rate limit）
    """
    # 確保輸出目錄存在
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_success = 0
    failed_files = []

    print(f"\n{'=' * 60}")
    print(f"開始處理 {len(txt_files)} 個 txt 檔案")
    print(f"使用模型: Azure OpenAI {AZURE_DEPLOYMENT_NAME}")
    print(f"輸出目錄: {OUTPUT_DIR}")
    print(f"{'=' * 60}\n")

    for idx, file_info in enumerate(txt_files, 1):
        filename = file_info["filename"]
        answer = file_info["content"]

        # 產生對應的 jsonl 檔名（將 .txt 改為 .jsonl）
        output_filename = filename.replace('.txt', '.jsonl')
        output_path = OUTPUT_DIR / output_filename

        print(f"[{idx}/{len(txt_files)}] 處理: {filename} -> {output_filename}")

        try:
            # 呼叫 Azure OpenAI API
            full_prompt = f"{REVERSE_PROMPT}\n\n{answer}"
            response_text = call_azure_openai(full_prompt)

            # 提取 JSON 資料
            parsed = extract_json_from_response(response_text)

            if parsed and all(k in parsed for k in ["system", "user"]):
                # 構建標準格式
                entry = {
                    "messages": [
                        {"role": "system", "content": parsed["system"]},
                        {"role": "user", "content": parsed["user"]},
                        {"role": "assistant", "content": answer}
                    ]
                }

                # 寫入單個 jsonl 檔案
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')

                total_success += 1
                print(f"  ✓ 成功生成")
                print(f"    System: {parsed['system'][:60]}...")
                print(f"    User: {parsed['user'][:60]}...")

            else:
                print(f"  ✗ 格式不完整，跳過")
                failed_files.append(filename)

            # 避免超過 API rate limit
            if idx < len(txt_files):
                time.sleep(delay)

        except Exception as e:
            print(f"  ✗ 處理失敗: {e}")
            failed_files.append(filename)
            continue

    # 輸出統計
    print(f"\n{'=' * 60}")
    print(f"處理完成！")
    print(f"  總檔案數: {len(txt_files)}")
    print(f"  成功生成: {total_success} 個 jsonl 檔案")
    print(f"  失敗數量: {len(failed_files)} 筆")
    if failed_files:
        print(f"  失敗檔案: {', '.join(failed_files)}")
    print(f"  輸出目錄: {OUTPUT_DIR}")
    print(f"{'=' * 60}")

    return total_success


# 使用範例
if __name__ == "__main__":
    print("開始讀取 txt 檔案...")

    # 讀取所有 txt 檔案
    txt_files = read_txt_files(INPUT_DIR)

    if not txt_files:
        print("未找到任何 txt 檔案，程式結束")
    else:
        print(f"\n共找到 {len(txt_files)} 個 txt 檔案\n")

        # 執行生成（每個 txt 產生對應的 jsonl）
        generate_training_data_per_file(
            txt_files,
            delay=1.0  # 每次 API 呼叫間隔 1 秒
        )