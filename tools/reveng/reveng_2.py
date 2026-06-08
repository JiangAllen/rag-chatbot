import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from typing import List, Dict, Tuple
from pathlib import Path
from config import *
import google.generativeai as genai
import json
import time
import re
import os

GEMINI_API_KEY = gemini_key
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")
INPUT_DIR = "./dataset/PROCESS_TXT"
OUTPUT_DIR = "./dataset/FINEJSONL"

INDEPENDENT_PROMPT = """你現在是一位頂尖的「AI 微調資料生成器 (Fine-tuning Data Generator for Patent Domain)」，專門協助專利代理人與技術顧問製作高品質的訓練資料。

你的任務是進行**「反向專利工程 (Reverse Patent Engineering)」**：我會提供一份【標準答案】，也就是 `assistant` 欄位的內容（通常是一段正式撰寫好的「專利請求項」）。  
你的工作是根據該標準答案，**自動生成對應的 `system` 與 `user` 欄位內容**，讓這三者形成一組可用於專利文本生成模型的高品質訓練樣本。

--------------------------------
【生成規則】

1. **`system` 欄位（角色設定與任務描述）**
   - 角色必須是一位具備專業背景的知識產權領域人士，例如：「資深專利工程師」、「發明顧問」、「首席智權專員」等。
   - 角色的任務必須明確是：「將客戶的構想轉化為正式的專利文件」。
   - 每次生成時，請變化描述方式與措辭，避免出現完全相同的 `system` 內容。
   - 整體語氣需正式且具專業感，能反映出該角色的職能。

2. **`user` 欄位（客戶或工程師的非正式構想）**
   - 內容需為**自然口語化的描述**，像是工程師的隨手筆記或客戶在會議中口述的發明構想。
   - 內容應「完整對應」到【標準答案】中提到的所有**關鍵技術構件 (components)**及**其之間的關係 (relations)**。
   - **嚴禁**出現正式專利術語，例如：「係包括」、「設置於」、「電性連接」、「其特徵在於」等。
   - 請用日常語言替換，舉例如下：
       - 「係包括」 → 「要有...」、「裡面有...」
       - 「設置於」 → 「放在...上面」、「裝在...」
       - 「鄰近」 → 「在...旁邊」、「靠近...」
   - 敘述開頭需像「請求協助撰寫專利」的語氣，例如：
       - 「我有個新點子，幫我整理一下能不能申請專利？」
       - 「我畫了個草圖，大概是這樣的...」
       - 「我們最近在做一個模組，我想保護它的設計...」

--------------------------------
【輸出格式】

請嚴格依照以下格式輸出，**不得添加多餘文字、說明或對話**：

【生成的 System 內容】: <此處填入生成的 system 欄位內容>  
【生成的 User 內容】: <此處填入生成的 user 欄位內容>

--------------------------------
【任務開始】

【標準答案 (Assistant 內容)】:
{claim_text}
"""
DEPENDENT_PROMPT = """你現在是一位頂尖的「AI 微調資料生成專家」，專注於專利文本生成與強化學習資料構建。  
你的核心任務是執行**「反向工程 (Reverse Engineering)」**：  
我將提供【獨立項】原文與【依附項】原文（即 assistant 的回覆內容）。  
你必須據此自動生成與之匹配的 `system` 與 `user` 內容，用以構建微調樣本。

────────────────────────────
【生成規格定義】

1. **System 內容（角色設定與任務說明）**
   - 你必須設計出一段自然、具有變化度的 system 指令。
   - System 角色必須為以下之一：
     - 「專利撰寫助手」
     - 「專利修改專家」
   - 該角色的任務：根據使用者提供的【基礎請求項】與【新增限定】，撰寫一個新的依附項（即 assistant 生成的條文）。
   - System 內容應具有明確任務導向，並在不同樣本中使用不同措辭（避免重複模板化語句）。

2. **User 內容（結構化修改指令）**
   - 格式必須嚴格遵守下列結構，且以中文書寫：
     【基礎請求項】: <完整複製我提供的【獨立項】原文>  
     【新增限定】: <由你推導出的新增技術特徵，需以自然口語方式描述>
   - **推導方法：**
     - 對比【獨立項】與【依附項】之間的差異。
     - 擷取依附項「其中...」之後的新增技術特徵。
     - 將該新增內容轉化為自然語言指令，例如：
       - 專利文本差異：「其中，該元件之材質為銅。」
       - 對應的【新增限定】：  
         「請幫我加一個限定，說明那個元件的材料是『銅』。」
   - 請勿使用專利用語，改以人類工程師口吻表達修改需求。

────────────────────────────
【輸出格式規範】

請**嚴格**依照下列輸出格式，不得添加多餘敘述、註解或聊天文字：

【生成的 System 內容】: <放置你生成的 system 內容>  
【生成的 User 內容】: <放置你生成的 user 內容>

────────────────────────────
【任務開始】

【獨立項原文】:  
{base_claim_text}

【依附項 (Assistant 內容)】:  
{dependent_claim_text}
"""

def parse_claims(text: str) -> Tuple[List[Dict], List[Dict], Dict]:
    claims = re.findall(r'【請求項(\d+)】\s*(.+?)(?=【請求項\d+】|$)', text, re.DOTALL)
    independent_claims = []
    dependent_claims = []
    claims_dict = {}
    for claim_num, claim_text in claims:
        claim_text = claim_text.strip()
        full_text = f"【請求項{claim_num}】 {claim_text}"
        claims_dict[claim_num] = full_text
        if re.match(r'^如請求項', claim_text):
            depends_on = re.findall(r'如請求項\s*(\d+)', claim_text)
            dependent_claims.append({
                'number': claim_num,
                'text': full_text,
                'depends_on': depends_on[0] if depends_on else None
            })
        else:
            independent_claims.append({
                'number': claim_num,
                'text': full_text
            })
    return independent_claims, dependent_claims, claims_dict

def call_gemini(prompt: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"    API 呼叫失敗 (嘗試 {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2)
            else:
                raise
    return ""

def parse_gemini_response(response: str) -> Tuple[str, str]:
    system_match = re.search(r'【生成的 System 內容】:\s*(.+?)(?=【生成的 User 內容】)', response, re.DOTALL)
    user_match = re.search(r'【生成的 User 內容】:\s*(.+?)$', response, re.DOTALL)
    system_content = system_match.group(1).strip() if system_match else ""
    user_content = user_match.group(1).strip() if user_match else ""
    return system_content, user_content

def process_single_file(input_file: str, output_file: str) -> int:
    with open(input_file, 'r', encoding='utf-8') as f:
        patent_text = f.read()
    print(f"  正在解析請求項...")
    independent_claims, dependent_claims, claims_dict = parse_claims(patent_text)
    if not independent_claims and not dependent_claims:
        print(f"  ⚠️ 警告: 未找到任何請求項")
        return 0
    print(f"  找到 {len(independent_claims)} 個獨立項和 {len(dependent_claims)} 個依附項")
    training_samples = []
    if independent_claims:
        print(f"  處理獨立項...")
        for idx, claim in enumerate(independent_claims, 1):
            print(f"    [{idx}/{len(independent_claims)}] 請求項 {claim['number']}", end='')
            try:
                prompt = INDEPENDENT_PROMPT.format(claim_text=claim['text'])
                response = call_gemini(prompt)
                system_content, user_content = parse_gemini_response(response)
                sample = {
                    "messages": [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": claim['text']}
                    ]
                }
                training_samples.append(sample)
                print(" ✓")
                time.sleep(1)
            except Exception as e:
                print(f" ✗ (錯誤: {e})")

    if dependent_claims:
        print(f"  處理依附項...")
        for idx, claim in enumerate(dependent_claims, 1):
            print(f"    [{idx}/{len(dependent_claims)}] 請求項 {claim['number']}", end='')

            base_claim = claims_dict.get(claim['depends_on'], "")
            if not base_claim:
                print(f" ✗ (找不到基礎請求項 {claim['depends_on']})")
                continue
            try:
                prompt = DEPENDENT_PROMPT.format(
                    base_claim_text=base_claim,
                    dependent_claim_text=claim['text']
                )
                response = call_gemini(prompt)
                system_content, user_content = parse_gemini_response(response)
                sample = {
                    "messages": [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": claim['text']}
                    ]
                }
                training_samples.append(sample)
                print(" ✓")
                time.sleep(1)
            except Exception as e:
                print(f" ✗ (錯誤: {e})")

    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in training_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"  ✓ 完成！共生成 {len(training_samples)} 筆訓練資料")
    return len(training_samples)

def batch_process_patents(input_dir: str, output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    txt_files = list(Path(input_dir).glob("*.txt"))
    if not txt_files:
        print(f"⚠️ 在 {input_dir} 中未找到任何 .txt 檔案")
        return
    print(f"找到 {len(txt_files)} 個 TXT 檔案")
    print("=" * 80)
    total_samples = 0
    success_count = 0
    for idx, txt_file in enumerate(txt_files, 1):
        filename = txt_file.stem
        output_file = Path(output_dir) / f"{filename}.jsonl"
        print(f"\n[{idx}/{len(txt_files)}] 處理檔案: {txt_file.name}")
        print("-" * 80)
        try:
            sample_count = process_single_file(str(txt_file), str(output_file))
            total_samples += sample_count
            success_count += 1
            print(f"  輸出至: {output_file}")
        except Exception as e:
            print(f"  ✗ 處理失敗: {e}")
        print("-" * 80)
    print("\n" + "=" * 80)
    print("批次處理完成")
    print("=" * 80)
    print(f"成功處理: {success_count}/{len(txt_files)} 個檔案")
    print(f"總共生成: {total_samples} 筆訓練資料")
    print(f"輸出目錄: {output_dir}")

if __name__ == "__main__":
    if GEMINI_API_KEY == "your-api-key-here":
        print("❌ 請先設定你的 Gemini API Key！")
        print("在程式開頭的 GEMINI_API_KEY 變數中填入你的 API Key")
    else:
        if not os.path.exists(INPUT_DIR):
            print(f"❌ 輸入資料夾不存在: {INPUT_DIR}")
            print("請確認路徑是否正確")
        else:
            batch_process_patents(INPUT_DIR, OUTPUT_DIR)