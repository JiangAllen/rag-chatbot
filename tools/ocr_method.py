"""
PDF OCR 文字解析工具 - 簡化版
解析所有頁面 → 合併成一個字串 → 移除空白 → 存成 txt
"""

import os
from pathlib import Path
from pdf2image import convert_from_path
import re


def clean_text(text):
    """移除文字之間的空白間隔"""
    # 移除字與字之間的空白（保留換行）
    text = re.sub(r'([^\s\n])\s+([^\s\n])', r'\1\2', text)
    # 移除多餘的空行
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    return text.strip()

# ===== 方法 1: Tesseract OCR =====
def parse_with_tesseract(pdf_path, poppler_path=None):
    """
    需要安裝:
    pip install pytesseract pdf2image pillow
    還需安裝 Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
    """
    import pytesseract

    print("正在轉換 PDF 為圖片...")
    images = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)

    all_text = []
    for page_num, image in enumerate(images, start=1):
        print(f"正在處理第 {page_num}/{len(images)} 頁...")
        text = pytesseract.image_to_string(image, lang='chi_tra+eng', config='--psm 6')
        all_text.append(text)

    return '\n\n'.join(all_text)


# ===== 方法 2: EasyOCR =====
def parse_with_easyocr(pdf_path, poppler_path=None):
    """
    需要安裝: pip install easyocr pdf2image pillow
    """
    import easyocr

    print("正在載入 EasyOCR 模型...")
    reader = easyocr.Reader(['ch_tra', 'en'], gpu=False)

    print("正在轉換 PDF 為圖片...")
    images = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)

    all_text = []
    for page_num, image in enumerate(images, start=1):
        print(f"正在處理第 {page_num}/{len(images)} 頁...")

        # 暫存圖片
        temp_path = f"temp_page_{page_num}.png"
        image.save(temp_path)

        # OCR
        result = reader.readtext(temp_path)
        text = '\n'.join([item[1] for item in result])
        all_text.append(text)

        # 刪除暫存
        os.remove(temp_path)

    return '\n\n'.join(all_text)


# ===== 方法 3: PaddleOCR =====
def parse_with_paddleocr(pdf_path, poppler_path=None):
    """
    需要安裝: pip install paddleocr pdf2image pillow
    """
    from paddleocr import PaddleOCR

    print("正在載入 PaddleOCR 模型...")
    ocr = PaddleOCR(use_angle_cls=True, lang='chinese_cht') # , use_gpu=False

    print("正在轉換 PDF 為圖片...")
    images = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)

    all_text = []
    for page_num, image in enumerate(images, start=1):
        print(f"正在處理第 {page_num}/{len(images)} 頁...")

        # 暫存圖片
        temp_path = f"temp_page_{page_num}.png"
        image.save(temp_path)

        # OCR
        result = ocr.ocr(temp_path, cls=True)

        text_lines = []
        if result and result[0]:
            for line in result[0]:
                text_lines.append(line[1][0])

        all_text.append('\n'.join(text_lines))

        # 刪除暫存
        os.remove(temp_path)

    return '\n\n'.join(all_text)


# ===== 主程式 =====
if __name__ == "__main__":
    pdf_path = "./dataset/PATENT/TA001016706_AN__1_113100101_202528752.pdf"
    output_txt = "output_ocr_result.txt"  # 輸出檔名

    # Windows 用戶：指定 poppler 路徑（如果需要）
    poppler_path = None  # 例如: r"C:\Program Files\poppler\Library\bin"

    # 檢查檔案
    if not os.path.exists(pdf_path):
        print(f"❌ 找不到 PDF 檔案: {pdf_path}")
        exit(1)

    print("=" * 60)
    print("PDF OCR 文字解析工具")
    print("=" * 60)
    print("\n選擇 OCR 引擎:")
    print("1. Tesseract OCR")
    print("2. EasyOCR (推薦)")
    print("3. PaddleOCR (中文最強)")

    choice = input("\n請輸入選項 (1/2/3): ").strip()

    # Windows 詢問 poppler 路徑
    if poppler_path is None and os.name == 'nt':
        print("\n⚠️  Windows 提醒: 如果尚未安裝 poppler，請輸入路徑")
        print("例如: C:\\Program Files\\poppler\\Library\\bin")
        print("已安裝則直接按 Enter")
        user_path = input("\nPoppler 路徑: ").strip()
        if user_path:
            poppler_path = user_path

    try:
        # 執行 OCR
        if choice == "1":
            full_text = parse_with_tesseract(pdf_path, poppler_path)
        elif choice == "2":
            full_text = parse_with_easyocr(pdf_path, poppler_path)
        elif choice == "3":
            full_text = parse_with_paddleocr(pdf_path, poppler_path)
        else:
            print("❌ 無效選項")
            exit(1)

        # 清理文字（移除空白）
        print("\n正在清理文字...")
        cleaned_text = clean_text(full_text)

        # 印出結果
        print("\n" + "=" * 60)
        print("解析結果:")
        print("=" * 60)
        print(cleaned_text)

        # 儲存成 txt
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)

        print("\n" + "=" * 60)
        print(f"✅ 完成！已儲存至: {output_txt}")
        print(f"📄 總字數: {len(cleaned_text)} 字")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()