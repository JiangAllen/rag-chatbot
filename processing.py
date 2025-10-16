from dateutil.relativedelta import relativedelta
from datetime import date, datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
from pathlib import Path
from typing import List
from config import *
import pandas as pd
import calendar
import googlesearch
import requests
import json

def get_messages_for_qa(system_prompt: str, user_conv: str) -> list:
    message = [{"role": "system", "content": system_prompt}]
    message.append({"role": "user", "content": user_conv})
    return message

def get_messages_for_neo4j(my_prompt: str, situation: str, question: str, reference: str) -> list:
    message = [{"role": "system", "content": """你是一位資深人資顧問兼 MBTI 應用專家（角色），熟悉企業人才盤點、轉調評估、績效改善、職涯規劃與團隊組建。"""}]
    message.append({"role": "user", "content": my_prompt.format(situation=situation, question=question, reference=reference)})
    return message

def get_messages_for_images(system_prompt: str, image_base64_list: List[str], companyname: str) -> list:
    messages = [{"role": "system", "content": system_prompt}]
    user_content = [
        {"type": "text", "text": "請根據以下來自 {} 的多張流程圖圖片，整合成一份邏輯清晰的圖像流程說明文檔。".format(companyname)}
    ]
    for img in image_base64_list:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": img}
        })
    messages.append({"role": "user", "content": user_content})
    return messages

def get_messages_for_excel(system_prompt: str, companyname: str, excel_json: str) -> list:
    message = [{"role": "system", "content": system_prompt}]
    user_content = [
        {
            "type": "text",
            "text": "請根據以下來自 {} 的表格型式，整合成一份邏輯清晰的圖像流程說明文檔。".format(companyname)
        },
        {
            "type": "text",
            "text": excel_json
        }
    ]
    message.append({"role": "user", "content": user_content})
    return message

def call_diffbot(url):
    print("call diffbot ! ")
    response = requests.get(diffbot.format(url), headers={"accept": "application/json"})
    body = response.json().get("objects", [{}])[0].get("text", "內文未找到")
    return body

def fetch_article_content(json_unit):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"}
    try:
        # json_unit["content"] = call_diffbot(json_unit["url"])
        response = requests.get(json_unit["url"], headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        soups = (
                soup.find("div", class_="article-content") or
                soup.find("article") or
                soup.find("div", id="main-content") or
                soup.find("div", class_="post-body") or
                soup.find("div", {"id": "article-body"}) or
                soup.find("div", {"id": "articletxt"}) or
                soup.find("div", {"class": "article_body", "id": "articleBody"}) or
                soup.find("div", class_="container med post-content") or
                soup.find("div", id="body") or
                soup
        )
        paragraphs = soups.find_all("p")
        body = "\n".join(p.get_text().strip() for p in paragraphs[:5])
        if not body.strip():
            from readability import Document
            doc = Document(response.text)
            summary_html = doc.summary()
            summary_soup = BeautifulSoup(summary_html, "html.parser")
            body = "\n".join(p.get_text().strip() for p in summary_soup.find_all("p"))
        json_unit["content"] = body if body.strip() else call_diffbot(json_unit["url"])
    except Exception as crawl_err:
        json_unit["content"] = f"(抓取失敗：{crawl_err})"
    return json_unit

def google_search_crawler(query):
    results = []
    print("===== 開始 Google 搜尋 =====")
    try:
        one_year_ago = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        today = datetime.now().strftime("%Y-%m-%d")
        query += " after:{} before:{}".format(one_year_ago, today)
        search_results = list(googlesearch.search(query, advanced=True, num_results=5))
        if not search_results:
            return "未找到相關搜尋結果。"
        for k, v in enumerate(search_results):
            if any(domain in v.url for domain in blacklist_domains):
                continue
            try:
                head = requests.head(v.url, timeout=5)
                content_type = head.headers.get("Content-Type", "").lower()
            except Exception:
                content_type = ""
            if ("pdf" in content_type or any(v.url.lower().endswith(ext) for ext in [".pdf", ".ppt", ".doc", ".docx", ".xls", ".xlsx"])):
                continue
            if any(existing["url"] == v.url for existing in results):
                continue
            json_unit = {"title": v.title, "foreword": v.description, "content": "", "url": v.url}
            results.append(json_unit)
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(fetch_article_content, results))
        return results
    except Exception as e:
        return "搜尋出現錯誤"

def save_output(data, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def json_to_markdown(jformat):
    mformat = ""
    for k, v in enumerate(jformat, 1):
        mformat += "### Article {}\n".format(k)
        mformat += "- 標題：{}\n".format(v.get("title", ""))
        mformat += "- 前言：{}\n".format(v.get("foreword", ""))
        mformat += "- 內容：\n{}\n".format(v.get("content", "").strip())
        mformat += "- 來源：{}\n".format(v.get("url", ""))
        mformat += "---\n"
    return mformat

def excel_to_json(file):
    excel_file = pd.ExcelFile(file)
    sheets_data = {}
    for sheet in excel_file.sheet_names:
        df = pd.read_excel(file, sheet_name=sheet)
        sheets_data[sheet] = df.to_string(index=False)
    exjs = ""
    for sheet, content in sheets_data.items():
        exjs += "--- Sheet: {} ---\n{}\n\n".format(sheet, content)
    print(exjs)
    return exjs

def cypher_to_markdown(data):
    if not data:
        return "查無資料"
    headers = data[0].keys()
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for d in data:
        lines.append("| " + " | ".join(str(d[h]) if d[h] is not None else "" for h in headers) + " |")
    return "\n".join(lines)

def mbti_to_markdown(data):
    md_output = []
    for idx, ns in enumerate(data, start=1):
        node = ns["node"]
        score = ns["score"]
        text = node.get("text", "").strip()
        article_md = "# 文章{} (相似度: {:.4f})\n{}\n".format(idx, score, text)
        md_output.append(article_md)
    return "\n\n".join(md_output)

def get_first_and_last_day_three_months_ago(months):
    months_ago = datetime.now() - relativedelta(months=months)
    year = months_ago.year
    month = months_ago.month
    first_day = datetime(year, month, 1)
    last_day = datetime(year, month, calendar.monthrange(year, month)[1])
    first_day = datetime.strftime(first_day,"%Y-%m-%d")
    last_day = datetime.strftime(last_day,"%Y-%m-%d")
    del months_ago, year, month
    return first_day, last_day

def get_first_and_last_day_lastweek_list():
    year, week_num, day_of_week = date.today().isocalendar()
    lastweek_list = []
    for i in range(7):
        lastweek_list.append(str(date.today()-timedelta(days=i+day_of_week+1)))
    del year, week_num, day_of_week
    return min(lastweek_list), max(lastweek_list)

def get_first_and_last_day_thissweek_list():
    year, week_num, day_of_week = date.today().isocalendar()
    thisweek_list = []
    for i in range(day_of_week):
        thisweek_list.append(str(date.today()-timedelta(days=i)))
    del year, week_num, day_of_week
    return min(thisweek_list), max(thisweek_list)

def full_to_half(inputs):
    return "".join(
        chr(0x0020 if ord(c) == 0x3000 else ord(c) - 0xFEE0 if 0xFF01 <= ord(c) <= 0xFF5E else ord(c))
        for c in inputs
    )

def news_key_filter(NKS):
    import ast
    NKS = ast.literal_eval(NKS)
    for n in NKS:
        if n["news_key"] != "":
            n["news_key"] = "-".join(n["news_key"].split("-")[:-1])
    return str(NKS)

def generate_subject(k, file_key_words):
    subject = []
    for i in range(len(k["sourcefile"])):
        if "col" in k["news_key"][i]:
            col_key = k["news_key"][i].split("-")[0]
            col_key = col_key.split("_")[1]
            subject.append({"news_key": col_key, "filename": "https://www.digitimes.com.tw/col/article.asp?id={}".format(col_key), "title": k["subject"][i],"type": k["sourcetype"][i]})
            del col_key
        elif "external" in k["news_key"][i]:
            subject.append({"news_key": k["news_key"][i].split("-")[0], "filename": k["reporter"][i], "title": k["subject"][i],"type": "external"})
        elif k["news_key"][i].count("-") == 1 and not any(j in k["news_key"][i] for j in file_key_words):
            subject.append({"news_key": k["news_key"][i].split("-")[0], "filename": "https://www.digitimes.com.tw/tech/dt/n/shwnws.asp?id={}".format(k["news_key"][i].split("-")[0]), "title": k["subject"][i],"type": k["sourcetype"][i]})
        elif k["news_key"][i].count("-") > 1 and not any(j in k["news_key"][i] for j in file_key_words):
            if k["news_key"][i].count("-") == 4:
                subject.append({"news_key": "-".join(k["news_key"][i].split("-")[:-3]), "filename": "https://www.digitimes.com.tw/tech/rpt/rpt_slideshow_2020.asp?v={}&seq={}".format("-".join(k["news_key"][i].split("-")[:-3]), k["news_key"][i].split("-")[-2]), "title": k["subject"][i]+"#"+k["news_key"][i].split("-")[-2],"type": k["sourcetype"][i]})
            else:
                subject.append({"news_key": "-".join(k["news_key"][i].split("-")[:-1]), "filename": "https://www.digitimes.com.tw/tech/rpt/rpt_show.asp?v={}".format("-".join(k["news_key"][i].split("-")[:-1])), "title": k["subject"][i],"type": k["sourcetype"][i]})
        else:
            pass
    subject = pd.DataFrame(subject)
    subject = subject.drop_duplicates(subset = ["title"]).reset_index(drop=True)
    subject = subject.to_dict("records")
    return subject