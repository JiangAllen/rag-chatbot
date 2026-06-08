# news-rag-api

AI 問答後端服務，整合 Azure Cognitive Search、Neo4j 知識圖譜與多來源網路爬蟲，提供新聞 RAG 問答、Google 搜尋問答與 MBTI/HR 情境分析。

## 服務架構

```
├── pipline.py      FastAPI 主服務 (port 8888)
├── crawler.py      Flask 爬蟲服務 (port 4455)
│
├── core/
│   ├── service.py          Azure OpenAI / Cognitive Search / Blob Storage 封裝
│   ├── llm_initial.py      LLM 初始化（Azure OpenAI、本機模型）
│   ├── neo.py              Neo4j 連線與查詢
│   ├── prompt.py           Prompt 模板
│   └── processing.py       工具函式（日期處理、Google 搜尋、爬蟲）
│
├── tools/                  手動執行的獨立腳本
│   ├── datapreprocessing.py
│   ├── apsprocess.py
│   ├── ocr_method.py
│   └── reveng/
│
├── dataset/
│   ├── MBTI/
│   └── IMDB Movies/
│
└── model/                  本機模型
    ├── meta-llamaMeta-Llama-3-8B-Instruct/
    ├── phi-3-mini-4k-instruct/
    └── tinyLlama-1.1B-Chat-v1.0/
```

## API

### FastAPI — `pipline.py` (port 8888)

| Endpoint | Method | 說明 |
|----------|--------|------|
| `/chat` | POST | 新聞 RAG 問答。改寫問題 → Azure Cognitive Search 撈新聞 → GPT-4.1 串流回答 |
| `/crawl` | POST | Google 搜尋問答。搜尋網頁 → 補全內文 → LLM 整理回答 |
| `/graph` | POST | MBTI/HR GraphRAG。從 Neo4j 查詢性格資料 → GPT-4.1 串流回答 |

**Request body（所有 endpoint 共用）：**
```json
{
  "history": [
    { "user": "上一輪問題", "bot": "上一輪回答" },
    { "user": "這一輪問題" }
  ]
}
```

**Response：** `/chat` 與 `/graph` 為 `text/event-stream` 串流；`/crawl` 為 JSON。

### Flask — `crawler.py` (port 4455)

| Endpoint | Method | 說明 |
|----------|--------|------|
| `/webcrawler` | POST | 依 URL domain 爬取新聞標題與內文 |

支援來源：Reuters、CNBC、TechCrunch、The Verge、Tom's Hardware、Electrek、9to5Mac、9to5Google、AppleInsider、MacRumors、The Register、Yahoo JP、ZDNet JP、Sankei、Monoist、Newswitch、ddaily、Sedaily、Hankyung、ETNews

**Request body：**
```json
{ "url": "https://www.reuters.com/..." }
```

**Response：**
```json
{ "subject": "標題", "body": "內文" }
```

## 環境設定

複製 `config.py` 並填入實際金鑰：

| 變數 | 說明 |
|------|------|
| `openai_key` | Azure OpenAI API Key |
| `openai_resource` | Azure OpenAI 資源名稱 |
| `api_version` | Azure OpenAI API 版本 |
| `search_key` | Azure Cognitive Search API Key |
| `search_service` | Azure Cognitive Search 服務名稱 |
| `index_name` | 搜尋索引名稱 |
| `blob_key` | Azure Blob Storage Key |
| `storage_account` | Azure Storage Account 名稱 |
| `blob_container` | Blob Container 名稱 |
| `gemini_key` | Google Gemini API Key |
| `NEO4J_URI` | Neo4j 連線 URI |
| `NEO4J_AUTH` | Neo4j 帳號密碼 `("neo4j", "password")` |

## 安裝與啟動

```bash
pip install -r requirements.txt
```

啟動 FastAPI 主服務：
```bash
python pipline.py
# 或
uvicorn pipline:app --host 0.0.0.0 --port 8888
```

啟動爬蟲服務：
```bash
python crawler.py
```

## 依賴服務

- **Azure OpenAI** — GPT-4o-mini（改寫）、GPT-4.1（回答）、text-embedding-ada-002（向量搜尋）
- **Azure Cognitive Search** — 新聞全文語意搜尋索引
- **Azure Blob Storage** — 新聞原始文件儲存
- **Neo4j Aura** — MBTI 知識圖譜
- **Diffbot** — 無規則網頁內文擷取備援
