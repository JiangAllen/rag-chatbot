# 🤖 RAG Chatbot API System

## 🧭 Overview

This project is a **prototype backend system** built with **FastAPI** or **Flask**, designed for **natural language processing (NLP)** tasks such as **news analysis**, **MBTI personality inference**, and **intelligent chatbot interactions**.  

It integrates **Large Language Models (LLMs)** with **Azure cloud services**, supporting information retrieval, summarization, and response generation.  

> ⚠️ **Disclaimer:**  
> This software is **not intended for production or commercial use**.  
> It is a **research and demonstration prototype**, developed to showcase the conceptual and technical differences between **traditional RAG (Retrieval-Augmented Generation)** and **GraphRAG** architectures, as well as their research applications.

---

## ✨ Features

- **Flexible Architecture** – Modular design for data processing and interpretation, providing high configurability and readability.  
- **Text Preprocessing** – Multi-strategy **embedding-based chunking** ensures contextual integrity during text segmentation.  
- **Intelligent Search** – Integrates **Azure Cognitive Search**, **Neo4j**, and external web search for comprehensive information retrieval and analysis.  
- **Multi-Model Support** – Configurable backends (OpenAI, Anthropic, or local LLMs) for text generation, embeddings, and more.  
- **Chunking & Streaming Responses** – Delivers high-quality, context-aware responses with optimized speed through chunk-based retrieval and streaming.  
- **Advanced Document & Image Parsing** – OCR-enabled framework for analyzing diverse document formats and images beyond plain text.

---

## 🏗️ Architecture

The backend is organized into several key modules:

| File | Description |
|------|--------------|
| **config.py** | Global configuration file for API keys and general parameters. |
| **prompt.py** | Prompt templates and feature engineering definitions. |
| **neo.py** | Automation module for **Neo4j** graph database operations. |
| **service.py** | Cloud service initialization and parameter configuration. |
| **llm_initial.py** | Model initialization module for cloud-based and local LLMs. |
| **datapreprocessing.py** | Data processing and conversion into database-compatible formats. |
| **processing.py** | Pipeline utility framework for crawlers and data format conversions. |
| **pipline.py** | Flask/FastAPI application containing research endpoints. |
| **Chunking-Text-Splitting.py** | Example script for semantic text chunking. |
| **yolo_clip_crop.py** | Small AI project utilizing **OpenCV** and **YOLO** for intelligent image cropping. |

---

## ⚡ Quick Start

### 🧩 Prerequisites
- Python **3.10+**
- API keys for your selected LLM provider and services
- Authentication credentials for **Neo4j**
- Python IDE (optional but recommended)

---

### 🐍 Manual Setup

####  1. Clone the repository
```bash
git clone <repository-url>
cd <your-project>
```

####  2. Create a virtual environment
```
python3 -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Linux/macOS
```

#### 3. Install dependencies
```
pip install -r requirements.txt
```

#### 4. Copy the example environment file and configure it
```
cp env.example .env
```

####  5. Set up API keys and run the server
```
echo "your-api-key"
python pipline.py
```

👉 The server will be available at http://localhost:8888

## 🐳 Docker Simple Deployment

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <your-project>
```

### 2. Verify Docker Installation
Make sure Docker is installed on your system and that the Docker service is running.

### 3. Build the Docker Image
From the project root directory, run:
```
docker build -t my-project .
```

### 4. Run the Docker Container
```
docker run -d -p 8000:8000 my-project
```

### 5. Access the Application
Once the container is running, open your browser and visit:
👉 http://localhost:8000

## ⚙️ API Endpoints
### POST /api/chat — News Text Analysis & Q&A
```
{
    "history": [
        {
            "user": "",
            "bot": ""
        },
        {
            "user": ""
        }
    ]
}
```
- user: User question
- bot: AI response

### POST /api/graph — MBTI Analysis & Q&A
```
{
    "history": [
        {
            "user": "",
            "hr": true
        }
    ]
}
```
- user: User question
- hr: Default is true

## 🧩 Troubleshooting
### Common Issues
- API Key Error: Ensure that the API key file exists and contains a valid key.
- CORS Error: Check your FRONTEND_URL configuration.
- Service Endpoint Error: Verify the model configuration in service.py.

## 📄 License
### This software is provided for research and demonstration purposes only. Do not use this code in production environments.

## 💬 Support
If you encounter issues or have questions:
- Create an issue in the repository
- Review the configuration settings
- Contact me at: wwlwyovwkn83999@gmail.com
