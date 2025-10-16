from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Form, File, UploadFile
from pydantic import BaseModel
from typing import List, Dict, Any
from typing import Sequence
from opencc import OpenCC
from prompt import *
from processing import *
from service import azure_service
from llm_initial import model_application
from neo import Neo4jConnection
import pandas as pd
import uvicorn
import re

class InputData(BaseModel):
    history: List[Dict[str, Any]]

def run_rag(history: Sequence[dict[str, str]]):
    azure = azure_service()
    history = [{**item, "user": full_to_half(item["user"]), "bot": full_to_half(item["bot"])} if "bot" in item else {**item, "user": full_to_half(item["user"])} for item in history]

    if len(history) >= 1:
        conversation = "".join("question: {},\nanswer: {},\n".format(h["user"], h["bot"]) if h.get("bot") else "question: {},\n".format(h["user"]) for h in history[:-1])
        conversation += "last question: {}\n".format(history[-1]["user"])
        print("conversation = \n{}".format(conversation))

        # rewrite
        llm = model_application(azure.openai_client, "gpt-4o-mini", None, None)
        message = get_messages_for_qa(system_message_search_query, conversation)
        query_text = llm.openai_chunk(message)
        print("after rewrite = {}".format(query_text))

        # time range
        llm = model_application(azure.openai_client, "gpt-4o-mini", None, None)
        one_month_ago_first_day, one_month_ago_last_day = get_first_and_last_day_three_months_ago(months=1)
        three_month_ago_first_day, three_month_ago_last_day = get_first_and_last_day_three_months_ago(months=3)
        lastweek_first_day, lastweek_last_day = get_first_and_last_day_lastweek_list()
        thisweek_first_day, thisweek_last_day = get_first_and_last_day_thissweek_list()
        message = get_messages_for_qa(system_message_for_time.format(
            year=datetime.strftime(datetime.today(),"%Y-%m-%d"),
            start=one_month_ago_first_day, end=one_month_ago_last_day,
            start2=three_month_ago_first_day, end2=three_month_ago_last_day,
            start3=lastweek_first_day, end3=lastweek_last_day,
            start4=datetime.strftime(datetime.now()-relativedelta(months=3),"%Y-%m-%d"), end4=datetime.strftime(datetime.today(),"%Y-%m-%d"),
            start5=datetime.strftime(datetime.now()-relativedelta(months=4),"%Y-%m-%d"), end5=datetime.strftime(datetime.today(),"%Y-%m-%d"),
            start6=datetime.strftime(datetime.today()-timedelta(days=14),"%Y-%m-%d"), end6=datetime.strftime(datetime.today(),"%Y-%m-%d"),
            start7=datetime.strftime(datetime.now()-relativedelta(months=12),"%Y"), end7=datetime.strftime(datetime.now()-relativedelta(months=12),"%Y"),
            start8=datetime.strftime(datetime.today(),"%Y-%m-%d"), end8=datetime.strftime(datetime.today(),"%Y-%m-%d"),
            start9=thisweek_first_day, end9=thisweek_last_day),
            query_text
        )
        query_text = llm.openai_chunk(message)
        print("after timerange = {}".format(query_text))

        query_text = OpenCC("s2t").convert(query_text.upper())

        if query_text.strip() == "0":
            query_text = history[-1]["user"]

        if "FROM" in query_text and "TO" in query_text:
            start = re.findall(r"[\d]{4}[\-][\d]{1,2}[\-][\d]{1,2}", query_text, re.I | re.M)[0]
            end = re.findall(r'[\d]{4}[\-][\d]{1,2}[\-][\d]{1,2}', query_text, re.I | re.M)[1]
            filters = f"datepublish ge {start + 'T00:00:00Z'} and datepublish le {end + 'T23:59:59Z'}"
            query_text = query_text[:query_text.find("FROM")]
        elif not "FROM" in query_text and not "TO" in query_text:
            start = datetime.now() - relativedelta(months=3)
            end = datetime.now()
            filters = f"datepublish ge {start.strftime('%Y-%m-%d') + 'T00:00:00Z'} and datepublish le {end.strftime('%Y-%m-%d') + 'T23:59:59Z'}"
        print("filters = {}".format(filters))

        search_result = azure.cognitive_search(query_text, filters, top_number)
        if not search_result.empty:
            search_result["datepublish"] = pd.to_datetime(search_result["datepublish"]).dt.strftime("%Y-%m-%d")
            search_result = search_result[~search_result["news_key"].str.contains("external|statistics")].reset_index(drop=True)
            search_result = search_result[search_result["@search.reranker_score"] >= threshold_score].reset_index(drop=True)[:8]
            results = news_key_filter(search_result[[]].to_json(orient="records", force_ascii=False))
            subject = generate_subject(search_result, [])
            if any(results):
                system_message = system_message_chat_conversation.format(injected_prompt="", follow_up_questions_prompt="", year=datetime.strftime(datetime.today(), "%Y-%m-%d"))
                message = get_messages_for_qa("{}\n\n資料來源:\n{}".format(system_message, results), query_text)
            else:
                system_message = system_message_chat_conversation_default.format(injected_prompt="", follow_up_questions_prompt="")
                message = get_messages_for_qa(system_message, query_text)
        elif search_result.empty:
            search_result = pd.DataFrame()
            results, subject = [], []
        # print(search_result.columns)
        # print(search_result[["datepublish", "subject", "@search.reranker_score"]])
        llm = model_application(azure.openai_client, "gpt-4.1", None, None)
        response = llm.openai_chunk(message)
        return subject, response

def run_goolge(history: Sequence[dict[str, str]]):
    azure = azure_service()
    history = [{**item, "user": full_to_half(item["user"]), "bot": full_to_half(item["bot"])} if "bot" in item else {**item, "user": full_to_half(item["user"])} for item in history]

    if len(history) >= 1:
        conversation = "".join("question: {},\nanswer: {},\n".format(h["user"], h["bot"]) if h.get("bot") else "question: {},\n".format(h["user"]) for h in history[:-1])
        conversation += "last question: {}\n".format(history[-1]["user"])
        print("conversation = \n{}".format(conversation))

        # rewrite
        llm = model_application(azure.openai_client, "gpt-4o-mini", None, None)
        message = get_messages_for_qa(system_message_search_query, conversation)
        query_text = llm.openai_chunk(message)
        print("after rewrite = {}".format(query_text))

        query_text = OpenCC("s2t").convert(query_text.upper())
        if query_text.strip() == "0":
            query_text = history[-1]["user"]

        json_result = google_search_crawler(query_text)
        for json in json_result:
            conversation = "subject={}, \nbody={}".format(json["title"].strip(), json["content"].strip())
            message = get_messages_for_qa(system_message_judge_news, conversation)
            result = llm.openai_chunk(message)
            if "NO" in result:
                json["content"] = call_diffbot(json["url"].strip())
        markdown_result = json_to_markdown(json_result)
        print(system_message_search_answer.format(year=datetime.strftime(datetime.today(), "%Y-%m-%d"), content=markdown_result))
        message = get_messages_for_qa(system_message_search_answer.format(year=datetime.strftime(datetime.today(), "%Y-%m-%d"), content=markdown_result), query_text)
        response = llm.openai_chunk(message)
        return response
        # else:
        #     print("初始回應已解答問題，無需搜尋。")
        #     return response

def run_graphrag(history: Sequence[dict[str, str]]):
    history = [{**item, "user": full_to_half(item["user"]), "bot": full_to_half(item["bot"])} if "bot" in item else {**item, "user": full_to_half(item["user"])} for item in history]
    if len(history) >= 1:
        # conversation = "".join("question: {},\nanswer: {},\n".format(h["user"], h["bot"]) if h.get("bot") else "question: {},\n".format(h["user"]) for h in history[:-1])
        # conversation += "last question: {}\n".format(history[-1]["user"])
        is_hr = history[-1]["hr"]
        user_question = history[-1]["user"]
        pattern = re.compile(r"情境:\s*(.*?)\s*問題:\s*(.*)", re.DOTALL)
        match = pattern.match(user_question)
        if match:
            situation = match.group(1).strip()
            question = match.group(2).strip()
            print("<情境>\n{}\n".format(situation))
            print("<問題>\n{}\n".format(question))
        n4c = Neo4jConnection(NEO4J_URI, NEO4J_AUTH)
        azure = azure_service()
        llm = model_application(azure.openai_client, "gpt-4.1", None, None)
        if is_hr is True:
            analyzed_mbti = re.findall(r'(?<![A-Z])(' + mbti_types + r')(?![A-Z])', situation)
            if len(analyzed_mbti) != 0:
                print("<已知> MBTI 的情境\n")
                print(analyzed_mbti)
                neosearches = n4c.query_neo4j(four_letter=analyzed_mbti, method="mbti")
                reference_neosearch = json.dumps([{k: " ".join(v) if isinstance(v, list) else v for k, v in ns.items()} for ns in neosearches], ensure_ascii=False)
                print("<參考資料>\n{}".format(reference_neosearch))
                message = get_messages_for_neo4j(system_message_hr_mbti_expert_answer_case1, situation, question, reference_neosearch)
                llm_answer = llm.openai_chunk(message)
                return reference_neosearch, llm_answer
            else:
                print("<未知> MBTI 的情境\n")
                query_embed = azure.openai_client.embeddings.create(model="text-embedding-ada-002", input=situation)
                vectsearches = n4c.query_neo4j(index_name="passage_embedding", embedding=query_embed.data[0].embedding, top_k=5, method="mbti_embedding")
                print("<向量搜尋>\n{}".format(vectsearches))
                ped_personality = list(set(source["source"].replace(".json", "") for source in vectsearches))
                print(ped_personality)
                neosearches = n4c.query_neo4j(four_letter=ped_personality, method="mbti")
                reference_neosearch = json.dumps([{k: " ".join(v) if isinstance(v, list) else v for k, v in ns.items()} for ns in neosearches], ensure_ascii=False)
                print("<參考資料>\n{}".format(reference_neosearch))
                message = get_messages_for_neo4j(system_message_hr_mbti_expert_answer_case2, situation, question, reference_neosearch)
                llm_answer = llm.openai_chunk(message)
                return reference_neosearch, llm_answer

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"], )

@app.post("/chat")
async def chatbot(input: InputData):
    source, result = run_rag(input.history)
    return {"response": result, "source": source}

@app.post("/crawl")
async def crawlbot(input: InputData):
    result = run_goolge(input.history)
    return {"response": result}

@app.post("/graph")
async def graphbot(input: InputData):
    # command, source, result = run_graphrag(input.history)
    # return {"neo4j command": command, "neo4j source": source, "response": result}
    source, result = run_graphrag(input.history)
    return {"source": source, "response": result}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)