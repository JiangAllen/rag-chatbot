from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from transformers import AutoTokenizer, AutoModelForCausalLM
from opencc import OpenCC
from config import *
import torch
import json
import time
import io

class model_application():
    def __init__(self, openai_client, chatgpt_deployment, bedrock_client, AWS_CLAUDE_DEPLOYMENT):
        self.openai_client = openai_client
        self.chatgpt_deployment = chatgpt_deployment
        self.bedrock_client = bedrock_client
        self.claude_deployment = AWS_CLAUDE_DEPLOYMENT
        # self.form_recognizer_client = DocumentAnalysisClient(
        #     endpoint="https://{}.cognitiveservices.azure.com/".format(AZURE_FORMRECOG_ACCOUNT),
        #     credential=AzureKeyCredential(AZURE_FORMRECOG_CREDENTIAL),
        #     headers={"x-ms-useragent": "azure-search-chat-demo/1.0.0"}
        # )

    def chunk_response_for_pdfdata(self, system_prompt):
        contents = [{"role": "system", "content": "{}".format(system_prompt)}, {"role": "user", "content": "help clean and organize the Context"}]
        response = self.openai_client.chat.completions.create(
            model=self.chatgpt_deployment,
            messages=contents,
            temperature=0.0,
            max_tokens=16384,
            presence_penalty=2.0,
            n=1
        )
        method = OpenCC("s2t")
        response.choices[0].message.content = method.convert(response.choices[0].message.content)
        del method
        return response.choices[0].message.content

    def openai_image(self, messages):
        chat_completion = self.openai_client.chat.completions.create(
            model=self.chatgpt_deployment,
            messages=messages,
            max_tokens=1024,
            stream=False
        )
        chat_completion.choices[0].message.content = OpenCC("s2t").convert(chat_completion.choices[0].message.content)
        return chat_completion.choices[0].message.content

    def openai_chunk(self, messages):
        chat_completion = self.openai_client.chat.completions.create(
            model=self.chatgpt_deployment,
            messages=messages,
            temperature=0.0,
            max_tokens=4096,
            presence_penalty=2.0,
            n=1
        )
        chat_completion.choices[0].message.content = OpenCC("s2t").convert(chat_completion.choices[0].message.content)
        return chat_completion.choices[0].message.content

    def openai_stream(self, messages):
        converter = OpenCC("s2t")
        stream = self.openai_client.chat.completions.create(
            model=self.chatgpt_deployment,
            messages=messages,
            temperature=0.0,
            max_tokens=4096,
            presence_penalty=2.0,
            stream=True,
            n=1
        )
        for s in stream:
            if s.choices and s.choices[0].delta.content:
                content = s.choices[0].delta.content
                time.sleep(0.005)
                if content not in ["", "@"]:
                    response = converter.convert(content)
                    yield f"data: {json.dumps({'response': response})}\n\n"
        yield "data: [DONE]\n\n"

    def bedrock_stream(self, sysmg, query_text):
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "temperature": 0,
            "top_p": 0.1,
            "system": sysmg,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": query_text
                        }
                    ]
                }
            ]
        }
        response = self.bedrock_client.invoke_model_with_response_stream(
            modelId=self.claude_deployment,
            body=json.dumps(body)
        )
        buffer = io.StringIO()
        for event in response.get("body"):
            chunk = json.loads(event["chunk"]["bytes"])
            if (chunk["type"] == "content_block_delta"):
                buffer.write(chunk["delta"]["text"])
            elif (chunk["type"] == "message_start"):
                inputTokens = chunk["message"]["usage"]["input_tokens"]
            elif (chunk["type"] == "message_stop"):
                outputTokens = chunk["amazon-bedrock-invocationMetrics"]["outputTokenCount"]
        return buffer.getvalue()

    def local_chunk(self, messages, temperature=0.7, max_tokens=1024):
        model = AutoModelForCausalLM.from_pretrained(llama_model, device_map={"": "cpu"}, torch_dtype="float32")
        tokenizer = AutoTokenizer.from_pretrained(llama_model)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) # "".join("{}: {}\n".format(m["role"], m["content"]) for m in messages)
        prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[-1]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate( # model
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
        )
        # response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
        return response
