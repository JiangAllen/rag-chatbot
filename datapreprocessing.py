from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from docling_core.types import DoclingDocument
from docling.datamodel.base_models import InputFormat
from docling.document_converter import (DocumentConverter, PdfFormatOption, WordFormatOption)
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer
from unstructured.partition.pdf import partition_pdf
from pptx import Presentation
from pathlib import Path
from PIL import Image
from io import BytesIO, StringIO
from typing import List, Dict, Optional
from tqdm import tqdm
from llm_initial import model_application
from pdf2image import convert_from_path
# from transformers import AutoModel, AutoTokenizer
# from sentence_transformers import SentenceTransformer, models
from neo import Neo4jConnection
from prompt import *
from config import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import comtypes.client
import pdfplumber
import pytesseract
import subprocess
import time
import fitz
import html
import json
import math
import glob
import re
import io
import os

# tokenizer = AutoTokenizer.from_pretrained("moka-ai/m3e-large")
# transformer = models.Transformer("moka-ai/m3e-large", max_seq_length=512)
# pooling = models.Pooling(transformer.get_word_embedding_dimension())
# m3e_large = SentenceTransformer(modules=[transformer, pooling])

class datapreprocessing:
    def __init__(self):
        self.llm = model_application(None, None, None, None)

    def convert_pdf_image(self, file_path, output_dir):
        os.environ["PATH"] = poppler_bin + os.pathsep + tesseract_path + os.pathsep + os.environ.get("PATH", "")
        images = convert_from_path(str(file_path), dpi=300)
        for i, img in enumerate(images):
            img.save(os.path.join(output_dir, "{}_page_{}.png".format(file_path.stem, i+1)))

    def analyze_pdf_with_pypdfloader(self, path):
        loader = PyPDFLoader(path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)
        for d in docs:
            print(d)

    def analyze_pdf_with_docling(self, path):
        converter = DocumentConverter()
        result = converter.convert(path)
        print("text = \n{}", format(result.document.export_to_text()))
        print("dict = \n{}".format(result.document.export_to_dict()))
        print("markdown = \n{}".format(result.document.export_to_markdown()))
        print("html = \n{}".format(result.document.export_to_html()))
        print("exportoken = \n{}".format(result.document.export_to_doctags()))
        print("element_tree = \n{}".format(result.document.print_element_tree()))
        print(result.document._export_to_indented_text(max_text_len=16))

        # result.document.save_as_json(filename=Path("./docs/test.json"))
        # result.document.save_as_markdown(filename=Path("./docs/test.md"))
        # result.document.save_as_html(filename=Path("./docs/test.html"))
        # result.document.save_as_document_tokens(filename=Path("./docs/test.tokens"))
        # result.document.save_as_yaml(filename=Path("./docs/test.yaml"))

    def split_text(self, text):
        SENTENCE_ENDINGS = ["。", "!", "?", "."]
        WORDS_BREAKS = ["，", ",", "；", ":", " ", "(", ")", "[", "]", "{", "}", "\t", "\n", ";"]
        all_text = text
        length = len(all_text)
        start = 0
        end = length
        if length > 1000:
            MAX_SECTION_LENGTH = int(length / (round(length / 1000)))
            SPLIT_LIMIT = 100
            while start < length:
                last_word = -1
                end = start + MAX_SECTION_LENGTH
                if end > length:
                    end = length
                else:
                    while end < length and (end - start - MAX_SECTION_LENGTH) < SPLIT_LIMIT and all_text[end] not in SENTENCE_ENDINGS:
                        if all_text[end] in WORDS_BREAKS:
                            last_word = end
                        end += 1
                    if end < length and all_text[end] not in SENTENCE_ENDINGS and last_word > 0:
                        end = last_word

                last_word = -1
                while start > 0 and start > end - MAX_SECTION_LENGTH - 2 * SPLIT_LIMIT and all_text[start] not in SENTENCE_ENDINGS:
                    if all_text[start] in WORDS_BREAKS:
                        last_word = start
                    start -= 1
                if all_text[start] not in SENTENCE_ENDINGS and last_word > 0:
                    start = last_word

                section_text = all_text[start:end]
                section_text_len = len(section_text)
                start += section_text_len
                yield (section_text)
            del all_text
            del start
            del end
            del section_text_len
            del length
        else:
            MAX_SECTION_LENGTH = 1000
            SPLIT_LIMIT = 100
            while start < length:
                last_word = -1
                end = start + MAX_SECTION_LENGTH
                if end > length:
                    end = length
                else:
                    while end < length and (end - start - MAX_SECTION_LENGTH) < SPLIT_LIMIT and all_text[end] not in SENTENCE_ENDINGS:
                        if all_text[end] in WORDS_BREAKS:
                            last_word = end
                        end += 1
                    if end < length and all_text[end] not in SENTENCE_ENDINGS and last_word > 0:
                        end = last_word
                last_word = -1
                while start > 0 and start > end - MAX_SECTION_LENGTH - 2 * SPLIT_LIMIT and all_text[start] not in SENTENCE_ENDINGS:
                    if all_text[start] in WORDS_BREAKS:
                        last_word = start
                    start -= 1
                if all_text[start] not in SENTENCE_ENDINGS and last_word > 0:
                    start = last_word
                section_text = all_text[start:end]
                section_text_len = len(section_text)
                start += section_text_len
                yield (section_text)
            del all_text
            del start
            del end
            del section_text_len
            del length

    def split_sentences(self, text):
        sentences = re.split(r"[。；？！\n]+", text)
        sentences = [s for s in sentences if s and s.strip()]
        return sentences

    def combine_sentences(self, sentences, buffer_size=1):
        combined_sentences = [
            ' '.join(sentences[j]["sentence"] for j in range(max(i - buffer_size, 0), min(i + buffer_size + 1, len(sentences)))) for i in range(len(sentences))
        ]
        for i, combined_sentence in enumerate(combined_sentences):
            sentences[i]["combined_sentence"] = combined_sentence
        return sentences

    def convert_to_vector(self, text):
        embeddings = m3e_large.encode([t["combined_sentence"] for t in text])
        for i, text in enumerate(text):
            text["combined_sentence_embedding"] = embeddings[i]
        return text

    def cosine_similarity_custom(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)

    def calculate_cosine_distances(self, embeddings):
        print(embeddings)
        distances = []
        for i in range(len(embeddings) - 1):
            embedding_current = embeddings[i]["combined_sentence_embedding"]
            embedding_next = embeddings[i + 1]["combined_sentence_embedding"]
            similarity = self.cosine_similarity_custom(embedding_current, embedding_next)
            distance = 1 - similarity
            distances.append(distance)
            embeddings[i]["distance_to_next"] = distance
        return distances, embeddings

    def semantic_chunk_split(self, text):
        single_sentences_list = self.split_sentences(text)
        for sg in single_sentences_list:
            print(sg)
        sentences = [{"index": i, "sentence": s} for i, s in enumerate(single_sentences_list)]
        sentences = self.combine_sentences(sentences)
        sentences = self.convert_to_vector(sentences)
        print(sentences)
        distances, sentences = self.calculate_cosine_distances(sentences)
        breakpoint_distance_threshold = np.percentile(distances, 80)
        indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold]
        start_index = 0
        chunks = []
        for index in indices_above_thresh:
            end_index = index
            group = sentences[start_index:end_index + 1]
            combined_text = ' '.join([d["sentence"] for d in group])
            chunks.append(combined_text)
            start_index = index + 1
        if start_index < len(sentences):
            combined_text = ' '.join([d["sentence"] for d in sentences[start_index:]])
            chunks.append(combined_text)
        for i, c in enumerate(chunks):
            print("#{} {}".format(i + 1, c))
            # yield c

    def get_pdfdata(self, file, is_image, is_table):
        print(is_image, is_table)
        data = {"body": []}
        def table_to_html(table):
            table_html = "<table>"
            rows = [sorted([cell for cell in table.cells if cell.row_index == i], key=lambda cell: cell.column_index) for i in range(table.row_count)]
            for row_cells in rows:
                table_html += "<tr>"
                for cell in row_cells:
                    tag = "th" if cell.kind in ("columnHeader", "rowHeader") else "td"
                    cell_spans = ""
                    if cell.column_span > 1:
                        cell_spans += f' colSpan="{cell.column_span}"'
                    if cell.row_span > 1:
                        cell_spans += f' rowSpan="{cell.row_span}"'
                    table_html += f"<{tag}{cell_spans}>{html.escape(cell.content)}</{tag}>"
                table_html += "</tr>"
            table_html += "</table>"
            return table_html

        if is_image or is_table:
            file.seek(0)
            poller = self.llm.form_recognizer_client.begin_analyze_document("prebuilt-layout", document=file.read())
            result = poller.result()

            def process_page(page_num):
                page = result.pages[page_num]
                page_offset = page.spans[0].offset
                page_length = page.spans[0].length
                tables_on_page = [
                    table for table in result.tables
                    if table.bounding_regions[0].page_number == page_num + 1
                ]
                table_chars = [-1] * page_length
                for table_id, table in enumerate(tables_on_page):
                    for span in table.spans:
                        for i in range(span.length):
                            idx = span.offset - page_offset + i
                            if 0 <= idx < page_length:
                                table_chars[idx] = table_id
                buf = StringIO()
                added_tables = set()
                for idx, table_id in enumerate(table_chars):
                    if table_id == -1:
                        buf.write(result.content[page_offset + idx])
                    elif table_id not in added_tables:
                        buf.write(table_to_html(tables_on_page[table_id]))
                        added_tables.add(table_id)
                return buf.getvalue()

            page_texts = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(process_page, i) for i in range(len(result.pages))]
                for future in tqdm(as_completed(futures), total=len(futures), desc="處理頁面中"):
                    page_texts.append(future.result())
            batch_size = 5
            for i in range(0, len(page_texts), batch_size):
                batch_text = "\n".join(page_texts[i:i + batch_size])
                system_prompt = extracted_pdf_prompt.format(inputs=batch_text)
                result_text = self.llm.chunk_response_for_pdfdata(system_prompt)
                data["body"].append(result_text)
                print("✅ 修正後:", result_text)
            return data
        else:
            file.seek(0)
            with pdfplumber.open(BytesIO(file.read())) as pdf:
                text_pages = []
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        text_pages.append(text)
                batch_size = 5
                for i in tqdm(range(0, len(text_pages), batch_size), desc="處理文字頁"):
                    batch_text = "\n".join(text_pages[i:i + batch_size])
                    system_prompt = extracted_pdf_prompt.format(inputs=batch_text)
                    result_text = self.llm.chunk_response_for_pdfdata(system_prompt)
                    data["body"].append(result_text)
                    print("✅ 修正後:", result_text)
            return data

    def contain_images(self, file):
        file_bytes = BytesIO(file.read())
        with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
            for page_num in range(len(pdf)):
                page = pdf[page_num]
                images = page.get_images(full=True)
                if images:
                    return True
            return False

    def contain_tables(self, file):
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                if tables and any(any(cell for cell in row) for row in tables):
                    return True
                else:
                    return False

    def pdf_to_txt(self, path, allcontent):
        content = allcontent["body"]
        with open(path, "w", encoding="utf-8") as file:
            if len(content) == 1:
                file.write(content[0])
            else:
                for ct in content:
                    file.write(ct + "\n\n=====next page=====\n\n")

    def chunk_for_mbti(self, text: str, key: str) -> List[str]:
        if key == "relationships":
            chunks = [seg.strip() for seg in re.split(r'[。\.?!]+', text) if seg.strip()]
        else:
            chunks = [seg.strip() for seg in re.split(r'[\n]+', text) if seg.strip()]
        # for i, c in enumerate(chunks, 1):
        #     print(f"---- Chunk {i} ----\n{c}\n")
        return chunks

    def embed_text(self, text: str) -> Optional[List[float]]:
        from service import azure_service
        azure = azure_service()
        if not text:
            return None
        try:
            resp = azure.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return resp.data[0].embedding
        except Exception as e:
            print(f"Embedding failed: {e}")
            return None

    def mbti_to_neo4j(self, csv_path, json_dir):
        n4c = Neo4jConnection(NEO4J_URI, NEO4J_AUTH)
        csv_rows = []
        if csv_path != "":
            def prepare_row(row):
                def clean_str(val):
                    return str(val).strip() if pd.notna(val) and str(val).strip() else None
                def clean_float(val):
                    try:
                        return float(val) if pd.notna(val) else None
                    except ValueError:
                        return None
                weight = {
                    clean_str(row.get("letter_1")): clean_float(row.get("letter_1_percentage")),
                    clean_str(row.get("letter_2")): clean_float(row.get("letter_2_percentage")),
                    clean_str(row.get("letter_3")): clean_float(row.get("letter_3_percentage")),
                    clean_str(row.get("letter_4")): clean_float(row.get("letter_4_percentage")),
                }
                weight = {k: v for k, v in weight.items() if k and v is not None}
                return {
                    "name": clean_str(row.get("name")),
                    "nameEmbedding": self.embed_text(clean_str(row.get("name"))),
                    "category": clean_str(row.get("category")),
                    "categoryEmbedding": self.embed_text(clean_str(row.get("category"))),
                    "subcategory": clean_str(row.get("subcategory")),
                    "subcategoryEmbedding": self.embed_text(clean_str(row.get("subcategory"))),
                    "four_letter": clean_str(row.get("four_letter")),
                    "enneagram": clean_str(row.get("enneagram")),
                    "weight": str(weight)
                }
            df = pd.read_csv(csv_path)
            df.drop(columns=["cat_id", "sub_cat_id", "four_letter_total_voted", "enneagram_total_voted", "socionics", "socionics_total_voted", "instinctual_variant", "instinctual_variant_total_voted", "tritype", "tritype_total_voted", "temperaments", "temperaments_total_voted", "attitudinal_psyche", "attitudinal_psyche_total_voted", "big_5_SLOAN", "big_5_SLOAN_total_voted", "classic_jungian", "classic_jungian_total_voted"], inplace=True)
            # newdf = df.head(20)
            newdf = df.sample(n=100, replace=False, random_state=42)
            csv_rows = [prepare_row(r) for _, r in newdf.iterrows()]
        if json_dir != "":
            personality_rows: List[dict] = []
            passage_rows: List[dict] = []
            files = glob.glob(os.path.join(json_dir, "*.json"))
            for file in files:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                four_letter = os.path.basename(file).replace(".json", "").upper()
                personality_rows.append({
                    "four_letter": four_letter,
                    "overview": data.get("overview"),
                    "strengths": data.get("strengths"),
                    "weaknesses": data.get("weaknesses"),
                    "relationships": data.get("relationships"),
                    "career": data.get("career"),
                    # "overviewEmbedding": self.embed_text(data.get("overview")),
                    # "strengthsEmbedding": self.embed_text(data.get("strengths")),
                    # "weaknessesEmbedding": self.embed_text(data.get("weaknesses")),
                    # "relationshipsEmbedding": self.embed_text(data.get("relationships")),
                    # "careerEmbedding": self.embed_text(data.get("career")),
                })
                for key, text in data.items():
                    print("Processing {} of {} ...".format(key, four_letter))
                    passages = self.chunk_for_mbti(text, key)
                    for idx, pag in enumerate(passages):
                        emb = None
                        if idx < 500:
                            emb = self.embed_text(pag)
                        passage_rows.append({
                            "four_letter": four_letter,
                            "key": key,
                            "chunk_index": "{}_{}".format(four_letter, idx),
                            "embedding": emb,
                            "text": pag,
                            "source_file": os.path.basename(file),
                        })
                    print(len(passage_rows))
        if len(csv_rows) != 0:
            print("Ingesting CSV...")
            n4c.ingest_neo4j(csv_rows, chunk_size=5000, mode="csv")
        if personality_rows:
            print("Ingesting {} passages to Neo4j ...".format(len(personality_rows)))
            n4c.ingest_neo4j(personality_rows, chunk_size=64, mode="json")
        if passage_rows:
            print("Ingesting {} passages to Neo4j ...".format(len(passage_rows)))
            n4c.ingest_neo4j(passage_rows, chunk_size=64, mode="json_passages")
        n4c.close_connection()
        print("✅ All ingest finished.")

    def imdb_movies_to_neo4j(self, path):
        def split_list(s):
            if pd.isna(s) or str(s).strip() == "":
                return []
            return [p.strip() for p in str(s).split(",") if p.strip()]
        def prepare_row(row):
            review_title = row["Review Title"] if pd.notna(row["Review Title"]) else None
            review_text = row["Review"] if pd.notna(row["Review"]) else None
            if not review_title:
                review_text = None
            return {
                "title": row["Title"],
                "year": int(row["Year"]) if pd.notna(row["Year"]) else None,
                "certificate": row["Certificate"] if pd.notna(row["Certificate"]) else None,
                "duration": int(row["Duration (min)"]) if pd.notna(row["Duration (min)"]) else None,
                "rating": float(row["Rating"]) if pd.notna(row["Rating"]) else None,
                "metascore": int(row["Metascore"]) if pd.notna(row["Metascore"]) else None,
                "description": row["Description"] if pd.notna(row["Description"]) else None,
                "genres": split_list(row["Genre"]),
                "directors": split_list(row["Director"]),
                "cast": split_list(row["Cast"]),
                "review_title": review_title,
                "review_text": review_text
            }
        df = pd.read_csv(path)
        print(df)
        df.drop(columns=["Poster", "Votes", "Review Count"])
        # newdf = df.head(15)
        # print(newdf)
        rows = [prepare_row(r) for _, r in df.iterrows()]
        n4c = Neo4jConnection(NEO4J_URI, NEO4J_AUTH)
        n4c.ingest_neo4j(rows, chunk_size=5000)
        n4c.close_connection()
        print("Ingest to Neo4j finished")