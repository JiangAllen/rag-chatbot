import re
import glob
import openai
import numpy as np
import matplotlib.pyplot as plt
# from datapreprocessing_ch import datapreprocessing
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer, models
# model = SentenceTransformer("moka-ai/m3e-large")
tokenizer = AutoTokenizer.from_pretrained("moka-ai/m3e-large")
transformer = models.Transformer("moka-ai/m3e-large", max_seq_length=512)
pooling = models.Pooling(transformer.get_word_embedding_dimension())
model = SentenceTransformer(modules=[transformer, pooling])

def _split_sentences_1(text):
    sentences = re.split(r'[。；？！\n]+', text)
    sentences = [s for s in sentences if s and s.strip()]
    return sentences

def _split_sentences_2(text):
    sentences = re.split(r'(?<=[。；？！\n])\s*', text)
    sentences = [s for s in sentences if s and s.strip()]
    return sentences

def _combine_sentences_1(sentences, buffer_size=1):
    combined_sentences = [
        ' '.join(sentences[j]["sentence"] for j in range(max(i-buffer_size, 0), min(i+buffer_size + 1, len(sentences)))) for i in range(len(sentences))
    ]
    for i, combined_sentence in enumerate(combined_sentences):
        sentences[i]["combined_sentence"] = combined_sentence
    return sentences

def _combine_sentences_2(sentences):
    combined_sentences = []
    for i in range(len(sentences)):
        combined_sentence = sentences[i]
        if i > 0:
            combined_sentence = sentences[i - 1] + combined_sentence
        if i < len(sentences) - 1:
            combined_sentence += sentences[i + 1]
        combined_sentences.append(combined_sentence)
    return combined_sentences

def convert_to_vector_1(texts):
    embeddings = model.encode([t["combined_sentence"] for t in texts])
    for i, text in enumerate(texts):
        text["combined_sentence_embedding"] = embeddings[i]
    return texts

def convert_to_vector_2(texts):
    try:
        response = openai.embeddings.create(
            input=texts,
            model="text-embedding-3-large"
        )
        embeddings = np.array([item.embedding for item in response.data])
        return embeddings
    except Exception as e:
        print("An error occurred:", e)
        return np.array([])

def cosine_similarity_custom(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def _calculate_cosine_distances_1(embeddings):
    distances = []
    for i in range(len(embeddings) - 1):
        embedding_current = embeddings[i]["combined_sentence_embedding"]
        embedding_next = embeddings[i + 1]["combined_sentence_embedding"]
        similarity = cosine_similarity_custom(embedding_current, embedding_next)
        distance = 1 - similarity
        distances.append(distance)
        embeddings[i]["distance_to_next"] = distance
    return distances, embeddings

def _calculate_cosine_distances_2(embeddings):
    distances = []
    for i in range(len(embeddings) - 1):
        similarity = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
        distance = 1 - similarity
        distances.append(distance)
    return distances

def draw_threshold(distances):
    plt.plot(distances)
    y_upper_bound = 0.15
    plt.ylim(0, y_upper_bound)
    plt.xlim(0, len(distances))

    breakpoint_percentile_threshold = 80
    breakpoint_distance_threshold = np.percentile(distances, breakpoint_percentile_threshold)

    plt.axhline(y=breakpoint_distance_threshold, color='r', linestyle='-')
    num_distances_above_theshold = len([x for x in distances if x > breakpoint_distance_threshold])
    plt.text(x=(len(distances) * .01), y=y_upper_bound / 50, s=f"{num_distances_above_theshold + 1} Chunks")

    indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold]

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i, breakpoint_index in enumerate(indices_above_thresh):
        start_index = 0 if i == 0 else indices_above_thresh[i - 1]
        end_index = breakpoint_index if i <= len(indices_above_thresh) - 1 else len(distances)
        plt.axvspan(start_index, end_index, facecolor=colors[i % len(colors)], alpha=0.25)
        plt.text(
            x=np.average([start_index, end_index]),
            y=breakpoint_distance_threshold + (y_upper_bound) / 20,
            s=f"Chunk #{i}", horizontalalignment="center",
            rotation="vertical"
        )
    if indices_above_thresh:
        last_breakpoint = indices_above_thresh[-1]
        if last_breakpoint < len(distances):
            plt.axvspan(last_breakpoint, len(distances), facecolor=colors[len(indices_above_thresh) % len(colors)], alpha=0.25)
            plt.text(
                x=np.average([last_breakpoint, len(distances)]),
                y=breakpoint_distance_threshold + (y_upper_bound) / 20,
                s=f"Chunk #{i + 1}",
                rotation="vertical"
            )
    plt.title("text Chunks Based On Embedding Breakpoints")
    plt.xlabel("Index of sentences in text (Sentence Position)")
    plt.ylabel("Cosine distance between sequential sentences")
    plt.show()
    return indices_above_thresh

def chunk_text_1(text):
    single_sentences_list = _split_sentences_1(text)
    sentences = [{"index": i, "sentence": s} for i, s in enumerate(single_sentences_list)]
    sentences = _combine_sentences_1(sentences)
    sentences = convert_to_vector_1(sentences)
    distances, sentences = _calculate_cosine_distances_1(sentences)
    indices_above_thresh = draw_threshold(distances)
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

def chunk_text_2(text):
    single_sentences_list = _split_sentences_2(text)
    print("文章切成 {} 段".format(len(single_sentences_list)))
    combined_sentences = _combine_sentences_2(single_sentences_list)
    print("合併上下文總共有 {} 段".format(len(combined_sentences)))
    embeddings = convert_to_vector_2(combined_sentences)
    distances = _calculate_cosine_distances_2(embeddings)
    print("轉成向量計算距離總共有 {} 段".format(len(distances)))
    breakpoint_percentile_threshold = 80
    breakpoint_distance_threshold = np.percentile(distances, breakpoint_percentile_threshold)
    print("閾值第80百分位數為 {}".format(breakpoint_distance_threshold))
    indices_above_thresh = [i for i, distance in enumerate(distances) if distance > breakpoint_distance_threshold]
    print("該切的有第 {} 段".format(" & ".join(map(str, indices_above_thresh))))

    chunks = []
    start_index = 0
    for index in indices_above_thresh:
        chunk = ' '.join(single_sentences_list[start_index:index + 1])
        chunks.append(chunk)
        start_index = index + 1
    if start_index < len(single_sentences_list):
        chunk = ' '.join(single_sentences_list[start_index:])
        chunks.append(chunk)
    for i, c in enumerate(chunks):
        print("#{} {}".format(i + 1, c))

text = ""

chunk_text_1(text)