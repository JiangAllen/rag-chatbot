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

def _split_sentences_2(text): # 使用正規表示式根據標點符號和空格將文字拆分為句子
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

def _combine_sentences_2(sentences): # 透過將每個句子與其前後句子結合起來來創建緩衝區來提供更廣泛的上下文
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

def convert_to_vector_2(texts): # 嘗試使用預先訓練的模型為文字清單產生嵌入並處理任何異常
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

def _calculate_cosine_distances_2(embeddings): # 計算連續嵌入之間的餘弦距離(1-餘弦相似度)
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
    single_sentences_list = _split_sentences_2(text) # 將輸入文字分成單獨的句子
    print("文章切成 {} 段".format(len(single_sentences_list)))
    combined_sentences = _combine_sentences_2(single_sentences_list) # 把相鄰的句子組合起來，在每個句子周圍形成一個上下文視窗
    print("合併上下文總共有 {} 段".format(len(combined_sentences)))
    embeddings = convert_to_vector_2(combined_sentences) # 使用神經網路模型將組合後的句子轉換成向量表徵
    distances = _calculate_cosine_distances_2(embeddings) # 計算連續組合句子嵌入之間的餘弦距離來測量相似度
    print("轉成向量計算距離總共有 {} 段".format(len(distances)))
    breakpoint_percentile_threshold = 80  # 根據所有距離的第80個百分位數來確定識別斷點的閾值距離
    breakpoint_distance_threshold = np.percentile(distances, breakpoint_percentile_threshold)
    print("閾值第80百分位數為 {}".format(breakpoint_distance_threshold))
    indices_above_thresh = [i for i, distance in enumerate(distances) if distance > breakpoint_distance_threshold] # 尋找所有距離超過計算閾值的索引，表示潛在的區塊斷點。
    print("該切的有第 {} 段".format(" & ".join(map(str, indices_above_thresh))))

    chunks = []
    start_index = 0
    for index in indices_above_thresh: # 循環遍歷已識別的斷點並相應地建立區塊
        chunk = ' '.join(single_sentences_list[start_index:index + 1])
        chunks.append(chunk)
        start_index = index + 1
    if start_index < len(single_sentences_list): # 如果最後一個斷點後還有任何句子，請將它們加為最後一個區塊
        chunk = ' '.join(single_sentences_list[start_index:])
        chunks.append(chunk)
    for i, c in enumerate(chunks):
        print("#{} {}".format(i + 1, c))

text1 = """業界分析師認為，華為最新發表的Mate 70系列智慧型手機的銷售表現，除了面臨地緣政治緊張情勢升溫下愈來愈高的供應鏈風險以外，還將受到產品發布太遲、處理器不敵競爭對手等不利因素的影響，預估2024年第4季銷量將落在300萬支上下。Mate 70系列是在華為11月26日舉行的Mate品牌盛典中發布，號稱是史上最強大的Mate手機。不過，發表會中並未透露有關處理器的任何細節，官方也並未解釋為何發布時間點會落在雙十一購物節之後。Mate 70系列預定在12月4日於中國全面上市，目前官方尚未公布海外市場銷售計畫。根據南華早報引用半導體研究機構TechInsights報告指出，華為Mate 70系列第4季銷售量，預估將落在300萬支上下，佔華為同期手機總出貨量的22%。另據Counterpoint Research預估，Mate 70系列整個產品生命周期的總出貨量，將達到1,000萬支以上。TechInsights分析師指出，包含Mate 70、70 Pro、70 Pro+與70 RS在內的系列新機，雖然硬體有升級，也導入新的人工智慧（AI）功能，但發布日期太遲的問題，再加上其中搭載的華為海思麒麟9010與9020兩款處理器，比不上高通（Qualcomm）、聯發科同期最新款處理器，這種種因素都會限制新機的銷售潛力。另外，Mate 70系列將有HarmonyOS 4.3與HarmonyOS NEXT 5.0兩種作業系統選項可選。TechInsights指出，由於HarmonyOS NEXT不支援Android應用程式，這可能令Mate 70對於中國市場以外消費者的吸引力大打折扣。"""
text2 = """**ESTJ 值得探索的職業機會**-ESTJ 的理 想職業是能夠培養以下特質的職業。如果工作符合 ESTJ 的天性，那麼成功的可能性就會更大。\n**準確性**-ESTJ 著重解決方案，而非問題導向。面對挑戰，他們能夠輕鬆地建立潛在解決方案的認知圖譜並制定計劃。他們做事一絲不苟，更注重準確性而非速度。當錯誤造成嚴重後果時，ESTJ 是最佳人選。\n**管理與組織**-ESTJ 通常能夠快速評估他人。他們敏銳的感知力使他們能夠甄別異常值並發現高績效機會。他們能夠坦誠地溝通責任，無論責任是否已確定。他們既不會粉飾真相，也不會屈服於偏見。\n**說服力**-許多 ESTJ 都精通說服的藝術。他們運用語言策略性強、敏感，通常能言善辯、雄辯有力。正因如此，ESTJ 在需要說服力的職業中表現出色——無論是產品銷售還是資產收購。\n**數據分析**-ESTJ 通常對數據分析充滿熱情，並且技藝嫻熟。他們具有分析思維，擅長將看似矛盾的事實組織起來，從而揭示其中的隱含意義。他們尤其擅長從事資料分析、資料探勘、統計或相關領域的工作。\n\n**ESTJ 職業應避免的因素**-與所有類型一樣，某些 ESTJ 職業並不那麼適合他們，因為他們不太適合某些任務和工作風格。如果能夠避免這些因素，ESTJ 更有可能擁有一個快樂且有效率的工作環境。\n**敏感性**-在某些情況下，說出善意的謊言或簡單地隱瞞真相可能很重要，例如在複雜的人際互動中。通常情況下，沒有人可以被單獨挑出來責怪——無論是無意的失誤還是系統性的缺陷。 ESTJ 可能在人力資源或調解等需要謹慎措辭處理敏感問題的職業中遇到困難。\n**理論與探索**-ESTJ 往往會因為無法用常識解決的任務和難題而失去動力。某些學科（主要是學術學科）要求對多種觀點同樣有效以及客觀真理是未經證實的斷言持開放態度。 ESTJ 最好避免這些。\n**創意表達**-ESTJ 型人格特質謹慎內斂。這對某些職業很有幫助，但對其他職業則不然。藝術表達需要某種程度的魯莽——一種對自身作品的信心，或一絲不切實際的自信。 ESTJ 型人確實有藝術天賦，但往往會被壓抑。\n**新奇感**-ESTJ 型人通常重視結構性和可預測性。他們重複一項任務的次數越多，就越擅長。他們往往對自己要求很高，並希望確保每個人都認可自己的能力。他們可能會對需要不斷適應和快速學習的職業感到害怕。"""

chunk_text_1(text2)