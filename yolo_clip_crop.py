import torch
import clip
import cv2
import sys
import os
import requests
import psycopg2
import pandas as pd
import shutil
from PIL import Image
from ultralytics import YOLO

filenames = []
decriptions = ["ai", "people", "dog", "human", "airplane", "people"]

def append_to_log(file_path, string):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(string + "\n")

def connect():
    connection = psycopg2.connect(
        database="",
        user="",
        password="",
        host=""
    )
    return connection

def get_picture():
    connection = connect()
    x = "select * from dnew_image where datepublish=current_date and is_photo ='Y' and (width/height) between 1.33 and 1.6"
    with connection.cursor() as cur:
        cur.execute(x)
        name = [desc[0] for desc in cur.description]
        rows = pd.DataFrame(cur.fetchall(), columns=name)
    return rows

def show_dataframe():
    df = get_picture()
    print(df)
    print(df.columns)
    for i in df["title"]:
        if i == None:
            continue
        print(i)

def download_img_to_local_folder():
    df = get_picture()
    print(df)
    for i, j, k in zip(df["datepublish"], df["filename"], df["title"]):
        filenames.append(j.lower())
        if k == None:
            decriptions.append("")
        else:
            decriptions.append(str(k))
        img_url = "".format(i.year, i.strftime("%m%d"), j.lower())
        print(img_url)
        response = requests.get(img_url, stream=True)
        img_name = os.path.join("D:\\Downloads\\news_imgs", "{}".format(j.lower()))
        with open(img_name, "wb") as out_file:
            shutil.copyfileobj(response.raw, out_file)
    print("image download finished!")

def load_picture():
    folder_path = ""
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))]
    filenames.extend(image_files)

def crop_images(filenames):
    from PIL import Image
    for f in filenames:
        image = Image.open("".format(f))
        resized_image = image.resize((1600, 900))
        resized_image.save("".format(f), format="JPEG", quality=85)
    print("resized image saved!")
    
def Saliency_Detection(filenames):
    import cv2
    for f in filenames:
        img = cv2.imread("".format(f))
        height, width = img.shape[:2]
        # 初始化顯著性檢測器
        saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        (success, saliencyMap) = saliency.computeSaliency(img)
        if not success:
            print("顯著性檢測失敗")
            exit()
        # 將顯著性圖轉換為二值圖
        saliencyMap = (saliencyMap * 255).astype("uint8")
        _, binaryMap = cv2.threshold(saliencyMap, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # 查找輪廓
        contours, _ = cv2.findContours(binaryMap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("未檢測到顯著區域")
            exit()
        # 找到最大的輪廓
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        # 繪製邊界框
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # 裁切並保存
        cropped = img[y:y + h, x:x + w]
        cv2.imwrite("".format(f), cropped)
        # 顯示結果
        # cv2.imshow("Saliency Map", saliencyMap)
        # cv2.imshow("Original Image with Bounding Box", img)
        # cv2.imshow("Cropped Image", cropped)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    print("saliency detection finished!")

def load_yolo_model():
    # model = torch.hub.load("ultralytics/yolov5", "yolov5m", pretrained=True)
    model = YOLO("yolov8m.pt")
    return model

def get_detected_objects(yolo_model, image_path, allowed_classes=None): # 使用 YOLO 檢測圖片中的物體
    # results = yolo_model(image_path)
    # detections = results.xyxy[0]
    # detected_objects = []
    # for *box, conf, cls in detections:
    #     x1, y1, x2, y2 = map(int, box)
    #     label = yolo_model.names[int(cls)]
    #     # if allowed_classes is None or label in allowed_classes:
    #     detected_objects.append({
    #         "label": label,
    #         "confidence": float(conf),
    #         "box": (x1, y1, x2, y2)
    #     })
    results = yolo_model(image_path)
    result = results[0]
    boxes = result.boxes
    detected_objects = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0]) # 置信度
        cls = int(box.cls[0]) # class
        label = yolo_model.names[cls]
        if not allowed_classes or label in allowed_classes:
            detected_objects.append({
                "label": label,
                "confidence": conf,
                "box": (int(x1), int(y1), int(x2), int(y2))
            })
    return detected_objects

def match_description_with_objects(image_path, description, detected_objects, clip_model, clip_preprocess, device):
    tokens = clip.tokenize([description], truncate=True).to(device) # 將描述轉換為 CLIP 的文本特徵
    with torch.no_grad():
        text_features = clip_model.encode_text(tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    # print(f"text features = \n{text_features}")
    similarities = [] # 計算每個檢測到的物體的相似度
    for obj in detected_objects:
        x1, y1, x2, y2 = obj["box"]  # 裁切物體部分
        pil_img = Image.open(image_path).crop((x1, y1, x2, y2))
        img_preprocessed = clip_preprocess(pil_img).unsqueeze(0).to(device)
        clip_model.eval()
        with torch.no_grad():
            image_features = clip_model.encode_image(img_preprocessed)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        # print(f"image features = \n{image_features}")
        # similarity = torch.cosine_similarity(text_features, image_features)
        # similarities.append(similarity.item())
        similarity = (text_features @ image_features.T).squeeze().item()
        similarities.append(similarity)
    for i, obj in enumerate(detected_objects): # 將相似度添加到物體訊息中
        obj["similarity"] = similarities[i]
    if similarities:  # 選擇相似度最高的物體
        best_match = max(detected_objects, key=lambda x: x["similarity"])
        return best_match
    else:
        return None

def adjust_bbox_to_aspect_ratio(box, target_aspect=16/9, img_width=None, img_height=None):
    x1, y1, x2, y2 = box
    current_width = x2 - x1
    current_height = y2 - y1
    current_aspect = current_width / current_height
    if current_aspect > target_aspect: # 當前寬度大於目標比例，擴展高度
        new_height = current_width / target_aspect
        delta_height = new_height - current_height
        y1 = max(0, int(y1 - delta_height / 2))
        y2 = min(img_height, int(y2 + delta_height / 2)) if img_height else y2 + int(delta_height / 2)
    else: # 當前高度大於目標比例，擴展寬度
        new_width = current_height * target_aspect
        delta_width = new_width - current_width
        x1 = max(0, int(x1 - delta_width / 2))
        x2 = min(img_width, int(x2 + delta_width / 2)) if img_width else x2 + int(delta_width / 2)
    return x1, y1, x2, y2

def crop_and_save_image(image_path, box, output_path, target_aspect=16/9):
    img = cv2.imread(image_path)
    img_height, img_width = img.shape[:2]
    x1, y1, x2, y2 = box
    adjusted_box = adjust_bbox_to_aspect_ratio(box, target_aspect, img_width, img_height)  # 調整邊界框以符合16:9比例
    x1, y1, x2, y2 = adjusted_box
    x1 = max(0, x1)  # 確保邊界不超出圖片範圍
    y1 = max(0, y1)  # 確保邊界不超出圖片範圍
    x2 = min(img_width, x2)  # 確保邊界不超出圖片範圍
    y2 = min(img_height, y2)  # 確保邊界不超出圖片範圍
    cropped = img[y1:y2, x1:x2]
    cropped_height, cropped_width = cropped.shape[:2]  # 如果裁切後的比例不正確，調整裁切區域
    cropped_aspect = cropped_width / cropped_height  # 如果裁切後的比例不正確，調整裁切區域
    if abs(cropped_aspect - target_aspect) > 0.01:
        if cropped_aspect > target_aspect:  # 進一步調整裁切區域
            new_width = int(cropped_height * target_aspect)  # 裁切寬度
            start_x = (cropped_width - new_width) // 2
            cropped = cropped[:, start_x:start_x + new_width]
        else:
            new_height = int(cropped_width / target_aspect)  # 裁切高度
            start_y = (cropped_height - new_height) // 2
            cropped = cropped[start_y:start_y + new_height, :]
    cv2.imwrite(output_path, cropped)
    return cropped

# show_dataframe()
# download_img_to_local_folder()
load_picture()
crop_images(filenames)
Saliency_Detection(filenames)

print(len(filenames))
print(len(decriptions))

device = "cuda" if torch.cuda.is_available() else "cpu"
for f, d in zip(filenames, decriptions):
    f = f[:f.find(".jpg")]
    image_path = "".format(f)
    print(image_path)
    print(d)

    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)  # load CLIP model
    yolo_model = load_yolo_model()  # load YOLO model

    allowed_classes = []
    detected_objects = get_detected_objects(yolo_model, image_path, allowed_classes)
    print("Detected objects:")
    append_to_log("".format(f), f"{d}\n")
    for obj in detected_objects:
        append_to_log("".format(f), f"{obj['label']} - Confidence: {obj['confidence']:.2f}")

    best_match = match_description_with_objects(image_path, d, detected_objects, clip_model, clip_preprocess, device)
    print(best_match)
    if best_match and best_match["similarity"] > 0.1:
        append_to_log("".format(f), f"Best match: {best_match['label']} with similarity {best_match['similarity']:.2f}")
        output_path = "".format(f)
        cropped_image = crop_and_save_image(image_path, best_match["box"], output_path, target_aspect=16/9)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No relevant part found for the description.")