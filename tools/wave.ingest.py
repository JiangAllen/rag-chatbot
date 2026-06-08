import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import MetadataQuery
from typing import List, Dict, Optional
from config import *

class WeaviateManager:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or weaviate_key
        self.client = None
        self.collection_name = "Article"

    def connect(self):
        self.client = weaviate.connect_to_local(
            auth_credentials=Auth.api_key(self.api_key)
        )
        if self.client.is_ready():
            print("✅ Weaviate 連線成功！")
            return True
        else:
            print("❌ 連線失敗")
            return False

    def create_schema(self, reset: bool = False):
        if reset and self.client.collections.exists(self.collection_name):
            self.client.collections.delete(self.collection_name)
            print(f"🗑️  已刪除既有的 {self.collection_name} collection")

        if not self.client.collections.exists(self.collection_name):
            self.client.collections.create(
                name=self.collection_name,
                description="新聞或文章內容",
                vectorizer_config=weaviate.classes.config.Configure.Vectorizer.text2vec_transformers(),
                properties=[
                    weaviate.classes.config.Property(
                        name="title",
                        data_type=weaviate.classes.config.DataType.TEXT
                    ),
                    weaviate.classes.config.Property(
                        name="content",
                        data_type=weaviate.classes.config.DataType.TEXT
                    )
                ]
            )
            print("✅ Schema 已建立成功！")
        else:
            print(f"ℹ️  {self.collection_name} collection 已存在")

    def insert_articles(self, articles: List[Dict[str, str]]):
        collection = self.client.collections.get(self.collection_name)
        for article in articles:
            collection.data.insert(article)
        print(f"✅ 已成功插入 {len(articles)} 筆資料！")

    def search_by_text(self, query: str, limit: int=5) -> List[Dict]:
        collection = self.client.collections.get(self.collection_name)
        response = collection.query.near_text(
            query=query,
            limit=limit,
            return_metadata=MetadataQuery(distance=True)
        )
        results = []
        for obj in response.objects:
            results.append({
                "uuid": str(obj.uuid),
                "title": obj.properties.get("title"),
                "content": obj.properties.get("content"),
                "distance": obj.metadata.distance
            })
        return results

    def search_by_keyword(self, keyword: str, properties: List[str] = None, limit: int=10) -> List[Dict]:
        collection = self.client.collections.get(self.collection_name)
        if properties is None:
            properties = ["title", "content"]
        response = collection.query.bm25(
            query=keyword,
            query_properties=properties,
            limit=limit,
            return_metadata=MetadataQuery(score=True)
        )
        results = []
        for obj in response.objects:
            results.append({
                "uuid": str(obj.uuid),
                "title": obj.properties.get("title"),
                "content": obj.properties.get("content"),
                "score": obj.metadata.score
            })
        return results

    def get_all_articles(self, limit: int=100) -> List[Dict]:
        collection = self.client.collections.get(self.collection_name)
        response = collection.query.fetch_objects(limit=limit)
        results = []
        for obj in response.objects:
            results.append({
                "uuid": str(obj.uuid),
                "title": obj.properties.get("title"),
                "content": obj.properties.get("content")
            })
        return results

    def get_count(self) -> int:
        collection = self.client.collections.get(self.collection_name)
        count = collection.aggregate.over_all(total_count=True)
        return count.total_count

    def close(self):
        if self.client:
            self.client.close()
            print("👋 Weaviate 連線已關閉")

if __name__ == "__main__":
    wmg = WeaviateManager()
    if wmg.connect():
        wmg.create_schema(reset=True)
        test_data = [
            {"title": "AI 技術快速發展", "content": "AI 正在改變世界，許多產業都受到影響。"},
            {"title": "Python 成為熱門語言", "content": "Python 因為簡單易用，被廣泛應用於數據科學與機器學習。"},
            {"title": "台灣科技產業現況", "content": "台灣在半導體與 AI 應用上具有強勁優勢。"}
        ]
        wmg.insert_articles(test_data)
        print(f"\n目前 Article 資料數量: {wmg.get_count()}")

        print("\n🔍 語義搜尋 - '人工智慧':")
        semantic_results = wmg.search_by_text("人工智慧", limit=2)
        for i, result in enumerate(semantic_results, 1):
            print(f"{i}. {result['title']} (distance: {result['distance']:.4f})")
            print(f"   {result['content']}\n")

        print("🔍 關鍵字搜尋 - 'Python':")
        keyword_results = wmg.search_by_keyword("Python", limit=2)
        for i, result in enumerate(keyword_results, 1):
            print(f"{i}. {result['title']} (score: {result['score']:.4f})")
            print(f"   {result['content']}\n")

        wmg.close()