import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from weaviate.classes.init import Auth
import weaviate
from config import *
import json

class WeaviateInspector:
    def __init__(self, api_key: str = None):
        self.client = weaviate.connect_to_local(
            auth_credentials=Auth.api_key(api_key or weaviate_key)
        )

    def list_all_collections(self):
        print("\n" + "=" * 80)
        print("📚 所有 Collections (Schemas)")
        print("=" * 80)

        collections = self.client.collections.list_all()
        if not collections:
            print("目前沒有任何 Collection")
            return

        for name, config in collections.items():
            print(f"\n🗂️  Collection: {name}")
            print(f"   描述: {config.description or 'N/A'}")
            print(f"   向量化器: {config.vectorizer}")

            collection = self.client.collections.get(name)
            count = collection.aggregate.over_all(total_count=True)
            print(f"   資料筆數: {count.total_count}")

            if config.properties:
                print(f"   欄位:")
                for prop in config.properties:
                    print(f"      - {prop.name} ({prop.data_type})")

    def show_collection_data(self, collection_name: str, limit: int = 10):
        print("\n" + "=" * 80)
        print(f"📄 Collection: {collection_name} 的資料內容")
        print("=" * 80)

        collection = self.client.collections.get(collection_name)
        response = collection.query.fetch_objects(limit=limit)

        if not response.objects:
            print(f"Collection '{collection_name}' 中沒有資料")
            return

        for i, obj in enumerate(response.objects, 1):
            print(f"\n--- 資料 {i} ---")
            print(f"UUID: {obj.uuid}")
            for key, value in obj.properties.items():
                if isinstance(value, str) and len(value) > 100:
                    value = value[:100] + "..."
                print(f"{key}: {value}")

    def show_collection_stats(self, collection_name: str):
        print("\n" + "=" * 80)
        print(f"📊 Collection: {collection_name} 統計資訊")
        print("=" * 80)

        collection = self.client.collections.get(collection_name)
        count = collection.aggregate.over_all(total_count=True)
        print(f"總資料筆數: {count.total_count}")

        sample = collection.query.fetch_objects(limit=1)
        if sample.objects:
            print(f"\n資料範例:")
            obj = sample.objects[0]
            for key, value in obj.properties.items():
                print(f"  {key}: {type(value).__name__}")

    def search_demo(self, collection_name: str, query: str, limit: int = 5):
        print("\n" + "=" * 80)
        print(f"🔍 搜尋: '{query}' in {collection_name}")
        print("=" * 80)

        collection = self.client.collections.get(collection_name)

        print("\n【語義搜尋結果】")
        from weaviate.classes.query import MetadataQuery

        response = collection.query.near_text(
            query=query,
            limit=limit,
            return_metadata=MetadataQuery(distance=True)
        )

        if not response.objects:
            print("沒有找到相關結果")
        else:
            for i, obj in enumerate(response.objects, 1):
                print(f"\n{i}. (distance: {obj.metadata.distance:.4f})")
                for key, value in obj.properties.items():
                    if isinstance(value, str) and len(value) > 100:
                        value = value[:100] + "..."
                    print(f"   {key}: {value}")

    def export_collection_to_json(self, collection_name: str, output_file: str):
        print(f"\n💾 匯出 {collection_name} 到 {output_file}...")

        collection = self.client.collections.get(collection_name)
        response = collection.query.fetch_objects(limit=10000)
        data = []
        for obj in response.objects:
            data.append({
                "uuid": str(obj.uuid),
                "properties": obj.properties
            })
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"✅ 成功匯出 {len(data)} 筆資料")

    def show_schema_json(self, collection_name: str):
        print("\n" + "=" * 80)
        print(f"📋 Collection: {collection_name} 完整 Schema")
        print("=" * 80)

        collections = self.client.collections.list_all()
        if collection_name in collections:
            config = collections[collection_name]
            schema = {
                "name": collection_name,
                "description": config.description,
                "vectorizer": str(config.vectorizer),
                "properties": [
                    {
                        "name": prop.name,
                        "dataType": str(prop.data_type),
                        "description": prop.description
                    }
                    for prop in config.properties
                ]
            }
            print(json.dumps(schema, indent=2, ensure_ascii=False))
        else:
            print(f"Collection '{collection_name}' 不存在")

    def close(self):
        self.client.close()

def main():
    inspector = WeaviateInspector()
    try:
        while True:
            print("\n" + "=" * 80)
            print("🔧 Weaviate Inspector - 選擇操作")
            print("=" * 80)
            print("1. 列出所有 Collections")
            print("2. 查看 Collection 資料")
            print("3. 查看 Collection 統計")
            print("4. 搜尋資料")
            print("5. 匯出 Collection 為 JSON")
            print("6. 顯示 Collection Schema")
            print("0. 離開")

            choice = input("\n請選擇 (0-6): ").strip()
            if choice == "0":
                break
            elif choice == "1":
                inspector.list_all_collections()
            elif choice == "2":
                name = input("輸入 Collection 名稱: ").strip()
                limit = input("顯示幾筆資料? (預設 10): ").strip() or "10"
                inspector.show_collection_data(name, int(limit))
            elif choice == "3":
                name = input("輸入 Collection 名稱: ").strip()
                inspector.show_collection_stats(name)
            elif choice == "4":
                name = input("輸入 Collection 名稱: ").strip()
                query = input("輸入搜尋關鍵字: ").strip()
                limit = input("顯示幾筆結果? (預設 5): ").strip() or "5"
                inspector.search_demo(name, query, int(limit))
            elif choice == "5":
                name = input("輸入 Collection 名稱: ").strip()
                output = input("輸入輸出檔名 (預設 output.json): ").strip() or "output.json"
                inspector.export_collection_to_json(name, output)
            elif choice == "6":
                name = input("輸入 Collection 名稱: ").strip()
                inspector.show_schema_json(name)
            else:
                print("❌ 無效的選項")
            input("\n按 Enter 繼續...")
    finally:
        inspector.close()
        print("\n👋 再見！")

if __name__ == "__main__":
    main()