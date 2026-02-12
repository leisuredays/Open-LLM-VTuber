"""ChromaDB에 저장된 RAG 지식 데이터를 확인하는 스크립트."""

import chromadb
import json
import sys


def main():
    client = chromadb.PersistentClient(path="chromadb_data")
    collections = client.list_collections()

    if not collections:
        print("ChromaDB에 저장된 컬렉션이 없습니다.")
        return

    print(f"=== ChromaDB 컬렉션 목록 ({len(collections)}개) ===\n")

    for col_info in collections:
        name = col_info.name if hasattr(col_info, "name") else str(col_info)
        collection = client.get_collection(name=name)
        count = collection.count()

        print(f"📁 컬렉션: {name}")
        print(f"   문서 수: {count}")

        if count == 0:
            print()
            continue

        # 전체 데이터 가져오기
        data = collection.get(include=["documents", "metadatas"])

        for i, (doc_id, doc, meta) in enumerate(
            zip(data["ids"], data["documents"], data["metadatas"])
        ):
            print(f"\n   [{i+1}] ID: {doc_id}")
            print(f"       메타: {json.dumps(meta, ensure_ascii=False)}")
            # 긴 문서는 200자까지만 표시
            preview = doc if len(doc) <= 200 else doc[:200] + "..."
            print(f"       내용: {preview}")

        print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
