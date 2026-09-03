"""
CLI для построения индекса: `python ingest.py`.

Логика переиндексации инкапсулирована в RAGEngine (rag/engine.py).
"""
from rag.engine import RAGEngine

if __name__ == "__main__":
    engine = RAGEngine()
    result = engine.reindex()
    print(result["message"])
