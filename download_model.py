"""
Скачивает модель эмбеддингов all-MiniLM-L6-v2 в ./models/all-MiniLM-L6-v2.

Запускать один раз, если вы хотите работать офлайн (без авто-загрузки
из Hugging Face). При наличии интернета app.py / ingest.py скачают модель
автоматически при первом обращении.
"""
from huggingface_hub import snapshot_download

from config import settings

if __name__ == "__main__":
    print(f"Загрузка модели {settings.embedding_model_id} -> {settings.model_dir}")
    path = snapshot_download(
        repo_id=settings.embedding_model_id,
        local_dir=settings.model_dir,
    )
    print(f"Готово: {path}")
