"""
Конфигурация проекта «Котобаза».

Все настройки вынесены в переменные окружения (.env) — больше нет
жёстко прописанных абсолютных путей и секретов, проект легко переносится
между машинами и публикуется на GitHub.
"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- API-сервер ---
    api_key: str = "88888888"     # ключ доступа к /ask и админ-эндпоинтам (смените в продакшене!)
    host: str = "0.0.0.0"
    port: int = 8000

    # --- Данные и векторная БД (пути относительны корня проекта) ---
    docs_dir: str = str(BASE_DIR / "docs")
    chroma_dir: str = str(BASE_DIR / "chroma_db")
    collection_name: str = "kotobaza"
    model_dir: str = str(BASE_DIR / "models" / "all-MiniLM-L6-v2")
    embedding_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Параметры эмбеддингов ---
    embed_device: str = "cpu"
    embed_normalize: bool = True

    # --- Провайдер эмбеддингов ---
    # "local" — HuggingFace MiniLM (полностью офлайн, локально). Слабоват для
    #   русского корпуса: короткие запросы ранжируются по длине вектора, а не по
    #   смыслу, из-за чего релевантные чанки не попадают в топ.
    # "api"   — OpenAI-совместимый endpoint (LM Studio / Ollama). Позволяет
    #   использовать сильные мультилингвальные модели (напр. nomic-embed-text),
    #   которые заметно лучше понимают русский и дают рабочий RAG.
    embed_provider: str = "local"
    embed_api_base_url: str = ""      # пусто -> берётся llm_base_url
    embed_api_model: str = "text-embedding-nomic-embed-text-v1.5"
    embed_api_key: str = "lm-studio"

    # --- Ретривер ---
    retriever_k: int = 8

    # --- LLM (любой OpenAI-совместимый endpoint) ---
    llm_provider: str = "local"      # "local" (LM Studio / Ollama) или "openai"
    llm_base_url: str = "http://localhost:1234/v1"  # пусто -> стандартный OpenAI
    llm_api_key: str = "lm-studio"   # для LM Studio подходит любое значение
    llm_model: str = "local-model"   # имя модели в выбранном сервере
    llm_temperature: float = 0.4
    llm_top_p: float = 0.9
    llm_max_tokens: int = 512        # ограничение длины генерации
    # Некоторые маленькие локальные модели не поддерживают system-роль;
    # при False инструкция переносится в пользовательское сообщение.
    llm_use_system_prompt: bool = True
    # Максимальный размер контекста (символы), передаваемого в LLM. Защищает
    # от переполнения контекстного окна модели и снижает задержку.
    llm_max_context_chars: int = 4000
    # Произвольный JSON, пробрасываемый в API (напр. для LM Studio:
    # {"chat_template_kwargs": {"enable_thinking": false}} — отключение
    # reasoning у моделей семейства Qwen3, если сервер это поддерживает).
    llm_extra_body: str = "{}"

    # --- CORS (через запятую; "*" — разрешить все, НЕ использовать с куками) ---
    allowed_origins: str = "*"

    # --- Telegram-бот ---
    # Токен и прокси читаются ТОЛЬКО из .env (через pydantic-settings) — в коде секретов нет.
    telegram_bot_token: str = ""                # токен от @BotFather
    telegram_proxy: str = ""                    # напр. http://127.0.0.1:7890; пусто = напрямую
    rag_api_url: str = "http://localhost:8000"  # адрес веб-сервера, к которому ходит бот

    @property
    def allowed_origins_list(self) -> list[str]:
        if self.allowed_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    def resolve_embedding_model(self) -> str:
        """Локальная модель (если скачана в models/), иначе HF repo-id с авто-загрузкой."""
        local = Path(self.model_dir)
        if local.exists() and (local / "config.json").exists():
            return str(local)
        return self.embedding_model_id


settings = Settings()
