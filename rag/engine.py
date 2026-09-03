"""
RAGEngine — переиспользуемое ядро движка RAG.

Инкапсулирует всю логику: эмбеддинги, векторную БД (Chroma), ретривер, LLM и
RAG-цепочку. FastAPI-приложение (`app.py`) и CLI (`ingest.py`) — лишь тонкие
обёртки над этим классом. Область применения (котики, юридические документы,
база знаний компании) задаётся только набором документов в `docs/`.

Ключевые свойства:
- локальная работа (эмбеддинги на устройстве, LLM — любой OpenAI-совместимый);
- гибридный ретривер: векторный поиск + BM25 (TF-IDF), слияние через RRF —
  это сильно повышает точность на коротких запросах по сравнению с чистым cosine;
- безопасная переиндексация без удаления папки целиком (нет блокировок файлов);
- потокобезопасное переподключение цепочки после изменения индекса.
"""
import os
import json
import threading
import httpx
import logging

import chromadb
import chardet
import numpy as np
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from sklearn.feature_extraction.text import TfidfVectorizer

from config import settings as default_settings

# Логгер модуля (используется watcher'ом авто-переиндексации).
log = logging.getLogger(__name__)


# --- Промпт ассистента (переиспользуется во всех интерфейсах) ---
SYSTEM_INSTRUCTION = """Ты — Котобаза 😺, эксперт по котам, особенно рыжим. Отвечай дружелюбно, но по делу.

Правила:
- Всегда отвечай по-русски (на русском языке).
- Отвечай СТРОГО на основе предоставленного контекста. Не используй факты, которых нет в контексте.
- Если в контексте нет нужной информации, честно напиши: «В предоставленных документах нет ответа на этот вопрос.»
- Пиши кратко и по существу. Эмодзи — умеренно, только для настроения.
- Не задавай встречных вопросов и не повторяй «согласно контексту» — просто отвечай как эксперт."""


def _build_prompt(use_system: bool) -> ChatPromptTemplate:
    """Собирает промпт. При use_system=False (некоторые локальные модели не
    поддерживают system-роль) инструкция переносится в пользовательское
    сообщение — это повышает совместимость с маленькими моделями."""
    if use_system:
        return ChatPromptTemplate.from_messages([
            ("system", SYSTEM_INSTRUCTION),
            ("human", "Контекст: {context}\n\nВопрос: {input}"),
        ])
    return ChatPromptTemplate.from_messages([
        ("human", SYSTEM_INSTRUCTION + "\n\nКонтекст: {context}\n\nВопрос: {input}"),
    ])


# --- Авто-определение кодировки (.txt могут быть в UTF-8 или Windows-1251) ---
def detect_encoding(file_path: str) -> str:
    with open(file_path, "rb") as f:
        raw = f.read(10000)
    result = chardet.detect(raw)
    return result["encoding"] or "utf-8"


class AutoDetectTextLoader(TextLoader):
    def __init__(self, file_path, **kwargs):
        encoding = detect_encoding(file_path)
        super().__init__(file_path, encoding=encoding, **kwargs)


# --- Фабрики ---
def build_embeddings(s=None):
    s = s or default_settings
    # API-провайдер (напр. LM Studio / Ollama) — сильные мультилингвальные
    # эмбеддинги (nomic-embed-text и др.), которые заметно лучше понимают
    # русский, чем локальный MiniLM. Для localhost отключаем trust_env, чтобы
    # httpx ходил напрямую (см. обоснование в build_llm).
    if getattr(s, "embed_provider", "local") == "api":
        from langchain_openai import OpenAIEmbeddings

        base = s.embed_api_base_url or s.llm_base_url
        kwargs = dict(
            model=s.embed_api_model,
            base_url=base,
            api_key=s.embed_api_key,
            check_embedding_ctx_length=False,
        )
        if base and ("localhost" in base or "127.0.0.1" in base):
            kwargs["http_client"] = httpx.Client(trust_env=False)
        return OpenAIEmbeddings(**kwargs)
    # Локальные эмбеддинги (MiniLM через sentence-transformers/torch) — только по
    # запросу. Импорт ленивый, чтобы при embed_provider=api не тянуть тяжёлый torch.
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=s.resolve_embedding_model(),
        model_kwargs={"device": s.embed_device},
        encode_kwargs={"normalize_embeddings": s.embed_normalize},
    )


def _parse_extra_body(raw: str) -> dict:
    """Парсит JSON из настройки llm_extra_body (на случай пустой/невалидной строки)."""
    try:
        data = json.loads(raw or "{}")
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def build_llm(s=None):
    s = s or default_settings
    extra = _parse_extra_body(s.llm_extra_body)
    # langchain-openai 1.6 по умолчанию шлёт `max_completion_tokens`, который
    # LM Studio и ряд OpenAI-совместимых серверов не понимают и отвечают 500.
    # Пробрасываем `max_tokens` через extra_body (raw-passthrough) — это
    # поддерживают и LM Studio, и OpenAI.com.
    extra["max_tokens"] = s.llm_max_tokens
    kwargs = dict(
        model=s.llm_model,
        temperature=s.llm_temperature,
        top_p=s.llm_top_p,
        api_key=s.llm_api_key,
        extra_body=extra,
    )
    if s.llm_base_url:
        kwargs["base_url"] = s.llm_base_url
    # Локальные endpoint'ы (LM Studio / Ollama на localhost) часто оказываются
    # недоступны из-за прокси, заданного в переменных окружения: httpx (клиент
    # OpenAI SDK) маршрутизирует localhost через прокси и получает 500. Для
    # локальных адресов отключаем trust_env, чтобы ходить напрямую.
    if s.llm_base_url and ("localhost" in s.llm_base_url or "127.0.0.1" in s.llm_base_url):
        # trust_env=False — ходить на localhost напрямую, минуя прокси из окружения.
        # max_keepalive_connections=0 — не пулить keep-alive соединения: локальные
        # LLM-серверы (LM Studio/Ollama) часто закрывают их после ответа, и повторное
        # переиспользование мёртвого соединения даёт 500 на 2-м и следующих запросах.
        # timeout — чтобы нестабильная локальная модель не вешала запрос бесконечно
        # (зависание превращается в быструю ошибку и триггерит повтор в ask()).
        kwargs["http_client"] = httpx.Client(
            trust_env=False,
            limits=httpx.Limits(max_keepalive_connections=0),
            timeout=httpx.Timeout(90.0),
        )
    return ChatOpenAI(**kwargs)


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


class RAGEngine:
    def __init__(self, settings=default_settings, embeddings=None, llm=None):
        self.settings = settings
        self.embeddings = embeddings or build_embeddings(self.settings)
        self.llm = llm or build_llm(self.settings)
        self.collection_name = self.settings.collection_name
        # Один PersistentClient на процесс — основа безопасной переиндексации
        self.client = chromadb.PersistentClient(path=self.settings.chroma_dir)
        self._lock = threading.Lock()
        self._watcher_running = False
        self._build()

    # --- Внутреннее построение цепочки (вызывается и при старте, и после реиндекса) ---
    def _build_chain(self):
        prompt = _build_prompt(self.settings.llm_use_system_prompt)

        def _ctx(q):
            ctx = format_docs(self.retrieve(q))
            limit = self.settings.llm_max_context_chars
            if limit and len(ctx) > limit:
                # усекаем по границе строки, чтобы не резать слова посередине
                ctx = ctx[:limit]
                if "\n" in ctx:
                    ctx = ctx[: ctx.rfind("\n")]
                ctx = ctx + "\n…(контекст усечён по лимиту)"
            return ctx

        self._prompt = prompt
        return (
            {"context": _ctx, "input": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def _build_keyword_index(self):
        """BM25-подобный индекс (TF-IDF) поверх всех чанков для гибридного поиска."""
        try:
            coll = self.client.get_collection(self.collection_name)
            data = coll.get(include=["documents"])
            texts = data.get("documents") or []
        except Exception:
            texts = []
        self._chunk_texts = texts
        if texts:
            self._vectorizer = TfidfVectorizer()
            self._tfidf_matrix = self._vectorizer.fit_transform(texts)
        else:
            self._vectorizer = None

    def _build(self):
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
        )
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.settings.retriever_k}
        )
        self._build_keyword_index()
        self.rag_chain = self._build_chain()

    # --- Гибридный ретривер: вектор + BM25, слияние через Reciprocal Rank Fusion ---
    # Раскрытие совершенного вида и синонимов для русского корпуса (без лемматизатора
    # и интернета): «помыть» -> «мыть купать шампунь...», «покормить» -> «кормить
    # кормление...» и т.п. Расширенный запрос идёт и в векторный, и в ключевой поиск.
    _EXPANSIONS = [
        ("мыть", "мыть купать шампунь уход за шерстью расчёсывать"),
        ("купать", "мыть купать шампунь уход за шерстью"),
        ("корм", "кормить кормление питание еда"),
        ("еда", "кормить кормление питание еда"),
        ("питани", "кормить кормление питание еда"),
        ("пить", "вода пить поилка фонтанчик"),
        ("вода", "вода пить поилка фонтанчик"),
        ("туалет", "туалет лоток наполнитель"),
        ("лоток", "туалет лоток наполнитель"),
        ("когт", "когти когтеточка точить"),
        ("дресс", "дрессировка воспитание приучить"),
        ("воспит", "дрессировка воспитание приучить"),
        ("приуч", "дрессировка воспитание приучить"),
        ("здоров", "здоровье ветеринар болезнь лечение"),
        ("боле", "здоровье ветеринар болезнь лечение"),
        ("ветерин", "здоровье ветеринар болезнь лечение"),
        ("леч", "здоровье ветеринар болезнь лечение"),
        ("игра", "игры активность игрушки"),
        ("активн", "игры активность игрушки"),
        ("сон", "сон отдых спать"),
        ("спит", "сон отдых спать"),
        ("отдых", "сон отдых спать"),
        ("гуля", "прогулка шлейка гулять"),
        ("прогул", "прогулка шлейка гулять"),
        ("шлейк", "прогулка шлейка гулять"),
        ("котёнок", "котёнок котенок маленький"),
        ("пожил", "пожилой старый возраст"),
        ("стар", "пожилой старый возраст"),
        ("зуб", "зубы зубной"),
        ("шерст", "шерсть вычёсывать расчёсывать линька"),
        ("линьк", "шерсть вычёсывать расчёсывать линька"),
        ("вычёс", "шерсть вычёсывать расчёсывать"),
        ("расчёс", "шерсть вычёсывать расчёсывать"),
        ("мяука", "мяукать общение голос мурлыкать"),
        ("мурлы", "мяукать общение голос мурлыкать"),
        ("общен", "общение мяукать ласка"),
        ("беремен", "беременность роды котята"),
        ("род", "беременность роды котята"),
        ("агресс", "агрессия кусать царапать"),
        ("куса", "агрессия кусать царапать"),
        ("царап", "агрессия кусать царапать"),
        ("аллерг", "аллергия"),
        ("перевоз", "перевозка переноска машина переезд"),
        ("переезд", "перевозка переноска машина переезд"),
        ("собак", "собака социализация животные"),
        ("других живот", "социализация животные"),
        ("ран", "первая помощь рана царапина"),
        ("помощ", "первая помощь рана царапина"),
        ("операц", "операция стерилизация кастрация"),
        ("стерилиз", "операция стерилизация кастрация"),
        ("кастр", "операция стерилизация кастрация"),
    ]

    def _expand_query(self, question: str) -> str:
        q = question.lower()
        extra = []
        for key, add in self._EXPANSIONS:
            if key in q and add not in extra:
                extra.append(add)
        return (question + " " + " ".join(extra)).strip()

    def _keyword_search(self, question: str, k: int):
        q = self._vectorizer.transform([question])
        sims = (self._tfidf_matrix @ q.T).toarray().ravel()
        order = np.argsort(sims)[::-1]
        out = []
        for i in order:
            if sims[i] <= 0:
                break
            out.append(Document(page_content=self._chunk_texts[i]))
            if len(out) >= k:
                break
        return out

    def _retrieve_hybrid(self, question: str, k: int):
        if self._vectorizer is None:
            return self.retriever.invoke(question)
        n = len(self._chunk_texts)
        # Расширяем запрос (совершенный вид/синонимы), чтобы векторный и
        # ключевой поиск лучше находили нужные чанки при русской морфологии.
        eq = self._expand_query(question)
        vs_docs = self.vectorstore.similarity_search(eq, k=min(20, n))
        kw_docs = self._keyword_search(eq, k=min(20, n))

        fused = {}
        # BM25 чуть весомее вектора: для русского корпуса ключевой поиск точно
        # ловит морфологические варианты (кормить/кормление, мыть/купать,
        # мяукать/мяуканье), которые слабый эмбеддинг иногда упускает. Без этого
        # релевантный чанок тонет в RRF-слиянии под «похожими по длине» векторами.
        vs_w, kw_w = 1.0, 2.5
        for rank, d in enumerate(vs_docs):
            fused[d.page_content] = fused.get(d.page_content, 0.0) + vs_w / (rank + 1 + 60)
        for rank, d in enumerate(kw_docs):
            fused[d.page_content] = fused.get(d.page_content, 0.0) + kw_w / (rank + 1 + 60)

        all_docs = {d.page_content: d for d in vs_docs + kw_docs}
        ranked = sorted(fused.keys(), key=lambda t: fused[t], reverse=True)
        return [all_docs[t] for t in ranked[:k]]

    # --- Публичный API движка ---
    def ask(self, question: str, max_retries: int = 3) -> str:
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                answer = (self.rag_chain.invoke(question) or "").strip()
            except Exception as e:  # любая ошибка сети/LLM — повторим
                last_error = e
                answer = ""
            if answer:
                return answer
            # Пустой ответ: часто это «холодный старт» локальной модели
            # (первый запрос после загрузки в LM Studio / Ollama возвращает
            # пустой content). Повторяем, чтобы пользователь не видел 500.
            last_error = ValueError(
                "LLM вернул пустой ответ. Вероятная причина — reasoning-модель "
                "(Qwen3-thinking и др.) без отключённого режима размышлений. "
                "Используйте не-reasoning модель или отключите thinking в настройках "
                "сервера LM Studio / Ollama (либо задайте LLM_EXTRA_BODY)."
            )
        raise last_error

    def retrieve(self, question: str, k: int | None = None):
        k = k or self.settings.retriever_k
        return self._retrieve_hybrid(question, k)

    def contexts_for(self, question: str, k: int | None = None) -> list[str]:
        return [d.page_content for d in self.retrieve(question, k)]

    def list_documents(self) -> list[str]:
        if not os.path.exists(self.settings.docs_dir):
            return []
        return os.listdir(self.settings.docs_dir)

    def add_document(self, filename: str, content: bytes) -> dict:
        if not filename.endswith(".txt"):
            raise ValueError("Только .txt файлы")
        os.makedirs(self.settings.docs_dir, exist_ok=True)
        path = os.path.join(self.settings.docs_dir, filename)
        with open(path, "wb") as f:
            f.write(content)
        return self.reindex()

    def remove_document(self, filename: str) -> dict:
        path = os.path.join(self.settings.docs_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл не найден: {filename}")
        os.remove(path)
        return self.reindex()

    def reindex(self, docs_dir=None, chunk_size=400, chunk_overlap=60) -> dict:
        """
        Полная переиндексация docs_dir -> коллекция Chroma.
        Пересоздаёт только коллекцию (без удаления папки), поэтому безопасно
        при работающем сервере. Потокобезопасно.
        """
        with self._lock:
            docs_dir = docs_dir or self.settings.docs_dir
            if not os.path.exists(docs_dir):
                return {"error": f"Папка {docs_dir} не найдена"}

            loader = DirectoryLoader(
                docs_dir, glob="**/*.txt", loader_cls=AutoDetectTextLoader
            )
            documents = loader.load()
            if not documents:
                return {"error": "Нет документов для индексации"}

            chunks = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            ).split_documents(documents)

            # Пересоздаём коллекцию через тот же PersistentClient
            try:
                self.client.delete_collection(self.collection_name)
            except Exception:
                pass  # коллекции ещё нет — нормально при первом запуске

            Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                client=self.client,
                collection_name=self.collection_name,
            )

            # Переподключаем vectorstore/retriever/rag_chain к свежей коллекции
            self._build()

        return {
            "status": "ok",
            "documents_loaded": len(documents),
            "chunks_created": len(chunks),
            "message": f"Индексация завершена: {len(documents)} документов, {len(chunks)} чанков",
        }

    # --- Авто-переиндексация при изменении docs/ ---
    def _docs_signature(self) -> str:
        """Сигнатура содержимого docs/ (число .txt + mtime самого свежего файла).
        Позволяет не переиндексировать заново, если другой процесс (web-сервер
        или бот) уже перестроил индекс под текущее состояние папки."""
        docs_dir = self.settings.docs_dir
        if not os.path.isdir(docs_dir):
            return "empty"
        newest, count = 0, 0
        for root, _dirs, files in os.walk(docs_dir):
            for f in files:
                if f.endswith(".txt"):
                    count += 1
                    try:
                        m = os.path.getmtime(os.path.join(root, f))
                    except OSError:
                        continue
                    if m > newest:
                        newest = m
        return f"{count}:{newest}"

    def _reindex_safe(self) -> dict:
        """Потокобезопасная (внутри процесса, через self._lock) и межпроцессная
        (через FileLock) переиндексация. Если другой процесс уже перестроил
        индекс под текущее состояние docs/ (см. .docs_sig), просто перезагружаем
        движок без повторного эмбеддинга."""
        lock_path = os.path.join(self.settings.chroma_dir, ".reindex.lock")
        os.makedirs(self.settings.chroma_dir, exist_ok=True)
        from filelock import FileLock

        flock = FileLock(lock_path, timeout=300)
        with flock:
            current = self._docs_signature()
            sig_file = os.path.join(self.settings.chroma_dir, ".docs_sig")
            shared = ""
            if os.path.exists(sig_file):
                try:
                    shared = open(sig_file, encoding="utf-8").read().strip()
                except OSError:
                    shared = ""
            if current and current == shared:
                # Кто-то другой уже пересобрал под это состояние docs/ —
                # просто переподключаем vectorstore/retriever/rag_chain.
                with self._lock:
                    self._build()
                return {
                    "status": "reloaded",
                    "message": "Индекс перезагружен без повторной переиндексации",
                }
            result = self.reindex()
            try:
                with open(sig_file, "w", encoding="utf-8") as fh:
                    fh.write(current or "empty")
            except OSError:
                pass
            return result

    def _wlog(self, msg: str) -> None:
        """Дублирует ключевые события наблюдателя в logs/watcher.log — мимо
        системы логирования uvicorn (которая глушит логгер rag.engine), чтобы
        можно было отладить авто-reindex в веб-сервере."""
        try:
            import datetime

            os.makedirs("logs", exist_ok=True)
            with open("logs/watcher.log", "a", encoding="utf-8") as fh:
                fh.write(f"{datetime.datetime.now().isoformat()} {msg}\n")
        except OSError:
            pass

    def start_watcher(self) -> None:
        """Запускает фоновый daemon-поток, который следит за папкой docs/ и при
        любом изменении .txt автоматически переиндексирует базу (с дебаунсом 2 с).
        Работает независимо в каждом процессе (web-сервер и бот), но благодаря
        FileLock не конфликтует при одновременной правке файлов."""
        if getattr(self, "_watcher_running", False):
            return
        try:
            from watchfiles import watch
        except ImportError:
            log.warning("watchfiles не установлен — авто-reindex отключён")
            return

        docs_dir = self.settings.docs_dir
        os.makedirs(docs_dir, exist_ok=True)
        self._watcher_running = True
        # uvicorn применяет logging.dictConfig(disable_existing_loggers=True) и
        # глушит логгер rag.engine, созданный до старта сервера. Возвращаем ему
        # вывод, чтобы сообщения авто-reindex были видны в логах веб-сервера.
        log.disabled = False
        log.info("Авто-reindex: слежу за изменениями в %s", docs_dir)
        self._wlog(f"watcher started for {docs_dir} (pid={os.getpid()})")

        def _loop():
            import time

            try:
                for _changes in watch(
                    docs_dir,
                    step=500,
                    watch_filter=lambda change, path: str(path).endswith(".txt"),
                ):
                    # Дебаунс: ждём, пока правка утихнет (редакторы шлют
                    # много событий подряд — не реиндексируем на каждое).
                    time.sleep(2)
                    if not getattr(self, "_watcher_running", False):
                        break
                    log.info(
                        "Авто-reindex: изменение в %s — пересобираю индекс…", docs_dir
                    )
                    self._wlog("change detected")
                    try:
                        res = self._reindex_safe()
                        log.info("Авто-reindex: %s", res.get("message"))
                        self._wlog(f"reindex done: {res.get('message')}")
                    except Exception:
                        log.exception("Авто-reindex: ошибка переиндексации")
                        self._wlog("reindex ERROR")
            except Exception:
                log.exception("Авто-reindex: watcher остановлен")
                self._wlog("watcher stopped with error")
                self._watcher_running = False

        threading.Thread(target=_loop, daemon=True).start()
