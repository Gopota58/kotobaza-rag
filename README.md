# 🐱 Котобаза — локальный RAG-движок + Telegram-бот (offline-first)

![CI](https://github.com/Gopota58/kotobaza-rag/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Stack](https://img.shields.io/badge/stack-FastAPI%20·%20LangChain%20·%20Chroma%20·%20LM%20Studio-informational)

> **Abstract (EN):** *Kotobaza* is a fully local, offline-first **RAG engine** —
> FastAPI + LangChain + Chroma, with a pluggable LLM (LM Studio / Ollama / OpenAI)
> and pluggable embeddings (local Sentence-Transformers **or** any OpenAI-compatible
> embedding API). It ships with a **Telegram bot**, a **web chat**, a **desktop
> client**, a **RAGAS-style evaluation harness**, `pytest` + **GitHub Actions CI**,
> and a **Docker** deployment. The bundled cat-care knowledge base is just a **demo
> corpus** — the engine is domain-agnostic: drop your own `.txt` into `docs/` and you
> get a private Q&A bot.

## Что это

«Котобаза» — не «очередной чатик про котов», а **переиспользуемый движок RAG**, который:

- работает **полностью локально** (генерация — на локальном LLM через LM Studio/Ollama,
  эмбеддинги — локально или через OpenAI-совместимый API);
- имеет **чёткое разделение**: переиспользуемое ядро `rag/engine.py` и тонкие транспорты
  (`app.py` — HTTP, `bot.py` — Telegram, `desktop_client.py` — GUI);
- **обходит блокировку Telegram в РФ** через HTTP-прокси (см. отдельный раздел);
- **сам переиндексирует** базу при изменении файлов в `docs/` (watch + filelock);
- поставляется с **оценкой качества** (faithfulness / answer relevancy / context
  precision / recall) и **зелёным CI**.

База знаний про уход за рыжим котом — демонстрационный корпус (сгенерирован с юмором,
см. `rewrite_cats.py`). Замените `docs/` — и движок будет отвечать по вашей области.

## Демо

> 📹 **Что записать для публикации:** экран с (1) вопросом в веб-чате → развёрнутым
> ответом по документам и (2) тем же в Telegram. Положите файлы `demo_web.gif` и
> `demo_tg.gif` в `docs/` и раскомментируйте блок ниже — GitHub покажет гиф прямо в README.
> Динамическая демка — то, что смотрят первым.

<!--
| Веб-чат | Telegram-бот |
|---|---|
| ![web demo](docs/demo_web.gif) | ![tg demo](docs/demo_tg.gif) |
-->


## Возможности

- **Гибридный ретривер**: векторный cosine + BM25/TF-IDF, слияние через Reciprocal Rank Fusion.
- **Сменный LLM**: LM Studio / Ollama (бесплатно, офлайн) или OpenAI — через `.env`.
- **Сменные эмбеддинги**: локальные `all-MiniLM-L6-v2` **или** любой OpenAI-совместимый
  embedding-API (напр. `nomic-embed-text` из LM Studio — заметно лучше по-русски).
- **Telegram-бот** (`python-telegram-bot`) с обходом блокировок через прокси.
- **Веб-чат** (vanilla JS) + **десктоп-клиент** (Tkinter, собирается в `.exe` через PyInstaller).
- **Авто-reindex**: правите `.txt` в `docs/` — индекс обновляется сам, без ручного запуска.
- **Управление корпусом по API**: загрузка/удаление `.txt` и переиндексация на лету.
- **Переносимо**: нет жёстких путей и секретов в коде — всё через переменные окружения.

## Архитектура

```
                 ┌──────────────┐        ┌───────────────────────────┐
   Telegram ───► │   bot.py     │ ─HTTP─►│        app.py (FastAPI)   │
  (через прокси) │ тонкий клиент│        │  /ask /ingest /documents  │
                 └──────────────┘        │  /health  + веб-чат /     │
   Браузер  ───────────────────────────► └─────────────┬─────────────┘
   (веб-чат)                                           │ владеет
                                                        ▼
                                              ┌───────────────────┐
                                              │  RAGEngine        │
                                              │  (rag/engine.py)  │
                                              │  ├ Retriever k=6  │  вектор + BM25 → RRF
                                              │  ├ Prompt         │
                                              │  └ LLM (ChatOpenAI-совместимый)
                                              └───────┬───────────┘
                                                      │
                        ┌─────────────────────────────┼─────────────────────────┐
                        ▼                             ▼                         ▼
                 ┌────────────┐            ┌──────────────────┐      ┌──────────────────┐
                 │   Chroma   │            │  Embeddings API  │      │   LLM server     │
                 │ (chroma_db)│            │ (LM Studio nomic │      │ (LM Studio /     │
                 └─────▲──────┘            │  / Ollama / ...) │      │  Ollama / OpenAI)│
                       │                   └──────────────────┘      └──────────────────┘
              ┌────────┴─────────┐
              │ watcher (поток)  │  watchfiles + filelock: docs/*.txt изменились → reindex
              └──────────────────┘
```

**Ключевая идея:** единственный владелец индекса и LLM-цепочки — **веб-сервер**.
Бот и десктоп — тонкие HTTP-клиенты к нему. Это исключает гонки при переиндексации
и дублирование тяжёлого состояния (Chroma-коллекция, эмбеддинги) в нескольких процессах.

## Технологии

FastAPI · Uvicorn · LangChain (`langchain`, `langchain-core`, `langchain-community`,
`langchain-chroma`, `langchain-openai`, опц. `langchain-huggingface`) · Chroma ·
`python-telegram-bot` · `watchfiles` + `filelock` · `httpx` · Pydantic /
`pydantic-settings` · `chardet` · Tkinter + PyInstaller · `pytest` · GitHub Actions · Docker.

## Telegram-бот и обход блокировок (РФ)

В России `api.telegram.org` блокируется на уровне IP/сетей, поэтому прямой запрос к
Telegram из кода уходит в таймаут. Решение — **HTTP-прокси** (в демо — локальный
VPN-прокси `KiberportalX` на `127.0.0.1:7890`), через который ходит только Telegram:

```python
# bot.py — python-telegram-bot v22 (httpx-транспорт)
request = HTTPXRequest(
    proxy=settings.telegram_proxy,          # http://127.0.0.1:7890
    httpx_kwargs={"trust_env": False},      # НЕ подхватывать системные HTTP(S)_PROXY
)
app = Application.builder().token(...).request(request).build()
```

Нюансы, которые легко упустить:
- `trust_env=False` — иначе httpx подхватит переменные окружения и попытается пустить
  **localhost-запросы к RAG через тот же прокси** (и всё сломается).
- Проксируется **только** Telegram; обращения к локальному RAG-серверу идут напрямую.
- Бот — **тонкий клиент**: сам не держит Chroma/LLM, а дергает `POST /ask` веб-сервера.
- `run_supervised()` — цикл-супервизор: при падении `polling` ждёт и перезапускается,
  `bootstrap_retries` — старт с ретраями, пока веб-сервер поднимается.

Настройка в `.env`: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_PROXY`, `RAG_API_URL`, `API_KEY`.

## Авто-reindex

Фоновый поток (`watchfiles`) следит за `docs/`. При изменении `.txt` считается
**сигнатура** набора файлов (имена+размеры+mtime); если она совпала — пересборка
пропускается (защита от лишних прогонов). Пересборка идёт под **`FileLock`** и пишет
только в `chroma_db/` (без `rm -rf` папки), поэтому нет блокировок файлов на Windows
и рассинхрона живой цепочки. Логи — в `logs/watcher.log`.

## LLM: два режима и выбор модели

- **Локально (рекомендуется):** LM Studio / Ollama → OpenAI-совместимый сервер на
  `:1234/v1` (или `:11434/v1`). `LLM_BASE_URL=...`, `LLM_MODEL=<имя>`.
- **OpenAI:** `LLM_BASE_URL=` (пусто), `LLM_API_KEY=sk-...`, `LLM_MODEL=gpt-4o-mini`.

### ⚠️ Избегайте reasoning-моделей (Qwen3-thinking и др.)

Reasoning-модели выносят весь вывод в служебное поле `reasoning_content` и возвращают
**пустой `content`** — через стандартный OpenAI-совместимый интерфейс RAG-ответа не
получится (движок честно вернёт «LLM вернул пустой ответ»). Что делать:
- используйте **не-reasoning** instruct-модель (Qwen2.5-Instruct, Llama-3-Instruct,
  Mistral-Instruct, `gpt-4o-mini`);
- либо в LM Studio **отключите «Thinking»** при загрузке модели;
- либо задайте `LLM_EXTRA_BODY={"chat_template_kwargs":{"enable_thinking":false}}`.

Маленькие модели (напр. ornith-1.5b) часто **не поддерживают system-роль** — поставьте
`LLM_USE_SYSTEM_PROMPT=false` (инструкция перенесётся в пользовательское сообщение).

## Быстрый старт

```bash
git clone <repo> kotobaza && cd kotobaza
python -m venv venv && source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                    # заполните LLM_MODEL / ключи
python ingest.py                                        # построить chroma_db из docs/
uvicorn app:app --reload --port 8000                    # веб-чат: http://localhost:8000/
python bot.py                                           # (опц.) Telegram-бот
```

Одной командой (ставит зависимости и качает модель): `start.bat` (Win) / `bash start.sh`.

## Структура проекта

```
kotobaza/
├── rag/engine.py          # ядро: эмбеддинги, Chroma, гибридный retriever, LLM, reindex, watcher
├── app.py                 # FastAPI: /ask /ingest /documents /health, CORS, auth, статика
├── bot.py                 # Telegram-бот (тонкий HTTP-клиент к app.py) + обход блокировок
├── ingest.py              # CLI: python ingest.py
├── config.py              # настройки из .env (pydantic-settings)
├── download_model.py      # скачивание модели эмбеддингов (huggingface_hub)
├── desktop_client.py      # GUI на Tkinter (+ KotobazaClient.spec → .exe через PyInstaller)
├── evaluation/            # RAGAS-стиль метрики + золотой набор (run_eval.py)
├── tests/                 # pytest: ingest, retrieval, API, ask с mock-LLM
├── .github/workflows/     # CI: ruff + pytest
├── static/index.html      # веб-чат
├── docs/                  # демо-корпус (.txt) — замените своим
├── ARCHITECTURE.md        # как разложены модули и потоки
├── DECISIONS.md           # ADR: почему такие решения
├── requirements.txt  .env.example  .gitignore  pyproject.toml  LICENSE
└── Dockerfile  docker-entrypoint.sh  docker-compose.yml  Makefile
```

## Как работает RAG-пайплайн

1. `reindex()` читает `docs/*.txt` (кодировка авто через `chardet`), режет на чанки
   (`RecursiveCharacterTextSplitter`, 400 симв., overlap 60), векторизует **выбранным
   провайдером эмбеддингов** (API или локальный MiniLM) и сохраняет в Chroma.
2. `ask()` запускает **гибридный ретривер**: векторный cosine по нормализованным
   векторам + ключевой BM25/TF-IDF, слияние через Reciprocal Rank Fusion, топ-6 чанков,
   контекст с защитой по длине → промпт + LLM.
3. Ответ парсится. При пустом `content` (reasoning-модель) движок поднимает понятную
   ошибку вместо тихого пустого ответа.

## Конфигурация (`.env`)

| Переменная        | По умолчанию                  | Назначение |
|-------------------|-------------------------------|------------|
| `API_KEY`         | `88888888`                    | ключ к `/ask` и админ-эндпоинтам (смените в проде) |
| `LLM_BASE_URL`    | `http://localhost:1234/v1`    | endpoint LLM (пусто → стандартный OpenAI) |
| `LLM_API_KEY`     | `lm-studio`                   | ключ LLM (для LM Studio — любой) |
| `LLM_MODEL`       | `local-model`                 | имя модели (см. «Выбор LLM») |
| `LLM_TEMPERATURE` / `LLM_TOP_P` / `LLM_MAX_TOKENS` | `0.4` / `0.9` / `512` | параметры генерации |
| `LLM_USE_SYSTEM_PROMPT` | `true`                 | `false` — для моделей без system-роли |
| `LLM_MAX_CONTEXT_CHARS` | `4000`                | защита от переполнения контекстного окна |
| `LLM_EXTRA_BODY`  | `{}`                          | произвольный JSON в API (напр. отключение thinking) |
| `RETRIEVER_K`     | `6`                           | сколько чанков брать в контекст |
| `EMBED_PROVIDER`  | `local`                       | `local` (MiniLM) или `api` (OpenAI-совместимый) |
| `EMBED_API_MODEL` / `EMBED_API_BASE_URL` | —    | модель/endpoint для `EMBED_PROVIDER=api` |
| `TELEGRAM_BOT_TOKEN` | —                          | токен бота (@BotFather) |
| `TELEGRAM_PROXY`  | —                             | `http://127.0.0.1:7890` — обход блокировки Telegram |
| `RAG_API_URL`     | `http://localhost:8000`       | адрес веб-сервера для бота |
| `ALLOWED_ORIGINS` | `*`                           | CORS (через запятую) |

## Оценка качества (RAGAS-стиль)

`evaluation/run_eval.py` прогоняет золотой набор через **тот же** движок и считает
метрики, используя LLM как «судью»:

- **faithfulness** — поддержан ли ответ только фактами из контекста;
- **answer_relevancy** — отвечает ли ответ на вопрос;
- **context_precision** — релевантны ли найденные фрагменты вопросу;
- **context_recall** — покрывает ли контекст эталонный ответ (token-F1).

```bash
python -m evaluation.run_eval     # -> evaluation/results.json + сводка
```

> Почему in-house, а не библиотека `ragas`? Актуальная `ragas` тянет
> `langchain_community.chat_models.vertexai`, которого нет в закреплённом стеке
> LangChain 1.x. Метрики реализованы прозрачно поверх тех же LLM/эмбеддингов проекта —
> офлайн, без хрупких зависимостей, и понятно, что именно считается.

## Тесты и CI

```bash
python -m pytest -q          # 8 тестов: ingest, retrieval, API, ask с mock-LLM
ruff check .                 # линтер (конфиг в pyproject.toml)
```

GitHub Actions (`.github/workflows/ci.yml`) на каждый push/PR ставит зависимости,
крутит `ruff` и `pytest`. Тесты **герметичны**: `tests/conftest.py` фиксирует
детерминированный хэш-эмбеддер (`HashingEmbeddings`) и временный каталог Chroma, а
сквозной `ask()` проверяется с **фиктивной LLM** (`tests/test_ask_mock.py`). Поэтому
сьют зелёный в CI **без GPU, без torch и без запущенного LLM-сервера** — и не трогает
рабочий индекс разработчика.

## Деплой (Docker)

```bash
docker compose up --build
docker compose exec ollama ollama pull qwen2.5:7b   # один раз
# открыть http://localhost:8000/
```

Либо только образ приложения: `docker build -t kotobaza-rag .` (LLM-сервер — ваш).

## Инженерные решения и ограничения

- **Гибридный ретривер + RRF** — заметно точнее чистого cosine на коротких/ключевых
  запросах, особенно для маленьких эмбеддингов.
- **Единый владелец индекса** — Chroma/LLM живёт только в веб-сервере; бот/десктоп —
  HTTP-клиенты. Меньше гонок и дублирования тяжёлого состояния.
- **Переиндексация без `rm -rf`** — пересоздаётся только коллекция через общий
  `PersistentClient` + переподключение цепочки на лету; под `FileLock` между процессами.
- **Обход блокировок Telegram** — проксируется только Telegram, `trust_env=False` для
  локальных вызовов, супервизор с ретраями.
- **Совместимость с моделями** — опциональная system-роль, защита контекста по длине,
  явная ошибка при пустом `content` reasoning-модели.
- **Чего нет (идеи для развития)**: реранкинг поверх гибридного списка, цитирование
  источников в ответе, стриминг токенов, облачный LLM-судья для стабильнее метрик,
  multistage-Docker поменьше.

## Возможные проблемы

- **LM Studio отвечает `500`, хотя `curl` работает** — httpx подхватывает системный
  прокси и пускает localhost через него. Для `localhost`/`127.0.0.1` в коде стоит
  `trust_env=False`. Также `500` бывает от нехватки RAM при нескольких загруженных
  моделях — держите в LM Studio загруженными только нужные (чат + эмбеддинги).
- **Пустой ответ LLM** — reasoning-модель: см. «Выбор LLM».
- **Бот не отвечает** — проверьте, что веб-сервер на `RAG_API_URL` запущен и прокси
  (`TELEGRAM_PROXY`) поднят.
- **Контекстное окно переполнено** — уменьшите `LLM_MAX_CONTEXT_CHARS` / `RETRIEVER_K`.

## Лицензия

MIT — см. [LICENSE](LICENSE).

## Автор

Учебно-портфельный проект для входа в профессию AI-инженера.
