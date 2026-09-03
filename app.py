"""
FastAPI-обёртка над RAGEngine.

Сама логика RAG вынесена в `rag/engine.py`, здесь только HTTP-интерфейс:
аутентификация, CORS, отдача статики и маршрутизация на методы движка.
"""
import os
import shutil
import logging

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Header
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from config import settings
from rag.engine import RAGEngine

# --- Инициализация движка (загружает эмбеддинги и открывает векторную БД) ---
engine = RAGEngine(settings)

# Прогрев LLM при старте: первый запрос к локальной GGUF-модели (LM Studio /
# Ollama) часто возвращает пустой ответ («холодный старт»). Один холостой
# вызов инициирует загрузку/компиляцию модели, чтобы первый вопрос
# пользователя сразу получил нормальный ответ, а не 500.
try:
    engine.ask("Привет")
except Exception:
    pass

# Авто-переиндексация: следим за папкой docs/ и при изменении .txt сами
# пересобираем индекс (см. rag/engine.py -> RAGEngine.start_watcher).
engine.start_watcher()

# --- FastAPI приложение ---
app = FastAPI(
    title="Котобаза API",
    description="Локальный RAG-движок по документам о котах (и не только)",
    version="1.0",
)

# --- CORS (из .env; credentials отключаются, если origin == "*") ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials="*" not in settings.allowed_origins_list,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Статика для фронтенда ---

@app.on_event("startup")
def _enable_rag_engine_logging():
    # uvicorn при старте применяет logging.dictConfig(disable_existing_loggers=True)
    # и глушит логгер rag.engine, созданный до конфигурации. Возвращаем ему
    # видимость, чтобы сообщения авто-reindex появлялись в логах сервера.
    rl = logging.getLogger("rag.engine")
    rl.disabled = False
    rl.propagate = True
    rl.setLevel(logging.INFO)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


# --- Аутентификация (API Key) ---
def verify_api_key(api_key: str = Header(None, alias="X-API-Key")):
    if not api_key or api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key


# --- Модели данных ---
class Question(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str


# ============================================
# ЭНДПОИНТЫ
# ============================================

@app.get("/", tags=["System"])
async def root():
    if os.path.exists(os.path.join("static", "index.html")):
        return RedirectResponse(url="/static/index.html")
    return {
        "message": "Котобаза API. Документация — /docs",
        "health": "/health",
        "ask": "POST /ask",
    }


@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AnswerResponse, tags=["RAG"])
async def ask(question: Question, api_key: str = Depends(verify_api_key)):
    """
    Задать вопрос по документам (retrieval + генерация).
    """
    try:
        return {"answer": engine.ask(question.question)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", tags=["Admin"])
async def ingest_documents(api_key: str = Depends(verify_api_key)):
    """
    Принудительная переиндексация всех документов из docs/.
    """
    try:
        result = engine.reindex()
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload", tags=["Admin"])
async def upload_file(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key),
):
    """
    Загрузить .txt и переиндексировать все документы.
    """
    content = await file.read()
    try:
        result = engine.add_document(file.filename, content)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return {
            "status": "ok",
            "message": f"Файл {file.filename} загружен и индексация выполнена",
            "details": result,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents", tags=["Admin"])
async def list_documents(api_key: str = Depends(verify_api_key)):
    """
    Список документов в docs/.
    """
    return {"documents": engine.list_documents()}


@app.delete("/documents/{filename}", tags=["Admin"])
async def delete_document(
    filename: str,
    api_key: str = Depends(verify_api_key),
):
    """
    Удалить документ и переиндексировать.
    """
    try:
        result = engine.remove_document(filename)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return {
            "status": "ok",
            "message": f"Файл {filename} удалён, индексация выполнена",
            "details": result,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
