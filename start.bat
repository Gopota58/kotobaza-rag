@echo off
title RAG-бот Котобаза
setlocal

:: Проверка наличия Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Python не найден! Установите Python 3.10+ и добавьте в PATH.
    pause
    exit /b
)

:: Проверка и создание виртуального окружения
if not exist venv (
    echo Создание виртуального окружения...
    python -m venv venv
    call venv\Scripts\activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate
)

:: Подготовка модели эмбеддингов
echo Подготовка модели эмбеддингов (при необходимости)...
python download_model.py || echo [info] Модель уже на месте или будет загружена автоматически.

:: Запуск API-сервера в фоне
echo Запуск API-сервера...
start "RAG API" cmd /k uvicorn app:app --reload --host 0.0.0.0 --port 8000

:: Ожидание, пока сервер поднимется
echo Ожидание запуска сервера...
timeout /t 5 /nobreak >nul

:: Открытие фронтенда в браузере
start http://127.0.0.1:8000/

echo.
echo ==========================================================
echo  Котобаза запущена!
echo  API:           http://127.0.0.1:8000
echo  Фронтенд:      http://127.0.0.1:8000/static/index.html
echo  Документация:  http://127.0.0.1:8000/docs
echo  Для остановки закройте окно "RAG API".
echo ==========================================================
pause
