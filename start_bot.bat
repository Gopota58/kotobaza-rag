@echo off
REM Запуск Telegram-бота «Котобаза» в фоновом окне.
REM Бот сам держит соединение с Telegram через прокси (TELEGRAM_PROXY в .env),
REM обходя блокировку в РФ. Для работы должен быть запущен KiberportalX
REM (локальный прокси 127.0.0.1:7890). Логи — в папке logs\.
cd /d %~dp0
start "" /MIN venv\Scripts\python.exe bot.py
echo Bot started. Logs: logs\bot.err.log / logs\bot.out.log. PID: logs\bot.pid
