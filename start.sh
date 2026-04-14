#!/bin/sh
set -e
echo "[boot] PORT=$PORT"
echo "[boot] DATABASE_URL tail: $(echo $DATABASE_URL | tail -c 40)"
python -c "import orchestrator.main; print('[boot] imports OK', flush=True)"
exec uvicorn orchestrator.main:app --host 0.0.0.0 --port "${PORT:-8000}" --log-level info
