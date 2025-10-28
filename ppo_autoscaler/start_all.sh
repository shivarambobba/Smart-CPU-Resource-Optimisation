#!/usr/bin/env bash
# Start model server and Flask app in the background (development helper)
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT_DIR"

echo "Starting model server (FastAPI) on port 8000..."
uvicorn model_server_real:app --app-dir "$ROOT_DIR" --host 127.0.0.1 --port 8000 &
MODEL_PID=$!
echo "model server pid=$MODEL_PID"

echo "Starting Flask app (app_fixed.py) on port 5002..."
python3 app_fixed.py &
FLASK_PID=$!
echo "flask pid=$FLASK_PID"

echo "Servers started. PIDs: model=$MODEL_PID flask=$FLASK_PID"
echo "To stop: kill $MODEL_PID $FLASK_PID"
