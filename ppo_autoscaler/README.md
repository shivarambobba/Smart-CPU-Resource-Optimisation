# Smart CPU Autoscaler — Extracted Project

This repository is a runnable extraction of the model and server code referenced in the attached paper "Smart CPU Autoscaling using Transformer LSTM and PPO".

What I created:
- A compact PyTorch implementation of a Transformer+LSTM model (`src/transformer_lstm.py`).
- A small script to create and save a dummy model checkpoint (`scripts/save_dummy_model.py`).
- A model server that can load the real model if available and otherwise falls back to `DummyModel` (`model_server_real.py`).
- Updated `requirements.txt` with Torch (CPU) and FastAPI/uvicorn.

Quick start (development / local):

1) Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) (Optional) Create a dummy checkpoint the server can load:

```bash
python3 scripts/save_dummy_model.py
```

3) Run the model server (FastAPI):

```bash
uvicorn model_server_real:app --host 127.0.0.1 --port 8000 --reload
```

4) Health check:

```bash
curl http://127.0.0.1:8000/health
```

Notes:
- The provided model implementation is intentionally small and CPU-friendly so you can run it locally even without GPU.
- The server will automatically fall back to the existing `DummyModel` behavior if a checkpoint is missing.

Files added/changed:
- `src/transformer_lstm.py` — PyTorch model implementation
- `scripts/save_dummy_model.py` — saves a small random checkpoint to `models/transformer_lstm.pth`
- `model_server_real.py` — FastAPI server that loads the model
- `requirements.txt` — added torch and related packages (CPU-friendly)
# PPO Autoscaler (small project)

This folder contains a small Flask UI (`app.py`) and supporting files for
running a tiny PPO-based autoscaler demo. It is derived from the research
notes in the attached PDF (Smart CPU Autoscaling using Transformer/LSTM and PPO).

What is included
- `app.py` - small Flask app with a `/predict` endpoint and `/ui` template.
  Uses a safe `DummyModel` fallback to avoid crashing on systems without a
  compatible PyTorch/Stable-Baselines installation.
- `model_loader.py` - helper to try loading the real PPO model in a subprocess
  (safe against native-level crashes during import).
- `model_server.py` - a FastAPI-based model server that will load the PPO
  model if available and expose `/predict` for other services to call.
- `test_run_app.py` - quick test script that imports the Flask app and
  exercises endpoints using Flask's test client.
- `ppo_cpu_autoscale.zip` - (existing) trained PPO model artifact.

Why this structure
- On some macOS/Python combinations importing PyTorch and stable-baselines3
  can abort the Python process with a C++ runtime error. To make the web UI
  reliable we avoid top-level imports of torch in the Flask process and either:
  1. Use a DummyModel for local UI/testing; or
  2. Run a separate model server process that imports torch (if available).

Quick start (minimal, no torch required)
1. Create and activate a venv:

```bash
cd ppo_autoscaler
python3 -m venv .venv
source .venv/bin/activate
```

2. Install minimal requirements and run the Flask app:

```bash
pip install --upgrade pip
pip install -r requirements.txt
# run the Flask app
python app.py
```

3. In another terminal, test endpoints:

```bash
curl -i http://127.0.0.1:5001/
curl -i -X POST -H "Content-Type: application/json" -d '{"cpu": 72}' http://127.0.0.1:5001/predict
```

Optional: run the model server (if you have torch/stable-baselines3 installed)

```bash
# start model server with uvicorn
uvicorn model_server:app --host 127.0.0.1 --port 8000
# then point your UI or other clients to http://127.0.0.1:8000/predict
```

Next steps and ideas from the paper
- Add a training/inference pipeline based on the Transformer+LSTM PPO design
  in the paper: implement data preprocessing, reward shaping, and training
  loop (not included here).
- Create Docker images for the web UI and a separate model server (recommended).
- Add a small admin UI to view request history and model decisions.
- Add tests and CI to validate model loading on supported environments.

If you'd like, I can:
- Scaffold Dockerfiles + docker-compose to run the Flask UI and model server.
- Implement the Transformer/LSTM training pipeline from the paper into a
  reproducible training script with unit tests.
- Make the Flask app forward predict requests to the model server automatically
  (detecting it via an env var) and add retry/timeout logic.

Tell me which next step you'd like me to implement and I'll proceed.
