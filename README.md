🚀 Smart CPU Autoscaler using Transformer-LSTM & PPO

An AI-powered cloud autoscaling system that combines Transformer-LSTM forecasting with Proximal Policy Optimization (PPO) reinforcement learning to make intelligent CPU scaling decisions.

⸻

🌟 Overview

Traditional autoscaling systems rely on static threshold rules and reactive monitoring. This project introduces a smarter AI-driven approach inspired by modern cloud research.

Key Components

* 🔮 Transformer-LSTM for CPU usage forecasting
* 🤖 PPO Reinforcement Learning Agent for intelligent scaling decisions
* ⚡ FastAPI Model Server for real-time inference
* 🌐 Flask Dashboard for testing and visualization
* 🛡️ Safe fallback mode using a DummyModel when AI dependencies are unavailable
* 💻 Lightweight and CPU-friendly for local development

⸻

🏗️ Architecture

                    Historical CPU Metrics
                              │
                              ▼
                  ┌─────────────────────┐
                  │ Transformer-LSTM    │
                  │ CPU Forecasting     │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ PPO RL Agent        │
                  │ Scaling Decision    │
                  └──────────┬──────────┘
                             │
                             ▼
                    Scale Up / Scale Down

⸻

✨ Features

🔮 AI-Powered Forecasting

Predict future CPU utilization trends using a Transformer-LSTM model.

🤖 Intelligent Autoscaling

Learn optimal scaling policies through PPO reinforcement learning.

⚡ FastAPI Inference Server

Serve predictions through REST APIs.

🌐 Interactive Dashboard

Simple Flask UI for testing autoscaling decisions.

🛡️ Reliable Fallback Mechanism

Automatically switches to a DummyModel if the trained model is unavailable.

💻 Local Development Friendly

Runs efficiently on CPUs without requiring expensive hardware.

⸻

📂 Project Structure

smart-cpu-autoscaler/
│
├── src/
│   └── transformer_lstm.py
│
├── scripts/
│   └── save_dummy_model.py
│
├── models/
│   └── transformer_lstm.pth
│
├── app.py
├── model_loader.py
├── model_server.py
├── model_server_real.py
├── test_run_app.py
├── requirements.txt
└── README.md

⸻

🚀 Quick Start

1. Clone Repository

git clone https://github.com/shivarambobba/Smart-CPU-Resource-Optimisation.git
cd smart-cpu-autoscaler

2. Create Virtual Environment

python3 -m venv .venv
source .venv/bin/activate

3. Install Dependencies

pip install -r requirements.txt

⸻

🧠 Generate a Sample Model

python scripts/save_dummy_model.py

This creates:

models/transformer_lstm.pth

⸻

⚡ Run FastAPI Model Server

uvicorn model_server_real:app --host 127.0.0.1 --port 8000 --reload

Health Check:

curl http://127.0.0.1:8000/health

Expected Output:

{
  "status": "healthy"
}

⸻

🌐 Run Flask Dashboard

python app.py

Open in browser:

http://127.0.0.1:5001/ui

⸻

🔍 Test Prediction Endpoint

curl -X POST \
-H "Content-Type: application/json" \
-d '{"cpu":72}' \
http://127.0.0.1:5001/predict

Example Response:

{
  "action": "scale_up",
  "predicted_cpu": 81.4
}

⸻

🛠 API Endpoints

Health Check

GET /health

Response:

{
  "status": "healthy"
}

⸻

Predict Scaling Decision

POST /predict

Request:

{
  "cpu": 72
}

Response:

{
  "action": "scale_up",
  "predicted_cpu": 81.4
}

⸻

🎯 Why This Project?

Most autoscaling solutions use fixed CPU thresholds:

CPU > 80% → Scale Up
CPU < 30% → Scale Down

This project takes a smarter approach:

✅ Forecasts future demand

✅ Learns optimal actions through reinforcement learning

✅ Reduces unnecessary scaling events

✅ Improves resource utilization

✅ Supports future cloud-native deployments

⸻

📊 Roadmap

Current Features

* Transformer-LSTM model
* PPO integration framework
* FastAPI server
* Flask dashboard
* DummyModel fallback

Future Enhancements

* Complete training pipeline
* Kubernetes integration
* Docker deployment
* Grafana dashboards
* Prometheus monitoring
* Multi-node autoscaling
* CI/CD pipeline
* Benchmarking against Kubernetes HPA

⸻

🐳 Docker Support (Coming Soon)

docker compose up

Planned deployment stack:

* Flask UI Container
* FastAPI Model Server Container
* Monitoring Dashboard
* One-Command Deployment

⸻

📚 Research Inspiration

This project is inspired by the research paper:

Smart CPU Autoscaling using Transformer-LSTM and PPO

The repository provides a practical and lightweight implementation for experimentation, learning, and future research.

⸻

🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Push to GitHub
5. Create a Pull Request

⸻

⭐ Support

If you find this project useful:

⭐ Star the repository

🍴 Fork it

🛠️ Contribute

📢 Share it with others

⸻

Built with ❤️ using PyTorch, FastAPI, Flask, and Reinforcement Lear

g
:::
