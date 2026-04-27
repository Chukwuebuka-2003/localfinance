# LocalFinance Assistant

An AI-powered, privacy-first personal finance assistant that analyzes your bank statements locally.

<img width="1600" height="900" alt="Dashboard" src="https://github.com/user-attachments/assets/e7aad0c5-d2af-4514-8550-63790a10f13c" />

<img width="1600" height="900" alt="Chat UI" src="https://github.com/user-attachments/assets/b30b4678-42e4-4bf4-b309-52d17d212a0b" />

## 🌟 Overview

LocalFinance is a personal finance tool that lets you interact with your financial data using natural language. It processes your bank statements (CSV/PDF) locally, stores them in a vector database, and uses a local LLM (Large Language Model) to provide insights, detect trends, and answer questions—all without your sensitive data ever leaving your machine.

## ✨ Features

- **Automated Statement Parsing**: Support for uploading and extracting transactions from **PDF** and **CSV** statements.
- **Semantic Financial Search**: Ask questions in plain English (e.g., *"How much did I spend at Amazon in 2025?"*).
- **AI Tool-Calling**: The assistant uses specialized tools to perform accurate calculations, anomaly detection, and period comparisons.
- **Privacy-First**: Designed to run with local LLM providers like **Ollama** or **LM Studio**.
- **Vector Search**: Uses **Qdrant** to enable fast, relevant retrieval of transaction context.
- **Advanced Insights**: Detect recurring subscriptions, spending velocity, merchant trends, and more.

## 🛠 Tech Stack

- **Backend**: FastAPI (Python)
- **Dependency Management**: [uv](https://github.com/astral-sh/uv)
- **Vector Database**: Qdrant
- **LLM Engine**: Ollama / LM Studio
- **Frontend**: Minimalistic HTML/JS with CSS

## 🚀 Getting Started

### Prerequisites

1.  **Docker Desktop**: For running the vector database.
2.  **Ollama**: (Recommended) For running local models.
3.  **uv**: For local Python environment management.

---

### 1. Model Setup (Ollama)

Before running the app, pull the required models:

```bash
# Pull the chat model
ollama pull granite4:latest

# Pull the embedding model
ollama pull nomic-embed-text-v2-moe
```

---

### 2. Configuration

Copy the example environment file and adjust your settings:

```bash
cp .env.example .env
```

---

### 3. Running with Docker (Easiest)

Build and start the entire stack:

```bash
docker compose up --build -d
```
Access the dashboard at **[http://localhost:1578](http://localhost:1578)**.

---

### 4. Running Locally (Development)

1.  **Start Qdrant**:
    ```bash
    docker compose up qdrant -d
    ```
2.  **Install Dependencies**:
    ```bash
    uv sync
    ```
3.  **Run the Server**:
    ```bash
    uv run python main.py
    ```
Access the dashboard at **[http://localhost:1578](http://localhost:1578)**.

## ⚙️ Environment Variables

| Variable | Description | Default |
| :--- | :--- | :--- |
| `AI_PROVIDER` | `ollama` or `lmstudio` | `ollama` |
| `OLLAMA_MODEL` | The chat model to use | `granite4:latest` |
| `OLLAMA_EMBEDDING_MODEL`| Embedding model for transactions | `nomic-embed-text-v2-moe` |
| `QDRANT_HOST` | Hostname for Qdrant | `localhost` |
| `VECTOR_SIZE` | Dimensions for your model | `768` |

## 📜 License

MIT License. See `LICENSE` for details.
