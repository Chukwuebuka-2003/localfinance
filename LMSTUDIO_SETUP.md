# LM Studio Setup Guide

This guide explains how to configure the Financial Assistant to use LM Studio instead of Ollama.

## Overview

The Financial Assistant now supports multiple AI providers:
- **Ollama** (default) - Open-source LLM runner
- **LM Studio** - Desktop app with OpenAI-compatible API

You can configure these via environment variables.

## Quick Start with LM Studio

### 1. Install LM Studio

Download and install LM Studio from: https://lmstudio.ai/

### 2. Configure LM Studio

1. Open LM Studio
2. Download a model (recommended: IBM Granite or Llama 3.1)
3. Start the **Local Inference Server**:
   - Go to the "Local Inference Server" tab
   - Set the port to `1234` (default)
   - Click "Start Server"

### 3. Configure the App

Create a `.env` file in the project root:

```bash
# Use LM Studio for chat
AI_PROVIDER=lmstudio
LMSTUDIO_HOST=http://host.docker.internal:1234
LMSTUDIO_MODEL=granite

# Use Ollama for embeddings (recommended - LM Studio may not support embeddings well)
EMBEDDING_PROVIDER=ollama
OLLAMA_HOST=http://host.docker.internal:11434
```

### 4. Run the App

```bash
docker-compose up
```

## Configuration Options

### Using LM Studio for Everything (Chat + Embeddings)

If your LM Studio model supports embeddings:

```bash
AI_PROVIDER=lmstudio
EMBEDDING_PROVIDER=lmstudio
LMSTUDIO_HOST=http://host.docker.internal:1234
LMSTUDIO_MODEL=your-model-name
LMSTUDIO_EMBEDDING_MODEL=your-model-name
```

### Using LM Studio for Chat, Ollama for Embeddings (Recommended)

This is the most reliable setup:

```bash
# Chat via LM Studio
AI_PROVIDER=lmstudio
LMSTUDIO_HOST=http://host.docker.internal:1234
LMSTUDIO_MODEL=granite

# Embeddings via Ollama
EMBEDDING_PROVIDER=ollama
OLLAMA_HOST=http://host.docker.internal:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text-v2-moe
```

### Using Ollama for Everything (Original Setup)

```bash
AI_PROVIDER=ollama
EMBEDDING_PROVIDER=ollama
OLLAMA_HOST=http://host.docker.internal:11434
OLLAMA_MODEL=granite4:latest
```

## Networking Notes

### Docker Desktop (Mac/Windows)

Use `host.docker.internal` to access services on your host machine:
- `http://host.docker.internal:11434` for Ollama
- `http://host.docker.internal:1234` for LM Studio

### Linux

On Linux, `host.docker.internal` may not work. Use your host's IP address:

```bash
# Find your host IP
ip addr show

# Use the IP in your .env
LMSTUDIO_HOST=http://192.168.1.100:1234
OLLAMA_HOST=http://192.168.1.100:11434
```

## Troubleshooting

### Connection Refused Errors

1. **Check LM Studio is running**: The inference server must be started
2. **Check the port**: Default is 1234
3. **Check CORS settings**: In LM Studio, enable CORS in the server settings
4. **Firewall**: Ensure your firewall allows connections on port 1234

### Model Not Found

The `LMSTUDIO_MODEL` name must match what LM Studio expects. Common values:
- `granite`
- `llama-3.1`
- `qwen2.5`

Check LM Studio's server logs to see what model name it expects.

### Embeddings Not Working

Not all LM Studio models support embeddings. If you get errors:
1. Set `EMBEDDING_PROVIDER=ollama` to use Ollama for embeddings only
2. Keep `AI_PROVIDER=lmstudio` for chat
3. Run both services on your host

## Environment Variables Reference

| Variable | Description | Default |
|----------|-------------|---------|
| `AI_PROVIDER` | Chat provider: `ollama` or `lmstudio` | `ollama` |
| `EMBEDDING_PROVIDER` | Embeddings provider: `ollama` or `lmstudio` | `ollama` |
| `OLLAMA_HOST` | Ollama server URL | `http://host.docker.internal:11434` |
| `OLLAMA_MODEL` | Chat model for Ollama | `granite4:latest` |
| `OLLAMA_EMBEDDING_MODEL` | Embedding model for Ollama | `nomic-embed-text-v2-moe` |
| `LMSTUDIO_HOST` | LM Studio server URL | `http://host.docker.internal:1234` |
| `LMSTUDIO_MODEL` | Model name for LM Studio | `granite` |
| `LMSTUDIO_EMBEDDING_MODEL` | Embedding model for LM Studio | `nomic-embed-text-v2-moe` |
| `LMSTUDIO_API_KEY` | API key for LM Studio | `lm-studio` |
| `VECTOR_SIZE` | Embedding dimensions | `768` |
