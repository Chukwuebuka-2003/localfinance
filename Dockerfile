FROM python:3.11-slim

# Install uv via pip
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy only dependency files first (for layer caching)
COPY pyproject.toml uv.lock ./

# Install all dependencies into the system python (not a venv)
# --system flag tells uv to install into the system Python, avoiding venv path issues
RUN uv pip install --system -r pyproject.toml

COPY . .

EXPOSE 1578

# Use uv run to ensure the uvicorn runs within the managed virtual environment
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "1578"]
