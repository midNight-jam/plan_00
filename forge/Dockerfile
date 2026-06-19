# 1. Start with a heavy NVIDIA image that includes the 'nvcc' compiler
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# install the system dependencies that vLLM or PyTorch might need
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# COPY the uv binary directly from the official image (fastest way to install uv)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# COPY onlyu dependency files first to cache the isntallation layer
COPY pyproject.toml uv.lock ./

# Install dependencies into a virtual env inside the container
RUN uv sync --frozen --no-dev

# COPY the actual application code a& config
COPY src/ ./src/
COPY config.yaml ./

# Environment Variables
# Hardcode the RTX 5080 / sm_120 workaround so we never miss it
ENV VLLM_USE_FLASHINFER_SAMPLER=0
# Python runs unbuffered so logs stream instantly
ENV PYTHONUNBUFFERED=1
# Putting the virtual env onthe system path
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "src.forge.server:app", "--host", "0.0.0.0", "--port", "8000"]