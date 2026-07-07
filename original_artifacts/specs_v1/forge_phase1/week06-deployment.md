# Week 6: Deployment Stack
> Phase: 1 | Project: Forge | Estimated Duration: 7 days

## Context

Weeks 1-5 built the full application. This week is about proper deployment: Docker Compose for development, K3s + Helm for production-like deployment, CI/CD with GitHub Actions, and proper infrastructure configuration. This proves you can ship, not just code.

**Prerequisites**: Weeks 1-5 complete — full platform with auth, multi-model, RAG working locally.

**Builds on**: Containerizes and deploys the existing application properly.

## Learning Goals

- [ ] Understand Helm chart structure — templates, values, helpers, dependencies
- [ ] Understand K8s GPU scheduling — resource requests, device plugins, node selectors
- [ ] Understand readiness vs liveness probes — why ML services need custom probes
- [ ] Understand CI/CD for ML — testing with GPU dependencies, Docker layer caching
- [ ] Understand K3s — lightweight K8s, when to use it vs full K8s

## Implementation Goals

- [ ] Optimize Docker image (multi-stage build, layer caching, minimal size)
- [ ] Create complete Docker Compose stack (all services, volumes, networks, health checks)
- [ ] Install and configure K3s cluster with GPU support
- [ ] Write Helm chart for Forge with configurable values
- [ ] Configure GPU resource scheduling in K8s
- [ ] Set up GitHub Actions CI/CD (lint, test, build, integration test, benchmark)
- [ ] Implement proper secrets management (not hardcoded, not in .env committed)
- [ ] Create Makefile for common operations

## Acceptance Criteria

1. **Docker Compose**: `make dev` brings up entire stack, model loads, inference works
2. **Docker image**: Image size < 5GB (optimized layers, no unnecessary dependencies)
3. **K3s running**: `kubectl get nodes` shows ready node with GPU
4. **Helm install**: `helm install forge ./helm-charts/forge` deploys to K3s successfully
5. **GPU in K8s**: Pod spec requests `nvidia.com/gpu: 1` and gets scheduled
6. **Readiness probe**: Pod only becomes Ready after model is loaded (not just server started)
7. **CI passes**: Push to GitHub → Actions run → lint + test + build all pass
8. **Integration test in CI**: Docker Compose spins up in CI, runs test suite, tears down
9. **Makefile works**: `make lint`, `make test`, `make build`, `make deploy` all work
10. **Secrets**: No passwords/keys in committed code; uses env vars or K8s secrets

## Validation Commands

```bash
# Development mode
make dev
curl http://localhost:8000/health  # Returns healthy after model loads

# Check Docker image size
docker images forge-server --format "{{.Size}}"

# K3s cluster
kubectl get nodes -o wide
kubectl describe node | grep -A5 "Allocated resources"

# Helm deploy
helm install forge ./helm-charts/forge -f helm-charts/forge/values-dev.yaml
kubectl get pods -w  # Watch until Ready
kubectl logs -f deploy/forge-server  # See model loading

# Verify GPU scheduling
kubectl describe pod forge-server-xxx | grep "nvidia.com/gpu"

# Readiness probe
kubectl get pods  # Should show 0/1 Ready during model load, 1/1 after

# CI (local simulation)
act -j lint  # Run GitHub Actions locally with 'act'

# Full test
make test-all

# Clean teardown
make clean
```

## Technical Implementation Details

### Component 1: Optimized Docker Image (Day 1)

**File: `Dockerfile`**

Multi-stage build:
```dockerfile
# Stage 1: Build dependencies
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder
# Install build tools, compile any C extensions
# Install Python dependencies into a venv

# Stage 2: Runtime
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
# Copy only the venv and application code
# Non-root user
# Health check instruction
```

Optimization tricks:
- Pin all dependency versions in requirements.txt (reproducible)
- Use `.dockerignore` to exclude tests, docs, .git
- Separate dependency layer from code layer (dependencies change less often)
- Pre-download model weights into a separate volume (not in image)

### Component 2: Docker Compose (Day 1-2)

**File: `docker-compose.yml`**

```yaml
services:
  forge-server:
    build: .
    ports: ["8000:8000"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - model_cache:/models
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      qdrant:
        condition: service_healthy
    environment:
      - DATABASE_URL=postgresql+asyncpg://forge:${DB_PASSWORD}@postgres:5432/forge
      - REDIS_URL=redis://redis:6379
      - QDRANT_URL=http://qdrant:6333
      - MODEL_CACHE_DIR=/models
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 120s  # Model loading takes time

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: forge
      POSTGRES_USER: forge
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./db/init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U forge"]

  redis:
    image: redis:7-alpine
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]

  qdrant:
    image: qdrant/qdrant:latest
    volumes:
      - qdrant_data:/qdrant/storage
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/healthz"]

volumes:
  model_cache:
  postgres_data:
  qdrant_data:
  redis_data:
```

### Component 3: K3s + GPU Setup (Day 2-3)

Install K3s:
```bash
curl -sfL https://get.k3s.io | sh -
```

Install NVIDIA device plugin:
```bash
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml
```

Verify GPU is visible:
```bash
kubectl describe node | grep nvidia.com/gpu
```

### Component 4: Helm Chart (Day 3-5)

**Directory: `helm-charts/forge/`**

```
helm-charts/forge/
├── Chart.yaml
├── values.yaml
├── values-dev.yaml
├── templates/
│   ├── _helpers.tpl
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── configmap.yaml
│   ├── secret.yaml
│   ├── hpa.yaml
│   ├── pvc.yaml          # Model cache PVC
│   └── tests/
│       └── test-connection.yaml
└── charts/               # Subcharts (postgres, redis, qdrant)
```

Key template details:

**deployment.yaml** — GPU resource request:
```yaml
resources:
  requests:
    nvidia.com/gpu: {{ .Values.gpu.count }}
    memory: {{ .Values.resources.memory }}
  limits:
    nvidia.com/gpu: {{ .Values.gpu.count }}
```

**Readiness probe** (critical for ML services):
```yaml
readinessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 60  # Model needs time to load
  periodSeconds: 10
  failureThreshold: 30     # Allow up to 5 min for model load
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 120
  periodSeconds: 30
```

**values.yaml**:
```yaml
replicaCount: 1
image:
  repository: ghcr.io/YOUR_USERNAME/forge
  tag: latest
gpu:
  count: 1
  type: nvidia.com/gpu
model:
  name: mistral-7b
  path: mistralai/Mistral-7B-Instruct-v0.3
  cacheSize: 50Gi
resources:
  memory: 24Gi
```

### Component 5: CI/CD Pipeline (Day 5-6)

**File: `.github/workflows/ci.yml`**

```yaml
name: CI
on: [push, pull_request]
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync
      - run: uv run ruff check .
      - run: uv run mypy src/

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync
      - run: uv run pytest tests/unit/ -v

  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/build-push-action@v5
        with:
          context: .
          push: false
          tags: forge:test
          cache-from: type=gha
          cache-to: type=gha,mode=max

  integration:
    runs-on: ubuntu-latest
    needs: build
    steps:
      - uses: actions/checkout@v4
      - run: docker compose -f docker-compose.test.yml up -d
      - run: sleep 30 && curl --retry 10 --retry-delay 5 http://localhost:8000/health
      - run: pytest tests/integration/ -v
      - run: docker compose -f docker-compose.test.yml down
```

Note: Integration tests in CI won't have a GPU. Use a mock model or CPU-only small model for CI. Real GPU tests run locally.

### Component 6: Makefile (Day 6)

**File: `Makefile`**

```makefile
.PHONY: dev test lint build deploy clean

dev:
	docker compose up -d
	@echo "Waiting for services..."
	@sleep 5
	docker compose logs -f forge-server

test:
	uv run pytest tests/unit/ -v

test-integration:
	uv run pytest tests/integration/ -v

lint:
	uv run ruff check .
	uv run mypy src/

build:
	docker build -t forge:latest .

deploy:
	helm upgrade --install forge ./helm-charts/forge -f helm-charts/forge/values-dev.yaml

clean:
	docker compose down -v
	helm uninstall forge 2>/dev/null || true
```

### Component 7: Secrets Management (Day 7)

- `.env.example` in repo (template with placeholder values)
- `.env` in `.gitignore` (never committed)
- K8s: use `Secret` resources (referenced by deployment)
- CI: use GitHub Actions secrets
- Document how to set up secrets in README

## If You Get Stuck

**K3s GPU not detected**: Ensure nvidia-container-runtime is the default Docker runtime. Check `/etc/docker/daemon.json` has `"default-runtime": "nvidia"`.

**Helm chart complex**: Start with just the deployment + service + configmap. Add HPA and PVC later. Use `helm template` to debug rendered YAML.

**CI without GPU**: Create a `docker-compose.test.yml` that uses a tiny CPU model (or mocked engine) for integration tests. Real GPU tests are manual.

**Image too large**: The CUDA base image is ~4GB alone. Focus on not adding unnecessary layers on top. Model weights should be in a volume, not the image.

## Agent Handoff Template

```
I'm on Week 6 of Forge — deployment and infrastructure.
Spec: /Users/jmalviya/Documents/zz/dev/plan_00/forge/specs/phase1/week06-deployment.md
Current state: Full application working locally (inference + RAG + multi-model + auth + rate limiting).
I need: Optimized Dockerfile, Docker Compose (all services), K3s + GPU setup, Helm chart, GitHub Actions CI/CD, Makefile.
Hardware: RTX 5080, Ubuntu, K3s single-node cluster.
```

## Out of Scope

- ArgoCD/GitOps (Anvil project, Phase A)
- Multi-node cluster (Anvil project)
- Production SSL/TLS termination (not needed for portfolio demo)
- Custom domain/DNS (not needed)
- Terraform for cloud resources (Anvil project)
