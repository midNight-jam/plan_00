# Week 6: Storage Systems for ML

## Context

**Where it fits:** Phase A, Week 6 — storage underpins every other component: checkpoints (Week 3), model artifacts (deployment), datasets (training input). This week builds the data plane.

**Prerequisites:**
- Weeks 1-5 completed (cluster, orchestrator, networking functional)
- MinIO installed (S3-compatible object storage, runs locally)
- PostgreSQL basics (metadata store)
- Content-addressable storage concepts (Git's object model)
- Hardware: ASUS ROG Strix SCAR 16 (RTX 5080 16GB, 32GB RAM, 2TB SSD, Ubuntu)

**What it builds on:** Week 3's checkpoint manager now has a real storage backend. Week 4's Terraform provisions the storage infrastructure. Week 7's integration tests exercise the full checkpoint-restore pipeline.

---

## Learning Goals

- [ ] Explain tiered storage economics: NVMe ($$$) vs NFS ($$) vs Object Storage ($)
- [ ] Describe content-addressable storage (CAS) and deduplication strategies
- [ ] Articulate model versioning: why it's harder than code versioning (large binary blobs)
- [ ] Explain checkpoint lifecycle: hot (actively used) → warm (recent) → cold (archived)
- [ ] Describe dataset versioning challenges: lineage, reproducibility, quality drift
- [ ] Understand S3 API semantics: multipart upload, presigned URLs, lifecycle policies
- [ ] Compare approaches: DVC, MLflow, Weights & Biases, LakeFS

---

## Implementation Goals

- [ ] Model Registry: version-controlled storage with metadata, tags, promotion workflow
- [ ] Checkpoint storage with tiered lifecycle (hot local → warm NFS → cold MinIO)
- [ ] Checkpoint deduplication using content-addressable storage (SHA256 blocks)
- [ ] Dataset versioning: track versions, diff between versions, quality checks
- [ ] CLI tool: `anvil model push/pull/list/promote`
- [ ] CLI tool: `anvil checkpoint restore --job <name> --step <N>`
- [ ] CLI tool: `anvil dataset register/version/compare`
- [ ] PostgreSQL metadata store (models, versions, lineage, labels)
- [ ] Redis cache for hot metadata (latest checkpoint location, model tags)
- [ ] Garbage collection: remove unreferenced blobs after retention period

---

## Acceptance Criteria

1. `anvil model push --name resnet50 --version 1.0 --path ./model.pt` uploads the model to MinIO and records metadata in PostgreSQL.
2. `anvil model pull --name resnet50 --version 1.0 --output ./` downloads the exact same bytes (SHA256 matches).
3. `anvil model promote --name resnet50 --version 1.0 --stage production` updates the model's stage without re-uploading.
4. Model Registry stores metadata: creation time, size, SHA256, framework, metrics, lineage (which training job produced it).
5. Checkpoint tier migration works: a checkpoint older than 1 hour moves from local NVMe to MinIO automatically.
6. Deduplication: pushing two models that share 90% of layers stores only the unique blocks (verified by storage usage).
7. `anvil checkpoint restore --job my-training --step 5000` retrieves the correct checkpoint and makes it available at the expected path.
8. Dataset versioning tracks a new version when data changes, and `anvil dataset compare v1 v2` shows added/removed/modified records.
9. Garbage collection removes blobs with zero references after the retention period (default 7 days).
10. All CLI commands complete within 5 seconds for metadata operations and achieve >500 MB/s throughput for data transfer (local MinIO).

---

## Validation Commands

```bash
# Start infrastructure
cd ~/anvil/storage
docker compose up -d  # MinIO, PostgreSQL, Redis

# Initialize schema
python -m anvil_storage.db migrate

# Model registry operations
anvil model push --name gpt2-small --version 0.1.0 --path ./test-model/ --framework pytorch
anvil model list
anvil model pull --name gpt2-small --version 0.1.0 --output /tmp/pulled-model/
diff -r ./test-model/ /tmp/pulled-model/  # Should be identical

# Model promotion
anvil model promote --name gpt2-small --version 0.1.0 --stage staging
anvil model promote --name gpt2-small --version 0.1.0 --stage production
anvil model list --stage production

# Checkpoint operations
anvil checkpoint save --job training-123 --step 5000 --path /tmp/ckpt/
anvil checkpoint list --job training-123
anvil checkpoint restore --job training-123 --step 5000 --output /tmp/restored/

# Deduplication test
python tests/test_deduplication.py  # Push similar models, check storage savings

# Tier migration test
python tests/test_tier_migration.py --wait 60  # Wait for auto-migration

# Dataset versioning
anvil dataset register --name imagenet-subset --path ./data/v1/
anvil dataset version --name imagenet-subset --path ./data/v2/
anvil dataset compare --name imagenet-subset --v1 1 --v2 2

# Garbage collection
python -m anvil_storage.gc --dry-run
python -m anvil_storage.gc --execute --retention-days 0  # For testing

# Performance test
python tests/test_throughput.py --size 1GB
```

---

## Technical Implementation Details

### Project Structure

```
~/anvil/storage/
├── anvil_storage/
│   ├── __init__.py
│   ├── cli.py               # Click-based CLI entry point
│   ├── registry/
│   │   ├── __init__.py
│   │   ├── model_registry.py    # Model CRUD operations
│   │   ├── version_manager.py   # Versioning logic
│   │   └── promotion.py         # Stage promotion workflow
│   ├── checkpoint/
│   │   ├── __init__.py
│   │   ├── manager.py           # Checkpoint save/restore
│   │   ├── tiering.py           # Hot → warm → cold migration
│   │   └── gc.py                # Garbage collection
│   ├── dataset/
│   │   ├── __init__.py
│   │   ├── versioning.py        # Dataset version tracking
│   │   ├── comparison.py        # Diff between versions
│   │   └── quality.py           # Data quality checks
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── backend.py           # Abstract storage interface
│   │   ├── minio_backend.py     # S3/MinIO implementation
│   │   ├── local_backend.py     # Local filesystem (hot tier)
│   │   └── cas.py               # Content-addressable storage
│   ├── db/
│   │   ├── __init__.py
│   │   ├── models.py            # SQLAlchemy models
│   │   ├── migrations/          # Alembic migrations
│   │   └── connection.py        # DB connection pool
│   └── cache/
│       ├── __init__.py
│       └── redis_cache.py       # Redis caching layer
├── tests/
│   ├── test_model_registry.py
│   ├── test_checkpoint.py
│   ├── test_deduplication.py
│   ├── test_tier_migration.py
│   ├── test_dataset_versioning.py
│   └── test_throughput.py
├── docker-compose.yaml          # MinIO + PostgreSQL + Redis
├── pyproject.toml
└── README.md
```

### Docker Compose (Infrastructure)

```yaml
# docker-compose.yaml
services:
  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_ROOT_USER: anvil
      MINIO_ROOT_PASSWORD: anvilsecret
    volumes:
      - minio_data:/data

  postgres:
    image: postgres:16
    ports:
      - "5432:5432"
    environment:
      POSTGRES_DB: anvil_storage
      POSTGRES_USER: anvil
      POSTGRES_PASSWORD: anvilsecret
    volumes:
      - pg_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  minio_data:
  pg_data:
```

### Content-Addressable Storage

```python
# anvil_storage/storage/cas.py
import hashlib
from pathlib import Path
from dataclasses import dataclass

BLOCK_SIZE = 4 * 1024 * 1024  # 4MB blocks

@dataclass
class Block:
    hash: str
    size: int
    data: bytes | None = None

class ContentAddressableStore:
    """Deduplicating storage using content hashing."""

    def __init__(self, backend: "StorageBackend"):
        self.backend = backend
        self.bucket = "anvil-cas"

    def store_file(self, file_path: Path) -> list[str]:
        """Split file into blocks, store unique ones, return block hashes."""
        block_hashes = []
        with open(file_path, "rb") as f:
            while chunk := f.read(BLOCK_SIZE):
                block_hash = hashlib.sha256(chunk).hexdigest()
                block_hashes.append(block_hash)

                # Only upload if block doesn't exist (dedup)
                if not self.backend.exists(self.bucket, f"blocks/{block_hash}"):
                    self.backend.put(self.bucket, f"blocks/{block_hash}", chunk)

        return block_hashes

    def retrieve_file(self, block_hashes: list[str], output_path: Path):
        """Reassemble file from block hashes."""
        with open(output_path, "wb") as f:
            for block_hash in block_hashes:
                data = self.backend.get(self.bucket, f"blocks/{block_hash}")
                f.write(data)

    def dedup_ratio(self, block_hashes: list[str], total_size: int) -> float:
        """Calculate deduplication ratio (1.0 = all unique, 0.0 = all duplicate)."""
        unique_blocks = len(set(block_hashes))
        total_blocks = len(block_hashes)
        return unique_blocks / total_blocks if total_blocks > 0 else 1.0
```

### Model Registry

```python
# anvil_storage/registry/model_registry.py
from datetime import datetime
from dataclasses import dataclass
from typing import Optional
from sqlalchemy.orm import Session
from ..db.models import ModelVersion
from ..storage.cas import ContentAddressableStore
from ..cache.redis_cache import RedisCache

@dataclass
class ModelMetadata:
    name: str
    version: str
    framework: str
    size_bytes: int
    sha256: str
    stage: str = "none"  # none → staging → production
    created_at: datetime = None
    training_job_id: Optional[str] = None
    metrics: dict = None

class ModelRegistry:
    def __init__(self, db: Session, cas: ContentAddressableStore, cache: RedisCache):
        self.db = db
        self.cas = cas
        self.cache = cache

    def push(self, name: str, version: str, path: str,
             framework: str, training_job_id: str = None, metrics: dict = None) -> ModelMetadata:
        """Upload model artifacts and register metadata."""
        from pathlib import Path
        model_path = Path(path)

        # Store via CAS (deduplication)
        all_blocks = []
        total_size = 0
        for file in sorted(model_path.rglob("*")):
            if file.is_file():
                blocks = self.cas.store_file(file)
                all_blocks.append({"relative_path": str(file.relative_to(model_path)), "blocks": blocks})
                total_size += file.stat().st_size

        # Compute overall SHA
        sha256 = hashlib.sha256(
            "".join(b for f in all_blocks for b in f["blocks"]).encode()
        ).hexdigest()

        # Save metadata
        model_version = ModelVersion(
            name=name, version=version, framework=framework,
            size_bytes=total_size, sha256=sha256,
            stage="none", block_manifest=all_blocks,
            training_job_id=training_job_id, metrics=metrics or {}
        )
        self.db.add(model_version)
        self.db.commit()

        # Invalidate cache
        self.cache.delete(f"model:{name}:latest")

        return ModelMetadata(
            name=name, version=version, framework=framework,
            size_bytes=total_size, sha256=sha256, created_at=model_version.created_at
        )

    def pull(self, name: str, version: str, output_path: str):
        """Download model artifacts to local path."""
        model = self.db.query(ModelVersion).filter_by(name=name, version=version).first()
        if not model:
            raise ValueError(f"Model {name}:{version} not found")

        out = Path(output_path)
        out.mkdir(parents=True, exist_ok=True)

        for file_entry in model.block_manifest:
            file_path = out / file_entry["relative_path"]
            file_path.parent.mkdir(parents=True, exist_ok=True)
            self.cas.retrieve_file(file_entry["blocks"], file_path)

    def promote(self, name: str, version: str, stage: str):
        """Promote model to a stage (staging/production)."""
        # Demote current holder of this stage
        current = self.db.query(ModelVersion).filter_by(name=name, stage=stage).first()
        if current:
            current.stage = "archived"

        model = self.db.query(ModelVersion).filter_by(name=name, version=version).first()
        model.stage = stage
        self.db.commit()

        self.cache.set(f"model:{name}:{stage}", version)
```

### CLI Tool

```python
# anvil_storage/cli.py
import click
from pathlib import Path

@click.group()
def cli():
    """Anvil Storage CLI — manage models, checkpoints, and datasets."""
    pass

@cli.group()
def model():
    """Model registry operations."""
    pass

@model.command()
@click.option("--name", required=True, help="Model name")
@click.option("--version", required=True, help="Semantic version")
@click.option("--path", required=True, type=click.Path(exists=True), help="Path to model artifacts")
@click.option("--framework", default="pytorch", help="ML framework")
def push(name, version, path, framework):
    """Push model artifacts to registry."""
    registry = _get_registry()
    meta = registry.push(name, version, path, framework)
    click.echo(f"Pushed {name}:{version} ({meta.size_bytes / 1e6:.1f} MB, sha256:{meta.sha256[:12]})")

@model.command()
@click.option("--name", required=True)
@click.option("--version", required=True)
@click.option("--output", required=True, type=click.Path())
def pull(name, version, output):
    """Pull model artifacts from registry."""
    registry = _get_registry()
    registry.pull(name, version, output)
    click.echo(f"Pulled {name}:{version} → {output}")

@model.command()
@click.option("--name", required=True)
@click.option("--version", required=True)
@click.option("--stage", required=True, type=click.Choice(["staging", "production"]))
def promote(name, version, stage):
    """Promote model to a deployment stage."""
    registry = _get_registry()
    registry.promote(name, version, stage)
    click.echo(f"Promoted {name}:{version} → {stage}")

@cli.group()
def checkpoint():
    """Checkpoint operations."""
    pass

@checkpoint.command()
@click.option("--job", required=True, help="Training job name")
@click.option("--step", required=True, type=int, help="Training step to restore")
@click.option("--output", required=True, type=click.Path())
def restore(job, step, output):
    """Restore a training checkpoint."""
    mgr = _get_checkpoint_manager()
    path = mgr.restore(job, step, output)
    click.echo(f"Restored checkpoint for {job} step {step} → {path}")

@cli.group()
def dataset():
    """Dataset versioning operations."""
    pass

@dataset.command()
@click.option("--name", required=True)
@click.option("--path", required=True, type=click.Path(exists=True))
def register(name, path):
    """Register a new dataset."""
    ds = _get_dataset_manager()
    version = ds.register(name, path)
    click.echo(f"Registered dataset {name} v{version}")

@dataset.command()
@click.option("--name", required=True)
@click.option("--v1", required=True, type=int)
@click.option("--v2", required=True, type=int)
def compare(name, v1, v2):
    """Compare two dataset versions."""
    ds = _get_dataset_manager()
    diff = ds.compare(name, v1, v2)
    click.echo(f"Added: {diff.added}, Removed: {diff.removed}, Modified: {diff.modified}")
```

### Checkpoint Tiering

```python
# anvil_storage/checkpoint/tiering.py
import asyncio
from datetime import datetime, timedelta
from enum import Enum

class Tier(Enum):
    HOT = "hot"      # Local NVMe — fastest access
    WARM = "warm"    # NFS — shared, moderate speed
    COLD = "cold"    # MinIO (S3) — cheapest, slowest

TIER_THRESHOLDS = {
    Tier.HOT: timedelta(hours=1),    # Keep on NVMe for 1 hour
    Tier.WARM: timedelta(days=7),    # Keep on NFS for 7 days
    Tier.COLD: timedelta(days=365),  # Keep in MinIO for 1 year
}

class TierManager:
    def __init__(self, hot_backend, warm_backend, cold_backend, db):
        self.backends = {
            Tier.HOT: hot_backend,
            Tier.WARM: warm_backend,
            Tier.COLD: cold_backend,
        }
        self.db = db

    async def migrate_stale(self):
        """Move checkpoints to cheaper tiers based on age."""
        checkpoints = self.db.query_all_checkpoints()
        now = datetime.utcnow()

        for ckpt in checkpoints:
            age = now - ckpt.created_at
            target_tier = self._target_tier(age)

            if target_tier != ckpt.current_tier:
                await self._migrate(ckpt, target_tier)

    async def _migrate(self, checkpoint, target_tier: Tier):
        """Copy to target tier, verify, delete from source."""
        source = self.backends[checkpoint.current_tier]
        target = self.backends[target_tier]

        data = await source.get_async(checkpoint.storage_path)
        await target.put_async(checkpoint.storage_path, data)

        # Verify integrity
        target_data = await target.get_async(checkpoint.storage_path)
        assert hashlib.sha256(data).digest() == hashlib.sha256(target_data).digest()

        await source.delete_async(checkpoint.storage_path)
        checkpoint.current_tier = target_tier
        self.db.update_checkpoint(checkpoint)

    def _target_tier(self, age: timedelta) -> Tier:
        if age < TIER_THRESHOLDS[Tier.HOT]:
            return Tier.HOT
        elif age < TIER_THRESHOLDS[Tier.WARM]:
            return Tier.WARM
        return Tier.COLD
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| MinIO connection refused | Check `docker compose ps`. Ensure port 9000 is mapped. Test: `curl http://localhost:9000/minio/health/live`. |
| PostgreSQL migration fails | Check connection string. Run `alembic upgrade head` manually. Verify database exists. |
| Dedup ratio is 1.0 (no dedup) | Ensure block size is appropriate. For small test files, use 1MB blocks. Verify files actually share content. |
| CLI can't find config | Set `ANVIL_CONFIG=~/.anvil/config.yaml` or pass `--config` flag. |
| Redis cache stale | Clear with `redis-cli FLUSHDB`. Implement TTL on cache entries. |
| Large file upload OOM | Use streaming upload with multipart. Don't load entire file into memory. |

---

## Agent Handoff Template

```
Resume Anvil Phase A, Week 6: Storage Systems for ML.

Hardware: ASUS ROG Strix SCAR 16, RTX 5080 16GB, 32GB RAM, Ubuntu.
Project root: ~/anvil/storage/
Infrastructure: Docker Compose (MinIO :9000, PostgreSQL :5432, Redis :6379)

Current state: [DESCRIBE - e.g., "Model push works, but deduplication isn't reducing storage"]

What's done:
- [x/blank] Docker Compose infrastructure running
- [x/blank] PostgreSQL schema + migrations
- [x/blank] Model Registry (push/pull/list)
- [x/blank] Model promotion workflow
- [x/blank] Content-addressable storage (dedup)
- [x/blank] Checkpoint save/restore
- [x/blank] Checkpoint tiering (hot→warm→cold)
- [x/blank] Dataset versioning
- [x/blank] CLI tool (anvil model/checkpoint/dataset)
- [x/blank] Garbage collection

Next task: [SPECIFIC NEXT STEP]

Key files:
- anvil_storage/cli.py — CLI entry point
- anvil_storage/registry/model_registry.py — model operations
- anvil_storage/storage/cas.py — deduplication layer
- anvil_storage/checkpoint/tiering.py — tier migration
- docker-compose.yaml — infrastructure

Dependencies: click, sqlalchemy, alembic, boto3 (MinIO), redis, pydantic.
```

---

## Out of Scope

- Distributed file systems (Lustre, GPFS, BeeGFS) — we use MinIO/NFS
- Data lakehouse (Delta Lake, Iceberg) — simple versioning is sufficient
- Feature stores (Feast, Tecton)
- Training data pipelines (ETL, augmentation)
- Model serving/inference storage optimization
- Multi-region replication
- Encryption at rest (MinIO handles this if configured)
- Compliance (data residency, GDPR deletion)
- Large-scale dedup (rabin fingerprinting, variable-size chunking) — fixed 4MB blocks
