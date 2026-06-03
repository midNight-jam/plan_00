# Week 1: Data Engineering Foundations

## Context

**Where it fits:** This is the first week of the Conduit project (Phase 1 — Data + Training Pipelines). Everything downstream depends on solid data infrastructure.

**Prerequisites:**
- Python 3.11+ installed
- Docker and Docker Compose available
- Basic SQL knowledge
- Familiarity with pandas/polars dataframes

**What it builds on:** This is the foundation week. All subsequent weeks (feature store, versioning, orchestration) depend on the data ingestion and validation infrastructure built here.

**What comes next:** Week 2 (Feature Store) will consume the validated, profiled data produced by this week's pipelines.

---

## Learning Goals

- [ ] Understand columnar storage formats and why DuckDB is ideal for local analytics
- [ ] Understand data quality dimensions: completeness, accuracy, consistency, timeliness
- [ ] Understand schema evolution strategies (additive, backward-compatible, breaking)
- [ ] Understand the role of object storage (MinIO) vs relational databases (PostgreSQL) in ML pipelines
- [ ] Understand data profiling and why automated statistics catch silent data corruption

---

## Implementation Goals

- [ ] Set up DuckDB for local analytics queries over ingested data
- [ ] Build data ingestion pipeline accepting CSV, JSON, and Parquet formats
- [ ] Implement schema validation with configurable rules per dataset
- [ ] Build data quality checks: nulls, ranges, uniqueness, referential integrity
- [ ] Implement schema evolution: detect and handle new columns, type changes
- [ ] Build automated data profiling: distributions, missing rates, cardinality
- [ ] Deploy PostgreSQL (metadata) and MinIO (raw storage) via Docker Compose
- [ ] Build CLI commands: `conduit data ingest`, `conduit data validate`, `conduit data profile`

---

## Acceptance Criteria

1. `conduit data ingest --source data/raw/users.csv --dataset users` successfully ingests a CSV file into MinIO and registers metadata in PostgreSQL.
2. `conduit data ingest` handles CSV, JSON, and Parquet formats without code changes, auto-detecting format from file extension.
3. `conduit data validate --dataset users` runs all configured quality checks and produces a structured report (JSON) with pass/fail per check.
4. A null check on a NOT NULL column correctly flags rows with missing values and reports the exact count and percentage.
5. Schema evolution: ingesting a file with a new column auto-detects the change, logs it, and updates the schema registry without breaking existing queries.
6. `conduit data profile --dataset users` produces statistics including min, max, mean, median, null rate, cardinality, and top-k values for each column.
7. DuckDB queries over ingested data return results in under 100ms for datasets up to 1M rows.
8. PostgreSQL metadata store tracks: dataset name, schema version, ingestion timestamp, row count, file location in MinIO.
9. A referential integrity check correctly identifies orphaned foreign keys between two related datasets.
10. The full ingest → validate → profile pipeline completes in under 10 seconds for a 100K-row dataset.

---

## Validation Commands

```bash
# Start infrastructure
docker compose up -d postgres minio

# Run ingestion
conduit data ingest --source data/raw/transactions.csv --dataset transactions
conduit data ingest --source data/raw/users.json --dataset users
conduit data ingest --source data/raw/events.parquet --dataset events

# Validate data quality
conduit data validate --dataset transactions --config configs/quality/transactions.yaml

# Profile dataset
conduit data profile --dataset users --output reports/users_profile.json

# Query with DuckDB
conduit data query "SELECT COUNT(*), AVG(amount) FROM transactions WHERE date > '2024-01-01'"

# Check schema evolution
conduit data schema --dataset users --history

# Run tests
pytest tests/unit/data/ -v
pytest tests/integration/data/ -v --timeout=30
```

---

## Technical Implementation Details

### Project Structure

```
conduit/
├── src/
│   └── conduit/
│       ├── __init__.py
│       ├── cli/
│       │   ├── __init__.py
│       │   └── data.py          # CLI commands
│       ├── data/
│       │   ├── __init__.py
│       │   ├── ingestion.py     # Ingestion pipeline
│       │   ├── validation.py    # Quality checks
│       │   ├── profiling.py     # Data profiling
│       │   ├── schema.py        # Schema management
│       │   └── storage.py       # MinIO + PostgreSQL
│       └── config/
│           └── settings.py
├── configs/
│   └── quality/
│       └── transactions.yaml
├── docker-compose.yaml
├── pyproject.toml
└── tests/
```

### Core Ingestion Class

```python
# src/conduit/data/ingestion.py
from dataclasses import dataclass
from pathlib import Path
from enum import Enum
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

class FileFormat(Enum):
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"

@dataclass
class IngestionResult:
    dataset: str
    rows_ingested: int
    schema_version: int
    storage_path: str
    schema_changes: list[str]

class DataIngestionPipeline:
    def __init__(self, storage: "ObjectStorage", metadata: "MetadataStore"):
        self.storage = storage
        self.metadata = metadata
        self.conn = duckdb.connect()

    def ingest(self, source: Path, dataset: str) -> IngestionResult:
        fmt = self._detect_format(source)
        table = self._read_source(source, fmt)
        schema_changes = self._detect_schema_changes(dataset, table.schema)
        storage_path = self.storage.put(dataset, table)
        self.metadata.register_ingestion(dataset, table.num_rows, storage_path, table.schema)
        return IngestionResult(
            dataset=dataset,
            rows_ingested=table.num_rows,
            schema_version=self.metadata.get_schema_version(dataset),
            storage_path=storage_path,
            schema_changes=schema_changes,
        )

    def _read_source(self, source: Path, fmt: FileFormat) -> pa.Table:
        match fmt:
            case FileFormat.CSV:
                return self.conn.execute(f"SELECT * FROM read_csv_auto('{source}')").arrow()
            case FileFormat.JSON:
                return self.conn.execute(f"SELECT * FROM read_json_auto('{source}')").arrow()
            case FileFormat.PARQUET:
                return pq.read_table(source)

    def _detect_format(self, source: Path) -> FileFormat:
        return FileFormat(source.suffix.lstrip("."))

    def _detect_schema_changes(self, dataset: str, new_schema: pa.Schema) -> list[str]:
        existing = self.metadata.get_schema(dataset)
        if existing is None:
            return ["new_dataset"]
        changes = []
        existing_names = {f.name for f in existing}
        for field in new_schema:
            if field.name not in existing_names:
                changes.append(f"added_column:{field.name}:{field.type}")
        return changes
```

### Data Validation

```python
# src/conduit/data/validation.py
from dataclasses import dataclass
from typing import Any
import duckdb

@dataclass
class CheckResult:
    check_name: str
    passed: bool
    details: dict[str, Any]

class DataValidator:
    def __init__(self, conn: duckdb.DuckDBPyConnection):
        self.conn = conn

    def check_not_null(self, table: str, column: str) -> CheckResult:
        result = self.conn.execute(
            f"SELECT COUNT(*) as nulls, COUNT(*) * 100.0 / (SELECT COUNT(*) FROM {table}) as pct "
            f"FROM {table} WHERE {column} IS NULL"
        ).fetchone()
        return CheckResult(
            check_name=f"not_null:{column}",
            passed=result[0] == 0,
            details={"null_count": result[0], "null_percentage": result[1]},
        )

    def check_range(self, table: str, column: str, min_val: float, max_val: float) -> CheckResult:
        result = self.conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE {column} < {min_val} OR {column} > {max_val}"
        ).fetchone()
        return CheckResult(
            check_name=f"range:{column}[{min_val},{max_val}]",
            passed=result[0] == 0,
            details={"out_of_range_count": result[0]},
        )

    def check_unique(self, table: str, column: str) -> CheckResult:
        result = self.conn.execute(
            f"SELECT COUNT(*) - COUNT(DISTINCT {column}) as duplicates FROM {table}"
        ).fetchone()
        return CheckResult(
            check_name=f"unique:{column}",
            passed=result[0] == 0,
            details={"duplicate_count": result[0]},
        )

    def check_referential_integrity(self, table: str, column: str, ref_table: str, ref_column: str) -> CheckResult:
        result = self.conn.execute(
            f"SELECT COUNT(*) FROM {table} t "
            f"LEFT JOIN {ref_table} r ON t.{column} = r.{ref_column} "
            f"WHERE r.{ref_column} IS NULL"
        ).fetchone()
        return CheckResult(
            check_name=f"ref_integrity:{table}.{column}->{ref_table}.{ref_column}",
            passed=result[0] == 0,
            details={"orphaned_count": result[0]},
        )
```

### Docker Compose

```yaml
# docker-compose.yaml
services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_DB: conduit
      POSTGRES_USER: conduit
      POSTGRES_PASSWORD: conduit_dev
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: conduit
      MINIO_ROOT_PASSWORD: conduit_dev
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - miniodata:/data

volumes:
  pgdata:
  miniodata:
```

### Quality Config

```yaml
# configs/quality/transactions.yaml
dataset: transactions
checks:
  - type: not_null
    columns: [transaction_id, user_id, amount, timestamp]
  - type: range
    column: amount
    min: 0.01
    max: 100000.00
  - type: unique
    column: transaction_id
  - type: referential_integrity
    column: user_id
    references:
      table: users
      column: id
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| DuckDB can't read file | Check file path is absolute. DuckDB's `read_csv_auto` needs valid paths. Try `duckdb.execute("SELECT * FROM read_csv_auto('/absolute/path.csv') LIMIT 5")` |
| MinIO connection refused | Ensure Docker is running: `docker compose ps`. Check port 9000 is free: `lsof -i :9000` |
| PostgreSQL auth failed | Check `.env` matches docker-compose credentials. Try `psql -h localhost -U conduit -d conduit` |
| Schema detection wrong types | DuckDB infers types aggressively. Use explicit schema: `read_csv('/path', columns={'col': 'VARCHAR'})` |
| Slow ingestion for large files | Use Arrow for zero-copy reads. Avoid pandas for files > 100MB. DuckDB streams naturally. |
| Import errors | Ensure `pip install -e .` was run. Check `pyproject.toml` has all deps. |

---

## Agent Handoff Template

```
I'm working on the Conduit project (ML Systems Engineer track), Week 1: Data Engineering Foundations.

Hardware: ASUS ROG Strix SCAR 16, RTX 5080 16GB, 32GB RAM, Ubuntu.
Project root: ~/conduit/

Current state: [describe what's working/broken]

What I need help with: [specific issue]

Key files:
- src/conduit/data/ingestion.py — ingestion pipeline
- src/conduit/data/validation.py — quality checks
- src/conduit/data/profiling.py — automated profiling
- docker-compose.yaml — PostgreSQL + MinIO
- configs/quality/ — validation rule configs

The pipeline should: ingest CSV/JSON/Parquet → validate schema → store in MinIO → register in PostgreSQL → profile data.
```

---

## Out of Scope

- Cloud deployment (AWS S3, BigQuery) — this week is local-only
- Streaming ingestion (Kafka, Kinesis) — batch only this week
- Complex transformations (dbt-style) — just ingestion and validation
- Authentication/authorization for MinIO or PostgreSQL
- Web UI for data browsing — CLI only
- Distributed processing (Spark, Ray) — single-node DuckDB is sufficient
- Real-time data quality monitoring — batch validation only
