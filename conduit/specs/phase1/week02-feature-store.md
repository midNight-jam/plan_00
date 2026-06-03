# Week 2: Feature Store

## Context

**Where it fits:** Week 2 of Phase 1 (Data + Training Pipelines). The feature store sits between raw data ingestion (Week 1) and model training (Week 6). It ensures features are computed once, served consistently, and never leak future data.

**Prerequisites:**
- Week 1 complete: data ingestion pipeline working, MinIO + PostgreSQL running
- Redis installed or available via Docker
- Understanding of SQL joins and time-series data

**What it builds on:** Uses the validated datasets from Week 1's ingestion pipeline as raw inputs. The feature engineering pipeline transforms raw data into model-ready features.

**What comes next:** Week 3 (Data Versioning) will version the feature datasets. Week 5 (Experiment Tracking) will log which features were used per training run.

---

## Learning Goals

- [ ] Understand why feature stores exist: consistency between training and serving, feature reuse, avoiding training-serving skew
- [ ] Understand online vs offline serving: latency requirements, storage backends, access patterns
- [ ] Understand point-in-time correctness: why naive joins cause data leakage and how temporal joins fix it
- [ ] Understand entity-centric feature design: entities, feature views, data sources
- [ ] Understand feature freshness SLAs and how stale features degrade model performance

---

## Implementation Goals

- [ ] Install and configure Feast with local file backend + Redis online store
- [ ] Define entities (user, transaction, merchant) with proper join keys
- [ ] Define feature views with schemas, data sources, and TTLs
- [ ] Build feature engineering pipeline: raw tables → computed features → Feast materialization
- [ ] Implement online serving: sub-10ms feature retrieval for inference
- [ ] Implement offline serving: historical feature retrieval with point-in-time joins
- [ ] Demonstrate point-in-time correctness: show how naive vs correct joins differ
- [ ] Build feature registry with descriptions, owners, and freshness metadata

---

## Acceptance Criteria

1. `feast apply` successfully registers all entity and feature view definitions without errors.
2. `feast materialize` populates the Redis online store with latest feature values for all entities.
3. Online feature retrieval (`get_online_features`) returns results in under 10ms for a single entity lookup.
4. Offline feature retrieval (`get_historical_features`) correctly performs point-in-time joins, returning only features available at each event timestamp.
5. A deliberate data leakage test shows that naive joins include future data while the Feast temporal join does not.
6. Feature engineering pipeline transforms raw transaction data into aggregated features (7d rolling avg, 30d count, etc.) and writes to the offline store.
7. Feature registry lists all features with descriptions, data types, owners, and last-updated timestamps.
8. Adding a new feature (definition + backfill) works without disrupting existing feature retrieval.
9. Redis online store correctly handles TTL expiry — expired features return null rather than stale values.
10. End-to-end: raw data ingestion (Week 1) → feature computation → Feast materialization → online retrieval all complete in under 30 seconds for 100K entities.

---

## Validation Commands

```bash
# Start infrastructure (Redis added to existing stack)
docker compose up -d postgres minio redis

# Apply Feast definitions
cd feature_repo && feast apply

# Run feature engineering pipeline
conduit features compute --config configs/features/user_features.yaml

# Materialize to online store
feast materialize $(date -d '7 days ago' +%Y-%m-%dT%H:%M:%S) $(date +%Y-%m-%dT%H:%M:%S)

# Test online serving
python -c "
from feast import FeatureStore
store = FeatureStore('feature_repo/')
features = store.get_online_features(
    features=['user_features:transaction_count_7d', 'user_features:avg_amount_30d'],
    entity_rows=[{'user_id': 'user_001'}]
).to_dict()
print(features)
"

# Test offline serving (point-in-time)
conduit features get-historical --entity-df data/training_events.parquet --features user_features

# Verify point-in-time correctness
pytest tests/unit/features/test_point_in_time.py -v

# Run full test suite
pytest tests/unit/features/ -v
pytest tests/integration/features/ -v
```

---

## Technical Implementation Details

### Project Structure (additions to Week 1)

```
conduit/
├── feature_repo/
│   ├── feature_store.yaml      # Feast config
│   ├── entities.py             # Entity definitions
│   ├── features/
│   │   ├── user_features.py    # User feature views
│   │   ├── transaction_features.py
│   │   └── merchant_features.py
│   └── data_sources.py         # Source definitions
├── src/conduit/
│   └── features/
│       ├── __init__.py
│       ├── engineering.py      # Feature computation
│       ├── registry.py         # Feature catalog
│       └── serving.py          # Online/offline serving wrapper
└── configs/
    └── features/
        └── user_features.yaml
```

### Feast Configuration

```yaml
# feature_repo/feature_store.yaml
project: conduit
registry: data/registry.db
provider: local
online_store:
  type: redis
  connection_string: "localhost:6379"
offline_store:
  type: file
entity_key_serialization_version: 2
```

### Entity and Feature Definitions

```python
# feature_repo/entities.py
from feast import Entity, ValueType

user = Entity(
    name="user_id",
    value_type=ValueType.STRING,
    description="Unique user identifier",
)

merchant = Entity(
    name="merchant_id",
    value_type=ValueType.STRING,
    description="Unique merchant identifier",
)
```

```python
# feature_repo/features/user_features.py
from datetime import timedelta
from feast import FeatureView, Field
from feast.types import Float64, Int64
from data_sources import user_transactions_source
from entities import user

user_features = FeatureView(
    name="user_features",
    entities=[user],
    ttl=timedelta(days=1),
    schema=[
        Field(name="transaction_count_7d", dtype=Int64),
        Field(name="transaction_count_30d", dtype=Int64),
        Field(name="avg_amount_7d", dtype=Float64),
        Field(name="avg_amount_30d", dtype=Float64),
        Field(name="max_amount_7d", dtype=Float64),
        Field(name="distinct_merchants_7d", dtype=Int64),
    ],
    source=user_transactions_source,
    online=True,
)
```

### Feature Engineering Pipeline

```python
# src/conduit/features/engineering.py
import duckdb
from pathlib import Path
from dataclasses import dataclass

@dataclass
class FeatureConfig:
    name: str
    entity_column: str
    timestamp_column: str
    aggregations: list[dict]
    windows: list[int]  # days

class FeatureEngineer:
    def __init__(self, conn: duckdb.DuckDBPyConnection):
        self.conn = conn

    def compute_features(self, config: FeatureConfig, source_table: str, output_path: Path) -> Path:
        agg_expressions = []
        for agg in config.aggregations:
            for window in config.windows:
                col = agg["column"]
                func = agg["function"]
                alias = f"{func}_{col}_{window}d"
                agg_expressions.append(
                    f"{func.upper()}(CASE WHEN {config.timestamp_column} >= "
                    f"event_timestamp - INTERVAL '{window} days' THEN {col} END) AS {alias}"
                )

        agg_sql = ", ".join(agg_expressions)
        query = f"""
            SELECT
                {config.entity_column},
                event_timestamp,
                {agg_sql}
            FROM {source_table}
            GROUP BY {config.entity_column}, event_timestamp
        """
        self.conn.execute(f"COPY ({query}) TO '{output_path}' (FORMAT PARQUET)")
        return output_path

    def compute_windowed_aggregates(self, entity_col: str, timestamp_col: str,
                                     source: str, windows: list[int]) -> str:
        window_ctes = []
        for w in windows:
            cte = f"""
            w{w}d AS (
                SELECT {entity_col}, {timestamp_col} as event_timestamp,
                    COUNT(*) as txn_count_{w}d,
                    AVG(amount) as avg_amount_{w}d,
                    MAX(amount) as max_amount_{w}d,
                    COUNT(DISTINCT merchant_id) as distinct_merchants_{w}d
                FROM {source}
                WHERE {timestamp_col} >= CURRENT_DATE - INTERVAL '{w} days'
                GROUP BY {entity_col}, {timestamp_col}
            )"""
            window_ctes.append(cte)
        return "WITH " + ",\n".join(window_ctes)
```

### Point-in-Time Correctness Test

```python
# tests/unit/features/test_point_in_time.py
import pandas as pd
from feast import FeatureStore

def test_no_future_data_leakage():
    """Verify that features retrieved for a past event don't include future data."""
    store = FeatureStore("feature_repo/")

    entity_df = pd.DataFrame({
        "user_id": ["user_001", "user_001"],
        "event_timestamp": [
            pd.Timestamp("2024-01-15"),  # should only see data before Jan 15
            pd.Timestamp("2024-02-15"),  # should see data before Feb 15
        ],
    })

    features = store.get_historical_features(
        entity_df=entity_df,
        features=["user_features:transaction_count_7d"],
    ).to_df()

    jan_count = features.iloc[0]["transaction_count_7d"]
    feb_count = features.iloc[1]["transaction_count_7d"]

    # Feb should potentially have more data since it sees Jan transactions
    # but Jan should NOT see Feb transactions
    assert jan_count <= feb_count or jan_count >= 0
    # The key assertion: Jan's count should reflect only pre-Jan-15 data
    assert jan_count == get_ground_truth_count("user_001", "2024-01-08", "2024-01-15")
```

### Docker Compose Addition

```yaml
# Add to docker-compose.yaml
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --maxmemory 256mb --maxmemory-policy allkeys-lru
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| `feast apply` fails with registry error | Delete `data/registry.db` and re-run. Ensure `feature_store.yaml` path is correct. |
| Redis connection refused | `docker compose up -d redis`, verify with `redis-cli ping` |
| Point-in-time join returns NaN | Check that `event_timestamp` column exists in entity_df and feature source has overlapping time range. |
| Materialization slow | Reduce date range. For dev, materialize last 7 days only. |
| Feature view schema mismatch | After changing schema, `feast teardown` then `feast apply` for clean slate. |
| Import errors with Feast | `pip install feast[redis]` — the redis extra is required for online store. |

---

## Agent Handoff Template

```
I'm working on the Conduit project, Week 2: Feature Store.

Hardware: ASUS ROG Strix SCAR 16, RTX 5080 16GB, 32GB RAM, Ubuntu.
Project root: ~/conduit/

Current state: [describe what's working/broken]

What I need help with: [specific issue]

Key files:
- feature_repo/feature_store.yaml — Feast configuration
- feature_repo/features/user_features.py — Feature view definitions
- src/conduit/features/engineering.py — Feature computation pipeline
- src/conduit/features/serving.py — Online/offline serving wrapper
- docker-compose.yaml — includes Redis for online store

Infrastructure: PostgreSQL (metadata), MinIO (raw data), Redis (online features), DuckDB (computation).
The feature pipeline: raw data → DuckDB aggregation → Parquet → Feast offline store → Redis online store.
```

---

## Out of Scope

- Feature monitoring/drift detection (covered in Phase 2)
- Streaming features (Kafka → real-time aggregation) — batch only
- Multi-environment Feast deployment (dev/staging/prod)
- Feature store UI/web interface
- Access control on features (who can read/write)
- Complex feature transformations (embeddings, NLP features) — simple aggregates only
- Feast on Kubernetes — local development only
