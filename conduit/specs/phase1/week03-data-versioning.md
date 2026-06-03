# Week 3: Data Versioning and Lineage

## Context

**Where it fits:** Week 3 of Phase 1 (Data + Training Pipelines). Data versioning ensures reproducibility — given any model, you can trace back to the exact data that produced it. Lineage tracking makes the full data flow visible and auditable.

**Prerequisites:**
- Week 1 complete: data ingestion and validation working
- Week 2 complete: feature store with computed features
- Git fundamentals (branches, commits, tags)
- DVC installed (`pip install dvc[s3]`)

**What it builds on:** Versions the datasets produced by Week 1's ingestion pipeline and the features from Week 2's feature store. Adds traceability to the entire data flow.

**What comes next:** Week 4 (Pipeline Orchestration) will use versioned datasets as pipeline inputs. Week 5 (Experiment Tracking) will link model runs to specific dataset versions.

---

## Learning Goals

- [ ] Understand why data versioning is critical: reproducibility, auditing, rollback, compliance
- [ ] Understand DVC's approach: Git for metadata, object storage for data
- [ ] Understand data lineage: tracking transformations from raw source to model input
- [ ] Understand dataset diffing: how to detect meaningful changes between versions
- [ ] Understand the relationship between code versions (git) and data versions (DVC)

---

## Implementation Goals

- [ ] Set up DVC with MinIO as remote storage backend
- [ ] Version training datasets: track, commit, push, pull, checkout previous versions
- [ ] Build lineage graph: raw data → transformations → features → training set
- [ ] Implement lineage visualization (Mermaid/Graphviz DAG output)
- [ ] Build dataset comparison: row-level and distribution-level diffs between versions
- [ ] Implement reproducibility proof: given a model ID, retrieve exact data + code + config
- [ ] Build metadata catalog: searchable registry of all datasets with schema, size, freshness
- [ ] Integrate versioning into CLI: `conduit data version`, `conduit data diff`, `conduit lineage show`

---

## Acceptance Criteria

1. `dvc push` successfully uploads tracked dataset files to MinIO remote storage.
2. `dvc checkout <version>` restores a previous dataset version with byte-for-byte accuracy, verified by checksum.
3. `conduit lineage show --dataset training_set_v3` produces a DAG visualization showing the full transformation chain from raw sources.
4. `conduit data diff --v1 abc123 --v2 def456` reports row count changes, schema changes, and distribution shifts between two dataset versions.
5. Given a model registered in the metadata store, `conduit reproduce --model model_v5` retrieves the exact git commit, DVC data version, and config used to train it.
6. The metadata catalog returns all datasets matching a search query (by name, column, owner) in under 1 second.
7. A lineage graph correctly shows that `training_set_v3` depends on `user_features_v2` and `transactions_raw_v7`.
8. Dataset versioning adds less than 5% overhead to the ingestion pipeline (measured by wall-clock time).
9. `conduit data catalog list` shows all registered datasets with their current version, row count, schema, and last-updated timestamp.
10. Rolling back a dataset (`conduit data rollback --dataset users --to-version 2`) correctly restores the previous version and updates all metadata references.

---

## Validation Commands

```bash
# Initialize DVC with MinIO remote
dvc init
dvc remote add -d minio s3://conduit-data
dvc remote modify minio endpointurl http://localhost:9000
dvc remote modify minio access_key_id conduit
dvc remote modify minio secret_access_key conduit_dev

# Track and version a dataset
conduit data version --dataset transactions --message "Initial transaction dataset"
dvc push

# Make changes and create new version
conduit data ingest --source data/raw/transactions_v2.csv --dataset transactions
conduit data version --dataset transactions --message "Added December transactions"
dvc push

# Diff between versions
conduit data diff --dataset transactions --v1 HEAD~1 --v2 HEAD

# Show lineage
conduit lineage show --dataset training_set --format mermaid > lineage.md
conduit lineage show --dataset training_set --format dot | dot -Tpng > lineage.png

# Reproduce from model
conduit reproduce --model model_v5 --dry-run

# Search catalog
conduit data catalog search --column "user_id" --type "feature"

# Run tests
pytest tests/unit/versioning/ -v
pytest tests/integration/versioning/ -v
```

---

## Technical Implementation Details

### Project Structure (additions)

```
conduit/
├── src/conduit/
│   └── versioning/
│       ├── __init__.py
│       ├── dvc_manager.py      # DVC operations wrapper
│       ├── lineage.py          # Lineage graph building
│       ├── diff.py             # Dataset comparison
│       ├── catalog.py          # Metadata catalog
│       └── reproduce.py        # Reproducibility engine
├── .dvc/
│   └── config                  # DVC configuration
├── data/
│   ├── raw/
│   │   └── transactions.csv.dvc  # DVC tracking file
│   └── processed/
│       └── training_set.parquet.dvc
└── lineage/
    └── manifest.yaml           # Lineage definitions
```

### Lineage Tracking

```python
# src/conduit/versioning/lineage.py
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

class NodeType(Enum):
    RAW_DATA = "raw_data"
    TRANSFORMATION = "transformation"
    FEATURE = "feature"
    TRAINING_SET = "training_set"
    MODEL = "model"

@dataclass
class LineageNode:
    id: str
    name: str
    node_type: NodeType
    version: str
    metadata: dict = field(default_factory=dict)

@dataclass
class LineageEdge:
    source: str
    target: str
    transformation: str  # description of what was done

class LineageGraph:
    def __init__(self):
        self.nodes: dict[str, LineageNode] = {}
        self.edges: list[LineageEdge] = []

    def add_node(self, node: LineageNode):
        self.nodes[node.id] = node

    def add_edge(self, source_id: str, target_id: str, transformation: str):
        self.edges.append(LineageEdge(source_id, target_id, transformation))

    def get_ancestors(self, node_id: str) -> list[LineageNode]:
        """Trace back all upstream dependencies of a node."""
        ancestors = []
        visited = set()
        queue = [node_id]
        while queue:
            current = queue.pop(0)
            for edge in self.edges:
                if edge.target == current and edge.source not in visited:
                    visited.add(edge.source)
                    ancestors.append(self.nodes[edge.source])
                    queue.append(edge.source)
        return ancestors

    def to_mermaid(self) -> str:
        lines = ["graph LR"]
        for node in self.nodes.values():
            shape = {"raw_data": "[({})]", "transformation": "[/{}\\]",
                     "feature": "[[{}]]", "training_set": "[{}]", "model": "(({}))"}
            fmt = shape.get(node.node_type.value, "[{}]")
            lines.append(f"    {node.id}{fmt.format(node.name + ' v' + node.version)}")
        for edge in self.edges:
            lines.append(f"    {edge.source} -->|{edge.transformation}| {edge.target}")
        return "\n".join(lines)

    def to_dot(self) -> str:
        lines = ["digraph lineage {", "    rankdir=LR;"]
        for node in self.nodes.values():
            lines.append(f'    {node.id} [label="{node.name} v{node.version}"];')
        for edge in self.edges:
            lines.append(f'    {edge.source} -> {edge.target} [label="{edge.transformation}"];')
        lines.append("}")
        return "\n".join(lines)
```

### Dataset Diffing

```python
# src/conduit/versioning/diff.py
import duckdb
from dataclasses import dataclass

@dataclass
class DatasetDiff:
    rows_added: int
    rows_removed: int
    rows_modified: int
    schema_changes: list[str]
    distribution_changes: dict[str, dict]

class DatasetComparator:
    def __init__(self, conn: duckdb.DuckDBPyConnection):
        self.conn = conn

    def diff(self, path_v1: str, path_v2: str, key_column: str) -> DatasetDiff:
        self.conn.execute(f"CREATE OR REPLACE VIEW v1 AS SELECT * FROM read_parquet('{path_v1}')")
        self.conn.execute(f"CREATE OR REPLACE VIEW v2 AS SELECT * FROM read_parquet('{path_v2}')")

        added = self.conn.execute(
            f"SELECT COUNT(*) FROM v2 WHERE {key_column} NOT IN (SELECT {key_column} FROM v1)"
        ).fetchone()[0]

        removed = self.conn.execute(
            f"SELECT COUNT(*) FROM v1 WHERE {key_column} NOT IN (SELECT {key_column} FROM v2)"
        ).fetchone()[0]

        schema_v1 = set(self.conn.execute("DESCRIBE v1").fetchall())
        schema_v2 = set(self.conn.execute("DESCRIBE v2").fetchall())
        schema_changes = [f"+{c[0]}:{c[1]}" for c in schema_v2 - schema_v1]
        schema_changes += [f"-{c[0]}:{c[1]}" for c in schema_v1 - schema_v2]

        distribution_changes = self._compare_distributions()
        return DatasetDiff(added, removed, 0, schema_changes, distribution_changes)

    def _compare_distributions(self) -> dict[str, dict]:
        cols = self.conn.execute("SELECT column_name FROM information_schema.columns WHERE table_name='v1'").fetchall()
        changes = {}
        for (col,) in cols:
            try:
                stats_v1 = self.conn.execute(f"SELECT AVG({col}), STDDEV({col}) FROM v1").fetchone()
                stats_v2 = self.conn.execute(f"SELECT AVG({col}), STDDEV({col}) FROM v2").fetchone()
                if stats_v1[0] and stats_v2[0]:
                    pct_change = abs(stats_v2[0] - stats_v1[0]) / max(abs(stats_v1[0]), 1e-10)
                    if pct_change > 0.05:
                        changes[col] = {"mean_v1": stats_v1[0], "mean_v2": stats_v2[0], "pct_change": pct_change}
            except Exception:
                continue
        return changes
```

### DVC Manager

```python
# src/conduit/versioning/dvc_manager.py
import subprocess
from pathlib import Path
from dataclasses import dataclass

@dataclass
class VersionInfo:
    dataset: str
    version: str
    git_commit: str
    dvc_hash: str
    timestamp: str
    row_count: int

class DVCManager:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root

    def track(self, data_path: Path) -> str:
        result = subprocess.run(["dvc", "add", str(data_path)], capture_output=True, text=True, cwd=self.repo_root)
        if result.returncode != 0:
            raise RuntimeError(f"DVC add failed: {result.stderr}")
        return self._get_dvc_hash(data_path)

    def push(self) -> None:
        subprocess.run(["dvc", "push"], check=True, cwd=self.repo_root)

    def checkout(self, data_path: Path, git_rev: str) -> None:
        subprocess.run(["git", "checkout", git_rev, "--", f"{data_path}.dvc"], check=True, cwd=self.repo_root)
        subprocess.run(["dvc", "checkout", str(data_path)], check=True, cwd=self.repo_root)

    def _get_dvc_hash(self, data_path: Path) -> str:
        dvc_file = Path(f"{data_path}.dvc")
        import yaml
        with open(self.repo_root / dvc_file) as f:
            meta = yaml.safe_load(f)
        return meta["outs"][0]["md5"]
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| `dvc push` fails with S3 error | Verify MinIO is running and credentials match. Test with `aws --endpoint-url http://localhost:9000 s3 ls` |
| DVC tracking file not created | Run `dvc init` first. Ensure you're in a git repo. |
| Lineage graph missing nodes | Check `lineage/manifest.yaml` has all transformations registered. Each pipeline step must log its lineage. |
| Diff is slow for large datasets | Use sampling: compare statistics on a 10% sample first, full diff only if sample shows changes. |
| Git commit hash mismatch | Ensure DVC files are committed to git before `dvc push`. Sequence: `dvc add` → `git add .dvc` → `git commit` → `dvc push`. |
| Reproduce fails | Verify the model's metadata contains both git SHA and DVC version. Check `conduit reproduce --model X --dry-run` first. |

---

## Agent Handoff Template

```
I'm working on the Conduit project, Week 3: Data Versioning and Lineage.

Hardware: ASUS ROG Strix SCAR 16, RTX 5080 16GB, 32GB RAM, Ubuntu.
Project root: ~/conduit/

Current state: [describe what's working/broken]

What I need help with: [specific issue]

Key files:
- src/conduit/versioning/dvc_manager.py — DVC operations
- src/conduit/versioning/lineage.py — Lineage graph building/visualization
- src/conduit/versioning/diff.py — Dataset comparison
- src/conduit/versioning/catalog.py — Metadata catalog
- lineage/manifest.yaml — Lineage definitions
- .dvc/config — DVC remote configuration

Infrastructure: MinIO (DVC remote storage), PostgreSQL (metadata), Git (code + DVC files).
Flow: data files tracked by DVC → metadata in .dvc files committed to git → lineage registered in catalog.
```

---

## Out of Scope

- Cloud DVC remotes (S3, GCS) — MinIO only for local dev
- Branch-level dataset isolation (DVC experiments) — simple linear versioning only
- Automated lineage capture from code AST analysis
- Data governance (PII tagging, access policies, retention rules)
- Large-scale catalog (Apache Atlas, DataHub) — simple PostgreSQL catalog
- Delta Lake or Iceberg time travel — DVC + Parquet approach
- Cross-repository lineage tracking
