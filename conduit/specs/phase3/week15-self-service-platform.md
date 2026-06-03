# Week 15: Self-Service ML Platform

## Context

**Where it fits:** Phase 3, Week 15 — Platform Maturity + Portfolio
**Prerequisites:** Phases 1+2 complete (data pipelines, training infrastructure, model serving, monitoring, drift detection, auto-retraining all operational)
**What it builds on:** The infrastructure works but requires deep platform knowledge to use. This week abstracts away complexity behind a developer-friendly CLI/SDK so data scientists can ship models without understanding Kubernetes, GPU scheduling, or pipeline orchestration internals.

**Hardware:** ASUS ROG Strix SCAR 16, RTX 5080 16GB, 32GB RAM, Ubuntu

---

## Learning Goals

- [ ] Understand CLI framework design (Click/Typer) and subcommand routing
- [ ] Learn SDK design patterns: fluent interfaces, builder patterns, sensible defaults
- [ ] Study configuration validation with Pydantic and schema enforcement
- [ ] Explore plugin architectures: entry points, hook systems, registry patterns
- [ ] Understand guardrail systems that prevent common ML mistakes at config time
- [ ] Learn cookiecutter/copier templating for project scaffolding
- [ ] Study documentation-as-code approaches (MkDocs, Sphinx, doctest)

---

## Implementation Goals

- [ ] Build `conduit` CLI with Typer: `pipeline create`, `model train`, `experiment compare`
- [ ] Implement Python SDK with fluent interface for programmatic access
- [ ] Create project templates (classification, regression, NLP, time-series)
- [ ] Build guardrail system: data leakage detection, missing validation checks, no test set warnings
- [ ] Implement configuration validation with clear error messages
- [ ] Write getting-started guide and cookbook for common tasks
- [ ] Build plugin system with custom transformers, models, and evaluators
- [ ] Add shell completions (bash/zsh/fish)
- [ ] Create interactive `conduit init` wizard for new projects

---

## Acceptance Criteria

1. `conduit pipeline create --name my-pipeline --type batch` scaffolds a valid pipeline config and source directory in under 2 seconds
2. `conduit model train --config train.yaml` validates config, detects 5+ common misconfigurations, and launches training with a single command
3. `conduit experiment compare exp-001 exp-002` produces a formatted table of metrics, parameters, and resource usage differences
4. Project templates generate fully runnable projects: `conduit init --template classification` produces code that trains a model with zero edits
5. Guardrails detect data leakage (target column in features), missing train/test split, and missing validation set with actionable error messages
6. Configuration validation catches type errors, missing required fields, and invalid value ranges before any GPU allocation occurs
7. Plugin system loads custom components via entry points: a third-party transformer installed via pip is discoverable by the CLI
8. SDK allows full lifecycle in <20 lines: `Pipeline.create() -> train() -> evaluate() -> deploy()`
9. Documentation site builds with `mkdocs serve` and includes quickstart, API reference, and 5+ cookbook recipes
10. Shell completions install correctly and complete subcommands, flags, and dynamic values (model names, pipeline IDs)

---

## Validation Commands

```bash
# Install conduit CLI in development mode
cd ~/conduit && pip install -e ".[dev,cli]"

# Verify CLI is available and shows help
conduit --help
conduit pipeline --help
conduit model --help

# Test project scaffolding
conduit init --template classification --name test-project --output /tmp/test-project
ls /tmp/test-project/{src,configs,tests,data}

# Test configuration validation (should fail with clear error)
echo "model:\n  type: invalid_model\n  epochs: -1" > /tmp/bad_config.yaml
conduit model train --config /tmp/bad_config.yaml --dry-run 2>&1 | grep "ValidationError"

# Test guardrails
conduit validate --config /tmp/test-project/configs/train.yaml

# Test plugin discovery
conduit plugins list
conduit plugins install conduit-plugin-example
conduit plugins list | grep "example"

# Test experiment comparison
conduit experiment compare exp-001 exp-002 --format table

# Test SDK programmatic access
python -c "
from conduit import Pipeline, Model
p = Pipeline.create('test', template='classification')
print(f'Pipeline created: {p.name}')
print(f'Config valid: {p.validate()}')
"

# Test documentation build
cd ~/conduit/docs && mkdocs build --strict
echo "Docs built: $(find site/ -name '*.html' | wc -l) pages"

# Test shell completions
conduit --install-completion bash
source ~/.bash_completion.d/conduit
```

---

## Technical Implementation Details

### CLI Architecture (Typer)

```python
# src/conduit/cli/main.py
import typer
from rich.console import Console
from conduit.cli import pipeline, model, experiment, plugins

app = typer.Typer(
    name="conduit",
    help="Self-service ML platform CLI",
    no_args_is_help=True,
)

app.add_typer(pipeline.app, name="pipeline")
app.add_typer(model.app, name="model")
app.add_typer(experiment.app, name="experiment")
app.add_typer(plugins.app, name="plugins")

console = Console()

@app.command()
def init(
    template: str = typer.Option("classification", help="Project template"),
    name: str = typer.Option(..., prompt=True, help="Project name"),
    output: str = typer.Option(".", help="Output directory"),
):
    """Initialize a new ML project from template."""
    from conduit.templates import scaffold_project
    project_path = scaffold_project(template=template, name=name, output_dir=output)
    console.print(f"[green]✓[/green] Project created at {project_path}")
    console.print(f"  cd {project_path} && conduit model train --config configs/train.yaml")

@app.command()
def validate(config: str = typer.Argument(..., help="Config file path")):
    """Validate configuration without running anything."""
    from conduit.validation import validate_config
    results = validate_config(config)
    for issue in results.issues:
        icon = "✗" if issue.severity == "error" else "⚠"
        console.print(f"  [{issue.severity}] {icon} {issue.message}")
    if results.is_valid:
        console.print("[green]✓[/green] Configuration is valid")
    raise typer.Exit(code=0 if results.is_valid else 1)
```

### SDK Fluent Interface

```python
# src/conduit/sdk/pipeline.py
from dataclasses import dataclass, field
from typing import Optional
from conduit.sdk.config import PipelineConfig
from conduit.sdk.training import TrainingRun
from conduit.sdk.evaluation import EvaluationReport

@dataclass
class Pipeline:
    name: str
    config: PipelineConfig
    _runs: list = field(default_factory=list)

    @classmethod
    def create(cls, name: str, template: str = "classification", **kwargs) -> "Pipeline":
        config = PipelineConfig.from_template(template, **kwargs)
        return cls(name=name, config=config)

    def train(self, data_path: str, **overrides) -> TrainingRun:
        self.config.update(overrides)
        self._validate_before_train()
        run = TrainingRun.launch(pipeline=self, data_path=data_path)
        self._runs.append(run)
        return run

    def evaluate(self, run: Optional[TrainingRun] = None) -> EvaluationReport:
        run = run or self._runs[-1]
        return EvaluationReport.generate(run)

    def deploy(self, run: Optional[TrainingRun] = None, canary_pct: float = 10.0):
        run = run or self._runs[-1]
        from conduit.sdk.deployment import deploy_model
        return deploy_model(run.model_artifact, canary_percent=canary_pct)

    def _validate_before_train(self):
        from conduit.guardrails import run_guardrails
        issues = run_guardrails(self.config)
        if issues.has_errors:
            raise GuardrailError(issues)
```

### Guardrails System

```python
# src/conduit/guardrails/leakage.py
import pandas as pd
from conduit.guardrails.base import Guardrail, Issue, Severity

class DataLeakageGuardrail(Guardrail):
    """Detect target column leakage into feature set."""

    name = "data_leakage"

    def check(self, config) -> list[Issue]:
        issues = []
        target_col = config.target_column
        feature_cols = config.feature_columns

        if target_col in feature_cols:
            issues.append(Issue(
                severity=Severity.ERROR,
                message=f"Target column '{target_col}' found in feature columns. "
                        f"This is data leakage and will produce unrealistically high metrics.",
                fix=f"Remove '{target_col}' from feature_columns in your config.",
            ))

        # Check for highly correlated proxies
        if config.data_path:
            df = pd.read_parquet(config.data_path, columns=feature_cols + [target_col])
            correlations = df[feature_cols].corrwith(df[target_col]).abs()
            suspicious = correlations[correlations > 0.95]
            for col, corr in suspicious.items():
                issues.append(Issue(
                    severity=Severity.WARNING,
                    message=f"Feature '{col}' has {corr:.3f} correlation with target. "
                            f"Possible proxy leakage.",
                    fix=f"Investigate if '{col}' would be available at prediction time.",
                ))
        return issues


class TrainTestSplitGuardrail(Guardrail):
    """Ensure proper train/test/validation split exists."""

    name = "train_test_split"

    def check(self, config) -> list[Issue]:
        issues = []
        if not config.test_split and not config.test_path:
            issues.append(Issue(
                severity=Severity.ERROR,
                message="No test set configured. Cannot evaluate generalization.",
                fix="Add test_split: 0.2 or test_path to your config.",
            ))
        if not config.validation_split and not config.validation_path:
            issues.append(Issue(
                severity=Severity.WARNING,
                message="No validation set configured. Cannot tune hyperparameters safely.",
                fix="Add validation_split: 0.1 or validation_path to your config.",
            ))
        return issues
```

### Plugin System

```python
# src/conduit/plugins/registry.py
from importlib.metadata import entry_points
from typing import Protocol, runtime_checkable

@runtime_checkable
class ConduitPlugin(Protocol):
    name: str
    version: str
    def register(self, registry: "PluginRegistry") -> None: ...

class PluginRegistry:
    def __init__(self):
        self._transformers: dict[str, type] = {}
        self._models: dict[str, type] = {}
        self._evaluators: dict[str, type] = {}

    def register_transformer(self, name: str, cls: type):
        self._transformers[name] = cls

    def register_model(self, name: str, cls: type):
        self._models[name] = cls

    def register_evaluator(self, name: str, cls: type):
        self._evaluators[name] = cls

    def discover_plugins(self):
        """Load plugins from installed packages via entry_points."""
        eps = entry_points(group="conduit.plugins")
        for ep in eps:
            plugin_cls = ep.load()
            plugin = plugin_cls()
            if isinstance(plugin, ConduitPlugin):
                plugin.register(self)

    def get_transformer(self, name: str) -> type:
        if name not in self._transformers:
            available = ", ".join(self._transformers.keys())
            raise KeyError(f"Transformer '{name}' not found. Available: {available}")
        return self._transformers[name]
```

### Project file structure:
```
~/conduit/
├── src/conduit/
│   ├── cli/
│   │   ├── main.py
│   │   ├── pipeline.py
│   │   ├── model.py
│   │   └── experiment.py
│   ├── sdk/
│   │   ├── pipeline.py
│   │   ├── config.py
│   │   ├── training.py
│   │   └── deployment.py
│   ├── guardrails/
│   │   ├── base.py
│   │   ├── leakage.py
│   │   ├── split.py
│   │   └── runner.py
│   ├── plugins/
│   │   ├── registry.py
│   │   └── base.py
│   ├── templates/
│   │   ├── classification/
│   │   ├── regression/
│   │   ├── nlp/
│   │   └── timeseries/
│   └── validation/
│       ├── schema.py
│       └── validators.py
├── docs/
│   ├── mkdocs.yml
│   ├── getting-started.md
│   ├── cookbook/
│   └── api-reference/
└── pyproject.toml
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| Typer subcommands not routing correctly | Ensure `add_typer()` is called before `app()` — order matters for registration |
| Entry points not discovering plugins | Run `pip install -e .` after adding `[project.entry-points]` to pyproject.toml; cached metadata won't update without reinstall |
| Shell completions not working | Run `conduit --install-completion` and source the generated file; check `~/.bash_completion.d/` or `~/.zfunc/` |
| Config validation too strict for advanced users | Add `--skip-guardrails` flag and `strict: false` config option for power users who know what they're doing |
| Template rendering fails with Jinja errors | Ensure template variables use `{{ cookiecutter.project_name }}` format; test templates in isolation with `copier copy` |
| Plugin import errors at discovery time | Wrap `ep.load()` in try/except and log warnings rather than crashing the entire CLI |

---

## Agent Handoff Template

```
I'm building Week 15 of the Conduit ML platform: Self-Service CLI/SDK.

Current state:
- Full ML lifecycle infrastructure from Phases 1+2 is operational
- Training, serving, monitoring, drift detection, and auto-retraining work
- No developer-facing abstraction layer exists yet

What I need help with:
- [specific task: e.g., "implementing the guardrails system for data leakage detection"]

Key files:
- CLI entry point: src/conduit/cli/main.py
- SDK pipeline: src/conduit/sdk/pipeline.py
- Guardrails: src/conduit/guardrails/
- Plugin registry: src/conduit/plugins/registry.py
- Project templates: src/conduit/templates/

Tech stack: Python 3.11, Typer, Pydantic v2, Rich, MkDocs, copier
Hardware: RTX 5080 16GB, 32GB RAM, Ubuntu

The goal is a CLI that lets data scientists go from idea to production model
without understanding Kubernetes, GPU scheduling, or pipeline orchestration.
```

---

## Out of Scope

- Web UI/dashboard (future work — CLI/SDK first)
- Multi-tenant access control (single-team platform for now)
- Cloud deployment (local/on-prem only this week)
- IDE integrations (VS Code extension, Jupyter magic commands)
- Billing/chargeback system (cost tracking comes in Week 17)
- Model marketplace or sharing between teams
- Natural language interface ("train a model on this data")
