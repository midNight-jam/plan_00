# Week 15: Internal Developer Platform

## Context

**Where it fits:** Phase C, Week 15 of the Anvil AI Infrastructure project. This is the first week of the platform maturity phase, transitioning from raw infrastructure to a polished developer experience.

**Prerequisites:**
- All Phase A infrastructure operational (K8s cluster, GPU operator, storage, networking)
- All Phase B systems running (scheduler, checkpoint manager, federation, chaos engineering)
- Working training job submission via raw Kubernetes manifests
- Monitoring stack (Prometheus, Grafana) collecting GPU and job metrics

**What it builds on:** Until now, submitting a training job required writing K8s YAML, understanding GPU topology, storage classes, and network policies. This week wraps all of that behind a developer-friendly CLI and API layer so AI engineers can focus on models, not infrastructure.

---

## Learning Goals

- [ ] Understand CLI framework design (cobra/viper patterns in Go, or Click/Typer in Python)
- [ ] Learn platform abstraction principles: hiding complexity without losing power
- [ ] Study internal developer platforms (Backstage, Humanitec) for UX patterns
- [ ] Understand API design for infrastructure tooling (imperative vs declarative)
- [ ] Learn documentation-as-code practices (MkDocs, Docusaurus)
- [ ] Study plugin architectures (hashicorp/go-plugin, Python entry_points)

---

## Implementation Goals

- [ ] Build `anvil` CLI binary with subcommands: `train`, `model`, `cluster`, `cost`
- [ ] Implement `anvil train submit` — accepts model path, requirements, produces running job
- [ ] Implement `anvil model deploy` — takes checkpoint, creates inference endpoint
- [ ] Implement `anvil cluster status` — shows GPU availability, queue depth, running jobs
- [ ] Implement `anvil cost report` — per-team GPU-hours, cost attribution
- [ ] Create platform config layer that maps simple specs to full K8s manifests
- [ ] Build plugin system allowing third-party extensions
- [ ] Write developer documentation: getting started guide (0 to serving in 15 min)
- [ ] Generate API reference from code annotations
- [ ] Add shell completions (bash, zsh, fish)

---

## Acceptance Criteria

1. `anvil train submit --model ./my_model --gpu 4 --gpu-type a100` creates a TrainingJob CR within 5 seconds without the user writing any YAML.
2. `anvil model deploy --checkpoint s3://bucket/ckpt-final --replicas 2` creates an InferenceService with health checks and autoscaling configured.
3. `anvil cluster status` displays a table showing node count, GPU availability per type, queue depth, and estimated wait time.
4. `anvil cost report --team ml-research --period 30d` outputs GPU-hours consumed, estimated cost, and top-5 expensive jobs.
5. The CLI validates inputs and provides actionable error messages (e.g., "GPU type 'v100' not available, did you mean 'a100'?").
6. A new user can go from `anvil init` to a running inference endpoint in under 15 minutes following the getting-started guide.
7. Plugin system allows adding a new subcommand via a separate binary in PATH named `anvil-<plugin>`.
8. Shell completions work for all subcommands and flags in bash and zsh.
9. `anvil --help` and all subcommand help text is clear, consistent, and includes examples.
10. CI pipeline builds the CLI for linux/amd64, runs integration tests against a kind cluster, and publishes artifacts.

---

## Validation Commands

```bash
# Build the CLI
cd ~/anvil/cmd/anvil && go build -o /usr/local/bin/anvil .

# Verify CLI is operational
anvil version
anvil --help

# Submit a training job (uses test model)
anvil train submit \
  --model ./examples/mnist \
  --gpu 1 \
  --gpu-type rtx5080 \
  --name smoke-test-$(date +%s)

# Check job status
anvil train status --name smoke-test-*

# Cluster overview
anvil cluster status
anvil cluster nodes --show-gpus

# Cost report
anvil cost report --team default --period 7d

# Deploy a model
anvil model deploy \
  --checkpoint /mnt/checkpoints/mnist-final \
  --name mnist-serve \
  --replicas 1

# Test the endpoint
anvil model test --name mnist-serve --input '{"image": "base64..."}'

# Plugin discovery
anvil plugin list

# Shell completions
anvil completion zsh > /tmp/anvil_completion.zsh && source /tmp/anvil_completion.zsh

# Run integration tests
cd ~/anvil && make test-integration
```

---

## Technical Implementation Details

### Project Structure

```
~/anvil/
├── cmd/
│   └── anvil/
│       ├── main.go
│       ├── root.go
│       ├── train.go
│       ├── model.go
│       ├── cluster.go
│       ├── cost.go
│       └── plugin.go
├── pkg/
│   ├── platform/
│   │   ├── abstractions.go    # Maps simple specs → K8s resources
│   │   ├── defaults.go        # Smart defaults for GPU, memory, storage
│   │   └── validation.go      # Input validation with suggestions
│   ├── client/
│   │   ├── k8s.go             # Kubernetes API interactions
│   │   ├── metrics.go         # Prometheus query client
│   │   └── storage.go         # Checkpoint storage operations
│   ├── output/
│   │   ├── table.go           # Table formatting
│   │   ├── json.go            # JSON output mode
│   │   └── progress.go        # Progress bars and spinners
│   └── plugin/
│       ├── discovery.go       # Find plugins in PATH
│       ├── runner.go          # Execute plugin subcommands
│       └── registry.go        # Plugin metadata registry
├── docs/
│   ├── getting-started.md
│   ├── architecture.md
│   ├── troubleshooting.md
│   └── api-reference/
├── examples/
│   ├── mnist/
│   ├── llm-finetune/
│   └── multi-gpu/
└── Makefile
```

### Core Abstraction Layer

```go
// pkg/platform/abstractions.go
package platform

type TrainingSpec struct {
    ModelPath       string            `yaml:"model_path"`
    GPUCount        int               `yaml:"gpu_count"`
    GPUType         string            `yaml:"gpu_type"`        // "any", "a100", "rtx5080"
    MaxDuration     time.Duration     `yaml:"max_duration"`
    CheckpointEvery time.Duration     `yaml:"checkpoint_every"`
    Environment     map[string]string `yaml:"environment"`
    Requirements    string            `yaml:"requirements"`    // pip requirements file
    ScalingPolicy   ScalingPolicy     `yaml:"scaling_policy"`
}

type ScalingPolicy struct {
    MinWorkers int `yaml:"min_workers"`
    MaxWorkers int `yaml:"max_workers"`
    ScaleOnMetric string `yaml:"scale_on_metric"` // "gpu_util", "throughput"
}

func (s *TrainingSpec) ToKubernetesResources() ([]client.Object, error) {
    // Maps simplified spec to:
    // - TrainingJob CR with node affinity for GPU type
    // - PVC for checkpoint storage
    // - ConfigMap for environment
    // - ServiceAccount with minimal RBAC
    // - NetworkPolicy for worker communication
}
```

### CLI Command Example

```go
// cmd/anvil/train.go
package main

import (
    "github.com/spf13/cobra"
    "github.com/anvil-platform/anvil/pkg/platform"
)

var trainSubmitCmd = &cobra.Command{
    Use:   "submit",
    Short: "Submit a training job",
    Example: `  anvil train submit --model ./my_model --gpu 4
  anvil train submit --config training.yaml`,
    RunE: func(cmd *cobra.Command, args []string) error {
        spec := platform.TrainingSpec{
            ModelPath: modelPath,
            GPUCount:  gpuCount,
            GPUType:   gpuType,
        }
        if err := spec.Validate(); err != nil {
            return fmt.Errorf("invalid spec: %w\n\nDid you mean: %s", err, spec.Suggest())
        }
        resources, err := spec.ToKubernetesResources()
        if err != nil {
            return err
        }
        return client.Apply(cmd.Context(), resources)
    },
}
```

### Plugin System

```go
// pkg/plugin/discovery.go
package plugin

import (
    "os/exec"
    "path/filepath"
    "strings"
)

func Discover() []Plugin {
    var plugins []Plugin
    paths := filepath.SplitList(os.Getenv("PATH"))
    for _, dir := range paths {
        matches, _ := filepath.Glob(filepath.Join(dir, "anvil-*"))
        for _, m := range matches {
            name := strings.TrimPrefix(filepath.Base(m), "anvil-")
            plugins = append(plugins, Plugin{Name: name, Path: m})
        }
    }
    return plugins
}
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| Cobra command not registering | Ensure `init()` calls `rootCmd.AddCommand()` and the file is in `cmd/anvil/` |
| K8s client auth failing | Check `~/.kube/config` context; use `anvil cluster context` to switch |
| Plugin not discovered | Verify binary is executable and named `anvil-<name>` in PATH |
| Shell completions not working | Run `source <(anvil completion zsh)` or add to `.zshrc` |
| Integration tests timing out | Increase timeout; ensure kind cluster has GPU mock configured |
| YAML generation wrong | Add test cases in `pkg/platform/abstractions_test.go` comparing output |

---

## Agent Handoff Template

```
Resume Anvil Phase C, Week 15: Internal Developer Platform.

Hardware: ASUS ROG Strix SCAR 16, RTX 5080 16GB, 32GB RAM, Ubuntu.
State: Phases A+B complete. K8s cluster with GPU operator, scheduler, checkpoint manager, federation all running.

Current goal: Build the `anvil` CLI that abstracts Kubernetes complexity for AI engineers.
Key files: ~/anvil/cmd/anvil/, ~/anvil/pkg/platform/
Test with: `make test-integration` against local kind cluster.

Specific task: [DESCRIBE WHAT TO DO NEXT]
Constraints: CLI must work without K8s knowledge. Smart defaults. Actionable errors.
```

---

## Out of Scope

- Web UI / dashboard (separate project, may come later)
- Multi-cloud provider abstraction (this is single-cluster focused)
- User authentication / RBAC within the CLI (defer to K8s RBAC)
- Billing integration with cloud providers
- IDE plugins (VS Code extension, etc.)
- Windows support for the CLI binary
