# CLAUDE.md

Global Rules & Conventions

## Overview

mlProfiler is a machine learning system profiling and prediction framework that models GPU resource usage, performance metrics, and workload scheduling for multi-instance ML workloads. The system uses a two-stage prediction approach combining kernel-level profiling with shared throughput analysis. In mps_thread branch, we focus on prediction MPS partitions with nvidia dcgm and nvidia-smi metrics to predict shared throughput and power. 



## Purpose
These rules apply to every conversation and every agent in this repository. They define coding, testing, documentation, and safety norms.

## Operating Principles
1. **Small, safe loops.** Prefer short iterations with explicit checkpoints and diffs.
2. **Single intent per agent.** Each agent stays within its charter. Escalate or hand off when out of scope.
3. **Context first.** Read available planning docs (`/prompts`, `/specs`) before proposing changes.
4. **Ask before you build.** If the task is ambiguous, ask for focus and constraints.
5. **Determinism.** Propose step plans and obtain confirmation for irreversible actions (deleting/migrating code, schema changes).

## Directory Structure

- `tests/mps/` - MPS (Multi-Process Service) testing and analysis. Most logics we need locates in `tests/mps/freq-scaling` and `tests/mps/multiinstance`
- `output/` - Generated predictions and analysis results
- `configs/` - Configuration files
- `setup/` - Environment setup scripts
- `k8sJobs/` - Kubernetes job definitions (legacy)
- `llm-queue/` - LLM workload queue management system (legacy)

## Code & Structure Conventions
- Languages: Python (primary), Bash for glue.
- Project layout:
  - `tests/mps` main logic located. Only main.py 
  - `tests/mps/tests` mirrors `tests/mps` structure (one test file per module).
- Function boundaries:
  - Pure core logic in small functions; side effects isolated.
  - Dependency injection for IO/services.
- Documentation:
  - Docstrings (Google style) on public functions.
  - Each feature has an associated spec in `/specs`.

## Testing Requirements
- Unit tests for **critical functions** (algorithmic cores, validation, parsing).
- Use `pytest`. Require fast, hermetic tests (<1s each when possible).
- Coverage: Focus on decision branches and error handling.
- When refactoring incorrect logic: **no backward compatibility** guarantees—prefer correctness and clarity. Update or remove tests accordingly.

## Implementation Specs
- `specs/implementation-specs/*.md` are the single source of truth for features reproduced from papers.
- A spec must include: Problem statement, scope, datasets/fixtures, algorithms, critical functions, acceptance tests, and footnotes to source sections.

## Safety & Policy
- Follow Anthropic safety guidance: do not build or assist misuse (security bypasses, harmful code). If a request is risky, refuse and propose safe alternatives. 

## Handoffs Between Agents
- Reader → Reproducer handoff is always via a spec Markdown saved under `specs/implementation-specs/`.
- Reproducer must respect the spec; if a design is incorrect, propose a redesign and update the spec before coding.

## Communication Norms
- Use checklists and numbered plans.
- Cite file paths explicitly (no "some file").
- Prefer diffs/patches over prose for code changes.

## Tooling & Prompts
- Agents may load additional prompts from `/prompts/agents/*.md`.
- When running under Claude Code or Messages API, this file is part of the **system** prompt (highest priority). 



## Core Architecture

### Main Entry Point
- `main.py` - Central orchestrator handling training, prediction, and data processing workflows
- Supports multiple operational modes: training, prediction, data preprocessing, and end-to-end evaluation

### Key Components

#### Data Processing Pipeline
- `utils/loaddata.py` - Core data loading and preprocessing utilities
- `utils/trainer.py` - Model training logic with support for multiple ML algorithms (Linear Regression, Random Forest, H2O AutoML)
- `utils/predictor.py` - Prediction engine for both single-stage and multi-stage inference

#### Prediction Framework
The system implements a two-stage prediction architecture:
1. **Stage 1**: Kernel-level profiling and feature extraction
2. **Stage 2**: Shared throughput analysis and resource optimization

#### Workload Management
- `workloads/train/` - Training workloads for various ML tasks (sentiment analysis, speech recognition, image classification, etc.)
- `workloads/inference/` - Inference workloads with real-time queue management
- `workloads/cuda_samples/` - CUDA sample applications for low-level profiling

#### Model Support
- `models/` - Pre-trained models and model definitions
- Support for multiple model types: linear regression, random forest, H2O AutoML

## Development Setup

### Environment Setup
```bash
# Create conda environment
conda create -n ml python=3.10
pip install -r requirements.txt

# Configure Weights & Biases (required for experiment tracking)
export WANDB_API_KEY=[YOUR_WANDB_API_KEY]
export WANDB_SILENT=true
```

### Docker Environment
```bash
# Build container
docker build --build-arg API_KEY=[YOUR_WANDB_API_KEY] -t [dockerhub_Username/image_name] .

# Push to registry
docker push dockerhub_Username/image_name
```

## Common Development Commands

### Data Processing
```bash
# Parse baseline metrics
cd tests/mps/freq_scaling/baseline_metrics
python parse_baseline_sysmetrics_freqscale.py ../../ccv100_logs/baseline_nonDL/ 0505_cudasample 2

# Generate stage 2 shared throughput data
python stage2.py [output-prefix]

# Merge stage 2 label and feature files
python main.py -pd2 -power [power_file] -sd [throughput_file] -duration [duration_file] -energy [energy_file] -comb 2 -t 100 -mt threadclass --output_prefix [prefix] --output_dir [output_dir] --label_policy separate_throughputpower_regression --baseline_file [baseline_file]
```

### Training
```bash
# Train models
python main.py --output_prefix [prefix] --train_file [train_file] --test_file [test_file] --customsplit --nonSplitData [data_file] -comb 2 -t 100 -mt threadclass

# Stage 2 training
python main.py --output_prefix [prefix] --train_file [train_file] --test_file [test_file] -comb 2 -t 100 -mt linear --output_dir [output_dir] --label_policy power_regression --train
```

### Prediction
```bash
# Stage 1 prediction
python main.py --test_file [test_file] --train_file [train_file] -t 100 -m [model_path] -w [HPworkload]

# Stage 2 prediction
python main.py --test_file [test_file] --train_file [train_file] -t 100 -m [model_path] -w [HPworkload] --sharedThroughputData [filepath]

# Predict all test points
python main.py --train_file [train_file] --test_file [test_file] -comb 2 -t 100 -mt threadclass --output_dir [output_dir] --model [model_path] --label_policy power_regression --predAllacc

# End-to-end prediction
python main.py --train_file [train_file] --test_file [test_file] -comb 2 -t 100 --output_dir [output_dir] --label_policy power_regression -mt RF --throughput_model [throughput_model] --power_model [power_model] --end_to_end --debug
```

### Batch Processing
```bash
# Run all workloads
bash main.sh

# Count average throughput
python output/count_average.py [output_directory]

# Run partitioned workloads
bash run_train_splits.sh

# Run unseen partitions
bash run_unseen_partitions.sh
```

## Key Configuration Parameters

- `-comb` / `--n_combination`: Number of workload combinations (typically 2 or 3)
- `-t` / `--targetMPS`: Target MPS (Multi-Process Service) percentage (typically 100)
- `-mt` / `--modeltype`: Model type (linear, RF, threadclass, AutoML, KACE)
- `--label_policy`: Prediction target (power_regression, separate_throughputpower_regression, throughput)
- `--correlation`: Feature correlation threshold for exclusion
- `-rd` / `--randomseed`: Random seed for reproducibility



## Testing

The repository includes comprehensive test suites in `tests/` covering multi-instance scenarios, frequency scaling, and baseline metrics analysis. Test execution depends on the specific experimental setup and available GPU resources.