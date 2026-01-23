# TRM Code Assistant - R&D Project Specification

## Executive Summary

This document specifies the complete infrastructure, accounts, and resources required to build a production-quality TRM-based Python code repair assistant.

**Project Goal**: Train a 7M-parameter TRM model capable of detecting and fixing Python bugs with >80% accuracy on real-world code.

**Timeline**: 16 weeks (4 months)
**Estimated Budget**: $15,000 - $50,000 (depending on scale)

---

## 1. Hardware Resources

### 1.1 Development Machine

**Purpose**: Daily development, testing, small-scale data processing

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 8 cores | 16+ cores (AMD Ryzen 9 / Intel i9) |
| RAM | 32 GB | 64-128 GB |
| Storage | 1 TB NVMe | 2 TB NVMe |
| GPU | RTX 3080 (10GB) | RTX 4090 (24GB) |

**Estimated Cost**: Already owned or $3,000-$5,000

### 1.2 Data Processing Cluster

**Purpose**: Large-scale data collection, preprocessing, validation

**Option A: Cloud (Recommended for flexibility)**

```yaml
# AWS/GCP Configuration
Provider: AWS (or GCP/Azure)

Processing Nodes:
  Instance Type: c6i.8xlarge (32 vCPU, 64GB RAM)
  Count: 4-8 nodes
  Storage: 500GB EBS each
  Spot Pricing: ~$0.50/hr per node

Coordinator Node:
  Instance Type: c6i.4xlarge (16 vCPU, 32GB RAM)
  Count: 1
  Storage: 1TB EBS
  On-Demand: ~$0.68/hr

Storage:
  S3 Bucket: 2TB
  Cost: ~$46/month

Estimated Monthly Cost: $500-1,500 (depending on usage)
```

**Option B: On-Premise**

```yaml
# Self-hosted cluster
Nodes: 4x workstation
  CPU: AMD Ryzen 9 5950X (16 cores)
  RAM: 128GB DDR4
  Storage: 2TB NVMe + 8TB HDD
  Network: 10GbE

Cost: ~$12,000-16,000 one-time
```

### 1.3 Training Infrastructure

**Purpose**: Model training, hyperparameter search, evaluation

**Option A: Cloud GPU (Recommended)**

```yaml
# Cloud GPU Options

AWS:
  Instance: p4d.24xlarge (8x A100 40GB)
  Cost: ~$32/hr on-demand, ~$12/hr spot

  Alternative: g5.12xlarge (4x A10G 24GB)
  Cost: ~$5.67/hr on-demand

GCP:
  Instance: a2-highgpu-4g (4x A100 40GB)
  Cost: ~$14/hr on-demand

Lambda Labs:
  Instance: gpu_8x_a100_80gb
  Cost: ~$14.32/hr

RunPod:
  Instance: A100 80GB
  Cost: ~$1.99/hr (community cloud)

Vast.ai:
  Instance: A100 40GB
  Cost: ~$1.00-2.00/hr

Estimated Training Cost:
  - Initial training: 100 GPU-hours = $200-500
  - Hyperparameter search: 500 GPU-hours = $1,000-2,500
  - Final training runs: 200 GPU-hours = $400-1,000
  Total: $1,600-4,000
```

**Option B: Own Hardware**

```yaml
# Training Workstation
GPU: 2x RTX 4090 (24GB each) or 1x RTX A6000 (48GB)
CPU: AMD Threadripper 3960X
RAM: 256GB DDR4
Storage: 4TB NVMe
PSU: 1600W

Cost: $8,000-15,000
```

### 1.4 Storage Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Storage Layout                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Local Development (2TB NVMe)                               │
│  ├── /code           - Source code, configs                 │
│  ├── /data/raw       - Raw collected data (temporary)       │
│  ├── /data/processed - Processed datasets                   │
│  └── /models         - Model checkpoints                    │
│                                                             │
│  Cloud Storage (S3/GCS - 2TB)                               │
│  ├── s3://trm-data/raw/          - Raw data archive         │
│  ├── s3://trm-data/processed/    - Processed datasets       │
│  ├── s3://trm-data/checkpoints/  - Training checkpoints     │
│  └── s3://trm-data/releases/     - Released models          │
│                                                             │
│  Database (PostgreSQL)                                      │
│  ├── metadata        - Sample metadata, statistics          │
│  ├── quality_scores  - Validation results                   │
│  └── experiments     - Training experiment tracking         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. API Keys & Accounts

### 2.1 Required Accounts

| Service | Purpose | Cost | Priority |
|---------|---------|------|----------|
| GitHub | API access, commit mining | Free (with limits) | Critical |
| Stack Exchange | Q&A data | Free | High |
| AWS/GCP/Azure | Cloud compute & storage | Pay-as-you-go | Critical |
| Weights & Biases | Experiment tracking | Free tier | High |
| HuggingFace | Model hosting | Free tier | Medium |

### 2.2 GitHub Setup

```yaml
# GitHub Configuration

Personal Access Token (Classic):
  Required Scopes:
    - repo (full access for private repos, if needed)
    - read:org (for org repos)
    - read:user (for user info)

  Rate Limits:
    - Authenticated: 5,000 requests/hour
    - Search API: 30 requests/minute

  Note: For large-scale mining, consider multiple tokens
        or GitHub App installation

GitHub App (Recommended for scale):
  Benefits:
    - Higher rate limits (15,000 requests/hour per installation)
    - Can be installed on target organizations
    - Better for production use

  Setup:
    1. Create GitHub App in developer settings
    2. Generate private key
    3. Install on target organizations/repos

Enterprise Consideration:
  - GitHub Enterprise Server access if mining internal repos
  - Contact GitHub for research API access (academic discount)
```

**Token Generation Steps**:
1. Go to GitHub → Settings → Developer Settings → Personal Access Tokens
2. Generate new token (classic)
3. Select scopes: `repo`, `read:org`, `read:user`
4. Store securely in environment variable or secrets manager

### 2.3 Stack Exchange API

```yaml
# Stack Exchange API Setup

Registration:
  URL: https://stackapps.com/apps/oauth/register

API Key Benefits:
  - Without key: 300 requests/day
  - With key: 10,000 requests/day

Quota:
  - 10,000 requests per day per key
  - Backoff headers must be respected

Data Dump Alternative:
  URL: https://archive.org/details/stackexchange
  Size: ~18GB compressed for Stack Overflow
  Benefits: No rate limits, complete history
  Recommended: Download dump for bulk processing
```

### 2.4 Cloud Provider Setup

```yaml
# AWS Setup (Recommended)

Account Setup:
  1. Create AWS account
  2. Enable MFA on root account
  3. Create IAM user for programmatic access
  4. Generate access keys

Required Services:
  - EC2 (compute instances)
  - S3 (object storage)
  - ECR (container registry for Docker images)
  - IAM (access management)

IAM Policy (minimum required):
```

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ec2:*",
        "s3:*",
        "ecr:*"
      ],
      "Resource": "*",
      "Condition": {
        "StringEquals": {
          "aws:RequestedRegion": "us-west-2"
        }
      }
    }
  ]
}
```

```yaml
# GCP Alternative

Setup:
  1. Create GCP project
  2. Enable Compute Engine, Cloud Storage APIs
  3. Create service account
  4. Download JSON key file

Benefits:
  - BigQuery access to GitHub Archive (free 1TB/month queries)
  - Preemptible VMs (like AWS Spot, cheaper)
```

### 2.5 Experiment Tracking

```yaml
# Weights & Biases Setup

Account:
  URL: https://wandb.ai
  Tier: Free (100GB storage, unlimited experiments)

API Key:
  Location: Settings → API Keys
  Environment: export WANDB_API_KEY=xxx

Project Structure:
  Entity: your-username-or-team
  Project: trm-code-repair

Integration:
  pip install wandb
  wandb login
```

### 2.6 Model Hosting

```yaml
# HuggingFace Hub Setup

Account:
  URL: https://huggingface.co
  Tier: Free (unlimited public models)

Token:
  Location: Settings → Access Tokens
  Type: Write (for uploading models)

Repository:
  Name: your-username/trm-code-repair
  Type: Model
  License: Apache 2.0 (recommended)
```

---

## 3. Repository Structure

### 3.1 Main Repository

```
trm-code-repair/
├── .github/
│   ├── workflows/
│   │   ├── ci.yml              # Continuous integration
│   │   ├── data-pipeline.yml   # Scheduled data collection
│   │   └── training.yml        # Training workflow
│   └── CODEOWNERS
│
├── configs/
│   ├── data/
│   │   ├── collection.yaml     # Data collection config
│   │   ├── processing.yaml     # Processing pipeline config
│   │   └── validation.yaml     # Validation thresholds
│   ├── training/
│   │   ├── base.yaml           # Base training config
│   │   ├── curriculum/         # Curriculum stage configs
│   │   └── experiments/        # Experiment-specific configs
│   └── infrastructure/
│       ├── aws.yaml            # AWS resource definitions
│       └── docker-compose.yml  # Local development
│
├── src/
│   ├── trm/                    # Core TRM model
│   │   ├── model.py
│   │   ├── layers.py
│   │   └── config.py
│   │
│   ├── data/                   # Data pipeline
│   │   ├── collectors/
│   │   │   ├── github/
│   │   │   ├── stackoverflow/
│   │   │   ├── linters/
│   │   │   └── synthetic/
│   │   ├── processors/
│   │   │   ├── tokenizer.py
│   │   │   ├── encoder.py
│   │   │   └── validator.py
│   │   └── taxonomy/
│   │
│   ├── training/               # Training infrastructure
│   │   ├── trainer.py
│   │   ├── curriculum.py
│   │   ├── evaluation.py
│   │   └── distributed.py
│   │
│   └── inference/              # Inference & deployment
│       ├── predictor.py
│       ├── server.py
│       └── cli.py
│
├── scripts/
│   ├── collect_data.py         # Data collection entry point
│   ├── process_data.py         # Processing entry point
│   ├── train.py                # Training entry point
│   ├── evaluate.py             # Evaluation entry point
│   └── deploy.py               # Deployment script
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── benchmarks/
│
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── model_analysis.ipynb
│   └── error_analysis.ipynb
│
├── docker/
│   ├── Dockerfile.data         # Data processing container
│   ├── Dockerfile.train        # Training container
│   └── Dockerfile.inference    # Inference container
│
├── docs/
│   ├── EXPANSION_PLAN.md
│   ├── PROJECT_SPECIFICATION.md
│   ├── DATA_COLLECTION.md
│   └── TRAINING_GUIDE.md
│
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── Makefile
└── README.md
```

### 3.2 Data Repository (Separate)

```
trm-data/                       # Large data, separate from code
├── raw/
│   ├── github/
│   │   ├── commits/            # Raw commit data
│   │   └── repos/              # Cloned repositories
│   ├── stackoverflow/
│   │   └── dump/               # SO data dump
│   └── linters/
│       └── outputs/            # Linter scan results
│
├── processed/
│   ├── v1/                     # Dataset version 1
│   │   ├── train/
│   │   │   ├── shard_00000.parquet
│   │   │   └── ...
│   │   ├── val/
│   │   └── test/
│   └── metadata.json
│
├── vocabulary/
│   ├── bpe_32k.json
│   └── token_frequencies.json
│
└── benchmarks/
    ├── syntax_errors/
    ├── logic_errors/
    └── security_vulns/
```

### 3.3 Model Repository (HuggingFace)

```
huggingface.co/your-username/trm-code-repair/
├── config.json
├── model.safetensors
├── tokenizer.json
├── vocab.json
├── README.md                   # Model card
└── examples/
```

---

## 4. Environment & Secrets Management

### 4.1 Environment Variables

```bash
# .env.template (DO NOT COMMIT ACTUAL VALUES)

# GitHub
GITHUB_TOKEN=ghp_YOUR_TOKEN_HERE
GITHUB_APP_ID=123456
GITHUB_APP_PRIVATE_KEY_PATH=/path/to/private-key.pem

# Stack Exchange
STACKEXCHANGE_API_KEY=xxxxxxxxxxxxxxxx

# AWS
AWS_ACCESS_KEY_ID=AKIA_YOUR_KEY_HERE
AWS_SECRET_ACCESS_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
AWS_DEFAULT_REGION=us-west-2
S3_BUCKET=trm-code-repair-data

# GCP (alternative)
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
GCP_PROJECT_ID=trm-code-repair

# Experiment Tracking
WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
WANDB_PROJECT=trm-code-repair
WANDB_ENTITY=your-username

# HuggingFace
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/trm

# Docker Registry
DOCKER_REGISTRY=your-registry.com
```

### 4.2 Secrets Management

```yaml
# Option 1: AWS Secrets Manager (Production)
Secret Name: trm-code-repair/api-keys
Secret Value:
  github_token: ghp_xxx
  stackexchange_key: xxx
  wandb_key: xxx
  hf_token: hf_xxx

# Option 2: HashiCorp Vault (Enterprise)
Path: secret/trm-code-repair
Keys: github_token, aws_credentials, etc.

# Option 3: Local .env with git-crypt (Small team)
.env encrypted with git-crypt
Team members have GPG keys for decryption

# Option 4: 1Password/Bitwarden (Individual)
Store credentials in password manager
Use CLI to inject into environment
```

### 4.3 Configuration Files

```yaml
# configs/data/collection.yaml

github:
  token_env: GITHUB_TOKEN
  rate_limit_buffer: 100  # Stop 100 requests before limit

  repositories:
    priority_repos:
      - django/django
      - pallets/flask
      - tiangolo/fastapi
      - pandas-dev/pandas
      - numpy/numpy
      # ... more

    discovery:
      min_stars: 100
      language: Python
      pushed_after: "2020-01-01"
      max_repos: 10000

  commit_filters:
    message_patterns:
      - "fix"
      - "bug"
      - "issue"
      - "closes #"
    max_files_changed: 5
    max_lines_changed: 100
    exclude_paths:
      - "test_*"
      - "*_test.py"
      - "docs/*"

stackoverflow:
  api_key_env: STACKEXCHANGE_API_KEY
  use_dump: true
  dump_path: /data/raw/stackoverflow/dump

  filters:
    tags:
      - python
      - python-3.x
    min_score: 5
    has_accepted_answer: true
```

---

## 5. Development Environment Setup

### 5.1 Local Development

```bash
# 1. Clone repository
git clone https://github.com/your-org/trm-code-repair.git
cd trm-code-repair

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -e ".[dev]"

# 4. Setup pre-commit hooks
pre-commit install

# 5. Copy environment template
cp .env.template .env
# Edit .env with your credentials

# 6. Verify setup
make test
make lint
```

### 5.2 Docker Development

```yaml
# docker-compose.yml

version: '3.8'

services:
  dev:
    build:
      context: .
      dockerfile: docker/Dockerfile.dev
    volumes:
      - .:/workspace
      - ~/.aws:/root/.aws:ro
      - data:/data
    environment:
      - GITHUB_TOKEN
      - WANDB_API_KEY
    ports:
      - "8888:8888"  # Jupyter
      - "6006:6006"  # TensorBoard
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: trm
      POSTGRES_USER: trm
      POSTGRES_PASSWORD: devpassword
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7
    ports:
      - "6379:6379"

volumes:
  data:
  postgres_data:
```

### 5.3 Required Python Packages

```toml
# pyproject.toml

[project]
name = "trm-code-repair"
version = "0.1.0"
requires-python = ">=3.10"

dependencies = [
    # Core ML
    "torch>=2.0",
    "transformers>=4.30",
    "datasets>=2.14",
    "safetensors>=0.3",

    # Data Processing
    "pandas>=2.0",
    "pyarrow>=12.0",
    "numpy>=1.24",

    # Code Analysis
    "tree-sitter>=0.20",
    "tree-sitter-python>=0.20",
    "libcst>=1.0",
    "rope>=1.9",

    # API Clients
    "PyGithub>=1.59",
    "aiohttp>=3.8",
    "httpx>=0.24",

    # Infrastructure
    "ray[default]>=2.5",
    "docker>=6.1",
    "boto3>=1.28",

    # Experiment Tracking
    "wandb>=0.15",
    "tensorboard>=2.13",

    # Linting/Analysis
    "pylint>=2.17",
    "mypy>=1.4",
    "ruff>=0.0.280",
    "bandit>=1.7",

    # Utilities
    "pyyaml>=6.0",
    "python-dotenv>=1.0",
    "rich>=13.4",
    "typer>=0.9",
    "pydantic>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "pytest-asyncio>=0.21",
    "pytest-cov>=4.1",
    "pre-commit>=3.3",
    "black>=23.7",
    "isort>=5.12",
    "jupyter>=1.0",
    "ipywidgets>=8.0",
]

training = [
    "deepspeed>=0.10",
    "flash-attn>=2.0",
    "bitsandbytes>=0.41",
]
```

---

## 6. Data Source Details

### 6.1 GitHub Repositories to Mine

```yaml
# Priority 1: High-quality, well-maintained projects
tier_1:
  web_frameworks:
    - django/django           # 70k+ stars, extensive history
    - pallets/flask           # 63k+ stars
    - tiangolo/fastapi        # 60k+ stars
    - encode/starlette        # 8k+ stars
    - encode/httpx            # 11k+ stars

  data_science:
    - pandas-dev/pandas       # 40k+ stars
    - numpy/numpy             # 24k+ stars
    - scikit-learn/scikit-learn # 56k+ stars
    - matplotlib/matplotlib   # 18k+ stars

  utilities:
    - psf/requests            # 50k+ stars
    - python-attrs/attrs      # 5k+ stars
    - pydantic/pydantic       # 15k+ stars

# Priority 2: Good projects with clear bug-fix patterns
tier_2:
  testing:
    - pytest-dev/pytest
    - robotframework/robotframework

  async:
    - aio-libs/aiohttp
    - MagicStack/uvloop

  cli:
    - pallets/click
    - tiangolo/typer
    - willmcgugan/rich

  devtools:
    - python/mypy
    - PyCQA/pylint
    - astral-sh/ruff

# Priority 3: Diverse real-world applications
tier_3:
  - home-assistant/core
  - ansible/ansible
  - celery/celery
  - apache/airflow
  - spotify/luigi
  - great-expectations/great_expectations
```

### 6.2 Stack Overflow Data

```yaml
# Data Dump Download
url: https://archive.org/download/stackexchange/stackoverflow.com-Posts.7z
size: ~18GB compressed, ~90GB uncompressed

# Relevant Tables
Posts:
  - Filter: PostTypeId=1 (questions) with AcceptedAnswerId
  - Filter: Tags contains 'python'

# Processing Pipeline
1. Download and extract dump
2. Filter Python questions with accepted answers
3. Extract code blocks from question and answer
4. Match question code (buggy) with answer code (fixed)
5. Validate pairs have meaningful differences
```

### 6.3 BigQuery GitHub Archive (Alternative)

```sql
-- Query for Python bug-fix commits
SELECT
  repo_name,
  commit,
  message,
  author.date as commit_date
FROM
  `bigquery-public-data.github_repos.commits`
WHERE
  REGEXP_CONTAINS(message, r'(?i)(fix|bug|issue|closes?\s*#)')
  AND repo_name IN (
    SELECT repo_name
    FROM `bigquery-public-data.github_repos.languages`
    WHERE language.name = 'Python'
    AND language.bytes > 100000
  )
LIMIT 1000000;

-- Cost: ~$5 per TB scanned (first 1TB/month free)
```

---

## 7. Budget Breakdown

### 7.1 Minimal Viable Budget (~$15,000)

```
Hardware (assuming existing dev machine):
  - Cloud GPU training (200 hrs @ $5/hr)      $1,000
  - Cloud CPU processing (500 hrs @ $0.50/hr)   $250
  - Cloud storage (500GB for 6 months)          $150

APIs & Services:
  - GitHub API: Free
  - Stack Exchange API: Free
  - Weights & Biases: Free tier
  - HuggingFace: Free tier

Infrastructure:
  - Domain & SSL: $50
  - Miscellaneous cloud services: $200

Contingency (20%): $330

Total: ~$2,000 cloud costs
```

### 7.2 Recommended Budget (~$30,000)

```
Compute:
  - Dedicated GPU workstation               $8,000
    (RTX 4090 + Ryzen 9 + 128GB RAM)
  - Cloud GPU overflow (500 hrs)            $2,500
  - Cloud CPU cluster (1000 hrs)              $500

Storage:
  - 4TB NVMe (local)                          $400
  - Cloud storage 2TB/year                    $500

Services:
  - Weights & Biases Teams                    $600/yr
  - GitHub Enterprise (optional)            $2,000/yr

Human Resources:
  - Data annotation (contractor, 100 hrs)   $5,000
  - Code review/QA (contractor, 50 hrs)     $2,500

Contingency (20%):                          $4,400

Total: ~$26,000
```

### 7.3 Full-Scale Budget (~$50,000+)

```
Compute:
  - Multi-GPU workstation (2x A6000)       $15,000
  - Cloud GPU cluster (1000 hrs A100)      $10,000
  - Distributed processing cluster          $3,000

Storage:
  - High-speed NAS (20TB)                   $2,000
  - Cloud storage & CDN                     $1,500

Services:
  - All premium tiers                       $3,000

Human Resources:
  - ML Engineer (part-time, 3 months)      $15,000
  - Data annotation team                    $8,000

Contingency:                                $7,500

Total: ~$65,000
```

---

## 8. Timeline & Milestones

```
Week 1-2: Infrastructure Setup
├── Set up cloud accounts and credentials
├── Configure development environment
├── Set up CI/CD pipelines
├── Initialize data storage
└── Deliverable: Working dev environment

Week 3-4: Data Collection Infrastructure
├── Implement GitHub commit miner
├── Implement Stack Overflow extractor
├── Set up linter integration
├── Begin initial data collection
└── Deliverable: 50K raw samples

Week 5-6: Data Processing Pipeline
├── Implement BPE tokenizer training
├── Build validation pipeline
├── Create deduplication system
├── Process collected data
└── Deliverable: 100K processed samples

Week 7-8: Bug Generation Expansion
├── Implement 50 new bug mutations
├── Add framework-specific bugs
├── Add security vulnerability patterns
├── Generate synthetic samples
└── Deliverable: 500K total samples

Week 9-10: Training Infrastructure
├── Set up distributed training
├── Implement curriculum learning
├── Create evaluation benchmarks
├── Initial training runs
└── Deliverable: Baseline model

Week 11-12: Training & Iteration
├── Hyperparameter optimization
├── Curriculum stage training
├── Error analysis
├── Model improvements
└── Deliverable: Improved model

Week 13-14: Evaluation & Refinement
├── Comprehensive evaluation
├── Real-world testing
├── Edge case handling
├── Performance optimization
└── Deliverable: Production-ready model

Week 15-16: Deployment & Documentation
├── Model packaging
├── API server implementation
├── Documentation
├── Release preparation
└── Deliverable: Released model & tools
```

---

## 9. Success Metrics

### 9.1 Data Quality Metrics

| Metric | Target |
|--------|--------|
| Total training samples | 1M+ |
| Unique bug types covered | 100+ |
| Validation pass rate | >90% |
| Deduplication rate | <5% duplicates |
| Human-verified subset | 10K samples |

### 9.2 Model Performance Metrics

| Metric | Target |
|--------|--------|
| Syntax error fix accuracy | >95% |
| Logic error fix accuracy | >75% |
| Security vulnerability detection | >80% |
| End-to-end fix rate | >60% |
| False positive rate | <10% |
| Inference latency (p99) | <500ms |

### 9.3 Operational Metrics

| Metric | Target |
|--------|--------|
| Training time (full) | <48 hours |
| Data pipeline throughput | 10K samples/hour |
| API availability | 99.9% |
| Model size | <50MB |

---

## 10. Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| GitHub rate limiting | High | Multiple tokens, caching, GitHub App |
| Data quality issues | High | Multi-stage validation, human review |
| Training instability | Medium | Checkpointing, curriculum learning |
| Budget overrun | Medium | Spot instances, incremental scaling |
| Model underperformance | High | Extensive evaluation, iteration budget |
| Legal/licensing issues | Medium | License filtering, legal review |

---

## 11. Quick Start Checklist

```markdown
## Before Starting

- [ ] GitHub account with personal access token
- [ ] AWS/GCP account with billing enabled
- [ ] Weights & Biases account
- [ ] HuggingFace account
- [ ] Development machine with GPU (or cloud instance)
- [ ] 500GB+ available storage

## Day 1 Setup

- [ ] Clone repository
- [ ] Install dependencies
- [ ] Configure environment variables
- [ ] Run test suite
- [ ] Collect first 1K samples (smoke test)

## Week 1 Goals

- [ ] Full infrastructure operational
- [ ] 10K samples collected and processed
- [ ] Training pipeline verified with small run
- [ ] Monitoring and logging configured
```
