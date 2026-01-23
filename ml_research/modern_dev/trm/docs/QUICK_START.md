# TRM Code Repair - Quick Start Guide

## Minimum Requirements

### Hardware
- **CPU**: 8+ cores
- **RAM**: 32GB minimum, 64GB recommended
- **GPU**: RTX 3080+ (10GB VRAM) for training
- **Storage**: 500GB SSD

### Accounts Needed
1. **GitHub** - Personal Access Token (free)
2. **AWS or GCP** - Cloud compute & storage (pay-as-you-go)
3. **Weights & Biases** - Experiment tracking (free tier)

---

## Step 1: Get API Keys (30 minutes)

### GitHub Token
```
1. Go to: github.com → Settings → Developer Settings → Personal Access Tokens
2. Generate new token (classic)
3. Select scopes: repo, read:org, read:user
4. Copy token (starts with ghp_)
```

### AWS Account
```
1. Create account at: aws.amazon.com
2. Go to: IAM → Users → Create User
3. Attach policies: AmazonEC2FullAccess, AmazonS3FullAccess
4. Create access key, save both Key ID and Secret
```

### Weights & Biases
```
1. Sign up at: wandb.ai
2. Go to: Settings → API Keys
3. Copy your API key
```

---

## Step 2: Setup Environment (15 minutes)

```bash
# Clone and enter directory
cd /path/to/trm

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Create environment file
cp configs/environment-template.txt .env

# Edit .env with your credentials
nano .env  # or your preferred editor
```

**Required in .env:**
```
GITHUB_TOKEN=ghp_your_token_here
AWS_ACCESS_KEY_ID=your_key_id
AWS_SECRET_ACCESS_KEY=your_secret_key
WANDB_API_KEY=your_wandb_key
```

---

## Step 3: Verify Setup (5 minutes)

```bash
# Check environment
make check-env

# Run tests
make test-fast

# Test data collection (small sample)
python3 -c "
from data import SyntheticBugGenerator
gen = SyntheticBugGenerator(seed=42)
code = '''
def add(a, b):
    return a + b
'''
result = gen.generate(code.strip())
print('Generator works!' if result else 'Issue with generator')
"
```

---

## Step 4: Collect Initial Data (1-2 hours)

```bash
# Generate 10K synthetic samples (fast, no API needed)
python3 scripts/collect_synthetic.py --count 10000 --output data/raw/synthetic

# OR use the Makefile
make collect-synthetic
```

---

## Step 5: Process Data (30 minutes)

```bash
# Process raw data into training format
make process-data

# Validate quality
make validate-data

# Check statistics
make stats
```

---

## Step 6: Train Model (4-8 hours on GPU)

```bash
# Basic training
make train

# Monitor with TensorBoard
make watch-training  # Open localhost:6006

# Or with W&B (view at wandb.ai)
```

---

## Budget Quick Reference

| Scale | Cloud Cost | What You Get |
|-------|------------|--------------|
| Minimal | ~$200/month | 100K samples, basic model |
| Standard | ~$500/month | 500K samples, good model |
| Full | ~$1500/month | 1M+ samples, production model |

---

## Common Issues

### "Rate limit exceeded" (GitHub)
- Wait 1 hour, or use multiple tokens
- Consider GitHub App for higher limits

### "Out of memory" (Training)
- Reduce batch size in config
- Use gradient checkpointing
- Use smaller model config

### "No GPU found"
- Check CUDA installation: `nvidia-smi`
- Verify PyTorch GPU: `python -c "import torch; print(torch.cuda.is_available())"`

---

## Next Steps

1. **Scale data collection**: Add GitHub commit mining
2. **Expand bug types**: Implement more mutations
3. **Curriculum training**: Train in difficulty stages
4. **Evaluate**: Test on real-world benchmarks

See `docs/PROJECT_SPECIFICATION.md` for full details.
