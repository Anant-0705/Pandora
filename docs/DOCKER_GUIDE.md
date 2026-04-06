# Docker Quick Start Guide for Pandora

## Prerequisites

- Docker and Docker Compose installed
- `.env` file with `GROQ_API_KEY` (copy from `.env.example`)

## Quick Commands

### 1. Run Dashboard (View Trained Models)

```bash
# Start the dashboard
docker-compose up pandora-dashboard

# Or run in background
docker-compose up -d pandora-dashboard

# View at: http://localhost:8501
```

### 2. Train Models

#### Train PPO (1M timesteps, ~2-3 hours):
```bash
docker-compose --profile training up pandora-train-ppo
```

#### Train DQN (300k timesteps, ~30-60 min):
```bash
docker-compose --profile training up pandora-train-dqn
```

#### Train Both (PPO: 1M, DQN: 300k):
```bash
docker-compose --profile training up pandora-train-both
```

### 3. Stop Services

```bash
# Stop dashboard
docker-compose down

# Stop all services including training
docker-compose --profile training down
```

## Updated Features

### New Actions in Environment:
- ✅ **INCREASE_POPULATION**: Pro-natalist policies (+15% population)
- ✅ **POPULATION_CONTROL**: Birth control policies (-5% pop, better resources)

### Updated Training Configs:
- ✅ **PPO**: 1M timesteps (was 100k) - needs more training!
- ✅ **DQN**: 300k timesteps (was 100k) - optimized
- ✅ **4 parallel environments** for faster training
- ✅ **Improved hyperparameters** (entropy, batch size, etc.)

### LLM Agent:
- ✅ **Updated prompt** with new population control actions
- ✅ **Strategic hints** for when to use each action
- ✅ **Better action descriptions** in prompt

## Service Details

### `pandora-dashboard`
- **Purpose**: Interactive Streamlit dashboard
- **Port**: 8501
- **Access**: http://localhost:8501
- **Models Required**: `models/pandora_ppo.zip`, `models/pandora_dqn.zip` (optional)
- **Restarts**: Automatically unless stopped

### `pandora-train-ppo`
- **Purpose**: Train PPO agent
- **Timesteps**: 1,000,000 (recommended for PPO)
- **Time**: ~2-3 hours (4 parallel envs)
- **Output**: `models/pandora_ppo.zip`
- **Profile**: `training` (use `--profile training`)

### `pandora-train-dqn`
- **Purpose**: Train DQN agent
- **Timesteps**: 300,000 (sufficient for DQN)
- **Time**: ~30-60 minutes (4 parallel envs)
- **Output**: `models/pandora_dqn.zip`
- **Profile**: `training` (use `--profile training`)

### `pandora-train-both`
- **Purpose**: Train both PPO and DQN sequentially
- **Config**: Uses `train_quick.py --both --standard`
- **Time**: ~2.5-3.5 hours total
- **Output**: Both model files
- **Profile**: `training` (use `--profile training`)

## Volumes

### Models Directory
```yaml
./models:/app/models
```
- Models are saved on your host machine
- Persists even if container is removed
- Share models between dashboard and training

### Logs Directory
```yaml
./logs:/app/logs
```
- TensorBoard logs (if tensorboard installed)
- Training progress logs
- View with: `tensorboard --logdir ./logs/`

## Environment Variables

Required in `.env` file:

```bash
GROQ_API_KEY=your_api_key_here
```

Optional:
```bash
# Python path (already set in docker-compose)
PYTHONPATH=/app
```

## Example Workflows

### Workflow 1: Fresh Start (No Models)

```bash
# 1. Train models
docker-compose --profile training up pandora-train-both

# 2. Wait for completion (~3 hours)

# 3. Start dashboard to view results
docker-compose up -d pandora-dashboard

# 4. Open http://localhost:8501
```

### Workflow 2: Quick Test (Existing Models)

```bash
# Just start the dashboard
docker-compose up pandora-dashboard

# Open http://localhost:8501
```

### Workflow 3: Retrain One Model

```bash
# Train only PPO
docker-compose --profile training up pandora-train-ppo

# Or train only DQN
docker-compose --profile training up pandora-train-dqn
```

## Troubleshooting

### Dashboard won't start:
```bash
# Check logs
docker-compose logs pandora-dashboard

# Common issues:
# - Missing .env file → Copy from .env.example
# - Missing GROQ_API_KEY → Add to .env
# - Port 8501 in use → Change port in docker-compose.yml
```

### Training fails:
```bash
# Check logs
docker-compose --profile training logs pandora-train-ppo

# Common issues:
# - Out of memory → Reduce n_envs in docker-compose.yml
# - Slow performance → Normal, training takes time
```

### Models not showing in dashboard:
```bash
# Check if models exist
ls -la models/

# Should see:
# - pandora_ppo.zip
# - pandora_dqn.zip

# If missing, train them first
```

## Building from Scratch

```bash
# Rebuild containers
docker-compose build

# Force rebuild (no cache)
docker-compose build --no-cache

# Rebuild and start
docker-compose up --build pandora-dashboard
```

## Production Deployment

### Using Docker Compose (Recommended):
```bash
# Run dashboard in background
docker-compose up -d pandora-dashboard

# Check status
docker-compose ps

# View logs
docker-compose logs -f pandora-dashboard
```

### Using Docker directly:
```bash
# Build image
docker build -t pandora:latest .

# Run dashboard
docker run -d \
  --name pandora-live \
  -p 8501:8501 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  --env-file .env \
  pandora:latest
```

## Cleanup

```bash
# Stop all services
docker-compose down

# Remove volumes (WARNING: deletes models!)
docker-compose down -v

# Remove images
docker-compose down --rmi all
```

## Advanced: Custom Training

Edit `docker-compose.yml` to customize training:

```yaml
pandora-train-ppo:
  entrypoint: ["python", "agents/rl_agent.py", 
               "--algo", "ppo", 
               "--timesteps", "2000000",  # ← Increase for better results
               "--n_envs", "8"]           # ← More envs = faster (needs more CPU)
```

## System Requirements

### Minimal (Dashboard only):
- 2 CPU cores
- 4 GB RAM
- 1 GB disk space

### Recommended (Training):
- 4-8 CPU cores
- 8-16 GB RAM
- 2 GB disk space

### Optimal (Fast Training):
- 8+ CPU cores
- 16+ GB RAM
- GPU support (modify Dockerfile for torch-cuda)

---

**Ready to start? Run:**
```bash
docker-compose up pandora-dashboard
```

Then open: **http://localhost:8501** 🚀
