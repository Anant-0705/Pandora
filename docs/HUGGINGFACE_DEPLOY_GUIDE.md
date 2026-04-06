# HuggingFace Spaces Deployment - Step by Step

## Prerequisites

✅ You've completed `OPENENV_SETUP_COMPLETE.md`  
✅ Local server tested and working  
✅ Python client tested successfully  
✅ HuggingFace account (free): https://huggingface.co/join

## Method 1: Git Push to HF Spaces (Recommended)

### Step 1: Create New Space

1. Go to: https://huggingface.co/new-space
2. Fill in:
   - **Space name**: `pandora-rl` (or your choice)
   - **License**: MIT
   - **Select the Space SDK**: **Docker**
   - **Hardware**: **CPU basic - Free** (2 vCPU, 16GB RAM)
3. Click **Create Space**

### Step 2: Clone Your New Space

```bash
# Install git-lfs if not already installed
git lfs install

# Clone the empty space
git clone https://huggingface.co/spaces/YOUR-USERNAME/pandora-rl
cd pandora-rl
```

### Step 3: Copy Pandora Files

```powershell
# From PowerShell (Windows)
cd c:\Users\AnantS\Desktop\pandora

# Copy all necessary directories
robocopy . ..\pandora-rl /E /XD .git __pycache__ logs models agent_workspace .env /XF *.pyc

# Or manually copy these folders:
# - agents/
# - core/
# - environment/
# - grader/
# - openenv_wrapper/
# - rewards/
# - requirements.txt
# - README.md (optional)
```

### Step 4: Copy Dockerfile to Root

HuggingFace Spaces looks for Dockerfile in the root directory:

```bash
cd ..\pandora-rl

# Copy Dockerfile
copy openenv_wrapper\server\Dockerfile Dockerfile

# Or create a new one at root
```

**Create `Dockerfile` at root** (copy this):

```dockerfile
FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Set Python path
ENV PYTHONPATH=/app

# Expose port (HF Spaces uses 7860 by default, but we'll use 8000)
ENV PORT=7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run OpenEnv server (HF Spaces sets PORT env var)
CMD uvicorn openenv_wrapper.server.app:app \
    --host 0.0.0.0 \
    --port ${PORT:-7860} \
    --workers 2
```

### Step 5: Create README.md for Space

Create `README.md` in the Space root:

```markdown
---
title: Pandora RL
emoji: 🌍
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# Pandora: RL-Driven Civilization Simulation

A reinforcement learning environment simulating 10,000 years of civilization.
Balance population, resources, technology, equality, and climate through
strategic decision-making.

## Using This Space

### Via Python Client

\`\`\`python
pip install git+https://huggingface.co/spaces/YOUR-USERNAME/pandora-rl

from openenv_wrapper.client import PandoraEnvClient
from openenv_wrapper.models import PandoraAction

# Connect to this Space
with PandoraEnvClient(
    base_url="https://YOUR-USERNAME-pandora-rl.hf.space"
).sync() as env:
    
    obs = env.reset(seed=42)
    print(f"Year {obs.year}, Population: {obs.population:,.0f}")
    
    # Choose 3 actions from 12 available
    action = PandoraAction(
        action_1=0,   # BOOST_FOOD_PRODUCTION
        action_2=6,   # PUSH_SCIENCE
        action_3=10   # INCREASE_POPULATION
    )
    
    result = env.step(action)
    print(f"Reward: {result.reward:.2f}")
\`\`\`

### Available Actions (0-11)

0. BOOST_FOOD_PRODUCTION - +30% food (hurts climate)
1. INVEST_IN_WATER - +50% water supply
2. EXPAND_TERRITORY - +20% all resources (risks conflict)
3. PROMOTE_EQUALITY - Reduce inequality by 10%
4. ENFORCE_ORDER - Clear conflicts (hurts happiness)
5. ENCOURAGE_RELIGION - +10% happiness boost
6. PUSH_SCIENCE - +1 tech level
7. INDUSTRIALIZE - +2 tech (heavy climate damage)
8. GO_GREEN - +10% climate health
9. INSTALL_AI_GOVERNANCE - AI-governed state
10. INCREASE_POPULATION - +15% growth (costs food)
11. POPULATION_CONTROL - -5% pop, better resources/capita

## API Endpoints

- **Health**: `GET /health`
- **Reset**: `POST /reset`
- **Step**: `POST /step`
- **State**: `GET /state`
- **API Docs**: `GET /docs`
- **WebSocket**: `WS /ws`

## GitHub Repository

[github.com/your-username/pandora](https://github.com/your-username/pandora)
```

### Step 6: Commit and Push

```bash
cd ..\pandora-rl

# Add all files
git add .

# Commit
git commit -m "Deploy Pandora RL environment with OpenEnv"

# Push to HuggingFace
git push

# If you get auth errors:
# git config credential.helper store
# Then git push again and enter your HF token
```

### Step 7: Wait for Build

1. Go to your Space: `https://huggingface.co/spaces/YOUR-USERNAME/pandora-rl`
2. Watch the build logs (takes 5-10 minutes)
3. Once built, you'll see "Running" status

### Step 8: Test Your Deployed Space

```python
# test_deployed.py
from openenv_wrapper.client import PandoraEnvClient
from openenv_wrapper.models import PandoraAction

# Connect to YOUR deployed Space
with PandoraEnvClient(
    base_url="https://YOUR-USERNAME-pandora-rl.hf.space"
).sync() as env:
    
    print("Connected to deployed Pandora!")
    
    obs = env.reset(seed=42)
    print(f"\nInitial State:")
    print(f"  Year: {obs.year}")
    print(f"  Population: {obs.population:,.0f}")
    print(f"  Happiness: {obs.happiness:.2f}")
    print(f"  Tech Level: {obs.tech_level}")
    print(f"  Climate: {obs.climate_health:.2f}")
    
    # Play 5 turns
    for turn in range(5):
        action = PandoraAction(
            action_1=0,   # Food
            action_2=6,   # Science
            action_3=10   # Population
        )
        
        result = env.step(action)
        obs = result.observation
        
        print(f"\nTurn {turn + 1}:")
        print(f"  Year: {obs.year}")
        print(f"  Population: {obs.population:,.0f}")
        print(f"  Reward: {result.reward:.2f}")
        print(f"  Events: {obs.recent_events[-1] if obs.recent_events else 'None'}")
        
        if obs.done:
            print("\nEpisode ended!")
            break
    
    print("\n✅ Deployment successful!")
```

Run: `python test_deployed.py`

---

## Method 2: Using `openenv push` (If Available)

If openenv CLI is installed:

```bash
cd c:\Users\AnantS\Desktop\pandora\openenv_wrapper
openenv push --repo-id YOUR-USERNAME/pandora-rl
```

This automatically:
- Creates the Space
- Uploads all files
- Configures Dockerfile
- Starts the build

---

## Troubleshooting

### Build Fails with "Module not found"

**Check**:
1. All dependencies in `requirements.txt`
2. Dockerfile has `COPY . .` to copy all files
3. `PYTHONPATH=/app` is set

**Fix**: Add missing imports to requirements.txt

### Port Issues

HF Spaces uses port **7860** by default. Make sure Dockerfile has:

```dockerfile
ENV PORT=7860
EXPOSE 7860
CMD uvicorn ... --port ${PORT:-7860}
```

### "Application startup failed"

**Check logs** in Space → "Logs" tab

Common issues:
- Missing openenv-core: `pip install openenv-core`
- Import errors: Check sys.path.insert in environment.py
- Circular imports: Check __init__.py files

### Connection Timeout

Free tier (CPU basic) can be slow on first request. Wait 30s and retry.

Upgrade to "CPU Upgrade" for faster startup if needed.

---

## Next Steps After Deployment

### 1. Share Your Space

Share the URL: `https://YOUR-USERNAME-pandora-rl.hf.space`

Anyone can now:
- View API docs: `/docs`
- Connect via Python client
- Train RL agents against your environment

### 2. Pin to Your Profile

In Space settings, click **"Pin to profile"** to showcase it.

### 3. Add to Environment Hub

Submit to OpenEnv hub: https://huggingface.co/collections/openenv/environment-hub

### 4. Monitor Usage

Check Space → Analytics to see:
- Number of requests
- Active sessions
- Response times

### 5. Upgrade if Needed

If you get lots of traffic:
- Upgrade to **CPU Upgrade** (8 vCPU, 32GB RAM) - $0.03/hour
- Or **GPU** for faster inference - $0.60/hour

---

## Cost Summary

| Tier | Hardware | Sessions | Cost |
|------|----------|----------|------|
| **CPU basic** | 2 vCPU, 16GB | ~128 concurrent | **Free** |
| CPU Upgrade | 8 vCPU, 32GB | ~500 concurrent | $0.03/hr |
| GPU T4 | 1 GPU, 4 vCPU | 1000+ concurrent | $0.60/hr |

**Recommendation**: Start with free tier, upgrade if needed.

---

## Example: Full Usage

```python
# Someone anywhere in the world can now use your environment!

from openenv_wrapper.client import PandoraEnvClient
from openenv_wrapper.models import PandoraAction, PANDORA_ACTIONS

# Connect
with PandoraEnvClient(
    base_url="https://YOUR-USERNAME-pandora-rl.hf.space"
).sync() as env:
    
    # Play a full episode
    obs = env.reset(seed=123)
    total_reward = 0
    
    for turn in range(100):  # 100 turns = 10,000 years
        # Your RL agent or strategy here
        action = PandoraAction(
            action_1=turn % 12,
            action_2=(turn + 1) % 12,
            action_3=(turn + 2) % 12
        )
        
        result = env.step(action)
        total_reward += result.reward
        
        if result.observation.done:
            break
    
    print(f"Episode ended at year {result.observation.year}")
    print(f"Final population: {result.observation.population:,.0f}")
    print(f"Total reward: {total_reward:.2f}")
```

---

## Success Checklist

- [ ] Dockerfile in root directory
- [ ] requirements.txt includes openenv-core
- [ ] README.md with usage examples
- [ ] Git pushed to HF Space
- [ ] Build completed successfully
- [ ] Health endpoint returns {"status": "healthy"}
- [ ] API docs visible at /docs
- [ ] Python client can connect
- [ ] Test episode runs successfully

---

**Congratulations!** Your Pandora environment is now:
✅ Deployed to HuggingFace Spaces  
✅ Accessible worldwide via URL  
✅ Pip-installable as a package  
✅ Ready for RL training and research  

🌍 **Share your Space and let others build civilizations!** 🚀
