# Quick Start: Deploy Pandora to HuggingFace Spaces with OpenEnv

## TL;DR

**Current status**: ❌ Pandora uses Gymnasium (not OpenEnv compatible)  
**Required**: ✅ Add OpenEnv wrapper for HF Spaces deployment  
**Docker**: ✅ Works same as offline for training  
**Timeline**: ~2 hours to integrate OpenEnv

## Answer to Your Questions

### 1. "How to deploy to HuggingFace Spaces?"

**Two options**:

**Option A: Streamlit Dashboard (Easy, Limited)**
- Deploy current `visualization/live_dashboard.py` as Streamlit Space
- Works immediately, no code changes
- ❌ Can't train remotely, only view pre-trained models
- ✅ Good for demos

**Option B: OpenEnv Environment (Recommended, Full Features)**
- Wrap Pandora in OpenEnv interface
- Serves environment via WebSocket API
- ✅ Others can train against your environment
- ✅ Worldwide access via URL
- ⏱️ Requires ~100 lines of wrapper code

### 2. "Will Docker version work same as offline?"

**YES** - Docker Compose works identically to local:

| Aspect | Local | Docker Compose | OpenEnv/Spaces |
|--------|-------|----------------|----------------|
| Training speed | ✅ Fast | ✅ Same | ❌ Slow (network) |
| Model quality | ✅ Best | ✅ Same | ✅ Same |
| Models saved | ✅ ./models/ | ✅ ./models/ (volume) | ❌ Not for training |
| Dashboard | ✅ localhost | ✅ localhost:8501 | ✅ Public URL |

**Recommendation**: Use Docker for production training, OpenEnv for sharing.

### 3. "Are we using OpenEnv?"

**NO** - Currently using Gymnasium (standard RL framework)

**Why OpenEnv is required for HF Spaces**:
- HF Spaces expects WebSocket-based serving
- OpenEnv provides `reset()`, `step()`, `state()` via FastAPI
- Gymnasium runs locally only, can't be accessed remotely

**Solution**: Add OpenEnv wrapper (keeps existing Gymnasium code)

## Implementation Steps

### Step 1: Create Directory Structure

```bash
cd c:\Users\AnantS\Desktop\pandora

# Create OpenEnv wrapper directory
mkdir openenv_wrapper
mkdir openenv_wrapper\server
```

### Step 2: Install Dependencies

```bash
pip install openenv-core fastapi uvicorn websockets
```

Add to `requirements.txt`:
```
openenv-core>=0.1.0
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
websockets>=12.0
```

### Step 3: Create Files

You need to create 6 files in `openenv_wrapper/`:

1. **`models.py`** - Type definitions (Action, Observation, State)
2. **`client.py`** - Python client for remote access  
3. **`server/environment.py`** - Adapter: OpenEnv → Gymnasium Pandora
4. **`server/app.py`** - FastAPI server (2 lines!)
5. **`server/Dockerfile`** - Container for HF Spaces
6. **`openenv.yaml`** - Deployment manifest

**Full code for all 6 files is in**: `OPENENV_INTEGRATION_GUIDE.md`

### Step 4: Test Locally

```bash
# Run OpenEnv server
cd openenv_wrapper
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

# In another terminal, test:
curl http://localhost:8000/health
# Should return: {"status": "healthy"}
```

### Step 5: Deploy to HF Spaces

**Method 1: Using openenv push (if available)**
```bash
cd openenv_wrapper
openenv push --repo-id your-username/pandora-rl
```

**Method 2: Manual (always works)**

1. Go to https://huggingface.co/new-space
2. Create Space:
   - Name: `pandora-rl`
   - SDK: **Docker**
   - Hardware: **CPU basic (free)**

3. Clone and setup:
```bash
git clone https://huggingface.co/spaces/your-username/pandora-rl
cd pandora-rl

# Copy entire Pandora project
xcopy /E /I c:\Users\AnantS\Desktop\pandora\* .

# Copy OpenEnv Dockerfile to root (HF Spaces requirement)
copy openenv_wrapper\server\Dockerfile Dockerfile

# Commit
git add .
git commit -m "Deploy Pandora with OpenEnv"
git push
```

4. Wait 5-10 minutes for build

5. Test at: `https://your-username-pandora-rl.hf.space/docs`

### Step 6: Use Your Deployed Environment

```python
# From anywhere in the world:
from openenv_wrapper.client import PandoraEnvClient
from openenv_wrapper.models import PandoraAction

with PandoraEnvClient(
    base_url="https://your-username-pandora-rl.hf.space"
).sync() as env:
    
    obs = env.reset(seed=42)
    print(f"Year {obs.year}, Population: {obs.population:,.0f}")
    
    # Take action
    action = PandoraAction(action_1=0, action_2=6, action_3=10)
    result = env.step(action)
    print(f"Reward: {result.reward:.2f}")
```

## Alternative: Quick Deploy Streamlit Only

If you just want to show the dashboard publicly (no training API):

1. Go to https://huggingface.co/new-space
2. Choose SDK: **Streamlit** (not Docker)
3. Clone and copy:

```bash
git clone https://huggingface.co/spaces/your-username/pandora-dashboard
cd pandora-dashboard

# Copy only what's needed for dashboard
xcopy /E /I c:\Users\AnantS\Desktop\pandora\agents .
xcopy /E /I c:\Users\AnantS\Desktop\pandora\environment .
xcopy /E /I c:\Users\AnantS\Desktop\pandora\core .
xcopy /E /I c:\Users\AnantS\Desktop\pandora\rewards .
xcopy /E /I c:\Users\AnantS\Desktop\pandora\grader .
xcopy /E /I c:\Users\AnantS\Desktop\pandora\visualization .
xcopy /E /I c:\Users\AnantS\Desktop\pandora\models .
copy c:\Users\AnantS\Desktop\pandora\requirements.txt .

# Create app.py (HF Spaces entry point)
echo import streamlit as st > app.py
echo exec(open("visualization/live_dashboard.py").read()) >> app.py

# Commit
git add .
git commit -m "Pandora Dashboard"
git push
```

Access at: `https://your-username-pandora-dashboard.hf.space`

**Pros**: Simple, works immediately  
**Cons**: No training API, just visualization

## What to Do Next

**Choose your path**:

### Path A: Just Share Dashboard (15 minutes)
1. Deploy Streamlit to HF Spaces (instructions above)
2. Share URL with others
3. ✅ Done

### Path B: Full OpenEnv Integration (2 hours)
1. Create `openenv_wrapper/` directory
2. Copy 6 files from `OPENENV_INTEGRATION_GUIDE.md`
3. Test locally with uvicorn
4. Deploy to HF Spaces as Docker Space
5. ✅ Now anyone can train against your environment via API

### Path C: Both (Best)
1. Do Path A first (quick demo)
2. Then Path B when you have time (proper API)

## Files I Created for You

1. ✅ **`OPENENV_INTEGRATION_GUIDE.md`** - Complete technical guide with all code
2. ✅ **`HUGGINGFACE_SPACES_QUICK_START.md`** (this file) - Quick start instructions
3. ✅ **`DOCKER_GUIDE.md`** - Docker deployment guide (already exists)

## Summary

| Feature | Current (Gymnasium) | With OpenEnv | Streamlit Only |
|---------|--------------------| -------------|----------------|
| Local training | ✅ Yes | ✅ Yes | ✅ Yes |
| Docker training | ✅ Yes | ✅ Yes | ❌ No |
| Remote API | ❌ No | ✅ Yes | ❌ No |
| Public demo | ❌ No | ✅ Yes | ✅ Yes |
| Setup time | 0 min | 2 hours | 15 min |

**My recommendation**:
1. ✅ **Now**: Deploy Streamlit dashboard (15 min) - Path A
2. ⏳ **Next week**: Add OpenEnv wrapper (2 hours) - Path B
3. 🚀 **Result**: Public demo + training API

---

**Ready to start?** 

Choose Path A (quick demo) or Path B (full API) and follow the steps above.

All code is in `OPENENV_INTEGRATION_GUIDE.md` - just copy/paste the 6 files!
