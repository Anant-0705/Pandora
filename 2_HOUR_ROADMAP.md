# 🚀 OpenEnv Integration - 2 Hour Roadmap

## Timeline

- **0:00-0:15** (15 min): Setup & Installation
- **0:15-0:45** (30 min): Create Files & Test Locally  
- **0:45-1:15** (30 min): Deploy to HuggingFace Spaces
- **1:15-2:00** (45 min): Test Deployment & Buffer

---

## Phase 1: Setup (15 minutes)

### What You Need

- ✅ 2 hours
- ✅ Windows PowerShell or CMD
- ✅ HuggingFace account (free): https://huggingface.co/join
- ✅ Git installed
- ✅ Python with pip

### Step 1: Create Directories (2 min)

```powershell
cd c:\Users\AnantS\Desktop\pandora

New-Item -ItemType Directory -Force -Path "openenv_wrapper"
New-Item -ItemType Directory -Force -Path "openenv_wrapper\server"
New-Item -ItemType File -Force -Path "openenv_wrapper\__init__.py"
New-Item -ItemType File -Force -Path "openenv_wrapper\server\__init__.py"
```

### Step 2: Install Dependencies (5 min)

```bash
pip install openenv-core fastapi "uvicorn[standard]" websockets pydantic
```

### Step 3: Verify Installation (3 min)

```bash
python -c "import fastapi, uvicorn, websockets, pydantic; print('✅ All dependencies installed')"
```

**Checkpoint**: ✅ Directories created, dependencies installed

---

## Phase 2: Create Files (30 minutes)

Open `OPENENV_SETUP_COMPLETE.md` and copy the 6 files:

### File 1: models.py (5 min)
- Path: `openenv_wrapper\models.py`
- Size: ~80 lines
- Copy from: "FILE 1" section

### File 2: server/environment.py (10 min)
- Path: `openenv_wrapper\server\environment.py`
- Size: ~150 lines
- Copy from: "FILE 2" section

### File 3: server/app.py (2 min)
- Path: `openenv_wrapper\server\app.py`
- Size: ~30 lines
- Copy from: "FILE 3" section

### File 4: client.py (5 min)
- Path: `openenv_wrapper\client.py`
- Size: ~60 lines
- Copy from: "FILE 4" section

### File 5: server/Dockerfile (3 min)
- Path: `openenv_wrapper\server\Dockerfile`
- Size: ~25 lines
- Copy from: "FILE 5" section

### File 6: openenv.yaml (5 min)
- Path: `openenv_wrapper\openenv.yaml`
- Size: ~60 lines
- Copy from: "FILE 6" section

**Checkpoint**: ✅ All 6 files created

---

## Phase 3: Test Locally (15 minutes)

### Step 1: Start Server (2 min)

```bash
cd c:\Users\AnantS\Desktop\pandora
uvicorn openenv_wrapper.server.app:app --host 0.0.0.0 --port 8000
```

Should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

### Step 2: Test Health Endpoint (1 min)

In another terminal:
```bash
curl http://localhost:8000/health
```

Should return:
```json
{"status": "healthy"}
```

### Step 3: View API Docs (2 min)

Open browser: http://localhost:8000/docs

Should see:
- POST /reset
- POST /step  
- GET /state
- GET /health
- WS /ws

### Step 4: Test Python Client (10 min)

Create `test_openenv_local.py`:

```python
from openenv_wrapper.client import PandoraEnvClient
from openenv_wrapper.models import PandoraAction

print("Connecting to local Pandora...")

with PandoraEnvClient(base_url="http://localhost:8000").sync() as env:
    print("✅ Connected!")
    
    # Reset
    obs = env.reset(seed=42)
    print(f"\n📊 Initial State:")
    print(f"  Year: {obs.year}")
    print(f"  Population: {obs.population:,.0f}")
    print(f"  Tech: {obs.tech_level}")
    
    # Play 5 turns
    for turn in range(5):
        action = PandoraAction(
            action_1=0,   # BOOST_FOOD_PRODUCTION
            action_2=6,   # PUSH_SCIENCE
            action_3=10   # INCREASE_POPULATION
        )
        
        result = env.step(action)
        obs = result.observation
        
        print(f"\n🔄 Turn {turn + 1}:")
        print(f"  Year: {obs.year}")
        print(f"  Population: {obs.population:,.0f}")
        print(f"  Reward: {result.reward:.2f}")
        
        if obs.done:
            print("\n🏁 Episode ended!")
            break
    
    print("\n✅ Local test successful!")
```

Run: `python test_openenv_local.py`

**Checkpoint**: ✅ Server running, API working, client tested

---

## Phase 4: Deploy to HuggingFace (30 minutes)

### Step 1: Create Space (5 min)

1. Go to: https://huggingface.co/new-space
2. Settings:
   - Name: `pandora-rl`
   - SDK: **Docker**
   - Hardware: **CPU basic (free)**
3. Click "Create Space"

### Step 2: Clone Space (2 min)

```bash
git lfs install
git clone https://huggingface.co/spaces/YOUR-USERNAME/pandora-rl
```

### Step 3: Copy Files (5 min)

```powershell
cd c:\Users\AnantS\Desktop

# Copy entire Pandora project
robocopy pandora pandora-rl /E /XD .git __pycache__ logs models agent_workspace .env /XF *.pyc
```

Or manually copy these folders to `pandora-rl/`:
- agents/
- core/
- environment/
- grader/
- openenv_wrapper/
- rewards/
- requirements.txt

### Step 4: Create Root Dockerfile (3 min)

In `pandora-rl/`, create `Dockerfile`:

```dockerfile
FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
ENV PORT=7860
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD uvicorn openenv_wrapper.server.app:app \
    --host 0.0.0.0 \
    --port ${PORT:-7860} \
    --workers 2
```

### Step 5: Create README (5 min)

See `HUGGINGFACE_DEPLOY_GUIDE.md` → "Create README.md for Space"

### Step 6: Commit and Push (5 min)

```bash
cd ..\pandora-rl

git add .
git commit -m "Deploy Pandora RL with OpenEnv"
git push
```

### Step 7: Wait for Build (5 min)

Watch build logs at: `https://huggingface.co/spaces/YOUR-USERNAME/pandora-rl`

**Checkpoint**: ✅ Space created, files pushed, building...

---

## Phase 5: Test Deployment (30 minutes)

### While Build is Running

Read documentation:
- Your Space's README
- API docs will be at: `/docs`
- Test plan for after deployment

### After Build Completes

Create `test_deployed.py`:

```python
from openenv_wrapper.client import PandoraEnvClient
from openenv_wrapper.models import PandoraAction

SPACE_URL = "https://YOUR-USERNAME-pandora-rl.hf.space"

print(f"Connecting to {SPACE_URL}...")

with PandoraEnvClient(base_url=SPACE_URL).sync() as env:
    print("✅ Connected to deployed Space!")
    
    obs = env.reset(seed=42)
    print(f"\n📊 Year {obs.year}, Population: {obs.population:,.0f}")
    
    # Play 10 turns
    for turn in range(10):
        action = PandoraAction(
            action_1=turn % 12,
            action_2=(turn + 4) % 12,
            action_3=(turn + 8) % 12
        )
        
        result = env.step(action)
        obs = result.observation
        
        print(f"Turn {turn+1}: Year {obs.year}, Pop: {obs.population:,.0f}, "
              f"Tech: {obs.tech_level}, Reward: {result.reward:+.2f}")
        
        if obs.done:
            break
    
    print(f"\n🎉 Deployment successful!")
    print(f"📍 Share this URL: {SPACE_URL}")
```

Run: `python test_deployed.py`

**Final Checkpoint**: ✅ Deployment works, URL shareable!

---

## Success Criteria

At the end of 2 hours, you should have:

- [x] OpenEnv wrapper created locally
- [x] Local server running and tested
- [x] Python client working
- [x] HuggingFace Space created
- [x] Pandora deployed to HF Spaces
- [x] Public URL accessible worldwide
- [x] API documentation at `/docs`
- [x] Test script confirms it works

---

## Quick Reference

### All Files You're Creating

```
pandora/
├── openenv_wrapper/
│   ├── __init__.py                    ✅ Empty file
│   ├── models.py                      ✅ 80 lines
│   ├── client.py                      ✅ 60 lines
│   ├── openenv.yaml                   ✅ 60 lines
│   └── server/
│       ├── __init__.py                ✅ Empty file
│       ├── environment.py             ✅ 150 lines
│       ├── app.py                     ✅ 30 lines
│       └── Dockerfile                 ✅ 25 lines
```

**Total new code**: ~405 lines across 6 files

### Commands Cheat Sheet

```bash
# Setup
pip install openenv-core fastapi uvicorn[standard] websockets

# Test locally
uvicorn openenv_wrapper.server.app:app --port 8000

# Deploy to HF
git clone https://huggingface.co/spaces/USERNAME/pandora-rl
cd pandora-rl
# ... copy files ...
git add . && git commit -m "Deploy" && git push
```

### Documentation Created

1. **OPENENV_SETUP_COMPLETE.md** - All 6 files with full code
2. **HUGGINGFACE_DEPLOY_GUIDE.md** - Deployment walkthrough
3. **INSTALL_OPENENV.md** - Quick install commands
4. **THIS FILE** - 2-hour roadmap

---

## Let's Go! 🚀

**Start now**: Run the PowerShell commands from Phase 1

**Questions?** Check:
- `OPENENV_SETUP_COMPLETE.md` for file contents
- `HUGGINGFACE_DEPLOY_GUIDE.md` for deployment
- `INSTALL_OPENENV.md` for troubleshooting

**You've got this!** In 2 hours, Pandora will be live on HuggingFace! 🌍
