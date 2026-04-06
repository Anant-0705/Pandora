# ✅ OpenEnv Integration Checklist

Use this to track your progress through the 2-hour integration.

## Phase 1: Setup (15 min) ⏱️ 0:00-0:15

- [ ] Navigate to pandora directory
- [ ] Run PowerShell/CMD commands to create directories
- [ ] Verify `openenv_wrapper/` exists
- [ ] Verify `openenv_wrapper/server/` exists
- [ ] Create empty `__init__.py` files
- [ ] Run `pip install openenv-core fastapi uvicorn[standard] websockets pydantic`
- [ ] Verify installation: `python -c "import fastapi; print('OK')"`

**Status**: ⬜ Not started | 🟡 In progress | ✅ Complete

---

## Phase 2: Create Files (30 min) ⏱️ 0:15-0:45

Open `OPENENV_SETUP_COMPLETE.md` and copy each file:

- [ ] **File 1**: `openenv_wrapper\models.py` (80 lines)
  - Copy from "FILE 1" section
  - Contains: PandoraAction, PandoraObservation, PandoraState
  
- [ ] **File 2**: `openenv_wrapper\server\environment.py` (150 lines)
  - Copy from "FILE 2" section
  - Contains: PandoraOpenEnv class
  
- [ ] **File 3**: `openenv_wrapper\server\app.py` (30 lines)
  - Copy from "FILE 3" section
  - Contains: FastAPI app creation
  
- [ ] **File 4**: `openenv_wrapper\client.py` (60 lines)
  - Copy from "FILE 4" section
  - Contains: PandoraEnvClient class
  
- [ ] **File 5**: `openenv_wrapper\server\Dockerfile` (25 lines)
  - Copy from "FILE 5" section
  - Contains: Docker container config
  
- [ ] **File 6**: `openenv_wrapper\openenv.yaml` (60 lines)
  - Copy from "FILE 6" section
  - Contains: Deployment manifest

**Verification**:
- [ ] All 6 files created
- [ ] No syntax errors (check in IDE)
- [ ] File sizes match approximately

---

## Phase 3: Test Locally (15 min) ⏱️ 0:45-1:00

- [ ] Start server: `uvicorn openenv_wrapper.server.app:app --port 8000`
- [ ] Verify server starts without errors
- [ ] See "Uvicorn running on http://0.0.0.0:8000"
- [ ] Test health: `curl http://localhost:8000/health`
- [ ] Returns: `{"status": "healthy"}`
- [ ] Open browser: `http://localhost:8000/docs`
- [ ] See API documentation page
- [ ] Verify endpoints: /reset, /step, /state, /health, /ws

**Python Client Test**:
- [ ] Create `test_openenv_local.py` (from 2_HOUR_ROADMAP.md)
- [ ] Run: `python test_openenv_local.py`
- [ ] See "✅ Connected!"
- [ ] See 5 turns of output
- [ ] No errors

---

## Phase 4: Deploy to HF Spaces (30 min) ⏱️ 1:00-1:30

### HuggingFace Account
- [ ] Have HF account (or create: https://huggingface.co/join)
- [ ] Logged in
- [ ] Know your username

### Create Space
- [ ] Go to: https://huggingface.co/new-space
- [ ] Name: `pandora-rl` (or your choice)
- [ ] SDK: **Docker** (important!)
- [ ] Hardware: **CPU basic - Free**
- [ ] Click "Create Space"

### Prepare Files
- [ ] Clone space: `git clone https://huggingface.co/spaces/YOUR-USERNAME/pandora-rl`
- [ ] Copy Pandora files to cloned space
- [ ] Create `Dockerfile` at root (from HUGGINGFACE_DEPLOY_GUIDE.md)
- [ ] Create `README.md` for space (from HUGGINGFACE_DEPLOY_GUIDE.md)
- [ ] Verify all folders copied: agents/, core/, environment/, grader/, openenv_wrapper/, rewards/
- [ ] Verify requirements.txt copied

### Push to HF
- [ ] `cd pandora-rl`
- [ ] `git add .`
- [ ] `git commit -m "Deploy Pandora RL with OpenEnv"`
- [ ] `git push`
- [ ] If auth error: enter HF token

### Watch Build
- [ ] Go to: `https://huggingface.co/spaces/YOUR-USERNAME/pandora-rl`
- [ ] See "Building..." status
- [ ] Watch build logs
- [ ] Wait 5-10 minutes
- [ ] See "Running" status ✅

---

## Phase 5: Test Deployment (30 min) ⏱️ 1:30-2:00

### Verify Endpoints
- [ ] Health: `https://YOUR-USERNAME-pandora-rl.hf.space/health`
- [ ] Returns: `{"status": "healthy"}`
- [ ] Docs: `https://YOUR-USERNAME-pandora-rl.hf.space/docs`
- [ ] See API documentation

### Python Client Test
- [ ] Create `test_deployed.py` (from 2_HOUR_ROADMAP.md)
- [ ] Update SPACE_URL with your username
- [ ] Run: `python test_deployed.py`
- [ ] See "✅ Connected to deployed Space!"
- [ ] See 10 turns of output
- [ ] See "🎉 Deployment successful!"

### Share
- [ ] Copy Space URL: `https://YOUR-USERNAME-pandora-rl.hf.space`
- [ ] Test URL in browser
- [ ] See Space homepage
- [ ] Pin to profile (optional)

---

## Troubleshooting

### If Server Won't Start
- [ ] Check all files created
- [ ] Check no syntax errors
- [ ] Try: `python -c "from openenv_wrapper.server.environment import PandoraOpenEnv"`
- [ ] If import fails, check sys.path.insert in environment.py

### If Health Fails
- [ ] Server running?
- [ ] Port 8000 available?
- [ ] Try different port: `--port 8001`

### If Client Can't Connect
- [ ] Server URL correct?
- [ ] Server running?
- [ ] Firewall blocking?

### If HF Build Fails
- [ ] Check build logs in Space
- [ ] Common: Missing dependency in requirements.txt
- [ ] Check Dockerfile syntax
- [ ] Verify all files pushed: `git status`

### If Deployed But Not Working
- [ ] Check logs in HF Space
- [ ] Try: `curl https://YOUR-SPACE.hf.space/health`
- [ ] Wait 30s after startup (cold start)
- [ ] Free tier can be slow on first request

---

## Success Metrics

After 2 hours, you should have:

✅ **Local Development**
- [ ] OpenEnv wrapper code created (6 files)
- [ ] Server runs locally without errors
- [ ] Python client connects successfully
- [ ] Test script passes

✅ **Deployment**
- [ ] HuggingFace Space created
- [ ] All files pushed to Space
- [ ] Build completed successfully
- [ ] Space shows "Running" status

✅ **Functionality**
- [ ] `/health` endpoint works
- [ ] `/docs` page accessible
- [ ] Python client connects to deployed Space
- [ ] Can run full episodes remotely

✅ **Sharing**
- [ ] Have public URL to share
- [ ] API documentation available
- [ ] Others can install and use: `pip install git+https://huggingface.co/spaces/...`

---

## Next Steps (After 2 Hours)

- [ ] Share Space URL on social media / Discord / GitHub
- [ ] Add Space to OpenEnv Environment Hub
- [ ] Write blog post about your RL environment
- [ ] Add more documentation to Space README
- [ ] Monitor usage in HF Space Analytics
- [ ] Consider upgrading hardware if needed
- [ ] Train RL agents using deployed environment
- [ ] Add example notebooks to Space

---

## Time Checkpoints

| Time | Milestone | Status |
|------|-----------|--------|
| 0:15 | Setup complete | ⬜ |
| 0:45 | Files created | ⬜ |
| 1:00 | Local test passes | ⬜ |
| 1:30 | Deployed to HF | ⬜ |
| 2:00 | Deployment tested | ⬜ |

---

## Quick Links

- **Setup Guide**: `OPENENV_SETUP_COMPLETE.md`
- **Deploy Guide**: `HUGGINGFACE_DEPLOY_GUIDE.md`
- **Roadmap**: `2_HOUR_ROADMAP.md`
- **This Checklist**: `OPENENV_CHECKLIST.md`

---

**Current Time**: ___:___

**Target Completion**: ___:___ (2 hours from now)

**Let's do this!** 🚀

Start with Phase 1 and check off each item as you go.

Good luck! You've got all the documentation and code you need. 💪
