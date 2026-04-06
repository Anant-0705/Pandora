# OpenEnv Integration - Complete Installation Commands

## Quick Copy-Paste Setup (Windows PowerShell)

```powershell
# Navigate to Pandora directory
cd c:\Users\AnantS\Desktop\pandora

# Create OpenEnv directory structure
New-Item -ItemType Directory -Force -Path "openenv_wrapper"
New-Item -ItemType Directory -Force -Path "openenv_wrapper\server"

# Create __init__.py files
New-Item -ItemType File -Force -Path "openenv_wrapper\__init__.py"
New-Item -ItemType File -Force -Path "openenv_wrapper\server\__init__.py"

# Install dependencies
pip install openenv-core fastapi "uvicorn[standard]" websockets pydantic

Write-Host "✅ Directory structure created!"
Write-Host "✅ Dependencies installed!"
Write-Host ""
Write-Host "Next: Copy the 6 files from OPENENV_SETUP_COMPLETE.md"
```

## Quick Copy-Paste Setup (Windows CMD)

```cmd
cd c:\Users\AnantS\Desktop\pandora

mkdir openenv_wrapper
mkdir openenv_wrapper\server

type nul > openenv_wrapper\__init__.py
type nul > openenv_wrapper\server\__init__.py

pip install openenv-core fastapi uvicorn[standard] websockets pydantic

echo Directory structure created!
echo Dependencies installed!
echo.
echo Next: Copy the 6 files from OPENENV_SETUP_COMPLETE.md
```

## Files to Create

After running the above commands, create these 6 files (copy content from `OPENENV_SETUP_COMPLETE.md`):

1. ✅ **openenv_wrapper\models.py**
2. ✅ **openenv_wrapper\server\environment.py**
3. ✅ **openenv_wrapper\server\app.py**
4. ✅ **openenv_wrapper\client.py**
5. ✅ **openenv_wrapper\server\Dockerfile**
6. ✅ **openenv_wrapper\openenv.yaml**

## Test Installation

```bash
# Test 1: Check imports
python -c "from openenv_wrapper.models import PandoraAction, PandoraObservation, PandoraState; print('✅ Models imported')"

# Test 2: Start server
uvicorn openenv_wrapper.server.app:app --host 0.0.0.0 --port 8000

# Test 3: Health check (in another terminal)
curl http://localhost:8000/health

# Test 4: View API docs
# Open browser: http://localhost:8000/docs
```

## Verification Checklist

- [ ] `openenv_wrapper/` directory exists
- [ ] `openenv_wrapper/server/` directory exists
- [ ] Both `__init__.py` files created
- [ ] 6 files created from OPENENV_SETUP_COMPLETE.md
- [ ] Dependencies installed (pip install ...)
- [ ] Server starts without errors
- [ ] `/health` endpoint returns success
- [ ] `/docs` page loads in browser

## Common Issues

### "ModuleNotFoundError: No module named 'openenv'"

**Fix**: `pip install openenv-core`

### "Cannot import name 'Environment'"

**Fix**: Check `openenv_wrapper/server/environment.py` has the fallback class

### Server won't start

**Check**:
1. All files created
2. No syntax errors
3. PYTHONPATH set correctly

**Run**: `python -c "import sys; sys.path.insert(0, '.'); from openenv_wrapper.server.environment import PandoraOpenEnv; print('✅ OK')"`

## Next Steps After Installation

1. ✅ Test locally (commands above)
2. ✅ Create test script (from OPENENV_SETUP_COMPLETE.md)
3. 🚀 Deploy to HuggingFace (see HUGGINGFACE_DEPLOY_GUIDE.md)

---

**Estimated time**: 10-15 minutes for installation, 30 minutes for testing, 30 minutes for deployment = ~1 hour total

You have 2 hours, so you'll finish with time to spare! 🚀
