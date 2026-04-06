# OpenEnv Integration - Complete Setup Script

## Step 1: Create Directory Structure

Open PowerShell or Command Prompt and run:

```powershell
cd c:\Users\AnantS\Desktop\pandora

# Create OpenEnv wrapper directories
New-Item -ItemType Directory -Force -Path "openenv_wrapper"
New-Item -ItemType Directory -Force -Path "openenv_wrapper\server"
New-Item -ItemType File -Path "openenv_wrapper\__init__.py"
New-Item -ItemType File -Path "openenv_wrapper\server\__init__.py"
```

Or using CMD:
```cmd
cd c:\Users\AnantS\Desktop\pandora
mkdir openenv_wrapper
mkdir openenv_wrapper\server
type nul > openenv_wrapper\__init__.py
type nul > openenv_wrapper\server\__init__.py
```

## Step 2: Install Dependencies

```bash
pip install openenv-core fastapi uvicorn[standard] websockets pydantic
```

## Step 3: Create Files

Now create these 7 files (I'll provide content below):

1. `openenv_wrapper\models.py` - Type definitions
2. `openenv_wrapper\server\environment.py` - Environment adapter  
3. `openenv_wrapper\server\app.py` - FastAPI server
4. `openenv_wrapper\client.py` - Python client
5. `openenv_wrapper\server\Dockerfile` - Container
6. `openenv_wrapper\openenv.yaml` - Manifest
7. Update `requirements.txt` - Add dependencies

---

## FILE 1: openenv_wrapper\models.py

```python
"""
Pandora OpenEnv Integration - Type Definitions
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class PandoraAction(BaseModel):
    """Action for Pandora - 3 actions from 12 options"""
    action_1: int = Field(ge=0, lt=12, description="First action (0-11)")
    action_2: int = Field(ge=0, lt=12, description="Second action (0-11)")
    action_3: int = Field(ge=0, lt=12, description="Third action (0-11)")
    
    @property
    def as_array(self):
        return [self.action_1, self.action_2, self.action_3]


class PandoraObservation(BaseModel):
    """Observation of civilization state"""
    # Episode control
    done: bool
    reward: Optional[float] = None
    
    # Core metrics
    year: int
    population: float
    happiness: float
    inequality: float
    tech_level: int
    climate_health: float
    
    # Resources
    food: float
    water: float
    energy: float
    minerals: float
    
    # Narrative
    recent_events: List[str] = Field(default_factory=list)


class PandoraState(BaseModel):
    """Full internal state"""
    episode_id: Optional[str] = None
    step_count: int = 0
    current_year: int = 0
    max_year: int = 10000
    is_extinct: bool = False
    total_turns: int = 0
    cumulative_reward: float = 0.0


PANDORA_ACTIONS = [
    'BOOST_FOOD_PRODUCTION', 'INVEST_IN_WATER', 'EXPAND_TERRITORY',
    'PROMOTE_EQUALITY', 'ENFORCE_ORDER', 'ENCOURAGE_RELIGION',
    'PUSH_SCIENCE', 'INDUSTRIALIZE', 'GO_GREEN', 'INSTALL_AI_GOVERNANCE',
    'INCREASE_POPULATION', 'POPULATION_CONTROL'
]
```

---

## FILE 2: openenv_wrapper\server\environment.py

```python
"""
Pandora OpenEnv Server - Environment Adapter
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import uuid
import numpy as np
from typing import Optional

try:
    from openenv.core.env_server import Environment
except ImportError:
    class Environment:
        SUPPORTS_CONCURRENT_SESSIONS = True

from environment.pandora_env import PandoraEnv
from environment.wrappers import (
    NormalizeObservationWrapper,
    RewardScalingWrapper,
    EarlyTerminationWrapper
)
from openenv_wrapper.models import PandoraAction, PandoraObservation, PandoraState


class PandoraOpenEnv(Environment):
    """OpenEnv wrapper around Gymnasium PandoraEnv"""
    
    SUPPORTS_CONCURRENT_SESSIONS = True
    
    def __init__(self):
        self._gym_env = None
        self._state = PandoraState()
        self._cumulative_reward = 0.0
        
    def _init_gym_env(self, seed: Optional[int] = None):
        """Initialize Gymnasium environment with wrappers"""
        base_env = PandoraEnv(seed=seed or 42)
        env = NormalizeObservationWrapper(base_env)
        env = RewardScalingWrapper(env)
        env = EarlyTerminationWrapper(env)
        return env
    
    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs) -> PandoraObservation:
        """Reset environment"""
        self._gym_env = self._init_gym_env(seed)
        obs, info = self._gym_env.reset()
        
        self._state = PandoraState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            current_year=0,
            is_extinct=False,
            cumulative_reward=0.0
        )
        
        self._cumulative_reward = 0.0
        return self._obs_to_openenv(obs, 0.0, False, info)
    
    def step(self, action: PandoraAction, timeout_s: Optional[int] = None, **kwargs) -> PandoraObservation:
        """Execute one turn"""
        if self._gym_env is None:
            raise RuntimeError("Call reset() first")
        
        gym_action = action.as_array
        obs, reward, terminated, truncated, info = self._gym_env.step(gym_action)
        
        self._state.step_count += 1
        self._state.total_turns += 1
        self._state.current_year = int(obs[0] * 10000)
        self._cumulative_reward += reward
        self._state.cumulative_reward = self._cumulative_reward
        
        done = terminated or truncated
        
        if obs[1] < 0.001:  # Extinction
            self._state.is_extinct = True
            done = True
        
        return self._obs_to_openenv(obs, reward, done, info)
    
    @property
    def state(self) -> PandoraState:
        return self._state
    
    def _obs_to_openenv(self, obs: np.ndarray, reward: float, done: bool, info: dict) -> PandoraObservation:
        """Convert Gymnasium obs to OpenEnv obs (denormalize)"""
        year = int(obs[0] * 10000)
        population = obs[1] * 1e12
        happiness = float(obs[2])
        inequality = float(obs[3])
        tech_level = int(obs[4] * 20)
        climate_health = float(obs[5])
        food = obs[6] * 1e6
        water = obs[7] * 1e6
        energy = obs[8] * 1e6
        minerals = obs[9] * 1e6
        
        recent_events = []
        if 'state_obj' in info and hasattr(info['state_obj'], 'history_log'):
            recent_events = info['state_obj'].history_log[-5:]
        
        return PandoraObservation(
            done=done,
            reward=reward,
            year=year,
            population=population,
            happiness=happiness,
            inequality=inequality,
            tech_level=tech_level,
            climate_health=climate_health,
            food=food,
            water=water,
            energy=energy,
            minerals=minerals,
            recent_events=recent_events
        )
```

---

## FILE 3: openenv_wrapper\server\app.py

```python
"""
FastAPI server for Pandora OpenEnv
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from openenv.core.env_server import create_fastapi_app
    from openenv_wrapper.server.environment import PandoraOpenEnv
    
    # Create FastAPI app with Pandora environment
    app = create_fastapi_app(PandoraOpenEnv)
    
except ImportError as e:
    # Fallback if openenv-core not installed
    from fastapi import FastAPI
    app = FastAPI(title="Pandora RL Environment")
    
    @app.get("/health")
    def health():
        return {"status": "healthy", "error": "openenv-core not installed"}
    
    @app.get("/")
    def root():
        return {
            "name": "Pandora RL Environment",
            "error": "openenv-core not installed",
            "install": "pip install openenv-core"
        }

# This gives you:
# - WebSocket: /ws
# - HTTP: /reset, /step, /state, /health
# - Docs: /docs
# - Web UI: /web
```

---

## FILE 4: openenv_wrapper\client.py

```python
"""
Python client for remote Pandora environment
"""

from typing import Optional

try:
    from openenv.core.env_client import EnvClient
    from openenv.core.client_types import StepResult
    from openenv_wrapper.models import PandoraAction, PandoraObservation, PandoraState
    
    
    class PandoraEnvClient(EnvClient[PandoraAction, PandoraObservation, PandoraState]):
        """Client for connecting to remote Pandora environment"""
        
        def _step_payload(self, action: PandoraAction) -> dict:
            return {
                "action_1": action.action_1,
                "action_2": action.action_2,
                "action_3": action.action_3
            }
        
        def _parse_result(self, payload: dict) -> StepResult:
            obs_data = payload.get("observation", {})
            
            observation = PandoraObservation(
                done=payload.get("done", False),
                reward=payload.get("reward"),
                year=obs_data.get("year", 0),
                population=obs_data.get("population", 0),
                happiness=obs_data.get("happiness", 0),
                inequality=obs_data.get("inequality", 0),
                tech_level=obs_data.get("tech_level", 0),
                climate_health=obs_data.get("climate_health", 0),
                food=obs_data.get("food", 0),
                water=obs_data.get("water", 0),
                energy=obs_data.get("energy", 0),
                minerals=obs_data.get("minerals", 0),
                recent_events=obs_data.get("recent_events", [])
            )
            
            return StepResult(
                observation=observation,
                reward=payload.get("reward"),
                done=payload.get("done", False)
            )
        
        def _parse_state(self, payload: dict) -> PandoraState:
            return PandoraState(
                episode_id=payload.get("episode_id"),
                step_count=payload.get("step_count", 0),
                current_year=payload.get("current_year", 0),
                max_year=payload.get("max_year", 10000),
                is_extinct=payload.get("is_extinct", False),
                total_turns=payload.get("total_turns", 0),
                cumulative_reward=payload.get("cumulative_reward", 0.0)
            )

except ImportError:
    # Fallback if openenv-core not installed
    class PandoraEnvClient:
        def __init__(self, *args, **kwargs):
            raise ImportError("openenv-core not installed. Run: pip install openenv-core")
```

---

## FILE 5: openenv_wrapper\server\Dockerfile

```dockerfile
FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Set Python path
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run OpenEnv server
CMD ["uvicorn", "openenv_wrapper.server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4"]
```

---

## FILE 6: openenv_wrapper\openenv.yaml

```yaml
name: pandora-rl
version: "1.0.0"
description: "RL-driven civilization simulation across 10,000 years. Balance population, resources, technology, equality, and climate through strategic decision-making."
author: "Your Name"
license: "MIT"

# Environment metadata
environment:
  type: "multi-step-episodic"
  max_steps: 100
  action_space: "discrete-multi"
  observation_space: "continuous"
  num_actions: 12
  
# 12 available actions
actions:
  - name: "BOOST_FOOD_PRODUCTION"
    id: 0
    description: "Increase food production (+30%, hurts climate)"
  - name: "INVEST_IN_WATER"
    id: 1
    description: "Improve water supply (+50%)"
  - name: "EXPAND_TERRITORY"
    id: 2
    description: "Gain all resources (+20%, risks conflict)"
  - name: "PROMOTE_EQUALITY"
    id: 3
    description: "Reduce inequality (-10%)"
  - name: "ENFORCE_ORDER"
    id: 4
    description: "Resolve conflicts (hurts happiness)"
  - name: "ENCOURAGE_RELIGION"
    id: 5
    description: "Boost morale (+10% happiness)"
  - name: "PUSH_SCIENCE"
    id: 6
    description: "Incremental tech progress (+1 level)"
  - name: "INDUSTRIALIZE"
    id: 7
    description: "Rapid advancement (+2 tech, heavy climate cost)"
  - name: "GO_GREEN"
    id: 8
    description: "Restore climate (+10%, reduces energy)"
  - name: "INSTALL_AI_GOVERNANCE"
    id: 9
    description: "Switch to AI-controlled state"
  - name: "INCREASE_POPULATION"
    id: 10
    description: "Pro-natalist policies (+15% growth, costs food)"
  - name: "POPULATION_CONTROL"
    id: 11
    description: "Birth control (-5% pop, better resources/capita)"

# HuggingFace Spaces deployment config
deployment:
  hardware: "cpu-basic"  # Free tier: 2 vCPU, 16GB RAM
  port: 8000
  workers: 4
  max_concurrent_envs: 100
  
# Links
repository: "https://github.com/your-username/pandora"
documentation: "https://github.com/your-username/pandora/blob/main/README.md"
```

---

## FILE 7: Update requirements.txt

Add these lines to your existing `requirements.txt`:

```txt
# OpenEnv integration (for HuggingFace Spaces deployment)
openenv-core>=0.1.0
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
websockets>=12.0
pydantic>=2.0.0
```

---

## Step 4: Test Locally

```bash
# Start server
cd c:\Users\AnantS\Desktop\pandora
uvicorn openenv_wrapper.server.app:app --host 0.0.0.0 --port 8000 --reload

# Test health endpoint (in another terminal)
curl http://localhost:8000/health
# Should return: {"status": "healthy"}

# View API docs
# Open browser: http://localhost:8000/docs
```

## Step 5: Test with Python Client

Create `test_openenv.py`:

```python
from openenv_wrapper.client import PandoraEnvClient
from openenv_wrapper.models import PandoraAction

# Connect to local server
with PandoraEnvClient(base_url="http://localhost:8000").sync() as env:
    # Reset
    obs = env.reset(seed=42)
    print(f"Year {obs.year}, Population: {obs.population:,.0f}")
    
    # Take 10 steps
    for i in range(10):
        action = PandoraAction(
            action_1=0,  # BOOST_FOOD_PRODUCTION
            action_2=6,  # PUSH_SCIENCE
            action_3=10  # INCREASE_POPULATION
        )
        
        result = env.step(action)
        obs = result.observation
        
        print(f"Turn {i+1}: Year {obs.year}, Pop: {obs.population:,.0f}, "
              f"Tech: {obs.tech_level}, Reward: {result.reward:.2f}")
        
        if obs.done:
            print("Episode ended!")
            break
```

Run: `python test_openenv.py`

---

## Next Steps After Local Testing

1. ✅ Verify all endpoints work: /health, /docs, /ws
2. ✅ Test Python client connects and plays episodes
3. 🚀 Deploy to HuggingFace Spaces (instructions in next file)

---

**READY TO GO!** Follow these steps and you'll have OpenEnv integrated in 30 minutes!
