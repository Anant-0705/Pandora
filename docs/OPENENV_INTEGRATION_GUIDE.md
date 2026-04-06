# Pandora OpenEnv Integration & HuggingFace Spaces Deployment Guide

## Why OpenEnv is Required

OpenEnv is HuggingFace's standard for deploying RL environments. It provides:

1. **WebSocket-based serving** - Efficient persistent sessions (vs HTTP overhead)
2. **Type-safe API** - Pydantic models for actions/observations
3. **Automatic deployment** - One command to HF Spaces
4. **Pip-installable client** - `pip install git+https://huggingface.co/spaces/your-space`
5. **Docker registry** - `docker pull registry.hf.space/your-space:latest`

**Current Status**: Pandora uses Gymnasium (OpenAI Gym), which works locally but isn't compatible with HF Spaces deployment.

**Solution**: Create an OpenEnv wrapper around our existing Pandora environment.

## Architecture Overview

```
pandora/
├── environment/
│   ├── pandora_env.py          # Existing Gymnasium env (KEEP)
│   └── wrappers.py             # Existing wrappers (KEEP)
├── openenv_wrapper/            # NEW - OpenEnv interface
│   ├── models.py               # Action/Observation/State types
│   ├── client.py               # Python client for remote use
│   ├── server/
│   │   ├── environment.py      # OpenEnv → Pandora adapter
│   │   ├── app.py              # FastAPI server
│   │   └── Dockerfile          # Container for HF Spaces
│   └── openenv.yaml            # Manifest
├── visualization/
│   └── live_dashboard.py       # Streamlit (works alongside OpenEnv)
└── docker-compose.yml          # Docker Compose (both versions)
```

**Key insight**: We keep the existing Gymnasium environment and add an OpenEnv wrapper for HF Spaces deployment.

## Step 1: Install OpenEnv

```bash
pip install openenv-core
```

Add to `requirements.txt`:
```
openenv-core>=0.1.0
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
websockets>=12.0
```

## Step 2: Create OpenEnv Models

Create `openenv_wrapper/models.py`:

```python
from typing import List, Optional, Dict
from openenv.core.env_server import Action, Observation, State
from pydantic import Field

class PandoraAction(Action):
    """3 actions per turn from 12 available"""
    action_1: int = Field(ge=0, lt=12, description="First action (0-11)")
    action_2: int = Field(ge=0, lt=12, description="Second action (0-11)")
    action_3: int = Field(ge=0, lt=12, description="Third action (0-11)")
    
    @property
    def as_array(self):
        return [self.action_1, self.action_2, self.action_3]

class PandoraObservation(Observation):
    """State of the civilization"""
    year: int
    population: float
    happiness: float
    inequality: float
    tech_level: int
    climate_health: float
    food: float
    water: float
    energy: float
    minerals: float
    
    # Recent events for narrative context
    recent_events: List[str] = Field(default_factory=list)
    
    # Episode metadata (inherited from Observation)
    # done: bool
    # reward: Optional[float]

class PandoraState(State):
    """Full internal state (for debugging/analysis)"""
    current_year: int = 0
    max_year: int = 10000
    is_extinct: bool = False
    total_turns: int = 0
    
    # Inherited from State:
    # episode_id: Optional[str]
    # step_count: int
```

## Step 3: Create OpenEnv Environment Adapter

Create `openenv_wrapper/server/environment.py`:

```python
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import uuid
import numpy as np
from openenv.core.env_server import Environment
from environment.pandora_env import PandoraEnv, ACTIONS
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
        self._last_obs = None
        self._cumulative_reward = 0.0
        
    def _init_gym_env(self, seed=None):
        """Initialize the Gymnasium environment with wrappers"""
        base_env = PandoraEnv(seed=seed or 42)
        
        # Apply the same wrappers used in training
        env = NormalizeObservationWrapper(base_env)
        env = RewardScalingWrapper(env)
        env = EarlyTerminationWrapper(env)
        
        return env
    
    def reset(self, seed=None, episode_id=None, **kwargs) -> PandoraObservation:
        """Reset the environment"""
        self._gym_env = self._init_gym_env(seed)
        obs, info = self._gym_env.reset()
        
        self._state = PandoraState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            current_year=0,
            max_year=10000,
            is_extinct=False,
            total_turns=0
        )
        
        self._last_obs = obs
        self._cumulative_reward = 0.0
        
        return self._obs_to_openenv(obs, 0.0, False, info)
    
    def step(self, action: PandoraAction, timeout_s=None, **kwargs) -> PandoraObservation:
        """Execute one turn (100 years)"""
        if self._gym_env is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        # Convert OpenEnv action to Gymnasium action
        gym_action = action.as_array
        
        # Step the Gymnasium environment
        obs, reward, terminated, truncated, info = self._gym_env.step(gym_action)
        
        # Update state
        self._state.step_count += 1
        self._state.total_turns += 1
        self._state.current_year = int(obs[0] * 10000)  # Denormalize year
        self._cumulative_reward += reward
        
        done = terminated or truncated
        
        # Check extinction
        if obs[1] < 0.001:  # Population near zero (normalized)
            self._state.is_extinct = True
            done = True
        
        self._last_obs = obs
        
        return self._obs_to_openenv(obs, reward, done, info)
    
    @property
    def state(self) -> PandoraState:
        """Return current internal state"""
        return self._state
    
    def _obs_to_openenv(self, obs: np.ndarray, reward: float, done: bool, info: dict) -> PandoraObservation:
        """Convert Gymnasium observation to OpenEnv observation"""
        # Denormalize observations (reverse the normalization wrapper)
        year = int(obs[0] * 10000)
        population = obs[1] * 1e12  # Max population
        happiness = obs[2]
        inequality = obs[3]
        tech_level = int(obs[4] * 20)  # Max tech
        climate_health = obs[5]
        food = obs[6] * 1e6
        water = obs[7] * 1e6
        energy = obs[8] * 1e6
        minerals = obs[9] * 1e6
        
        # Extract recent events from info
        recent_events = []
        if 'state_obj' in info:
            state_obj = info['state_obj']
            if hasattr(state_obj, 'history_log'):
                recent_events = state_obj.history_log[-5:]  # Last 5 events
        
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

## Step 4: Create FastAPI Server

Create `openenv_wrapper/server/app.py`:

```python
from openenv.core.env_server import create_fastapi_app
from openenv_wrapper.server.environment import PandoraOpenEnv

# Create the FastAPI app with our environment
app = create_fastapi_app(PandoraOpenEnv)

# This gives you:
# - WebSocket endpoint: /ws
# - HTTP endpoints: /reset, /step, /state, /health
# - Web UI: /web
# - API docs: /docs
```

## Step 5: Create Python Client

Create `openenv_wrapper/client.py`:

```python
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from openenv_wrapper.models import PandoraAction, PandoraObservation, PandoraState

class PandoraEnvClient(EnvClient[PandoraAction, PandoraObservation, PandoraState]):
    """Client for connecting to remote Pandora environment"""
    
    def _step_payload(self, action: PandoraAction) -> dict:
        """Convert action to wire format"""
        return {
            "action_1": action.action_1,
            "action_2": action.action_2,
            "action_3": action.action_3
        }
    
    def _parse_result(self, payload: dict) -> StepResult:
        """Parse step response"""
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
        """Parse state response"""
        return PandoraState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            current_year=payload.get("current_year", 0),
            max_year=payload.get("max_year", 10000),
            is_extinct=payload.get("is_extinct", False),
            total_turns=payload.get("total_turns", 0)
        )
```

## Step 6: Create OpenEnv Manifest

Create `openenv_wrapper/openenv.yaml`:

```yaml
name: pandora-rl
version: "1.0.0"
description: "RL-driven civilization simulation across 10,000 years"
author: "Your Name"
license: "MIT"

# Environment metadata
environment:
  type: "multi-step-episodic"
  max_steps: 100
  action_space: "discrete-multi"
  observation_space: "continuous"
  
# HF Spaces configuration
deployment:
  hardware: "cpu-basic"  # Free tier (2 vCPU, 16GB RAM)
  port: 8000
  workers: 4
  max_concurrent_envs: 100
```

## Step 7: Create Dockerfile for OpenEnv

Create `openenv_wrapper/server/Dockerfile`:

```dockerfile
FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
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

# Run OpenEnv server
CMD ["uvicorn", "openenv_wrapper.server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4"]
```

## Step 8: Deploy to HuggingFace Spaces

### Method 1: Using `openenv push` (Recommended)

```bash
# From project root
cd openenv_wrapper
openenv push --repo-id your-username/pandora-rl
```

### Method 2: Manual HF Spaces Setup

1. **Create new Space**: https://huggingface.co/new-space
   - Name: `pandora-rl`
   - SDK: `Docker`
   - Hardware: `CPU basic (free)`

2. **Clone and push**:
```bash
git clone https://huggingface.co/spaces/your-username/pandora-rl
cd pandora-rl

# Copy OpenEnv files
cp -r ../pandora/openenv_wrapper/* .
cp -r ../pandora/environment .
cp -r ../pandora/core .
cp -r ../pandora/rewards .
cp ../pandora/requirements.txt .

# Add Dockerfile at root for HF Spaces
cp server/Dockerfile ./Dockerfile

# Commit and push
git add .
git commit -m "Initial deployment"
git push
```

3. **Wait for build** (~5-10 minutes)

4. **Test your Space**:
   - API: `https://your-username-pandora-rl.hf.space/health`
   - Docs: `https://your-username-pandora-rl.hf.space/docs`
   - Web UI: `https://your-username-pandora-rl.hf.space/web`

## Step 9: Use Your Deployed Environment

### From Python (anywhere):

```python
from openenv_wrapper.client import PandoraEnvClient
from openenv_wrapper.models import PandoraAction

# Connect to your HF Space
with PandoraEnvClient(
    base_url="https://your-username-pandora-rl.hf.space"
).sync() as env:
    
    # Start episode
    obs = env.reset(seed=42)
    print(f"Year {obs.year}, Pop: {obs.population:,.0f}")
    
    # Play 10 turns
    for turn in range(10):
        # Choose actions (0-11 for 12 available actions)
        action = PandoraAction(
            action_1=0,  # BOOST_FOOD_PRODUCTION
            action_2=6,  # PUSH_SCIENCE
            action_3=10  # INCREASE_POPULATION
        )
        
        result = env.step(action)
        obs = result.observation
        
        print(f"Year {obs.year}, Pop: {obs.population:,.0f}, "
              f"Reward: {result.reward:.2f}")
        
        if obs.done:
            print("Episode ended!")
            break
```

### Or install as package:

```bash
pip install git+https://huggingface.co/spaces/your-username/pandora-rl
```

Then:
```python
from pandora_rl import PandoraEnvClient, PandoraAction
# Same usage as above
```

## Comparison: Docker vs OpenEnv vs Local

| Feature | Local Gymnasium | Docker Compose | OpenEnv/HF Spaces |
|---------|----------------|----------------|-------------------|
| **Setup** | pip install | docker-compose up | openenv push |
| **Access** | Python only | Local network | **Internet** |
| **URL** | N/A | localhost:8501 | **your-space.hf.space** |
| **Sharing** | Share code | Share image | **Share URL** |
| **Scalability** | 1 machine | 1 machine | **Auto-scale** |
| **Cost** | Free | Free | **Free (basic tier)** |
| **Use case** | Development | Production (local) | **Production (cloud)** |

**Key difference**: OpenEnv makes your environment accessible worldwide via URL, no server management needed.

## Does Docker Work Same as Offline?

**Short answer**: Yes, with caveats.

**Local (offline)**:
- Direct Python: `python train_quick.py`
- Full filesystem access
- No network overhead
- Streamlit on localhost

**Docker Compose**:
- Containerized: `docker-compose up`
- Isolated filesystem (volumes for /models, /logs)
- Minimal network overhead (localhost)
- Streamlit on localhost:8501
- **Same training results, same models**

**OpenEnv/HF Spaces**:
- WebSocket API: `env.step(action)`
- Remote execution (network latency ~50-100ms)
- Stateful sessions (each client gets own env instance)
- **Different interface, same underlying logic**

### Training Comparison:

| Method | Can Train RL? | Performance | Use Case |
|--------|---------------|-------------|----------|
| Local | ✅ Yes | Fastest | Development |
| Docker | ✅ Yes | ~Same | Production local |
| OpenEnv | ❌ Not recommended | Slow (network) | **Evaluation/Demo** |

**Important**: OpenEnv is for **serving** environments, not training. Train locally/Docker, deploy to HF Spaces for evaluation.

## Recommended Workflow

1. **Develop locally**: `python train_quick.py`
2. **Train models**: Docker or local (`train_quick.py --both --standard`)
3. **Test locally**: Streamlit dashboard
4. **Deploy to HF Spaces**: OpenEnv for worldwide access
5. **Evaluate remotely**: Share HF Space URL with researchers

## Next Steps

See:
- `HUGGINGFACE_SPACES_DEPLOYMENT.md` - Detailed deployment guide
- `OPENENV_INTEGRATION.md` - Technical integration details
- `DOCKER_COMPARISON.md` - Local vs Docker vs Cloud comparison

---

**Summary**: OpenEnv makes Pandora accessible via web API on HuggingFace Spaces. Docker Compose works identically to local for training. Use OpenEnv for sharing/evaluation, Docker/local for training.
