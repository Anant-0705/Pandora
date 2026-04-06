# Pandora: RL-Driven Civilization Simulation

A reinforcement learning environment for training AI agents to govern civilizations through thousands of years, balancing population, resources, technology, equality, and climate.

## 🎯 Overview

Pandora simulates a civilization across 10,000 years where AI agents make strategic decisions to maximize population survival and societal wellbeing. The environment supports multiple agent types:

- **RL Agents (PPO/DQN)**: Neural network policies trained via reinforcement learning
- **LLM Agent**: GPT-based agent using Groq API for natural language reasoning
- **Random Agent**: Baseline for comparison

## ✨ Features

### 12 Strategic Actions

**Resource Management:**
- `BOOST_FOOD_PRODUCTION` - Increase food (+30%, hurts climate)
- `INVEST_IN_WATER` - Improve water supply (+50%)
- `EXPAND_TERRITORY` - Gain all resources (+20%, risks conflict)

**Social Policies:**
- `PROMOTE_EQUALITY` - Reduce inequality (-10%)
- `ENFORCE_ORDER` - Resolve conflicts (hurts happiness)
- `ENCOURAGE_RELIGION` - Boost morale (+10% happiness)

**Technology:**
- `PUSH_SCIENCE` - Incremental tech progress (+1 level)
- `INDUSTRIALIZE` - Rapid advancement (+2 levels, heavy climate cost)
- `GO_GREEN` - Restore climate health (+10%, reduces energy)
- `INSTALL_AI_GOVERNANCE` - Switch to AI-controlled state

**Population Control (NEW):**
- `INCREASE_POPULATION` - Pro-natalist policies (+15% growth, costs food)
- `POPULATION_CONTROL` - Birth control (-5% pop, improves per-capita resources)

### Multi-Objective Rewards

The environment uses a sophisticated reward function balancing:
- **Utilitarian**: Population × Happiness (with extinction penalties)
- **Justice**: Equality factor (1 - inequality)
- **Sustainability**: Climate health
- **Progress**: Technology milestones

### Enhanced Training

- **Observation Normalization**: All observations scaled to [0,1]
- **Reward Scaling**: Clipped and scaled for stable gradients
- **Early Termination**: Ends episodes on extinction
- **Parallel Environments**: 4 environments for faster training
- **Optimized Hyperparameters**: Tuned for long-horizon tasks

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repo-url>
cd pandora

# Install dependencies
pip install -r requirements.txt

# Create .env file with Groq API key
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### Training Models

```bash
# Quick test (50k timesteps, ~5 min)
python train_quick.py --both --test

# Standard training (PPO: 1M, DQN: 300k, ~3 hours)
python train_quick.py --both --standard

# Long training (PPO: 1.5M, DQN: 500k, ~4 hours)
python train_quick.py --both --long

# Train individual algorithms
python agents/rl_agent.py --algo ppo --timesteps 1000000 --n_envs 4
python agents/rl_agent.py --algo dqn --timesteps 300000 --n_envs 4
```

### Testing Models

```bash
# Test PPO (stochastic policy required!)
python eval/test_ppo.py

# Advanced PPO testing (both deterministic and stochastic)
python eval/test_ppo_advanced.py

# Test DQN
python eval/test_dqn.py
```

### Running Dashboard

```bash
# Local
streamlit run visualization/live_dashboard.py

# Docker
docker-compose up pandora-dashboard

# Access at http://localhost:8501
```

## 🐳 Docker Deployment

See [DOCKER_GUIDE.md](DOCKER_GUIDE.md) for comprehensive Docker instructions.

Quick commands:
```bash
# Start dashboard
docker-compose up -d pandora-dashboard

# Train models
docker-compose --profile training up pandora-train-both

# Stop services
docker-compose down
```

## 📊 Project Structure

```
pandora/
├── agents/
│   ├── rl_agent.py          # PPO/DQN training script
│   ├── llm_agent.py         # Groq-powered LLM agent (UPDATED)
│   └── random_agent.py      # Baseline agent
├── core/
│   ├── civilization.py      # Core simulation logic
│   ├── event_system.py      # Actions and events (NEW: population control)
│   └── history_logger.py    # Event logging
├── environment/
│   ├── pandora_env.py       # Gymnasium environment (12 actions)
│   ├── multi_agent_env.py   # Multi-agent wrapper
│   └── wrappers.py          # Normalization, scaling, termination
├── rewards/
│   └── calculators.py       # Multi-objective reward functions
├── grader/
│   └── llm_grader.py        # LLM-based evaluation
├── eval/
│   ├── test_ppo.py          # PPO evaluation
│   ├── test_ppo_advanced.py # Deterministic vs stochastic testing
│   └── test_dqn.py          # DQN evaluation
├── visualization/
│   └── live_dashboard.py    # Streamlit dashboard
├── models/                  # Saved models (.zip files)
├── logs/                    # TensorBoard logs
├── docs/                    # Documentation
├── train_quick.py           # Easy training script
├── diagnose_models.py       # Model analysis tool
└── Dockerfile              # Docker configuration (UPDATED)
```

## 🔬 Key Insights

### Why PPO Needs More Training

PPO is an **on-policy** algorithm that learns from fresh experiences. For long-horizon tasks (100 steps × 10,000 years), it needs:
- **1M+ timesteps** (vs 300k for DQN)
- **Higher entropy** (0.02) for exploration
- **Stochastic policy** during deployment

DQN works faster because:
- **Off-policy** learning from replay buffer
- More sample efficient for discrete actions
- Deterministic policy works fine

### Stochastic vs Deterministic

**CRITICAL:** PPO must use `deterministic=False` during inference!

```python
# ✅ CORRECT (stochastic)
action, _ = ppo_model.predict(obs, deterministic=False)

# ❌ WRONG (deterministic - will fail!)
action, _ = ppo_model.predict(obs, deterministic=True)
```

Why? PPO trains with entropy regularization, learning to explore. Removing randomness during deployment breaks the learned policy.

### Training Times

| Algorithm | Timesteps | Parallel Envs | Time | Success Rate |
|-----------|-----------|---------------|------|--------------|
| PPO       | 1.5M      | 4             | ~3h  | 100% (stochastic) |
| DQN       | 300k      | 4             | ~1h  | 80-90% |

## 🎮 Using the Dashboard

1. **Start dashboard**: `streamlit run visualization/live_dashboard.py`
2. **Select agent** from sidebar: Random, LLM, PPO, or DQN
3. **Watch simulation** unfold over 10,000 years
4. **View metrics**: Population, happiness, equality, tech, climate
5. **Read history**: Natural language event log
6. **Get LLM grading**: Final performance evaluation

## 📈 Model Performance

### PPO (after 1.5M timesteps)

**Stochastic policy:**
- Average reward: +39.03
- Success rate: 100% (reaches year 10,000)
- Average population: 9,774

**Deterministic policy:**
- Average reward: -77.64
- Success rate: 0% (crashes at year 8,000)
- Average population: 99

### DQN (after 300k timesteps)

- Average reward: +35-45
- Success rate: 80-90%
- Population: 10k-288k

## 🛠️ Troubleshooting

### Import Errors

All scripts use `sys.path.insert(0, project_root)` to fix imports. If you still get errors:

```bash
# Verify Python can find modules
python test_setup.py
```

### Dashboard Agents Dying

Ensure models were trained with wrappers:
- `NormalizeObservationWrapper` (scales to [0,1])
- `RewardScalingWrapper` (prevents gradient explosion)
- `EarlyTerminationWrapper` (feedback on extinction)

Recent models (trained after updates) include these by default.

### PPO Fails but DQN Works

Normal! PPO needs:
1. **More training**: 1M+ timesteps (not 300k)
2. **Stochastic policy**: `deterministic=False`
3. **Proper wrappers**: Train with same wrappers used in deployment

### Low Population

Check if agent is using new population actions:
```bash
python diagnose_models.py
```

If not, retrain with current codebase (includes `INCREASE_POPULATION` and `POPULATION_CONTROL`).

## 📚 Documentation

- [Quick Start Guide](docs/QUICK_START.md)
- [RL Improvements](docs/RL_IMPROVEMENTS.md)
- [PPO Training Issues](docs/PPO_TRAINING_ISSUES.md)
- [Train-Test Mismatch](docs/TRAIN_TEST_MISMATCH.md)
- [Docker Guide](DOCKER_GUIDE.md)
- [Files Updated for Stochastic](FILES_UPDATED_FOR_STOCHASTIC.md)

## 🤝 Contributing

1. Train models: `python train_quick.py --both --long`
2. Test thoroughly: `python eval/test_ppo_advanced.py`
3. Update documentation if adding features
4. Ensure Docker builds: `docker-compose build`

## 📝 License

MIT License

## 🙏 Acknowledgments

Built with:
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) - RL algorithms
- [Gymnasium](https://gymnasium.farama.org/) - RL environments
- [Groq](https://groq.com/) - Fast LLM inference
- [Streamlit](https://streamlit.io/) - Interactive dashboard

---

**Ready to train?**

```bash
python train_quick.py --both --standard
```

Then watch your civilization evolve:

```bash
streamlit run visualization/live_dashboard.py
```

🌍 **Build a civilization that survives the ages!** 🚀
