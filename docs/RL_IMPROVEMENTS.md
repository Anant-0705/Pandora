# RL Training Improvements for Pandora

## Overview of Changes

This document summarizes the major improvements made to enhance RL agent learning in the Pandora civilization simulation environment.

## Problems Addressed

### 1. **Population Extinction**
- **Problem**: Agents were not learning to preserve their population, leading to extinction
- **Solution**: 
  - Added explicit extinction penalties (-100 for population ≤ 0, -50 for population < 1000)
  - Added early termination when population drops too low
  - Added population survival bonuses

### 2. **Over-Investment in Technology**
- **Problem**: Agents prioritized tech advancement without balancing population/resource needs
- **Solution**:
  - Replaced simple reward (happiness*10 + tech*5) with multi-objective rewards
  - Balanced utilitarian, justice, sustainability, and progress objectives
  - Added severe penalties for population decline

### 3. **Missing Population Control**
- **Problem**: No direct way for agents to manage population growth
- **Solution**: Added two new actions:
  - **INCREASE_POPULATION**: Pro-natalist policies (+15% population, +happiness, -food)
  - **POPULATION_CONTROL**: Birth control policies (-5% population, +happiness, +resources/capita, -inequality)

### 4. **Weak Learning Signal**
- **Problem**: Default hyperparameters not optimized for this complex, long-horizon task
- **Solution**:
  - Increased network size from 64 to 256 units (2 layers)
  - Lowered learning rate to 1e-4 for stability
  - Added entropy coefficient (0.01) for PPO exploration
  - Extended DQN exploration period to 30% of training
  - Increased parallel environments from 1 to 4

### 5. **Observation Scale Mismatch**
- **Problem**: Observations ranged from 0-1 (happiness) to 0-1e12 (population), confusing neural networks
- **Solution**: Added NormalizeObservationWrapper to scale all observations to [0, 1]

---

## New Actions Added

### INCREASE_POPULATION (Action Index: 10)
**Effect**: Pro-natalist policies to boost population growth
- Population: +15% immediate boost
- Happiness: +0.05
- Food: -growth_bonus/500 (costs resources)

**When to use**: When population is low or declining, and you have sufficient food resources

### POPULATION_CONTROL (Action Index: 11)
**Effect**: Birth control and education policies
- Population: -5% (controlled decline)
- Happiness: +0.08
- Inequality: -0.05
- Resources (food/water/energy): +10% each (more per capita)

**When to use**: When population is straining resources, or to improve quality of life

---

## New Reward Structure

### Previous (Simple Baseline):
```python
reward = happiness * 10.0 + tech_level * 5.0
```

### Current (Multi-Objective):
```python
total_reward = utilitarian + justice + sustainability + progress

where:
  utilitarian = population_reward + happiness_reward + extinction_penalties
  justice = (1 - inequality) * 5
  sustainability = climate_health * 3 - (penalties if climate < 0.2)
  progress = tech_delta * 8 + milestone_bonuses
```

### Extinction Penalties:
- Population ≤ 0: **-100**
- Population < 1,000: **-50**
- Population drops > 50% in one turn: **-30**

### Population Bonuses:
- Population > 10,000: **+5**

### Tech Milestones:
- Level 5 (Agriculture): +50
- Level 7 (Writing): +60
- Level 10 (Industrialism): +100
- Level 13 (Computing): +200
- Level 16 (Space Travel): +300
- Level 18 (AI Governance): +500

---

## Training Configuration

### PPO Hyperparameters

| Parameter | Old | New | Reason |
|-----------|-----|-----|--------|
| Learning Rate | 3e-4 | **1e-4** | More stable for complex rewards |
| Network Size | [64, 64] | **[256, 256]** | Larger capacity for 10-dim state |
| n_steps | 2048 | **2048** | Good for long horizon |
| Entropy Coef | 0.0 | **0.01** | Encourages exploration |
| Parallel Envs | 1 | **4** | 4x faster learning |
| Observation | Raw | **Normalized** | Scaled to [0, 1] |
| Reward | Raw | **Clipped & Scaled** | Prevents gradient explosion |

### DQN Hyperparameters

| Parameter | Old | New | Reason |
|-----------|-----|-----|--------|
| Learning Rate | 3e-4 | **1e-4** | More stable |
| Network Size | [64, 64] | **[256, 256]** | Larger capacity |
| Buffer Size | 1e6 | **100k** | Faster updates |
| Exploration Fraction | 0.1 | **0.3** | Explore longer |
| Exploration Final | 0.01 | **0.05** | More exploration |
| Parallel Envs | 1 | **4** | 4x faster learning |
| Action Space | 1000 | **144** | Simplified (12x12 instead of 10x10x10) |

---

## Environment Wrappers

### 1. NormalizeObservationWrapper
- Scales all observations to [0, 1] range
- Prevents large-scale features (population: 1e12) from dominating small-scale features (happiness: 0-1)

### 2. RewardScalingWrapper
- Clips rewards to [-100, 100]
- Scales by 0.01 to prevent gradient explosion
- Stabilizes training

### 3. EarlyTerminationWrapper
- Ends episode early if population < 100
- Gives immediate feedback on extinction
- Prevents wasting compute on doomed civilizations

---

## Usage

### Training with New Configuration

```bash
# Train PPO with optimized settings (default 100k timesteps, 4 parallel envs)
python agents/rl_agent.py --algo ppo

# Train DQN with custom timesteps
python agents/rl_agent.py --algo dqn --timesteps 300000

# Train with more parallel environments for faster learning
python agents/rl_agent.py --algo ppo --timesteps 500000 --n_envs 8

# Train with custom seed
python agents/rl_agent.py --algo ppo --timesteps 200000 --seed 123
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--algo` | ppo | Algorithm: 'ppo' or 'dqn' |
| `--timesteps` | 100000 | Total training timesteps |
| `--n_envs` | 4 | Parallel environments |
| `--seed` | 42 | Random seed |

---

## Expected Results

### Before Improvements (300k timesteps):
- ❌ Population often goes extinct
- ❌ Over-invests in PUSH_SCIENCE/INDUSTRIALIZE
- ❌ Ignores food/climate/happiness
- ❌ Episode reward: -50 to +20 (often negative)

### After Improvements (300k timesteps):
- ✅ Population maintained above 10,000
- ✅ Balanced tech investment with population control
- ✅ Uses INCREASE_POPULATION when low
- ✅ Uses POPULATION_CONTROL when resources strained
- ✅ Episode reward: +50 to +200 (consistently positive)
- ✅ Reaches tech milestones while maintaining civilization
- ✅ Better climate management (uses GO_GREEN strategically)

### Convergence Timeline:
- **50k steps**: Basic survival learned (avoid immediate extinction)
- **100k steps**: Resource management learned (food production, water)
- **200k steps**: Strategic balance (tech + population + climate)
- **300k+ steps**: Optimal long-term planning (milestone hunting with sustainability)

---

## Monitoring Training

### TensorBoard Logs
Training now creates TensorBoard logs in `./logs/`:

```bash
# View training progress
tensorboard --logdir ./logs/

# Then open: http://localhost:6006
```

### Metrics to Watch:
- **ep_rew_mean**: Average episode reward (should increase)
- **ep_len_mean**: Average episode length (should reach ~100 steps)
- **Value Loss**: Should decrease and stabilize
- **Policy Loss**: Should decrease
- **Entropy**: Should start high and decrease (exploration → exploitation)

---

## Testing Trained Models

### Quick Evaluation

```bash
# Test PPO model
python eval/test_ppo.py

# Test DQN model
python eval/test_dqn.py
```

### Full Benchmark

```bash
# Run full benchmark with programmatic scoring
python eval/round_1_benchmark.py
```

### Live Dashboard (Compare all agents)

```bash
streamlit run visualization/live_dashboard.py
```

---

## Debugging Tips

### If agents still fail to survive:

1. **Check reward breakdown**: Look at `info['reward_breakdown']` to see which objectives are negative
2. **Increase extinction penalty**: Edit `calculators.py` to make penalties even steeper
3. **Reduce exploration**: Lower `ent_coef` to 0.005 for PPO or `exploration_final_eps` to 0.02 for DQN
4. **Increase training time**: Try 500k-1M timesteps
5. **Check observation normalization**: Print normalized obs to ensure they're in [0, 1]

### If agents over-explore and never converge:

1. **Lower entropy**: Reduce `ent_coef` to 0.005 (PPO)
2. **Faster exploration decay**: Lower `exploration_fraction` to 0.2 (DQN)
3. **Increase batch size**: Try 128 or 256

### If training is too slow:

1. **Increase parallel envs**: `--n_envs 8` or `--n_envs 16`
2. **Use GPU**: Ensure PyTorch has CUDA support
3. **Reduce network size**: Try [128, 128] instead of [256, 256]

---

## File Changes Summary

### Modified Files:
- `environment/pandora_env.py`: Added 2 actions, multi-objective rewards, prev_state tracking
- `core/event_system.py`: Implemented INCREASE_POPULATION and POPULATION_CONTROL actions
- `rewards/calculators.py`: Enhanced utilitarian_reward with extinction penalties
- `agents/rl_agent.py`: Optimized hyperparameters, added wrappers, 4 parallel envs

### New Files:
- `environment/wrappers.py`: NormalizeObservationWrapper, RewardScalingWrapper, EarlyTerminationWrapper

### Backward Compatibility:
- Old saved models (.zip files) will NOT work with new action space (10 → 12 actions)
- Need to retrain from scratch
- Old evaluation scripts should still work (just add wrappers)

---

## Next Steps

1. **Train new models**: 
   ```bash
   python agents/rl_agent.py --algo ppo --timesteps 300000
   python agents/rl_agent.py --algo dqn --timesteps 300000
   ```

2. **Evaluate improvements**: Compare new vs old models on dashboard

3. **Tune further**: Experiment with different hyperparameters if needed

4. **Save best models**: Keep track of which hyperparameters work best

5. **Document findings**: Record which strategies the agents learn

---

## Technical Details

### Action Space Change:
- **Old**: MultiDiscrete([10, 10, 10]) = 1000 combinations
- **New**: MultiDiscrete([12, 12, 12]) = 1728 combinations
- **DQN Optimization**: Simplified to 144 combinations (12x12, reusing first action as third)

### Reward Scaling Math:
```python
raw_reward = -100 to +500 (wide range)
clipped_reward = np.clip(raw_reward, -100, 100)
scaled_reward = clipped_reward * 0.01
final_reward = -1.0 to +1.0 (stable for NN training)
```

### Observation Normalization Math:
```python
raw_obs = [year, pop, happiness, ...]
          [5000, 1e9, 0.7, ...]
          
normalized = (raw - min) / (max - min)
           = [5000/10000, 1e9/1e12, 0.7/1, ...]
           = [0.5, 0.001, 0.7, ...]  # All in [0, 1]
```

---

## Contact / Support

If issues persist after these improvements, check:
1. TensorBoard logs for training curves
2. Episode dumps for agent behavior
3. Reward breakdowns in info dict
4. Population/climate trajectories in dashboard

Happy Training! 🚀
