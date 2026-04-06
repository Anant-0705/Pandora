# 🚀 QUICK START GUIDE - Updated Pandora RL Training

## TL;DR - What Changed

Your RL agents now have:
1. ✅ **2 new actions** to control population (INCREASE_POPULATION, POPULATION_CONTROL)
2. ✅ **Better rewards** that heavily penalize extinction and reward survival
3. ✅ **Smarter training** with 4x parallel environments and 4x larger neural networks
4. ✅ **Normalized observations** so agents learn faster
5. ✅ **Optimized hyperparameters** specifically for long-term planning

## 🎯 Start Training NOW

```bash
# Recommended: Train both PPO and DQN with 300k timesteps
python train_quick.py --both --standard

# Or train individually
python agents/rl_agent.py --algo ppo --timesteps 300000
python agents/rl_agent.py --algo dqn --timesteps 300000
```

**Expected training time:**
- PPO: ~30-60 minutes (300k steps, 4 parallel envs)
- DQN: ~45-90 minutes (300k steps, 4 parallel envs)

## 📊 Monitor Progress

```bash
# Open TensorBoard
tensorboard --logdir ./logs/

# Then open browser: http://localhost:6006
```

**What to watch:**
- `ep_rew_mean`: Should increase from ~-20 to +50 or higher
- `ep_len_mean`: Should reach ~100 (full episodes)

## ✅ Test Your Models

```bash
# Test PPO
python eval/test_ppo.py

# Test DQN  
python eval/test_dqn.py
```

**Good results:**
- Final population: 10,000+ (ideally 100k+)
- Final year: 10,000 (completed full simulation)
- Total reward: +50 to +200

**Bad results (need more training):**
- Final population: 0 or <1,000
- Final year: <10,000 (early extinction)
- Total reward: negative or <+20

## 🎮 Compare All Agents

```bash
# Launch interactive dashboard
streamlit run visualization/live_dashboard.py
```

This shows PPO, DQN, and LLM agents running side-by-side!

## 📚 Key Action Indices

| Action | Index | Purpose |
|--------|-------|---------|
| BOOST_FOOD_PRODUCTION | 0 | +30% food (hurts climate) |
| INVEST_IN_WATER | 1 | +50% water |
| EXPAND_TERRITORY | 2 | +20% all resources (risk conflict) |
| PROMOTE_EQUALITY | 3 | -10% inequality |
| ENFORCE_ORDER | 4 | Clear conflicts (hurts happiness) |
| ENCOURAGE_RELIGION | 5 | +10% happiness |
| PUSH_SCIENCE | 6 | +1 tech level |
| INDUSTRIALIZE | 7 | +2 tech (major climate damage) |
| GO_GREEN | 8 | +10% climate (reduces energy) |
| INSTALL_AI_GOVERNANCE | 9 | Switch to AI rule |
| **INCREASE_POPULATION** | **10** | **+15% population (NEW)** |
| **POPULATION_CONTROL** | **11** | **-5% pop, better resources (NEW)** |

## 🎓 Training Tips

### If population keeps dying:
- Agents should learn to use **BOOST_FOOD_PRODUCTION** early
- Use **INCREASE_POPULATION** when pop < 10,000
- Balance tech with survival (don't spam INDUSTRIALIZE)

### If climate collapses:
- Use **GO_GREEN** when climate < 0.5
- Avoid **INDUSTRIALIZE** unless climate > 0.7
- Climate disasters kill 10-30% of population

### If agents explore too randomly:
- Increase training time to 500k or 1M steps
- Lower entropy coefficient in `agents/rl_agent.py`

### If training is too slow:
- Increase `--n_envs` to 8 or 16
- Use GPU (install PyTorch with CUDA)

## 📁 Important Files

| File | Purpose |
|------|---------|
| `agents/rl_agent.py` | Main training script |
| `train_quick.py` | Easy preset training |
| `environment/pandora_env.py` | Simulation environment |
| `environment/wrappers.py` | Normalization & scaling |
| `rewards/calculators.py` | Multi-objective rewards |
| `eval/test_ppo.py` | Test PPO model |
| `eval/test_dqn.py` | Test DQN model |
| `visualization/live_dashboard.py` | Interactive comparison |

## 🔥 Common Commands

```bash
# Quick test (50k steps, ~5-10 min)
python train_quick.py --test

# Standard training (300k steps, ~30-60 min) - RECOMMENDED
python train_quick.py --standard

# Long training (1M steps, ~2-3 hours)
python train_quick.py --long

# Train both algorithms
python train_quick.py --both --standard

# Custom training
python agents/rl_agent.py --algo ppo --timesteps 500000 --n_envs 8

# View logs
tensorboard --logdir ./logs/

# Test model
python eval/test_ppo.py

# Compare all
streamlit run visualization/live_dashboard.py
```

## 🎉 Success Indicators

Your training is working when:
- ✅ TensorBoard shows `ep_rew_mean` increasing
- ✅ Episodes reach 100 steps (year 10,000)
- ✅ Final population > 10,000
- ✅ Agents use diverse actions (not just tech spam)
- ✅ Multiple runs show consistent success

## ⚡ Quick Comparison

| Metric | Before | After |
|--------|--------|-------|
| Actions available | 10 | **12** (+population control) |
| Reward function | Simple | **Multi-objective** |
| Parallel envs | 1 | **4** (4x faster) |
| Network size | 64 units | **256 units** (4x larger) |
| Observations | Raw | **Normalized [0,1]** |
| Rewards | Raw | **Clipped & scaled** |
| Expected success | ~20% | **>80%** |
| Training time | Unpredictable | **~30-60 min for 300k** |

---

**Ready to train? Start here:**
```bash
python train_quick.py --both --standard
```

**Questions? Check:**
- `IMPROVEMENTS_SUMMARY.md` - Detailed explanations
- `RL_IMPROVEMENTS.md` - Technical deep dive
- `plan.md` - Implementation plan

Good luck! 🚀
