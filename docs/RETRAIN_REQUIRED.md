# 🔴 CRITICAL: Your Models Need Retraining!

## What's Wrong

Your test results show the **OLD models** that were trained BEFORE all my improvements:

### PPO Results (OLD MODEL):
- ❌ Population: 92,249 → 94 (99.9% death rate!)
- ❌ Died at year 6,600 (only 66% through)
- ❌ Total reward: **-85.01** (VERY BAD)

### DQN Results (OLD MODEL):
- ❌ Population: 91,800 → 2,422 (97.4% death rate)
- ⚠️ Survived to year 10,000 but barely
- ⚠️ Total reward: **15.19** (POOR)

## Why They're Failing

These models were trained with:
1. ❌ Only **10 actions** (now we have 12)
2. ❌ **Simple rewards** that rewarded tech over survival
3. ❌ **No population control** strategies
4. ❌ **Unnormalized observations** (bad learning)
5. ❌ **Poor hyperparameters** (1 env, small network)

They literally **don't know** about the new INCREASE_POPULATION and POPULATION_CONTROL actions!

## 🚀 The Solution: Retrain!

### Step 1: Diagnose Current Models (Optional)

```bash
python diagnose_models.py
```

This will confirm your models are old and show what actions they're taking.

### Step 2: Delete Old Models

```bash
# Windows (Git Bash)
rm models/pandora_*.zip

# Or manually delete files in models/ folder
```

### Step 3: Train NEW Models

```bash
# Recommended: Train both PPO and DQN with new configuration
python train_quick.py --both --standard
```

**What this does:**
- Trains with **12 actions** including population control
- Uses **multi-objective rewards** (survival + justice + sustainability + progress)
- Runs **4 parallel environments** (4x faster learning)
- Uses **256-unit networks** (4x larger, smarter)
- Trains for **300k timesteps** (~30-60 min each)

**Expected time:** ~1-2 hours total for both models

### Step 4: Monitor Training

While training runs, open another terminal:

```bash
# Install tensorboard if you want live monitoring (optional)
pip install tensorboard

# View training progress
tensorboard --logdir ./logs/
```

Open browser to `http://localhost:6006` and watch:
- `ep_rew_mean` should increase over time
- `ep_len_mean` should reach ~100 steps

### Step 5: Test New Models

```bash
python eval/test_ppo.py
python eval/test_dqn.py
```

**Expected results with NEW models:**
- ✅ Population: 10,000+ (maybe 100k+)
- ✅ Final year: 10,000 (full simulation)
- ✅ Total reward: +50 to +200 (POSITIVE!)

## Expected Behavior Timeline

### During Training (watch terminal output):

**0-50k steps:**
- Models exploring randomly
- Some population crashes (learning what NOT to do)

**50k-100k steps:**
- Learning basic survival (food production)
- Population staying above 1,000

**100k-200k steps:**
- Learning resource management
- Using INCREASE_POPULATION when needed
- Population more stable

**200k-300k steps:**
- Strategic play emerges
- Balancing tech + population + climate
- Consistent survival to year 10,000

## Streamlit Dashboard Issue

The dashboard probably crashes because:
1. Old models expect 10 actions, new environment has 12
2. Incompatible observation shapes
3. Model architecture mismatch

**After retraining**, the dashboard should work:

```bash
streamlit run visualization/live_dashboard.py
```

## Quick Commands Reference

```bash
# 1. Check what your current models are doing (diagnostic)
python diagnose_models.py

# 2. Delete old models
rm models/pandora_*.zip

# 3. Train new models (THIS IS WHAT YOU NEED!)
python train_quick.py --both --standard

# 4. (Optional) Monitor training
tensorboard --logdir ./logs/

# 5. After training completes, test
python eval/test_ppo.py
python eval/test_dqn.py

# 6. View in dashboard
streamlit run visualization/live_dashboard.py
```

## Why Population Collapses (in OLD models)

The old models learned to:
1. **Spam PUSH_SCIENCE** (easy tech rewards)
2. **Ignore BOOST_FOOD_PRODUCTION** (no immediate reward)
3. **Never use INCREASE_POPULATION** (didn't exist!)
4. **Let population starve** (no penalty in old rewards)

Result: Population → 0, civilization extinct 💀

## What NEW Models Will Learn

With the improvements, models will learn to:
1. ✅ **Produce food** early (avoids starvation)
2. ✅ **Boost population** when it drops (INCREASE_POPULATION action)
3. ✅ **Control population** when resources strained (POPULATION_CONTROL action)
4. ✅ **Balance tech with survival** (multi-objective rewards)
5. ✅ **Manage climate** (avoid disasters that kill population)

Result: Sustainable civilization to year 10,000 ✅

## Training Progress Indicators

### Good signs during training:
- ✅ `ep_rew_mean` increasing in TensorBoard
- ✅ `ep_len_mean` reaching ~100 steps
- ✅ Terminal shows "rollout/" metrics improving
- ✅ No Python errors/crashes

### Bad signs (need to investigate):
- ❌ Reward stays negative or decreasing
- ❌ Episodes always terminate early
- ❌ Python errors about action space
- ❌ Training crashes repeatedly

## Need Faster Training?

If you want faster results:

```bash
# Use more parallel environments (if you have CPU cores)
python agents/rl_agent.py --algo ppo --timesteps 300000 --n_envs 8

# Or reduce timesteps for quick test (not recommended for final models)
python train_quick.py --test  # Only 50k steps, ~5-10 min
```

## After Retraining

Your new models should show:

**PPO:**
- Population: 50,000+ consistently
- Reaches year 10,000
- Total reward: +100 to +200

**DQN:**
- Population: 10,000+ consistently  
- Reaches year 10,000
- Total reward: +50 to +150

---

## TL;DR - Just Do This

```bash
# Delete old models
rm models/pandora_*.zip

# Train new ones
python train_quick.py --both --standard

# Wait ~1-2 hours

# Test
python eval/test_ppo.py
python eval/test_dqn.py
```

Then your models will actually work! 🎉
