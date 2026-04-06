#!/usr/bin/env python3
"""
Test script to verify all imports work correctly.
Run this before starting training to catch any issues.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*60)
print("Testing Pandora Environment Setup")
print("="*60)

# Test 1: Basic imports
print("\n1. Testing basic imports...")
try:
    import gymnasium as gym
    import numpy as np
    from stable_baselines3 import PPO, DQN
    print("   ✅ Core libraries (gymnasium, numpy, stable-baselines3)")
except ImportError as e:
    print(f"   ❌ Failed: {e}")
    print("   Run: pip install -r requirements.txt")
    sys.exit(1)

# Test 2: Environment imports
print("\n2. Testing environment imports...")
try:
    from environment.pandora_env import PandoraEnv, ACTIONS
    print(f"   ✅ PandoraEnv ({len(ACTIONS)} actions available)")
except ImportError as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

# Test 3: Wrapper imports
print("\n3. Testing wrapper imports...")
try:
    from environment.wrappers import (
        NormalizeObservationWrapper,
        RewardScalingWrapper,
        EarlyTerminationWrapper
    )
    print("   ✅ All wrappers")
except ImportError as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

# Test 4: Core imports
print("\n4. Testing core imports...")
try:
    from core.world_engine import WorldEngine
    from core.civilization_state import CivilizationState
    from core.event_system import apply_action, trigger_random_events
    print("   ✅ Core modules")
except ImportError as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

# Test 5: Reward calculators
print("\n5. Testing reward calculators...")
try:
    from rewards.calculators import compute_total_turn_reward
    print("   ✅ Reward calculators")
except ImportError as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

# Test 6: Create environment
print("\n6. Testing environment creation...")
try:
    env = PandoraEnv(seed=42)
    obs, info = env.reset()
    print(f"   ✅ Environment created (observation shape: {obs.shape})")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

# Test 7: Test with wrappers
print("\n7. Testing wrapped environment...")
try:
    from environment.wrappers import NormalizeObservationWrapper
    env = PandoraEnv(seed=42)
    env = NormalizeObservationWrapper(env)
    obs, info = env.reset()
    print(f"   ✅ Wrapped environment (normalized obs: {obs.min():.2f} to {obs.max():.2f})")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

# Test 8: Test action space
print("\n8. Testing action space...")
try:
    env = PandoraEnv(seed=42)
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    print(f"   ✅ Action executed (reward: {reward:.2f}, year: {info['year']})")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

# Test 9: Test new actions
print("\n9. Testing new population control actions...")
try:
    from environment.pandora_env import ACTIONS
    assert 'INCREASE_POPULATION' in ACTIONS, "INCREASE_POPULATION not found"
    assert 'POPULATION_CONTROL' in ACTIONS, "POPULATION_CONTROL not found"
    print(f"   ✅ New actions present (total: {len(ACTIONS)} actions)")
    print(f"      - INCREASE_POPULATION at index {ACTIONS.index('INCREASE_POPULATION')}")
    print(f"      - POPULATION_CONTROL at index {ACTIONS.index('POPULATION_CONTROL')}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

# Test 10: Test reward system
print("\n10. Testing multi-objective rewards...")
try:
    env = PandoraEnv(seed=42)
    obs, info = env.reset()
    action = [0, 1, 2]  # Some actions
    obs, reward, done, truncated, info = env.step(action)
    
    assert 'reward_breakdown' in info, "reward_breakdown not in info"
    breakdown = info['reward_breakdown']
    
    expected_keys = ['utilitarian', 'justice', 'sustainability', 'progress', 'total']
    for key in expected_keys:
        assert key in breakdown, f"{key} not in reward breakdown"
    
    print(f"   ✅ Multi-objective rewards working")
    print(f"      Total reward: {breakdown['total']:.2f}")
    print(f"      - Utilitarian: {breakdown['utilitarian']:.2f}")
    print(f"      - Justice: {breakdown['justice']:.2f}")
    print(f"      - Sustainability: {breakdown['sustainability']:.2f}")
    print(f"      - Progress: {breakdown['progress']:.2f}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("🎉 All tests passed! Environment is ready for training.")
print("="*60)
print("\nNext steps:")
print("  python train_quick.py --both --standard")
print("  python agents/rl_agent.py --algo ppo --timesteps 300000")
print("="*60 + "\n")
