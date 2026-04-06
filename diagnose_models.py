#!/usr/bin/env python3
"""
Diagnostic script to see what actions a trained model is taking.
Helps understand why population is collapsing.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO, DQN
from environment.pandora_env import PandoraEnv, ACTIONS
from environment.wrappers import NormalizeObservationWrapper, RewardScalingWrapper, EarlyTerminationWrapper
from collections import Counter

def diagnose_model(model_path, model_type='ppo', max_steps=100):
    """Run a model and track which actions it takes."""
    
    print(f"\n{'='*70}")
    print(f"Diagnosing {model_type.upper()} Model: {model_path}")
    print('='*70)
    
    # Create environment with wrappers
    env = PandoraEnv(seed=42)
    env = NormalizeObservationWrapper(env)
    env = RewardScalingWrapper(env, scale=0.01, clip_range=(-100, 100))
    env = EarlyTerminationWrapper(env, min_population=100)
    
    if model_type.lower() == 'dqn':
        from agents.rl_agent import FlattenActionWrapper
        env = FlattenActionWrapper(env)
    
    # Load model
    try:
        if model_type.lower() == 'ppo':
            model = PPO.load(model_path)
        else:
            model = DQN.load(model_path)
    except Exception as e:
        print(f"\n❌ ERROR: Could not load model: {e}")
        print(f"\nThis is likely an OLD model trained before the improvements.")
        print(f"You need to RETRAIN with: python train_quick.py --both --standard")
        return
    
    obs, _ = env.reset()
    
    action_history = []
    population_history = []
    year_history = []
    reward_history = []
    
    print("\nRunning simulation...")
    print(f"{'Step':<6} {'Year':<7} {'Population':<15} {'Reward':<10} {'Actions Taken'}")
    print('-'*70)
    
    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        # Track data
        year = info.get('year', 0)
        population = info.get('population', 0)
        
        # Get action names
        if model_type.lower() == 'dqn':
            # DQN uses flattened action space
            act_val = int(action) if isinstance(action, (int, float)) else int(action.item())
            a1 = act_val // 12
            a2 = act_val % 12
            action_indices = [a1, a2, a1]
        else:
            # PPO uses MultiDiscrete
            action_indices = action if isinstance(action, list) else action.tolist()
        
        try:
            action_names = [ACTIONS[i] for i in action_indices]
            action_history.extend(action_names)
        except IndexError:
            print(f"\n⚠️  WARNING: Model tried to use invalid action indices {action_indices}")
            print(f"   This confirms the model is OLD (trained for 10 actions, now we have {len(ACTIONS)})")
            action_names = ["INVALID"] * len(action_indices)
        
        population_history.append(population)
        year_history.append(year)
        reward_history.append(reward)
        
        if step % 10 == 0:
            print(f"{step:<6} {year:<7} {population:<15,} {reward:<10.3f} {', '.join(action_names)}")
        
        if done or truncated:
            print(f"\n{'⚠️  EARLY TERMINATION' if population < 100 else '✅ COMPLETED'} at step {step}")
            break
    
    # Analysis
    print("\n" + "="*70)
    print("ACTION USAGE ANALYSIS")
    print("="*70)
    
    action_counts = Counter(action_history)
    total_actions = len(action_history)
    
    print(f"\nTotal actions taken: {total_actions}")
    print(f"\nAction breakdown:")
    for action_name, count in action_counts.most_common():
        percentage = (count / total_actions) * 100
        print(f"  {action_name:<25} {count:>4} times ({percentage:>5.1f}%)")
    
    # Check for new actions
    print("\n" + "="*70)
    print("NEW ACTION USAGE CHECK")
    print("="*70)
    
    new_actions = ['INCREASE_POPULATION', 'POPULATION_CONTROL']
    new_action_usage = {action: action_counts.get(action, 0) for action in new_actions}
    
    if sum(new_action_usage.values()) == 0:
        print("\n❌ Model NEVER used new population control actions!")
        print("   This confirms this is an OLD model that doesn't know about them.")
    else:
        print(f"\n✅ Model used new actions {sum(new_action_usage.values())} times")
        for action, count in new_action_usage.items():
            print(f"   {action}: {count} times")
    
    # Population trend
    print("\n" + "="*70)
    print("POPULATION TREND")
    print("="*70)
    
    if len(population_history) > 0:
        start_pop = population_history[0]
        end_pop = population_history[-1]
        change = end_pop - start_pop
        change_pct = (change / start_pop * 100) if start_pop > 0 else 0
        
        print(f"\nStarting population: {start_pop:,}")
        print(f"Ending population:   {end_pop:,}")
        print(f"Change:              {change:+,} ({change_pct:+.1f}%)")
        
        if end_pop < 100:
            print(f"\n❌ POPULATION COLLAPSED (below 100)")
        elif end_pop < 1000:
            print(f"\n⚠️  POPULATION CRITICAL (below 1,000)")
        elif end_pop < start_pop * 0.5:
            print(f"\n⚠️  POPULATION DECLINING RAPIDLY")
        else:
            print(f"\n{'✅' if end_pop > start_pop else '⚠️'}  Population {'growing' if end_pop > start_pop else 'stable/declining'}")
    
    # Reward analysis
    print("\n" + "="*70)
    print("REWARD ANALYSIS")
    print("="*70)
    
    total_reward = sum(reward_history)
    avg_reward = total_reward / len(reward_history) if reward_history else 0
    
    print(f"\nTotal reward:   {total_reward:.2f}")
    print(f"Average reward: {avg_reward:.3f}")
    print(f"Min reward:     {min(reward_history):.3f}")
    print(f"Max reward:     {max(reward_history):.3f}")
    
    if total_reward < 0:
        print(f"\n❌ NEGATIVE total reward - model is performing VERY POORLY")
    elif total_reward < 20:
        print(f"\n⚠️  LOW total reward - model needs more training")
    else:
        print(f"\n✅ Positive total reward - model is learning!")
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    if "INVALID" in action_history or sum(new_action_usage.values()) == 0:
        print("\n🔴 This is an OLD MODEL that needs retraining!")
        print("\n   The model was trained BEFORE the improvements with:")
        print("   - Only 10 actions (now we have 12)")
        print("   - Simple reward function (now multi-objective)")
        print("   - No population control strategies")
        print("\n   🚀 SOLUTION: Retrain with new configuration:")
        print("      python train_quick.py --both --standard")
    elif end_pop < 1000:
        print("\n⚠️  Model is NEW but needs MORE TRAINING")
        print("\n   The model knows about new actions but hasn't learned good strategies yet.")
        print("\n   🚀 SOLUTION: Train longer:")
        print("      python train_quick.py --both --long")
    else:
        print("\n✅ Model looks healthy! Continue training if you want better results.")
    
    print("="*70 + "\n")

if __name__ == "__main__":
    print("\n🔍 MODEL DIAGNOSTIC TOOL")
    print("="*70)
    print("This tool analyzes what actions your trained models are taking")
    print("and helps identify if they need retraining.")
    print("="*70)
    
    # Diagnose PPO
    if os.path.exists("models/pandora_ppo.zip"):
        diagnose_model("models/pandora_ppo", model_type='ppo', max_steps=100)
    else:
        print("\n⚠️  No PPO model found at models/pandora_ppo.zip")
    
    # Diagnose DQN
    if os.path.exists("models/pandora_dqn.zip"):
        diagnose_model("models/pandora_dqn", model_type='dqn', max_steps=100)
    else:
        print("\n⚠️  No DQN model found at models/pandora_dqn.zip")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\nIf your models are OLD or performing poorly:")
    print("  1. Delete old models: rm models/pandora_*.zip")
    print("  2. Train new models:  python train_quick.py --both --standard")
    print("  3. Wait ~1-2 hours for training to complete")
    print("  4. Test again:        python eval/test_ppo.py")
    print("="*70 + "\n")
