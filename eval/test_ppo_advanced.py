#!/usr/bin/env python3
"""
Advanced PPO testing that compares deterministic vs stochastic policies
and runs multiple seeds to check consistency.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from environment.pandora_env import PandoraEnv, ACTIONS
from environment.wrappers import NormalizeObservationWrapper, RewardScalingWrapper, EarlyTerminationWrapper
import numpy as np

def test_ppo(model_path="models/pandora_ppo", num_runs=5):
    """Test PPO with both deterministic and stochastic policies."""
    
    print("\n" + "="*70)
    print("COMPREHENSIVE PPO MODEL TESTING")
    print("="*70)
    
    model = PPO.load(model_path)
    
    # Test both deterministic and stochastic
    for policy_type in ["stochastic", "deterministic"]:
        deterministic = (policy_type == "deterministic")
        
        print(f"\n{'='*70}")
        print(f"Testing with {policy_type.upper()} policy")
        print('='*70)
        
        results = []
        
        for run in range(num_runs):
            # Create fresh environment for each run
            env = PandoraEnv(seed=42 + run)  # Different seeds
            env = NormalizeObservationWrapper(env)
            env = RewardScalingWrapper(env, scale=0.01, clip_range=(-100, 100))
            env = EarlyTerminationWrapper(env, min_population=100)
            
            obs, _ = env.reset()
            total_reward = 0
            step_count = 0
            
            for step in range(1000):
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                step_count += 1
                
                if done or truncated:
                    break
            
            final_pop = info.get('population', 0)
            final_year = info.get('year', 0)
            
            results.append({
                'run': run + 1,
                'seed': 42 + run,
                'steps': step_count,
                'final_year': final_year,
                'final_pop': final_pop,
                'total_reward': total_reward
            })
            
            print(f"Run {run+1} (seed {42+run}): Year {final_year:>5} | Pop {final_pop:>10,} | Reward {total_reward:>7.2f} | Steps {step_count}")
        
        # Calculate statistics
        print(f"\n{'-'*70}")
        print(f"STATISTICS ({policy_type.upper()}):")
        print(f"{'-'*70}")
        
        avg_reward = np.mean([r['total_reward'] for r in results])
        std_reward = np.std([r['total_reward'] for r in results])
        avg_pop = np.mean([r['final_pop'] for r in results])
        std_pop = np.std([r['final_pop'] for r in results])
        avg_year = np.mean([r['final_year'] for r in results])
        success_rate = sum(1 for r in results if r['final_year'] == 10000) / len(results) * 100
        
        print(f"Average Reward:     {avg_reward:>8.2f} ± {std_reward:.2f}")
        print(f"Average Population: {avg_pop:>8,.0f} ± {std_pop:,.0f}")
        print(f"Average Final Year: {avg_year:>8,.0f}")
        print(f"Success Rate:       {success_rate:>8.1f}% (reached year 10000)")
        print(f"{'-'*70}")
    
    print("\n" + "="*70)
    print("DIAGNOSIS")
    print("="*70)
    print("\nIf STOCHASTIC performs better than DETERMINISTIC:")
    print("  → Model relies on exploration/randomness during execution")
    print("  → Solution: Use stochastic policy for deployment")
    print("  → Or train longer with higher value function coefficient")
    print("\nIf both perform poorly:")
    print("  → Model hasn't learned a good strategy yet")
    print("  → Solution: Train even longer (2M+ timesteps)")
    print("  → Or adjust hyperparameters (see PPO_TRAINING_ISSUES.md)")
    print("\nIf both perform well:")
    print("  → Model has learned successfully!")
    print("="*70 + "\n")

if __name__ == "__main__":
    if not os.path.exists("models/pandora_ppo.zip"):
        print("\n❌ ERROR: No PPO model found at models/pandora_ppo.zip")
        print("Please train a model first:")
        print("  python train_quick.py --algo ppo --standard")
        exit(1)
    
    test_ppo(num_runs=5)
