#!/usr/bin/env python3
"""
Quick start script for training improved RL agents on Pandora.

This script provides easy shortcuts for common training scenarios.
"""

import subprocess
import sys
import argparse

def run_command(cmd):
    """Run command and print output in real-time."""
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print('='*60 + "\n")
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    return process.returncode

def main():
    parser = argparse.ArgumentParser(
        description="Quick start script for Pandora RL training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (50k steps)
  python train_quick.py --test
  
  # Standard training (300k steps)
  python train_quick.py --standard
  
  # Long training (1M steps)
  python train_quick.py --long
  
  # Train both PPO and DQN
  python train_quick.py --both --standard
  
  # Custom configuration
  python train_quick.py --algo dqn --steps 500000 --envs 8
        """
    )
    
    # Preset training modes
    presets = parser.add_mutually_exclusive_group()
    presets.add_argument('--test', action='store_true',
                        help='Quick test run (50k steps, good for testing)')
    presets.add_argument('--standard', action='store_true',
                        help='Standard training (300k DQN / 1M PPO, recommended)')
    presets.add_argument('--long', action='store_true',
                        help='Long training (500k DQN / 1.5M PPO, for best results)')
    
    # Custom configuration
    parser.add_argument('--algo', type=str, choices=['ppo', 'dqn'],
                       help='Algorithm to train (default: ppo)')
    parser.add_argument('--steps', type=int,
                       help='Custom number of timesteps')
    parser.add_argument('--envs', type=int,
                       help='Number of parallel environments (default: 4)')
    parser.add_argument('--both', action='store_true',
                       help='Train both PPO and DQN')
    
    args = parser.parse_args()
    
    # Determine configuration
    if args.test:
        timesteps = 50000
        mode_name = "Test"
    elif args.standard:
        timesteps = 300000  # Will be adjusted per-algorithm
        mode_name = "Standard"
    elif args.long:
        timesteps = 500000  # Will be adjusted per-algorithm
        mode_name = "Long"
    elif args.steps:
        timesteps = args.steps
        mode_name = "Custom"
    else:
        # Default to standard
        timesteps = 300000
        mode_name = "Standard"
    
    n_envs = args.envs if args.envs else 4
    
    algorithms = []
    if args.both:
        algorithms = ['ppo', 'dqn']
    elif args.algo:
        algorithms = [args.algo]
    else:
        algorithms = ['ppo']
    
    print("\n" + "="*60)
    print(f"PANDORA RL TRAINING - {mode_name} Mode")
    print("="*60)
    print(f"Algorithms: {', '.join([a.upper() for a in algorithms])}")
    print(f"Timesteps: {timesteps:,}")
    print(f"Parallel Environments: {n_envs}")
    print("="*60)
    
    # Train each algorithm
    for algo in algorithms:
        # Adjust timesteps per algorithm (PPO needs more!)
        algo_timesteps = timesteps
        if algo == 'ppo' and not args.steps:  # Only adjust if not custom
            if args.test:
                algo_timesteps = 50000
            elif args.standard:
                algo_timesteps = 1000000  # PPO needs 1M for standard
            elif args.long:
                algo_timesteps = 1500000  # PPO needs 1.5M for long
            else:
                algo_timesteps = 1000000  # Default for PPO
        
        cmd = [
            sys.executable,
            'agents/rl_agent.py',
            '--algo', algo,
            '--timesteps', str(algo_timesteps),
            '--n_envs', str(n_envs)
        ]
        
        returncode = run_command(cmd)
        
        if returncode != 0:
            print(f"\n❌ Training failed for {algo.upper()} with code {returncode}")
            return returncode
        else:
            print(f"\n✅ Training completed successfully for {algo.upper()}!")
    
    print("\n" + "="*60)
    print("🎉 All training completed!")
    print("="*60)
    print("\nNext steps:")
    print("1. Test your models:")
    print("   python eval/test_ppo.py")
    print("   python eval/test_dqn.py")
    print("\n2. View training logs:")
    print("   tensorboard --logdir ./logs/")
    print("\n3. Compare agents on dashboard:")
    print("   streamlit run visualization/live_dashboard.py")
    print("="*60 + "\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
