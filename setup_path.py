# Run this to set up your environment for local imports
# This adds the project root to PYTHONPATH

import os
import sys

# Get the project root directory (where this file is located)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Add to Python path if not already there
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print(f"✅ Added {PROJECT_ROOT} to Python path")
print("\nYou can now import modules like:")
print("  from environment.pandora_env import PandoraEnv")
print("  from agents.rl_agent import train_agent")
