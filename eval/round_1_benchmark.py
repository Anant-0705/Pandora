import os
import copy
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from environment.pandora_env import PandoraEnv
from agents.random_agent import RandomAgent
from rewards.calculators import compute_total_turn_reward, detect_phoenix_bonus
from grader.llm_grader import evaluate_history_with_groq

def main():
    print("🧬 PANDORA: Round 1 Evaluation Benchmark")
    print("Initializing environment...")
    
    env = PandoraEnv(seed=42)
    agent = RandomAgent(env.action_space)
    
    obs, info = env.reset()
    
    total_programmatic_score = 0
    prev_state = copy.deepcopy(env.engine.state)
    
    print("Simulating 10,000 years with Random Agent...")
    
    # Run for 100 turns (10,000 years)
    for _ in range(100):
        action = agent.act(obs)
        obs, reward, done, truncated, info = env.step(action)
        
        current_state = info['state_obj']
        
        # Calculate our programmatic reward logic
        turn_rewards = compute_total_turn_reward(current_state, prev_state)
        total_programmatic_score += turn_rewards['total']
        
        prev_state = copy.deepcopy(current_state)
        
        if done:
            break
            
    final_state = info['state_obj']
    history = final_state.history_log
    
    phoenix = detect_phoenix_bonus(history)
    total_programmatic_score += phoenix
    
    print("\n--- Simulation Complete ---")
    print(f"Final Year: {final_state.year}")
    print(f"Final Population: {final_state.population:,}")
    print(f"Final Tech Level: {final_state.tech_level}")
    print(f"Total Programmatic Score: {total_programmatic_score:.2f}")
    if phoenix > 0:
        print("  🔥 Phoenix Bonus Achieved! (+500.0)")
        
    print("\n--- LLM Judgment Court ---")
    print("Requesting verdict from Groq...")
    
    verdict = evaluate_history_with_groq(history, final_state.year)
    
    print(f"\nNarrative Score: {verdict.get('score', 0)} / 100")
    print(f"Verdict: {verdict.get('verdict', 'No verdict')}")
    print(f"Most Dramatic Moment: {verdict.get('most_dramatic_moment', 'N/A')}")
    print(f"Dimension Scores:")
    for dim, score in verdict.get('dimension_scores', {}).items():
        print(f"  - {dim}: {score}")

if __name__ == "__main__":
    main()
