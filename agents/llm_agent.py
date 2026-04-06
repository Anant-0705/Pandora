import os
import json
import logging
from groq import Groq
from environment.pandora_env import ACTIONS

class LLMGodAgent:
    def __init__(self, action_space):
        self.action_space = action_space
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is missing. The LLM Agent cannot play.")
        self.client = Groq(api_key=api_key)
        
    def act(self, observation, info):
        # Parse history log to provide narrative grounding
        history_log = info.get("history_log", [])
        state_obj = info.get("state_obj", None)
        
        if not state_obj:
            return self.action_space.sample() # Fallback if missing state metadata
            
        recent_history = "\n".join(history_log[-10:])
        if not recent_history.strip():
            recent_history = "The dawn of a new civilization."
            
        prompt = f"""
You are the invisible God shaping a simulation. Your goal is to balance the 5 metrics: 
Population, Happiness, Equality, Climate Health, and Tech Level.

Current Year: {state_obj.year}
Population: {state_obj.population:,}
Happiness: {state_obj.happiness:.2f}
Equality Factor (0-1): {1.0 - state_obj.inequality:.2f}
Tech Level: {state_obj.tech_level}
Climate Health: {state_obj.climate_health:.2f}

Recent History Highlights:
{recent_history}

You must pick exactly 3 actions from this list to aggressively shape the world:

RESOURCE ACTIONS:
- BOOST_FOOD_PRODUCTION: +30% food (hurts climate)
- INVEST_IN_WATER: +50% water supply
- EXPAND_TERRITORY: +20% all resources (risk conflict)

SOCIAL ACTIONS:
- PROMOTE_EQUALITY: Reduce inequality by 10%
- ENFORCE_ORDER: Clear conflicts (hurts happiness)
- ENCOURAGE_RELIGION: +10% happiness boost

TECHNOLOGY ACTIONS:
- PUSH_SCIENCE: +1 tech level (incremental)
- INDUSTRIALIZE: +2 tech levels (heavy climate damage)
- GO_GREEN: +10% climate health (reduces energy)
- INSTALL_AI_GOVERNANCE: Switch to AI-governed state

POPULATION CONTROL (NEW):
- INCREASE_POPULATION: +15% population growth (costs food)
- POPULATION_CONTROL: -5% population, +happiness, better resources/capita

Strategic notes:
- If population is LOW or declining, consider INCREASE_POPULATION
- If population is straining resources, consider POPULATION_CONTROL
- Balance tech advancement with population survival
- Climate disasters kill population when climate < 0.4

Return a JSON array of exactly 3 action strings. Output ONLY valid JSON containing the array. Do not include markdown formatting like ```json.
Example response:
["BOOST_FOOD_PRODUCTION", "INCREASE_POPULATION", "PUSH_SCIENCE"]
"""
        try:
            # We use 8b-instant variant because this runs frequently in the RL loop
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
            )
            
            content = response.choices[0].message.content.strip()
            
            # Defensive unpacking in case of markdown wrapping
            if content.startswith("```json"):
                content = content.replace("```json", "", 1).rsplit("```", 1)[0]
            elif content.startswith("```"):
                content = content.replace("```", "", 1).rsplit("```", 1)[0]
                
            chosen_actions = json.loads(content.strip())
            
            # Convert JSON string choices back to our Gym MultiDiscrete format integers
            indices = []
            for a in chosen_actions[:3]:
                if a in ACTIONS:
                    indices.append(ACTIONS.index(a))
                else:
                    indices.append(0) 
                    
            while len(indices) < 3:
                indices.append(0)
                
            return indices
            
        except Exception as e:
            logging.error(f"LLM Agent parsing error: {e}")
            return self.action_space.sample()
