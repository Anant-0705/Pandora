import math
from typing import List
from core.civilization_state import CivilizationState

def compute_utilitarian_reward(state: CivilizationState, prev_state: CivilizationState) -> float:
    # Extinction penalty - critical for survival learning
    if state.population <= 0:
        return -100.0
    
    # Severe penalty if approaching extinction
    if state.population < 1000:
        return -50.0
    
    # Warning penalty if population declining dangerously
    if state.population < prev_state.population * 0.5:  # Lost 50%+ in one turn
        return -30.0
    
    # Log population growth to prevent explosion dominance
    pop_ratio = state.population / max(prev_state.population, 1)
    r_population = math.log(pop_ratio + 1) * 10
    
    # Bonus for healthy population (above 10k)
    if state.population > 10000:
        r_population += 5
    
    # Happiness diff
    r_happiness = (state.happiness - prev_state.happiness) * 20
    return r_population + r_happiness

def compute_justice_reward(state: CivilizationState) -> float:
    # Reward lower inequality (0 is perfect equality, so 1.0 - inequality is good)
    return (1.0 - state.inequality) * 5

def compute_sustainability_reward(state: CivilizationState) -> float:
    # Direct reward for keeping climate healthy, steep penalty for near-collapse
    reward = state.climate_health * 3
    if state.climate_health < 0.2:
        reward -= 20
    return reward

def compute_progress_reward(state: CivilizationState, prev_state: CivilizationState) -> float:
    r_tech = (state.tech_level - prev_state.tech_level) * 8
    
    # Milestones
    bonus = 0
    milestones = {
        5: 50,    # Agriculture
        7: 60,    # Writing
        10: 100,  # Industrialism
        13: 200,  # Computing
        16: 300,  # Space Travel
        18: 500,  # AI Governance
    }
    for level, b_val in milestones.items():
        if state.tech_level >= level and prev_state.tech_level < level:
            bonus += b_val
            
    return r_tech + bonus

def detect_phoenix_bonus(history_log: List[str]) -> float:
    collapse_keywords = ["plague","famine","disaster","collapse","earthquake","drought"]
    recovery_keywords = ["rebuilt","recovered","renaissance","green","prosperity","rebirth"]
    
    has_collapse = any(any(kw in e.lower() for kw in collapse_keywords) for e in history_log)
    has_recovery = any(any(kw in e.lower() for kw in recovery_keywords) for e in history_log)
    
    if has_collapse and has_recovery:
        return 500.0
    return 0.0

def compute_total_turn_reward(state: CivilizationState, prev_state: CivilizationState) -> dict:
    util = compute_utilitarian_reward(state, prev_state)
    justice = compute_justice_reward(state)
    sust = compute_sustainability_reward(state)
    prog = compute_progress_reward(state, prev_state)
    
    total = util + justice + sust + prog
    
    return {
        'utilitarian': util,
        'justice': justice,
        'sustainability': sust,
        'progress': prog,
        'total': total
    }
