import random
from typing import List
from core.civilization_state import CivilizationState, PowerType

def apply_action(state: CivilizationState, action_name: str) -> CivilizationState:
    """Applies a specific action to mutate the state."""
    new_state = state.model_copy(deep=True)
    
    # RESOURCE ACTIONS
    if action_name == 'BOOST_FOOD_PRODUCTION':
        new_state.resources['food'] *= 1.3
        new_state.climate_health = max(0.0, new_state.climate_health - 0.05)
        new_state.history_log.append(f"Year {new_state.year}: Boosted food production, but slightly degraded the environment.")
    
    elif action_name == 'INVEST_IN_WATER':
        new_state.resources['water'] *= 1.5
        new_state.history_log.append(f"Year {new_state.year}: Invested heavily in water infrastructure.")
        
    elif action_name == 'EXPAND_TERRITORY':
        for r in new_state.resources:
            new_state.resources[r] *= 1.2
        if random.random() < 0.3:
            new_state.active_conflicts.append("Border Skirmish")
        new_state.history_log.append(f"Year {new_state.year}: Expanded territory, increasing both resources and the likelihood of conflict.")

    # SOCIAL ACTIONS
    elif action_name == 'PROMOTE_EQUALITY':
        new_state.inequality = max(0.0, new_state.inequality - 0.1)
        new_state.happiness = max(0.0, new_state.happiness - 0.05) # Angers elite
        new_state.history_log.append(f"Year {new_state.year}: Promoted equality programs. The working class rejoiced, but the elite resisted.")
        
    elif action_name == 'ENFORCE_ORDER':
        new_state.active_conflicts.clear()
        new_state.happiness = max(0.0, new_state.happiness - 0.1)
        new_state.history_log.append(f"Year {new_state.year}: Imposed strict order and ended all conflicts at the cost of personal freedoms.")
        
    elif action_name == 'ENCOURAGE_RELIGION':
        new_state.happiness = min(1.0, new_state.happiness + 0.1)
        new_state.history_log.append(f"Year {new_state.year}: Encouraged state religion, temporarily boosting morale.")

    # TECH/CLIMATE ACTIONS
    elif action_name == 'PUSH_SCIENCE':
        new_state.tech_level += 1
        new_state.history_log.append(f"Year {new_state.year}: Pushed scientific research heavily, advancing to tech level {new_state.tech_level}.")
        
    elif action_name == 'INDUSTRIALIZE':
        new_state.tech_level += 2
        new_state.climate_health = max(0.0, new_state.climate_health - 0.2)
        new_state.history_log.append(f"Year {new_state.year}: Rapid industrialization led to major technological leaps but devastated the climate.")
        
    elif action_name == 'GO_GREEN':
        new_state.climate_health = min(1.0, new_state.climate_health + 0.1)
        new_state.resources['energy'] *= 0.8
        new_state.history_log.append(f"Year {new_state.year}: Transitioned to green energy, prioritizing climate over immediate growth.")
        
    elif action_name == 'INSTALL_AI_GOVERNANCE':
        new_state.power_structure = PowerType.AI_GOVERNED
        new_state.history_log.append(f"Year {new_state.year}: Handed over all civil administration to an advanced AI governance system.")
        
    return new_state

def trigger_random_events(state: CivilizationState) -> CivilizationState:
    new_state = state.model_copy(deep=True)
    
    # Natural Population Growth
    food_ratio = new_state.resources['food'] / max(1.0, new_state.population / 1000.0)
    growth_rate = 0.05 * min(food_ratio, 2.0) * new_state.happiness
    new_state.population = int(new_state.population * (1 + growth_rate))
    
    # Climate events
    if new_state.climate_health < 0.4 and random.random() < 0.4:
        death_rate = random.uniform(0.1, 0.3)
        new_state.population = int(new_state.population * (1 - death_rate))
        new_state.happiness = max(0.0, new_state.happiness - 0.2)
        new_state.resources['food'] *= 0.5
        new_state.history_log.append(f"Year {new_state.year}: Severe climate disaster triggered by prolonged environmental degradation. Lost {int(death_rate*100)}% of the population.")
        
    # Conflict events
    if new_state.inequality > 0.7 and random.random() < 0.4:
        new_state.active_conflicts.append("Class Rebellion")
        new_state.resources['energy'] *= 0.7
        new_state.history_log.append(f"Year {new_state.year}: High inequality sparked a massive class rebellion, devastating infrastructure.")

    # Base resource consumption
    new_state.resources['food'] = max(0.0, new_state.resources['food'] - new_state.population / 1000.0 * 2)
    new_state.resources['water'] = max(0.0, new_state.resources['water'] - new_state.population / 1000.0 * 2)
    
    if new_state.resources['food'] == 0:
        new_state.population = int(new_state.population * 0.9)
        new_state.history_log.append(f"Year {new_state.year}: Famine strikes! The population starves due to lack of food.")
        
    return new_state
