from core.civilization_state import CivilizationState
from core.event_system import apply_action, trigger_random_events
from typing import List

class WorldEngine:
    def __init__(self, seed: int = 42):
        self.seed = seed
        import random
        random.seed(self.seed)
        self.state = CivilizationState()

    def step(self, action_names: List[str]) -> CivilizationState:
        """
        Advances the world by 1 turn (100 years), applying the chosen actions,
        triggering random events, and updating the state.
        """
        self.state.year += 100
        
        # Apply agent decisions
        for act in action_names:
            self.state = apply_action(self.state, act)
            
        # Apply world consequences (disasters, population dynamics, consumption)
        self.state = trigger_random_events(self.state)
        
        return self.state
        
    def reset(self) -> CivilizationState:
        self.state = CivilizationState()
        import random
        random.seed(self.seed)
        return self.state
