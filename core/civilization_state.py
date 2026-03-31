from pydantic import BaseModel, ConfigDict
from enum import Enum
from typing import List, Dict

class PowerType(Enum):
    TRIBAL      = 'tribal'
    MONARCHY    = 'monarchy'
    DEMOCRACY   = 'democracy'
    CORPORATE   = 'corporate'
    AI_GOVERNED = 'ai_governed'

class CivilizationState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    year:             int           = 0
    population:       int           = 100000
    happiness:        float         = 0.5
    inequality:       float         = 0.5
    tech_level:       int           = 1
    power_structure:  PowerType     = PowerType.TRIBAL
    climate_health:   float         = 1.0
    resources:        Dict[str, float] = {"food": 100.0, "water": 100.0, "energy": 10.0, "minerals": 10.0}
    active_conflicts: List[str]     = []
    history_log:      List[str]     = []
    total_score:      float         = 0.0
