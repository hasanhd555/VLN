from abc import ABC, abstractmethod
from typing import List, Tuple

class NavigationPipeline(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def run(self, instruction: str, scan: str, starting_viewpoint: str, 
            instruction_id: str, goal_viewpoint: str) -> List[Tuple[str, float, float]]:
        """Returns trajectory [(viewpoint_id, heading_rads, elevation_rads),]"""
        pass