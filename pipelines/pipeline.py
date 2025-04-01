from abc import ABC, abstractmethod

class NavigationPipeline(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def run(self,instruction,scan,starting_viewpoint,instruction_id):
        """Returns trajectory [(viewpoint_id, heading_rads, elevation_rads),]"""
        pass