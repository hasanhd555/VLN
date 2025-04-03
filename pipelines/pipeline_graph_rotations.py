from pipelines.pipeline import NavigationPipeline

class RotationsPipeline(NavigationPipeline):
    def __init__(self):
        super().__init__()

    
    
    def run(self,instruction,scan,starting_viewpoint,instruction_id):
        """Returns trajectory [(viewpoint_id, heading_rads, elevation_rads),]"""
        pass