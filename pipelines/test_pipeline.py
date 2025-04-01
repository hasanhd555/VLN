from pipeline import NavigationPipeline

class TestPipeline(NavigationPipeline):
    def __init__(self):
        super().__init__()

    
    
    def run(self,instruction,scan,starting_viewpoint):
        """Returns trajectory [(viewpoint_id, heading_rads, elevation_rads),]"""
        pass