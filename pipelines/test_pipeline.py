from pipeline import NavigationPipeline

class TestPipeline(NavigationPipeline):
    def __init__(self,json_file):
        super().__init__()
        self.json_file=json_file
        #read json file

    
    
    def run(self,instruction,scan,starting_viewpoint,instruction_id):
        """Returns trajectory [(viewpoint_id, heading_rads, elevation_rads),]"""
        #
        pass