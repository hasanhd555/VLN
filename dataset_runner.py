from typing import TypedDict, List, Optional
import json

class DatasetEntry(TypedDict):
    distance: float
    scan: str
    path_id: int
    path: List[str]
    heading: float
    instructions: List[str]
    goals: Optional[any]  # You can replace `any` with a specific type if known


class DatasetRunner:
    def __init__(self,pipeline,output_file):
        self.pipeline = pipeline
        self.output_file=output_file
    def _load_dataset(self, dataset_path: str) -> List[DatasetEntry]:
        with open(dataset_path, 'r', encoding='utf-8') as file:
            data: List[DatasetEntry] = json.load(file)
        return data
    
    def log_entry(self, trajectory, instruction_id):
        entry = {
            "instr_id": instruction_id,
            "trajectory": trajectory  # Assuming trajectory is a list of (viewpoint_id, heading_rads, elevation_rads)
        }

        # Append to the output file
        try:
            with open(self.output_file, 'r+', encoding='utf-8') as file:
                try:
                    existing_data = json.load(file)
                    if not isinstance(existing_data, list):
                        existing_data = []  # Ensure it's a list
                except json.JSONDecodeError:
                    existing_data = []  # Handle empty or corrupted JSON file

                existing_data.append(entry)
                file.seek(0)  # Move to the beginning of the file
                json.dump(existing_data, file, indent=4)
                file.truncate()  # Remove any excess data
                print(f"Ran instruction number: {instruction_id}")
        except FileNotFoundError:
            with open(self.output_file, 'w', encoding='utf-8') as file:
                json.dump([entry], file, indent=4)
        


    def run_split(self):
        #runs agent on each instruction
        paths_dataset:List[DatasetEntry]=self._load_dataset()
        for path in paths_dataset:
            for index, instruction in enumerate(path["instructions"]):
                trajectory = self.pipeline.run(instruction, path["scan"],path["path"][0])
                instruction_id = f"{path['path_id']}_{index}"
                self.log_entry(trajectory,instruction_id)

            
        