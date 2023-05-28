# missing_process/block_rule/__init__.py

import json
import os

# Path to the directory containing the JSON files
json_directory = os.path.dirname(__file__)

def load_json_file(filename):
    json_path = os.path.join(json_directory, filename)
    with open(json_path) as f:
        return json.load(f)