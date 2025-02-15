import json

def load_max_translations(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

