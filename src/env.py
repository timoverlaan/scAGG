import json

def get(key: str = None):
    with open('env.json') as f:
        d = json.load(f)
    
    if key is None:
        return d
    
    assert key in d, f"Key {key} not found in \'env.json\'"

    return d[key]