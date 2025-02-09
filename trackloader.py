import json
from trackdata import TrackData
from typing import List, Dict, Any

def load_tracks(config_path: str) -> List[TrackData]:
    config = load_config(config_path)
    tracks = []
    for track_data in config["tracks"]:
        tracks.append(TrackData(track_data, config))
    return tracks

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as data_file:
        config = json.load(data_file)
    return config