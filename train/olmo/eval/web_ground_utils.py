import json

def web_grounding_score(pred: str, metadata: dict) -> int:
    try:
        bbox = metadata['bbox']
        pred = json.loads(pred)
        if "action" in pred:
            pred = pred["action"]
        x, y = pred['x'], pred['y']
        
        # check if the predicted point is inside the bbox
        if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
            return 1
        else:
            return 0
    except Exception as e:
        print(f"Error parsing prediction: {pred}, error: {e}")
        return 0