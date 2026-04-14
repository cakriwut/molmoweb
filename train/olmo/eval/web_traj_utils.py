import json 

def fuzzy_match_for_texts(a: str, b: str) -> bool:
    """
    Check if two strings match with a fuzzy logic approach.
    This function allows for minor differences in the strings.
    """
    a = a.lower().strip()
    b = b.lower().strip()
    
    # Simple exact match
    if a == b:
        return True
    # Check if either string is empty
    if len(a) == 0 or len(b) == 0:
        return False
    # Check if one string is a substring of the other
    if a in b or b in a:
        return True
    # Check for common words
    common_words = set(a.split()).intersection(set(b.split()))
    if len(common_words) > 0:
        return True
    return False


def fuzzy_match_for_numbers(a: float, b: float, tolerance: float = 1) -> bool:
    """
    Check if two numbers match within a specified tolerance.
    """
    return abs(a - b) <= tolerance



def web_traj_step_score(pred: str, metadata: dict) -> int:
    score = {
        "format": 0,
        "name": 0,
        "args": 0,
        "values": 0,
    }
    gt_action = json.loads(metadata["answer"])["action"]
    try:
        pred_action = json.loads(pred)["action"]
        score['format'] = 1

        try:
            gt_action_name = gt_action["name"]
            pred_action_name = pred_action["name"]

            if gt_action_name == pred_action_name:
                score["name"] = 1
        except Exception:
            print(f"Error in action name comparison: {pred_action}")
            score["name"] = 0

        try:
            if set(gt_action.keys()) == set(pred_action.keys()):
                score["args"] = 1
        except Exception:
            print(f"Error in action args comparison: {pred_action}")
            score["args"] = 0
        
        try:
            if gt_action_name in ["click", "scroll", "keyboard_press", "goto"]:
                
                if all(gt_action.get(k) == pred_action.get(k) for k in gt_action):
                    score["values"] = 1
                
                elif gt_action_name == "goto":
                    gt_url = gt_action.get("url", "")
                    pred_url = pred_action.get("url", "")
                    if fuzzy_match_for_texts(gt_url, pred_url):
                        score["values"] = 1

                elif gt_action_name == "click":
                    gt_x = gt_action.get("x", 0)
                    gt_y = gt_action.get("y", 0)
                    pred_x = pred_action.get("x", 0)
                    pred_y = pred_action.get("y", 0)
                    if fuzzy_match_for_numbers(gt_x, pred_x) and fuzzy_match_for_numbers(gt_y, pred_y):
                        score["values"] = 1
                
                elif gt_action_name == "scroll":
                    gt_delta_x = gt_action.get("delta_x", 0)
                    gt_delta_y = gt_action.get("delta_y", 0)
                    pred_delta_x = pred_action.get("delta_x", 0)
                    pred_delta_y = pred_action.get("delta_y", 0)
                    if fuzzy_match_for_numbers(gt_delta_x, pred_delta_x) and fuzzy_match_for_numbers(gt_delta_y, pred_delta_y):
                        score["values"] = 1

            
            elif gt_action_name == "keyboard_type":
                gt_text = gt_action.get("text", "")
                pred_text = pred_action.get("text", "")
                if fuzzy_match_for_texts(gt_text, pred_text):
                    score["values"] = 1
        
            elif gt_action_name == "send_msg_to_user":
                gt_msg = gt_action.get("msg", "")
                pred_msg = pred_action.get("msg", "")
                if fuzzy_match_for_texts(gt_msg, pred_msg):
                    score["values"] = 1
            
        except Exception:
            print(f"Error in action values comparison: {pred_action}")
            score["values"] = 0

    except Exception:
        print(f"Format error for prediction: {pred}")
        score['format'] = 0
    
    return score

    
   