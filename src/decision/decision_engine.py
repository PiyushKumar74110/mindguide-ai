def decide_action(pred_state, intensity, stress, energy, time_of_day):

    if pred_state in ["stressed","anxious"]:
        if intensity >= 4:
            return "box_breathing","now"
        else:
            return "journaling","within_15_min"

    elif pred_state == "sad":
        if energy <= 2:
            return "rest","now"
        else:
            return "movement","within_15_min"

    elif pred_state in ["calm","relaxed"]:
        if energy >= 4:
            return "deep_work","now"
        else:
            return "light_planning","later_today"

    return "pause","now"
