def decision_action(pred_state, intensity, stress, energy, time_of_day):
    action = "pause"
    when = "now"

    if pred_state in ['stressed', 'anxious']:
        if intensity>=4:
            action = 'rest' if energy<=2 else 'box_breathing'
        else:
            action = 'journaling'
    elif pred_state in ['calm','relaxed']:
        action = 'deep_work' if energy>=4 else 'light_planning'

    if stress>=4 and energy<=2:
        action = 'rest'
    elif stress<=2 and energy >=4:
        action = 'movement'

    if action in ['rest', 'yoga', 'sound_therapy']:
        when = 'tonight' if time_of_day in [2,3] else 'later_today'
    elif action in ['deep_work', 'journaling', 'movement']:
        when = 'now' if time_of_day in [0,1] else 'later_today'
    else:
        when = 'within_15_min'

    return action,when


    

