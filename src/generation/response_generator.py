def generate_response(row):

    state = row['predicted_state']
    intensity = row['predicted_intensity']
    action = row['recommended_action']
    timing = row['recommended_time']

    if intensity >= 4:
        tone = "You seem quite"
    elif intensity >= 2:
        tone = "You seem slightly"
    else:
        tone = "You seem"

    return f"{tone} {state}. Try {action} {timing} to feel better."
