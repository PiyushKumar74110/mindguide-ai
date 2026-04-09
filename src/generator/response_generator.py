def generate_response(row):
    msg = f"You seem {row['predicted_state']} (intensity{row['predicted_intensity']:.1f}). "\
    f"Let's {row['recommend_action']} {row['recommend_time']}."

    if row['uncertain_flag']==1:
        msg+= " (Low confidence, please check how you feel.)"
        return msg
    
