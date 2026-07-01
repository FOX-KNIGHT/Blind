# Risk Categories
RISK_LOW = "Low"
RISK_MEDIUM = "Medium"
RISK_HIGH = "High"
RISK_CRITICAL = "Critical"

LETHAL_CLASSES = {"car", "bus", "truck", "motorcycle", "train", "airplane"}

def assess_risk(label, distance_z, time_to_collision, will_collide):
    """
    Categorizes the danger level based on distance, TTC, and object lethality.
    """
    is_lethal = label in LETHAL_CLASSES
    
    if not will_collide:
        # Even if not on a direct collision course, objects close by are High/Medium risk
        if distance_z < 2.0 and is_lethal:
            return RISK_HIGH
        if distance_z < 1.5:
            return RISK_HIGH
        if distance_z < 2.5:
            return RISK_MEDIUM
        return RISK_LOW
    
    if is_lethal:
        if time_to_collision < 5.0:
            return RISK_CRITICAL
        return RISK_HIGH
        
    # Standard objects (people, chairs, obstacles)
    if time_to_collision < 3.0 or distance_z < 2.0:
        return RISK_CRITICAL
    elif time_to_collision < 5.0 or distance_z < 3.5:
        return RISK_HIGH
    else:
        return RISK_MEDIUM
