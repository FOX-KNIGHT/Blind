def calculate_ttc(distance_z, velocity_z_mps):
    """
    Calculates Time-To-Collision (TTC).
    distance_z: Current depth in meters.
    velocity_z_mps: Velocity in the Z-axis (meters/second). Positive means moving towards the camera.
    """
    if velocity_z_mps <= 0.1:  # Not moving towards the user significantly
        return float('inf')
        
    ttc = distance_z / velocity_z_mps
    return ttc

def predict_collision(horizontal_deviation_x, distance_z, velocity_x_mps, velocity_z_mps, time_horizon=5.0):
    """
    Predicts if the object will intersect the user's walking path.
    Assumes user is walking straight and occupies X between -0.75m and +0.75m.
    
    Returns: (will_collide, collision_time, intersection_x)
    """
    # If the object is not moving, check if it's already in the user's path
    if abs(velocity_x_mps) < 0.1 and abs(velocity_z_mps) < 0.1:
        if abs(horizontal_deviation_x) < 0.75 and distance_z < 2.5:
            return True, 0.0, horizontal_deviation_x
        return False, float('inf'), horizontal_deviation_x
        
    ttc = calculate_ttc(distance_z, velocity_z_mps)
    
    # We care about collisions happening within our navigation horizon (5 seconds)
    if ttc > time_horizon:
        return False, ttc, 0.0
        
    # Project the X position at the Time-To-Collision
    predicted_x = horizontal_deviation_x + (velocity_x_mps * ttc)
    
    # Check if the predicted X is within the user's bodily corridor (-0.75m to 0.75m)
    if abs(predicted_x) < 0.75 or abs(horizontal_deviation_x) < 0.75:
        return True, ttc, predicted_x
        
    return False, ttc, predicted_x
