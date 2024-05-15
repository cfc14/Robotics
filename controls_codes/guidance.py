import numpy as np

def LOS(state):
    previous_waypoint = np.array([0, 0, 30])
    current_waypoint = np.array([1000, 1000, 30])
    pw = previous_waypoint
    cw = current_waypoint
    cps = state[0:3:1, 0]

    x1, y1, z1, x2, y2, z2, x3, y3, z3 = pw[0], pw[1], pw[2], cw[0], cw[1], cw[2], cps[0], cps[1], cps[2]
    l = ((x2 - x1) * (x3 - x1) + (y2 - y1) * (y3 - y1) + (z2 - z1) * (z3 - z1)) / (
            (x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    x4, y4, z4 = l * (x2 - x1) + x1, l * (y2 - y1) + y1, l * (z2 - z1) + z1

    intersection_point = np.array([x4, y4, z4])
    delta = 3
    Xp = np.arctan2(y4, x4)
    Vp = np.arctan2(-z4, (x4 ** 2 + y4 ** 2) ** 0.5)
    RXp = np.array([[np.cos(Xp), -np.sin(Xp), 0], [np.sin(Xp), np.cos(Xp), 0], [0, 0, 1]])
    RVp = np.array([[np.cos(Xp), 0, np.sin(Vp)], [0, 1, 0], [-np.sin(Vp), 0, np.cos(Vp)]])
    R = RXp @ RVp
    error_vectors = R.T @ (cps - intersection_point)
    yaw_desired = np.arctan2(-error_vectors[1], delta)
    pitch_desired = np.arctan2(error_vectors[2], ((0.1*delta )** 2 + (error_vectors[1]) ** 2) ** 0.5)
    
    return np.array([[pitch_desired],[yaw_desired]])

