import numpy as np
import math
import matplotlib.pyplot as plt
import time
from math import cos, sin, asin, atan2

def euler_to_rotation_matrix(alpha, beta, gamma):
    R_x = np.array([[1, 0, 0],
                     [0, cos(alpha), -sin(alpha)],
                     [0, sin(alpha), cos(alpha)]])

    R_y = np.array([[cos(beta), 0, sin(beta)],
                     [0, 1, 0],
                     [-sin(beta), 0, cos(beta)]])

    R_z = np.array([[cos(gamma), -sin(gamma), 0],
                     [sin(gamma), cos(gamma), 0],
                     [0, 0, 1]])

    return R_z.dot(R_y).dot(R_x)

def rotation_matrix_to_euler(R):
    beta = asin(-R[2,0])
    alpha = atan2(R[2,1]/cos(beta), R[2,2]/cos(beta))
    gamma = atan2(R[1,0]/cos(beta), R[0,0]/cos(beta))
    
    return alpha, beta, gamma

def timeTaken(fxn, params):
    t = time.time()
    res = fxn(**params)
    print(f'Time taken by {fxn}: {time.time() - t}')
    return res

def ssa(angle):
    return np.mod(angle + np.pi, 2 * np.pi) - np.pi

def getLines(bbox_points):
    return [(bbox_points[0], bbox_points[1]),
            (bbox_points[1], bbox_points[2]),
            (bbox_points[2], bbox_points[3]),
            (bbox_points[3], bbox_points[0]),
            ]

def euclidean_distance(point1, point2):
    x1,y1 = point1
    x2,y2 = point2

    return (((x1 - x2) ** 2) + ((y1 - y2) ** 2)) ** 0.5

def getBBoxPoints(center_x, center_y, bbox, yaw):
    cos_angle = math.cos(yaw)
    sin_angle = math.sin(yaw)

    min_x, min_y, min_z, max_x, max_y, max_z = bbox

    corner1_x = center_x + (min_x * cos_angle - min_y * sin_angle)
    corner1_y = center_y + (min_x * sin_angle + min_y * cos_angle)

    corner2_x = center_x + (max_x * cos_angle - min_y * sin_angle)
    corner2_y = center_y + (max_x * sin_angle + min_y * cos_angle)

    corner3_x = center_x + (max_x * cos_angle - max_y * sin_angle)
    corner3_y = center_y + (max_x * sin_angle + max_y * cos_angle)

    corner4_x = center_x + (min_x * cos_angle - max_y * sin_angle)
    corner4_y = center_y + (min_x * sin_angle + max_y * cos_angle)

    return [(corner1_x, corner1_y), (corner2_x, corner2_y), (corner3_x, corner3_y), (corner4_x, corner4_y)]

def plot_rectangle(centroid_x, centroid_y, bbox, yaw, style):

    [(corner1_x, corner1_y), (corner2_x, corner2_y), (corner3_x, corner3_y), (corner4_x, corner4_y)] = getBBoxPoints(centroid_x, centroid_y, bbox, yaw)

    plt.plot([corner1_x, corner2_x], [corner1_y, corner2_y], style)
    plt.plot([corner2_x, corner3_x], [corner2_y, corner3_y], style)
    plt.plot([corner3_x, corner4_x], [corner3_y, corner4_y], style)
    plt.plot([corner4_x, corner1_x], [corner4_y, corner1_y], style)

def checkCollisionBetweenLines(line1, line2):
    x1, y1 = line1[0]
    x2, y2 = line1[1]
    x3, y3 = line2[0]
    x4, y4 = line2[1]
    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if denom == 0:
        return False
    
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom

    if 0 <= ua <= 1 and 0 <= ub <= 1:
        return True
    else:
        return False
    
def checkPointInsideRectangle(point, rectangle, deflate = 0.0):
    centroid_x, centroid_y, min_x, min_y, min_z, max_x, max_y, max_z, yaw  = rectangle
    min_x = min_x + deflate
    min_y = min_y + deflate
    min_z = min_z + deflate
    max_x = max_x - deflate
    max_y = max_y - deflate
    max_z = max_z - deflate

    point_x, point_y = point
    width = max_x - min_x
    height = max_y - min_y
    relative_x = point_x - centroid_x
    relative_y = point_y - centroid_y

    angle_rad = -yaw
    rotated_x = relative_x * math.cos(angle_rad) - relative_y * math.sin(angle_rad)
    rotated_y = relative_x * math.sin(angle_rad) + relative_y * math.cos(angle_rad)

    half_width = width / 2
    half_height = height / 2

    if -half_width <= rotated_x <= half_width and -half_height <= rotated_y <= half_height:
        return True
    else:
        return False