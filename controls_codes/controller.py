import numpy as np
import guidance
import math
import control as ctrl

##################################################################################################
'''
########################
Thrust Allocation matrix
########################
'''
c1=1
c2=1
l2=l3=0.495/2
l4=l5=0.235/2
thrust_allocation_matrix = np.array([[c1,0,0,0,0],
                         [0,c2,c2,0,0],
                         [0,0,0,c2,c2],
                         [0,0,0,-l4*c2,l5*c2],
                         [0,l2*c2,-l3*c2,0,0]])

###################################################################################################
# PID CONTROLS START HERE
################################################################################################### 

'''
##################################
Variable and function Declarations 
##################################
'''
cross_track=[]
z_desired=[]

z_object=8
z_vel = 0

m_pid=0.5
c_pid=-2

def perpendicular(x):
    x0=x[0,0]
    y0=x[1,0]
    distance=(y0-m_pid*x0-c_pid)/((1+m_pid**2)**(0.5))
    return distance

'''
############################
Body frame Force Calculation
############################
'''
def get_force(control_output):
    
    f = np.array([[10],
                  [control_output[0,0]],
                  [control_output[1,0]],
                  [control_output[3,0]],
                  [control_output[2,0]]])
    
    
    thruster_force = np.dot(np.linalg.inv(thrust_allocation_matrix),f)
    #need to cap thrust
    thruster_force=np.clip(thruster_force,-80,80)
    Force = np.zeros((6,1))         
    
    intermediate = np.dot(thrust_allocation_matrix,thruster_force)
               
    Force[0,0] = intermediate[0,0]
    Force[1,0] = intermediate[1,0]
    Force[2,0] = intermediate[2,0]
    Force[3,0] = 0
    Force[4,0] = intermediate[3,0]
    Force[5,0] = intermediate[4,0]
    return Force

'''
###############################
PID Gains and Error declaration
###############################
'''

Kp=np.array([[90,0,0,0],[0,90,0,0],[0,0,400,0],[0,0,0,200]])
Kd=np.array([[50,0,0,0],[0,70,0,0],[0,0,200,0],[0,0,0,100]])
Ki=np.array([[2,0,0,0],[0,2,0,0],[0,0,1,0],[0,0,0,1]]) 

integral_error = np.array([[0],[0],[0],[0]])
prev_error = np.array([[0],[0],[0],[0]])

'''
##########################################
PID function that returns Body Frame Force
##########################################
'''
def PID(x):
    global integral_error
    global prev_error
    global z_object
    global cross_track
    
    dt=0.001
    cross_track.append(perpendicular(x))
    z_desired.append(z_object)
    current =np.array([[perpendicular(x)],[x[2,0]],[x[5,0]],[x[4,0]]])
    desired=np.array([[0],[z_object],[math.atan(m_pid)],[0]])
    error = desired-current
    
    integral_error = error*dt + integral_error
    error_derivative = (error-prev_error)/dt
    
    control_output = np.dot(Kp,error) + np.dot(Ki,integral_error)+ np.dot(Kd , error_derivative)
    prev_error=np.copy(error)
    z_object = z_object+z_vel*0.001
    Force = get_force(control_output)
    
    return Force

###################################################################################################
# LQR CONTROLS START HERE
################################################################################################### 
'''
#####################################
Transformation J matrix earth to Body
#####################################
'''

def J_matrix(state):
    X1 = state[0:6:1, 0]
    X1 = np.expand_dims(X1, 1)
    #print(X1[3,0])
    J1 = np.array([[np.cos(X1[5, 0]) * np.cos(X1[4, 0]),
                    -np.sin(X1[5, 0]) * np.cos(X1[3, 0]) + np.sin(X1[3, 0]) * np.sin(X1[4, 0]) * np.cos(X1[5, 0]),
                    np.sin(X1[5, 0]) * np.sin(X1[3, 0]) + np.sin(X1[4, 0]) * np.cos(X1[5, 0]) * np.cos(X1[3, 0])],
                   [np.sin(X1[5, 0]) * np.cos(X1[4, 0]),
                    np.cos(X1[5, 0]) * np.cos(X1[3, 0]) + np.sin(X1[3, 0]) * np.sin(X1[4, 0]) * np.sin(X1[5, 0]),
                    -np.cos(X1[5, 0]) * np.sin(X1[3, 0]) + np.sin(X1[4, 0]) * np.sin(X1[5, 0]) * np.cos(X1[3, 0])],
                   [-np.sin(X1[4, 0]), np.sin(X1[3, 0]) * np.cos(X1[4, 0]), np.cos(X1[3, 0]) * np.cos(X1[4, 0])]])
    J2 = np.array([[1, np.sin(X1[3, 0]) * np.tan(X1[4, 0]), np.cos(X1[3, 0]) * np.tan(X1[4, 0])],
                   [0, np.cos(X1[3, 0]), -np.sin(X1[3, 0])],
                   [0, np.sin(X1[3, 0]) / np.cos(X1[4, 0]), np.cos(X1[3, 0]) / np.cos(X1[4, 0])]])

    O3 = np.zeros((3, 3))  # 3 x 3 zero matrix
    J = np.concatenate([np.concatenate([J1, O3], axis=1), np.concatenate([O3, J2], axis=1)])
    return J, J1, J2


'''
###############################
to get the state matrix A and B
###############################
'''

def state_matrix(state, M, C):
    J, J1, J2 = J_matrix(state)
    O6 = np.zeros((6, 6))
    A = np.concatenate([np.concatenate([O6, J], axis=1), np.concatenate([O6, -np.linalg.inv(M) @ C], axis=1)])
    B = np.concatenate([O6, np.linalg.inv(M)])

    return A, B


z_obj_lqr = 8
z_vel_lqr = 0

'''
#############################
LQR Control
#############################
'''
m_lqr=1
c_lqr=0

def perpendicular_lqr(x):
    x0=x[0,0]
    y0=x[1,0]
    distance=(y0-m_lqr*x0-c_lqr)/((1+m_lqr**2)**(0.5))
    return distance

def LQR(x, v, M, D, C):  # Lqr control
    global z_obj_lqr, z_vel_lqr
    
    state = np.concatenate([x, v])
    velocity = state[6:12:1, 0]
    velocity = np.expand_dims(velocity, 1)
    
    linear_drag_matrix = D @ np.diag(np.squeeze(np.abs(velocity)))
    
    X1 = state[0:6:1, 0]
    X1 = np.expand_dims(X1, 1)
    V1 = state[6:12:1, 0]
    V1 = np.expand_dims(V1, 1)
    
    Q = 2000 * np.eye(12)
    R = 1 * np.eye(6)
    
    C = C + linear_drag_matrix
    A, B = state_matrix(state, M, C)
    
    distance = -perpendicular_lqr(X1)

    K, S, E = ctrl.lqr(A, B, Q, R)
    
    e_state = np.zeros((12, 1))
    
    e_state[4, 0] = -X1[4,0]
    e_state[5, 0] = np.arctan(1) - X1[5,0]
    e_state[7, 0] = distance - V1[1, 0]
    e_state[8, 0] = z_obj_lqr-X1[2,0] - V1[2, 0]
    e_state[10, 0] =  - V1[4, 0]
    e_state[11, 0] =  - V1[5, 0]
    
    force = K @ e_state
    force = np.delete(force,3,0)
    
    U = (np.linalg.inv(thrust_allocation_matrix)) @ (force)
    U[0,0] = 10
    U = np.clip(U,-80,80)

    z_obj_lqr += z_vel_lqr * 0.001

    intermediate = np.dot(thrust_allocation_matrix,U)
    Force = np.zeros((6,1))
    
    Force[0,0] = intermediate[0,0]
    Force[1,0] = intermediate[1,0]
    Force[2,0] = intermediate[2,0]
    Force[3,0] = 0
    Force[4,0] = intermediate[3,0]
    Force[5,0] = intermediate[4,0]
    
    return Force