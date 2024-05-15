import numpy as np
import guidance
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
#############################################
Body frame Force Calculation LOS Guidance law
#############################################
'''
def get_force_los(control_output):
    f = np.array([[10],
                  [0],
                  [0],
                  [control_output[0,0]],
                  [control_output[1,0]]])
    
    
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
####################################################
PID Gains and Error declaration for LOS Guidance law
####################################################
'''
Kp_los=np.array([[200,0],[0,200]])   
Kd_los=np.array([[80,0],[0,60]])
Ki_los=np.array([[4,0],[0,3]]) 

integral_error_los = np.array([[0],[0]])
prev_error_los = np.array([[0],[0]])

'''
###############################################################
PID function that returns Body Frame Force for LOS Guidance law
###############################################################
'''
def PID_los(x):
    global integral_error_los
    global prev_error_los
    
    dt=0.001
    current =np.array([[x[4,0]],[x[5,0]]])
    desired=guidance.LOS(x)
    error = desired-current         
    integral_error_los = error*dt + integral_error_los
    error_derivative_los = (error-prev_error_los)/dt
    
    control_output = np.dot(Kp_los,error) + np.dot(Ki_los,integral_error_los)+ np.dot(Kd_los, error_derivative_los)
    prev_error_los=np.copy(error)
    
    Force = get_force_los(control_output)
    
    return Force

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

'''
#############################
LQR Control For line of sight
#############################
'''

def LQR_los(x, v, M, D, C): 
    state = np.concatenate([x, v])
    velocity = state[6:12:1, 0]
    velocity = np.expand_dims(velocity, 1)
    
    linear_drag_matrix = D @ np.diag(np.squeeze(np.abs(velocity)))
    X1 = state[0:6:1, 0]
    
    Q = 2000 * np.eye(12)
    R = 1 * np.eye(6)
    
    C = C + linear_drag_matrix
    A, B = state_matrix(state, M, C)
    
    desired= guidance.LOS(state)
    
    K, S, E = ctrl.lqr(A, B, Q, R)
    
    e_state = np.zeros((12, 1))
    e_state[4, 0] = desired[0,0] - X1[4]
    e_state[5, 0] = desired[1,0] - X1[5]
    
    force = K @ e_state
    force = np.delete(force,3,0)
    
    U = (np.linalg.inv(thrust_allocation_matrix)) @ (force)
    U[0,0] = 10
    U=np.clip(U,-80,80)
    
    intermediate = np.dot(thrust_allocation_matrix,U)
    Force = np.zeros((6,1))
    
    Force[0,0] = intermediate[0,0]
    Force[1,0] = intermediate[1,0]
    Force[2,0] = intermediate[2,0]
    Force[3,0] = 0
    Force[4,0] = intermediate[3,0]
    Force[5,0] = intermediate[4,0]
    
    return Force

########################################################################################################
