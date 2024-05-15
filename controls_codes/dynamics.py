import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import controller_los
import controller
import dynamics_test

def dynamics_solve():
    '''
    ###################
    Set Simulation Time
    ###################
    '''

    sim_time=300

    '''
    #####################
    Variable Declarations
    #####################
    '''
    # Present Position and velocity matrices of AUV
    AUV_Velocity=np.array([[0],[0],[0],[0],[0],[0]])
    AUV_Position=np.array([[0],[0],[0],[0],[0],[0]])

    # Previous Position and velocity matrices of AUV
    AUV_prev_velocity=np.array([[0],[0],[0],[0],[0],[0]])
    AUV_prev_position=np.array([[0],[0],[1],[0],[0],[0]])

    #Force given out by each individual thruster
    thruster_force=np.array([[0],[0],[0],[0],[0]])


    '''
    #########################################
    Matrix declarations for Dynamics equation
    #########################################
    '''
    # F = M(d^e/dt^2) + (C+D)(de/dt) + G

    M=np.array([[32+2.8947,0,0,0,0,0],
                [0,32+41.87,0,0,0,2.8947],
                [0,0,32+41.87,0,-2.8947,0],
                [0,0,0,5.7069,0,0],
                [0,0,-2.8947,0,5.7069+12.541,0],
                [0,2.8947,0,0,0,0.3672+12.541]])

    # M=np.array([[32+2.8947,0,0,0,0,0],
    #             [0,32+41.87,0,0,0,0],
    #             [0,0,32+41.87,0,0,0],
    #             [0,0,0,5.7069,0,0],
    #             [0,0,0,0,5.7069+12.541,0],
    #             [0,0,0,0,0,0.3672+12.541]])

    D = np.array([[0.11,0,0,0,0,0],
                [0,0.1534,0,0,0,0],
                [0,0,0.1612,0,0,0],
                [0,0,0,9.604,0,0],
                        [0,0,0,0,79.62,0],
                        [0,0,0,0,0,14.58]])
    D=np.multiply(50,D)

    def get_G(x):
        G = np.array([[0],[0],[0],[0],[(-8.184515*x[4,0])/100],[0]])
        return G

    def get_C(AUV_Velocity):
        m=32
        Ix = 5.7069
        Iy = 5.7069
        Iz = 0.3672
        u = AUV_Velocity[0,0]
        v = AUV_Velocity[1,0]
        w = AUV_Velocity[2,0]
        p = AUV_Velocity[3,0]
        q = AUV_Velocity[4,0]
        r = AUV_Velocity[5,0]

        C= np.array([[0,0,0,0,m*w,m*v*(-1)],
                    [0,0,0,(-1)*m*w,0,m*u],
                    [0,0,0,m*v,(-1)*m*u,0],
                    [0,m*w,(-1)*m*v,0,Iz*r,(-1)*Iy*q],
                    [(-1)*m*w,0,m*u,Iz*r,0,Ix*p],
                    [m*v,(-1)*m*u,0,Iy*q,(-1)*Ix*p,0]])

        return C

    '''
    ###################################
    Declaring the transformation matrix
    ###################################
    '''

    def transformed(X1, v):
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
        return J @ v


    '''
    ##########################
    Defining SolveIVP function
    ##########################
    '''

    def f(t_span,y):
        x, v = np.split(y, 2)
        v=v.reshape((6,1))
        x=x.reshape((6,1))
        v_squared=np.array([[v[0,0]*abs(v[0,0])],
                            [v[1,0]*abs(v[1,0])],
                            [v[2,0]*abs(v[2,0])],
                            [v[3,0]*abs(v[3,0])],
                            [v[4,0]*abs(v[4,0])],
                            [v[5,0]*abs(v[5,0])]])

        c=get_C(v)
        
        #Force = controller_los.LQR_los(x, v, M, D, c)
        #Force = controller.LQR(x, v, M, D, c)
        #Force = controller.PID(x)
        Force = controller_los.PID_los(x)
        #Force = dynamics_test.Force
        
        dvdt = np.dot(np.linalg.inv(M),(Force - np.dot(c,v)-np.dot(D,v_squared)-get_G(x)))
        dxdt=transformed(x,v)
        return np.concatenate([dxdt.flatten(),dvdt.flatten()])

    t_span=np.array([0,sim_time])

    A0 = np.concatenate([AUV_prev_position, AUV_prev_velocity], axis=0).flatten()


    '''
    ###############
    Solution matrix
    ###############
    '''
    solution = solve_ivp(f,t_span,A0)
    solution_matrix = solution.y
    solution_time = solution.t 
    return solution_matrix,solution_time,sim_time
