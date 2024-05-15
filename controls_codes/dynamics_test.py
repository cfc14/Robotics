import numpy as np

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

def get_force():
    u=np.array([[0],[0],[0],[10],[-10]])
    u=np.clip(u,-40,55)
    
    intermediate = np.dot(thrust_allocation_matrix,u)
    Force = np.zeros(6).reshape(6,1)
    
    Force[0,0]=intermediate[0,0]
    Force[1,0]=intermediate[1,0]
    Force[2,0]=intermediate[2,0]
    Force[3,0]=0
    Force[4,0]=intermediate[3,0]
    Force[5,0]=intermediate[4,0]
    return Force

Force = get_force()
