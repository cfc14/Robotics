import matplotlib.pyplot as plt
import math
import dynamics

solution_matrix ,solution_time,sim_time=dynamics.dynamics_solve()

x = solution_matrix[0,:]
y = solution_matrix[1,:]
z = solution_matrix[2,:]
pitch = solution_matrix[4,:]*180/math.pi
yaw = solution_matrix[5,:]*180/math.pi

plt.figure(figsize=(10, 10)) 

plt.subplot(2, 2, 1) 
plt.plot(solution_time,z,'b',label='Present z')
plt.title('Z vs time')
plt.legend()
plt.grid()

plt.subplot(2, 2, 2)
plt.plot(solution_time,pitch,'b',label='Present pitch ')
plt.legend()
plt.title('Pitch vs time')
plt.grid()

plt.subplot(2, 2, 3)
plt.plot(solution_time,yaw,'b',label='Present yaw ')
plt.title('Yaw vs time')
plt.legend()
plt.grid()

plt.subplot(2, 2, 4)
plt.plot(x,y,'b',label='AUV motion ')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('X vs Y')
plt.grid()


plt.suptitle('Comparison plot')  
plt.show()
