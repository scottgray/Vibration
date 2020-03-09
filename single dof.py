#Single Degree of Freedom Numerical Simulation
#Scott Harris
#Created on 2/13/2020

import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt

#variables
m = 2.0
k = 2.0
c = 1.0  #critical damping = 2*sqrt(m*k)

F0 = 1.0
delta_t = 0.001
omega = 1.0
time = np.arange(0.0,40.0,delta_t)

#initial state
y = np.array([0,0]) #[velocity,displacement]

A = np.array([[m,0],[0,1]])
B = np.array([[c,k],[-1,0]])
F = np.array([0.0,0.0])

Y = []
force =[]
#time-stepping soluation
for t in time:
    
    if t<= 15:
        F[0] = F0 * np.cos(omega*t)
    else:
        F[0] = 0.0
        
    y = y + delta_t * inv(A).dot(F - B.dot(y))
    Y.append(y[1])
    force.append(F[0])
    
    KE = 0.5 * m * y[0]**2 #kinetic energy
    PE = 0.5 * k * y[1]**2 #potential energy
    
    if t % 1 <= 0.01:
        print('Total Energy: ', KE+PE)
    
    
#plot the result
t = []
for i in time:
    t.append(i)

plt.plot(t,Y)
plt.plot(t,force)
plt.grid(True)
plt.legend(['Displacement','Force'], loc='lower right')

plt.show()

print('Critical Damping: ', np.sqrt((-c**2 + 4*m*k)/(2.0*m)))
print('Natural Frequency: ', np.sqrt(k/m))
