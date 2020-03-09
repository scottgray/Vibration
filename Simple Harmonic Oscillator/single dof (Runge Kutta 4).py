#Single Degree of Freedom Numerical Simulation
#Scott Harris
#Created on 2/13/2020

import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt


def F(t):
    F = np.array([0.0,0.0])
    
    if t<= 15:
        F[0] = F0 * np.cos(omega*t)
    else:
        F[0] = 0.0
        
    return F

def G(y,t):
    return inv(A).dot(F(t) - B.dot(y))

def RK4_step(y,t,delta_t):
    K1 = G(y,t)
    K2 = G(y + 0.5*K1*delta_t, t + 0.5*delta_t)
    K3 = G(y + 0.5*K2*delta_t, t + 0.5*delta_t)
    K4 = G(y + K3*delta_t, t + delta_t)
    
    
    return delta_t * (K1 + 2*K2 + 2*K3 + K4)/6

    
#variables
m = 2.0
k = 2.0
c = 1.0  #critical damping = 2*sqrt(m*k)

F0 = 1.0
delta_t = 0.01
omega = 1.0
time = np.arange(0.0,40.0,delta_t)

#initial state
y = np.array([0,0]) #[velocity,displacement]
A = np.array([[m,0],[0,1]])
B = np.array([[c,k],[-1,0]])


Y = []
force =[]
#time-stepping soluation
for t in time:
            
    y = y + RK4_step(y, t, delta_t)
    Y.append(y[1])
    force.append(F(t)[0])
    
    KE = 0.5 * m * y[0]**2
    PE = 0.5 * k * y[1]**2
    
    if t % 1 <= 0.01:
        print('Total Energy: ', KE+PE)
    
    
#plot the result
plt.plot(time,Y)
plt.plot(time,force)
plt.grid(True)
plt.legend(['Displacement','Force'], loc='lower right')

plt.show()

print('Critical Damping: ', np.sqrt((-c**2 + 4*m*k)/(2.0*m)))
print('Natural Frequency: ', np.sqrt(k/m))
