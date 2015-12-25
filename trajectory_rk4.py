import matplotlib.pyplot as plt
import math

g = 9.8

def y_exact(v0,t): 
    return (v0*t-(0.5)*g*t**2)


def a(y, v, t):
    return -g
    
def yk1(y,v,t,dt):  
    return dt*v
    
def vk1(y,v,t,dt):    
    return dt*a(y,v,t)
    
def yk2(y,v,t,dt):
    return dt * (v + vk1(y,v,t,dt)/2)
    
def vk2(y,v,t,dt):
    return dt * a( y + yk1(y,v,t,dt)/2, v + vk1(y,v,t,dt)/2, t+dt/2)
    
def yk3(y,v,t,dt):
    return dt * (v + vk2(y,v,t,dt)/2)    
    
def vk3(y,v,t,dt):
    return dt * a( y + yk2(y,v,t,dt)/2, v + vk2(y,v,t,dt)/2, t+dt/2)

def yk4(y,v,t,dt):
    return dt * (v + vk3(y,v,t,dt))    
    
def vk4(y,v,t,dt):
    return dt * a(y + yk3(y,v,t,dt), v + vk3(y,v,t,dt), t+dt)
    
def y_RK4(y,v,t,dt):
    return (1./6) * (yk1(y,v,t,dt) + 2*yk2(y,v,t,dt) + 2*yk3(y,v,t,dt) + yk4(y,v,t,dt))
    
def v_RK4(y,v,t,dt):
    return (1./6) * (vk1(y,v,t,dt) + 2*vk2(y,v,t,dt) + 2*vk3(y,v,t,dt) + vk4(y,v,t,dt))
     
v0 = 300.
t, dt, y, v = 0., .1, 0., v0
ty_rk = []
ty_exact = []
while y >= 0:   
    if abs(round(t) - t) < 1e-5:
        print("y(%2.1f)\t= %4.6f \t error: %4.6g" % ( t, y, abs(y - y_exact(v0,t))))
        ty_rk.append( (t,y) )
        ty_exact.append( (t,y_exact(v0,t)) )
    t, y, v = t + dt, y + y_RK4(y,v,t,dt), v + v_RK4(y,v,t,dt)    
    
    
trajectory_rk, = plt.plot([xx for xx,yy in ty_rk], [yy/1000 for xx,yy in ty_rk], label='Runge Kutta Trajectory')
trajectory_exact, = plt.plot([xx for xx,yy in ty_exact], [yy/1000 for xx,yy in ty_exact], 'ro', label='Exact Trajectory')
plt.xlabel = 'Time (s)'
plt.ylabel = 'Altitude (km)'
plt.axis([0, math.ceil(max([xx for xx,yy in ty_rk])), 0, math.ceil(max([yy for xx,yy in ty_rk])/1000)])
plt.legend(handles=[trajectory_rk, trajectory_exact])
plt.show()