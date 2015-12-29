import matplotlib.pyplot as plt
from math import cos, sin, pi, sqrt
from numpy import arctan

h = .01          # stepsize (sec)
g = 9.81        # m/sec2
G = 6.672*10**-11
Re = 6378*1000.
Me = 5.976*10**24

def f(t,x0):
    
    x = x0['x']
    y = x0['y']
    r = sqrt(x*x + y*y)
    
    g = G*Me/(r*r)
    
    theta = 0
    if x0['x'] == 0 and x0['y'] >= 0:
        theta = 90.
    elif x0['x'] == 0 and x0['y'] < 0:
        theta = -90.
    else:
        theta = arctan(y/x)*180./pi
    
    res = {}
    res['x'] = x0['vx']
    res['y'] = x0['vy']
    res['vx'] = -g*cos(pi*theta/180.)
    res['vy'] = -g*sin(pi*theta/180.) 
            
        
    return res
    
def RK4(t,x0):
    k0 = f(t,x0)
    
    xtemp = {xi : x0[xi] + k0[xi]*(h/2.) for xi in k0}
    k1 = f(t+h/2., xtemp)
    
    xtemp = {xi : x0[xi] + k1[xi]*(h/2.) for xi in k1}
    k2 = f(t+h/2., xtemp)
    
    xtemp = {xi : x0[xi] + k2[xi]*h for xi in k2}  
    k3 = f(t+h, xtemp)
    
    x0 = {xi : x0[xi] + h/6. * (k0[xi] + 2 * k1[xi] + 2 * k2[xi] + k3[xi]) for xi in x0}
    
    return x0
   
v0 = 9000       # m/s 
v_theta = 60     # degrees


x0 = {
    'x':0.,
    'y':Re,  
    'vx': v0*cos(pi*v_theta/180),
    'vy': v0*sin(pi*v_theta/180),
}

ty_rk = []
ty_exact = []
t = 0.          # sec

while t<40000 and sqrt(x0['y']**2 + x0['y']**2) >= Re:   
       
    ty_rk.append( (t, x0) )
                            
    t = t + h
    x0 = RK4(t,x0)
  
rng = x0['x']    
    
fig = plt.figure()
ax1 = fig.add_subplot(111)
trajectory_rk, = ax1.plot([x0i['x'] for ti,x0i in ty_rk], [x0i['y'] for ti,x0i in ty_rk], label='Runge Kutta Trajectory')
ax1.set_xlabel('Range (m)')
ax1.set_ylabel('Altitude (m)')
plt.legend(handles=[trajectory_rk], loc='center left')
plt.show()


print 'range for theta=', str(v_theta), ' :', rng
print t
