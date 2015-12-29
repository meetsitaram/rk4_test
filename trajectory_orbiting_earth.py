import matplotlib.pyplot as plt
from math import cos, sin, pi, sqrt
from numpy import arctan, arange

h = .01          # stepsize (sec)
g_sl = 9.81        # m/sec2
G = 6.672*10**-11
Re = 6378*1000.
Me = 5.976*10**24


def f(t,x0):
    
    x = x0['x']
    y = x0['y']
    r = sqrt(x*x + y*y)
    
    g = G*Me/(r*r)
    
    theta = arctan(y/(x+0.00001))*180./pi
    
    if x<0 and y<0:
        theta = 180 - theta     #3rd quadrant
    elif x<0 and y>0:
        theta = theta - 180     #4th quadrant
    
    
    res = {}
    res['x'] = x0['vx']
    res['y'] = x0['vy']
    res['vx'] = -g*cos(pi*theta/180.)
    res['vy'] = -g*sin(pi*theta/180.) 
    
    if x < 0:
        if y < 0:
            res['vx'] =  abs(res['vx']) #3rd quadrant
            res['vy'] =  abs(res['vy']) #3rd quadrant
        else:
            res['vx'] =  abs(res['vx']) #4th quadrant
            res['vy'] =  -abs(res['vy']) #4th quadrant
            
        
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
   
#v0 = 9000       # m/s 
#v_theta = 60     # degrees


x0 = {
    'x':0.,
    'y':600000. + Re,  
    'vx': 7500,
    'vy': 0,
}

ty_rk = []
ty_exact = []
t = 0.          # sec

while t<7000 and sqrt(x0['x']**2 + x0['y']**2) >= Re:   
       
    ty_rk.append( (t, x0) )
                            
    t = t + h
    x0 = RK4(t,x0)
  
rng = x0['x']    
    
fig = plt.figure()
ax1 = fig.add_subplot(111)
trajectory_rk, = ax1.plot([x0i['x']/1000. for ti,x0i in ty_rk], [x0i['y']/1000. for ti,x0i in ty_rk], label='Orbital Trajectory')
earth, = ax1.plot([Re*sin(theta)/1000. for theta in arange(0,2.01*pi, 0.1)], [Re*cos(theta)/1000. for theta in arange(0,2.01*pi, 0.1)], label='Earth')
ax1.set_xlabel('Range (km)')
ax1.set_ylabel('Altitude (km)')
plt.legend(handles=[trajectory_rk, earth], loc='center left')
plt.show()


#print 'range for theta=', str(v_theta), ' :', rng
print t
