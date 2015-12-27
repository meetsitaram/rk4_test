import matplotlib.pyplot as plt
from math import cos, sin, pi

h = .01          # stepsize (sec)
g = 9.81        # m/sec2

def f(t,x0):
    res = {}
    res['x'] = x0['vx']
    res['y'] = x0['vy']
    res['vx'] = 0
    res['vy'] = -g    
        
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
   
v0 = 100        # m/s 
theta = 30      # degrees

x0 = {
    'x':0.,
    'y':0.,
    'vx': v0*cos(pi*theta/180),
    'vy': v0*sin(pi*theta/180)
}

ty_rk = []
ty_exact = []
t = 0.          # sec

while x0['y'] >= 0:   
       
    ty_rk.append( (t, x0) )
    
    ty_exact.append( (t, {
                            'x':v0*cos(pi*theta/180)*t, 
                            'y':v0*sin(pi*theta/180)*t - (0.5)*g*t**2, 
                            'vx':v0*cos(pi*theta/180), 
                            'vy':v0*sin(pi*theta/180)
                        }) )
                            
    t = t + h
    x0 = RK4(t,x0)
  
rng = x0['x']    
    
fig = plt.figure()
ax1 = fig.add_subplot(111)
trajectory_rk, = ax1.plot([x0i['x'] for ti,x0i in ty_rk], [x0i['y'] for ti,x0i in ty_rk], label='Runge Kutta Trajectory')
trajectory_exact, = ax1.plot([x0i['x'] for ti,x0i in ty_exact], [x0i['y'] for ti,x0i in ty_exact], 'r', label='Exact Trajectory')
ax1.set_xlabel('Range (m)')
ax1.set_ylabel('Altitude (m)')
plt.legend(handles=[trajectory_rk, trajectory_exact])
plt.show()

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
trajectory_rk, = ax2.plot([ti for ti,x0i in ty_rk], [x0i['y'] for ti,x0i in ty_rk], label='Runge Kutta Trajectory')
trajectory_exact, = ax2.plot([ti for ti,x0i in ty_exact], [x0i['y'] for ti,x0i in ty_exact], 'r', label='Exact Trajectory')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Altitude (m)')
plt.legend(handles=[trajectory_rk, trajectory_exact])
plt.show()

print 'range for theta=30 :', rng

#iteratively compute range for each angle and find the maximum range
angle = 0.
h_angle = 1.

optimal_angle = 0.
max_range = 0.
trajectories = []
while angle <= 90:
    
    x0 = {
        'x':0.,
        'y':0.,
        'vx': v0*cos(pi*angle/180),
        'vy': v0*sin(pi*angle/180)
    }
    
    t = 0.          # sec
    
    trajectory = []
    while x0['y'] >= 0:
        
        trajectory.append( (t, x0) )
        
        t = t + h
        x0 = RK4(t,x0)
        
    if x0['x'] >= max_range:
        max_range = x0['x']
        optimal_angle = angle
        
    if angle % 15 == 0:
        trajectories.append( (angle, trajectory) )
    
    angle = angle + h_angle
    
print 'optimal angle:', optimal_angle, ', max_range:', max_range

fig3 = plt.figure()
ax2 = fig3.add_subplot(111)
ax_handles = []
for t in trajectories:
    trajectory_rk, = ax2.plot([x0i['x'] for ti,x0i in t[1]], [x0i['y'] for ti,x0i in t[1]] , label='Theta ' + str(t[0]))
    ax_handles.append(trajectory_rk)
    
ax2.set_xlabel('Range (m)')
ax2.set_ylabel('Altitude (m)')
plt.legend(handles=ax_handles)
plt.show()