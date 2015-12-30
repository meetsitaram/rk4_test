import matplotlib.pyplot as plt
from math import cos, sin, pi, sqrt
from numpy import zeros, arange

h = .01          # stepsize (sec)
g = 9.81        # m/sec2

mass = 0.15    #  kg
ref_area = 0.005   #  m^2
draf_coefficient = 0.4    # drag coefficient
air_density = 1.22  # density of air kg/m^3

POS_X = 0
POS_Y = 1
POS_VX = 2
POS_VY = 3
NUM_VARS = 4

ENABLE_DRAG = True

def get_drag_acceleration(vx, vy):
    if not ENABLE_DRAG:
        return [0., 0.]
    
    v = sqrt(vx**2 + vy**2)
    
    f_drag = (1./2) * air_density * draf_coefficient * ref_area * v**2    
    a_drag = f_drag/mass
    
    return [-a_drag*vx/v, -a_drag*vy/v]
    
    
def f(t,x0):

    [ax_drag, ay_drag] = get_drag_acceleration(x0[POS_VX], x0[POS_VY] )
    
    res = zeros(NUM_VARS)
    res[POS_X] = x0[POS_VX]
    res[POS_Y] = x0[POS_VY]
    res[POS_VX] = ax_drag
    res[POS_VY] = ay_drag - g    
        
    return res
    
def RK4(t,x0):
    k0 = f(t,x0)
    
    xtemp = [x0[xi] + k0i*(h/2.) for xi,k0i in enumerate(k0)]
    k1 = f(t+h/2., xtemp)
    
    xtemp = [x0[xi] + k1i*(h/2.) for xi,k1i in enumerate(k1)]
    k2 = f(t+h/2., xtemp)
    
    xtemp = [x0[xi] + k2i*h for xi,k2i in enumerate(k2)]
    k3 = f(t+h, xtemp)
    
    x0 = [x0i + h/6. * (k0[xi] + 2 * k1[xi] + 2 * k2[xi] + k3[xi]) for xi,x0i in enumerate(x0)]
    
    return x0
   
def get_trajectory(x0, y0, v0, theta):

    args0 = zeros(NUM_VARS)
    args0[POS_X] = x0
    args0[POS_Y] = y0
    args0[POS_VX] = v0*cos(pi*theta/180)
    args0[POS_VY] = v0*sin(pi*theta/180)
    
    
    trajectory = []
    t = 0.          # sec
    
    while args0[POS_Y] >= 0:              
        trajectory.append( (t, args0) )        
        t = t + h
        args0 = RK4(t,args0)
        
    return trajectory

trajectories = []
optimal_theta = 0.
x0,y0,v0 = 0., 0., 50.
max_range = 0.
for angle in arange(0., 90., 1.):
    ENABLE_DRAG = True
    trajectory = get_trajectory(x0, y0, v0, angle) 
    tmp_max_range = max([xoi[POS_X] for ti,xoi in trajectory])
    if tmp_max_range > max_range:
        max_range = tmp_max_range
        optimal_theta = angle
        
    if angle == 45:
        trajectories.append( (angle, trajectory) )        

print 'optimal theta:', optimal_theta, ', range:', max_range
ENABLE_DRAG = True
trajectory_drag = get_trajectory(x0, y0, v0, optimal_theta) 

ENABLE_DRAG = False
trajectory_no_drag = get_trajectory(x0, y0, v0, optimal_theta) 


fig = plt.figure()
ax1 = fig.add_subplot(111)
plot_no_drag, = ax1.plot([x0i[POS_X] for ti,x0i in trajectory_no_drag], [x0i[POS_Y] for ti,x0i in trajectory_no_drag], label='Trajectory Without Drag')
plot_drag, = ax1.plot([x0i[POS_X] for ti,x0i in trajectory_drag], [x0i[POS_Y] for ti,x0i in trajectory_drag], 'r', label='Trajectory Under Drag')
ax1.set_xlabel('Range (m)')
ax1.set_ylabel('Altitude (m)')
plt.legend(handles=[plot_no_drag, plot_drag])
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(111)
plot_no_drag, = ax1.plot([ti for ti,x0i in trajectory_drag], [abs(x0i[POS_VX]) for ti,x0i in trajectory_drag], label='Vx under drag')
plot_drag, = ax1.plot([ti for ti,x0i in trajectory_drag], [abs(x0i[POS_VY]) for ti,x0i in trajectory_drag], 'r', label='Vy under drag')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Speed (m/s)')
plt.legend(handles=[plot_no_drag, plot_drag])
plt.show()


fig3 = plt.figure()
ax2 = fig3.add_subplot(111)
ax_handles = []
for t in trajectories:
    trajectory, = ax2.plot([x0i[POS_X] for ti,x0i in t[1]], [x0i[POS_Y] for ti,x0i in t[1]] , label='Trajectory at Theta ' + str(t[0]))
    ax_handles.append(trajectory)
    
plot_drag, = ax2.plot([x0i[POS_X] for ti,x0i in trajectory_drag], [x0i[POS_Y] for ti,x0i in trajectory_drag], 'r', label='Trajectory at Optimal Theta:' + str(optimal_theta) )
ax_handles.append(plot_drag)
ax2.set_xlabel('Range (m)')
ax2.set_ylabel('Altitude (m)')
plt.legend(handles=ax_handles)
plt.show()

fig3 = plt.figure()
ax2 = fig3.add_subplot(111)
ax_handles = []
for t in trajectories:
    plot_v, = ax2.plot([ti for ti,x0i in t[1]], [sqrt(x0i[POS_VX]**2 + x0i[POS_VY]**2) for ti,x0i in t[1]] , 'b', label='V at Theta ' + str(t[0]))
    ax_handles.append(plot_v)
    
plot_v, = ax2.plot([ti for ti,x0i in trajectory_drag], [sqrt(x0i[POS_VX]**2 + x0i[POS_VY]**2) for ti,x0i in trajectory_drag], 'r', label='V at Optimal Theta:' + str(optimal_theta) )

ax_handles.append(plot_v)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Speed (m/s)')
plt.legend(handles=ax_handles)
plt.show()
#
#
#fig2 = plt.figure()
#ax2 = fig2.add_subplot(111)
#trajectory_rk, = ax2.plot([ti for ti,x0i in ty_rk], [x0i[POS_Y] for ti,x0i in ty_rk], label='Runge Kutta Trajectory')
#trajectory_exact, = ax2.plot([ti for ti,x0i in ty_exact], [x0i[POS_Y] for ti,x0i in ty_exact], 'r', label='Exact Trajectory')
#ax2.set_xlabel('Time (s)')
#ax2.set_ylabel('Altitude (m)')
#plt.legend(handles=[trajectory_rk, trajectory_exact])
#plt.show()
#
#print 'range for theta=30 :', rng
#
##iteratively compute range for each angle and find the maximum range
#angle = 0.
#h_angle = 1.
#
#optimal_angle = 0.
#max_range = 0.
#trajectories = []
#while angle <= 90:
#    
#    x0 = zeros(NUM_VARS)
#    x0[POS_X] = 0.
#    x0[POS_Y] = 0.
#    x0[POS_VX] = v0*cos(pi*angle/180)
#    x0[POS_VY] = v0*sin(pi*angle/180)
#     
#    
#    t = 0.          # sec
#    
#    trajectory = []
#    while x0[POS_Y] >= 0:
#        
#        trajectory.append( (t, x0) )
#        
#        t = t + h
#        x0 = RK4(t,x0)
#        
#    if x0[POS_X] >= max_range:
#        max_range = x0[POS_X]
#        optimal_angle = angle
#        
#    if angle % 15 == 0:
#        trajectories.append( (angle, trajectory) )
#    
#    angle = angle + h_angle
#    
#print 'optimal angle:', optimal_angle, ', max_range:', max_range
#
#fig3 = plt.figure()
#ax2 = fig3.add_subplot(111)
#ax_handles = []
#for t in trajectories:
#    trajectory_rk, = ax2.plot([x0i[POS_X] for ti,x0i in t[1]], [x0i[POS_Y] for ti,x0i in t[1]] , label='Theta ' + str(t[0]))
#    ax_handles.append(trajectory_rk)
#    
#ax2.set_xlabel('Range (m)')
#ax2.set_ylabel('Altitude (m)')
#plt.legend(handles=ax_handles)
#plt.show()