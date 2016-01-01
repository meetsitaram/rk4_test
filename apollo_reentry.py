# Simulation of Apollo rentry
#  under the guidance of nasaspaceflight user meithan

import matplotlib.pyplot as plt
from math import cos, sin, pi, sqrt
from numpy import zeros, exp, arcsin

h = .01          # stepsize used in RK4 (sec)

mass = 5621.    #  mass of object (kg)
ref_area = 11.631   #  reference cross section area - used for drag (m^2)
draf_coefficient = 0.3    # drag coefficient
air_density_sl = 1.2  # density of air (kg/m^3)
H_FOR_DRAG = 7000.   # H value used in drag model (m)
ATMOSPHERE_HEIGHT = 90000 # 90 KM - we can ignore air drag beyond this
G = 6.672e-11   # universal gravitation constant
Re = 6372e3     # radius of earch (m)
Me = 5.976e24   # mass of earth (kg)
fpa = -6.1  # flight path angle


x0,y0,v0 = 0., Re + 121.8e3, 11130.     # initial altitude (m), Initial speed (m/s)

POS_X = 0
POS_Y = 1
POS_VX = 2
POS_VY = 3
NUM_VARS = 4
TMP_POS_A_DRAG = 4
TMP_POS_A = 5

ENABLE_DRAG = True

max_drag_acceleration = 0. # in g
max_acceleration = 0. # in g
max_altitude = 0.
down_range = 0.
terminal_speed = 0.
curr_acceleration = 0.
curr_drag_acceleration = 0.

def get_drag_acceleration(vx, vy, x, y):
    if not ENABLE_DRAG:
        return [0., 0.]
    
    r = sqrt(x*x+y*y)
    h = r - Re
    if h < 0:
        return [0., 0.] 
        
    if h > ATMOSPHERE_HEIGHT:
        return [0., 0.]
        
    v = sqrt(vx**2 + vy**2)
    
    air_density = air_density_sl * exp ( - h / H_FOR_DRAG)
    
    f_drag = (1./2) * air_density * draf_coefficient * ref_area * v**2    
    a_drag = f_drag/mass
    
    g_sl = G*Me/Re**2
    
    global max_drag_acceleration
    if a_drag > max_drag_acceleration*g_sl:
        max_drag_acceleration = a_drag / g_sl
        
    global curr_drag_acceleration 
    curr_drag_acceleration = a_drag / g_sl
        
    return [-a_drag*vx/v, -a_drag*vy/v]
   
def get_gravity_acceleration(x,y):
    
    r = sqrt(x*x+y*y)
    
    if r == 0:
        return [0., 0.]
    
    g = G*Me/r**2

    return [-g*x/r, -g*y/r]    
    
def get_acceleration(x, y, vx, vy):
    
    g_sl = G*Me/Re**2
    
    a_drag = get_drag_acceleration(vx, vy, x, y)
    a_gravity = get_gravity_acceleration(x,y)
    a = [a_drag[0]+a_gravity[0], a_drag[1]+a_gravity[1]]
    
    a_magnitude = sqrt(a[0]**2 + a[1]**2)
    global max_acceleration
    if a_magnitude > max_acceleration*g_sl:
        max_acceleration = a_magnitude  / g_sl
        
    global curr_acceleration 
    curr_acceleration = a_magnitude / g_sl    
        
    return a

#Derivative function   
def f(t,x0):

    ax,ay = get_acceleration(x0[POS_X], x0[POS_Y], x0[POS_VX], x0[POS_VY] )
    
    res = zeros(NUM_VARS)
    res[POS_X] = x0[POS_VX]
    res[POS_Y] = x0[POS_VY]
    res[POS_VX] = ax
    res[POS_VY] = ay
        
    return res
    
# 4th order Runge-Kutta
def RK4(t,x0):
    xtemp = zeros(NUM_VARS)
        
    k0 = f(t,x0)
    for i in range(NUM_VARS):
        xtemp[i] = x0[i] + k0[i]*(h/2.)        
    
    k1 = f(t+h/2., xtemp)
    for i in range(NUM_VARS):
        xtemp[i] = x0[i] + k1[i]*(h/2.)
        
    k2 = f(t+h/2., xtemp)
    for i in range(NUM_VARS):
        xtemp[i] = x0[i] + k2[i]*h
        
    k3 = f(t+h, xtemp)
    for i in range(NUM_VARS):
        x0[i] = x0[i] + (h/6.) * (k0[i] + 2 * k1[i] + 2 * k2[i] + k3[i])
    
    return x0
   
def get_trajectory(x0, y0, v0, theta):

    args0 = zeros(NUM_VARS)
    args0[POS_X] = x0
    args0[POS_Y] = y0
    args0[POS_VX] = v0*cos(pi*theta/180.)
    args0[POS_VY] = v0*sin(pi*theta/180.)
    
    
    trajectory = []
    t = 0.          # sec
    
    global curr_drag_acceleration
    global curr_acceleration
    
    while sqrt(args0[POS_X]**2 + args0[POS_Y]**2) >= Re and t < 700: 
        tmp_args = list(args0)
        tmp_args.append(curr_drag_acceleration)            
        tmp_args.append(curr_acceleration)
        trajectory.append( (t, tmp_args) )        
        
        
        global max_altitude
        altitude = sqrt((args0[POS_X])**2 + (args0[POS_Y])**2) - Re
        if altitude > max_altitude:
            max_altitude = altitude

        global down_range
        #get arc distance
        tmp_range = 2*Re*arcsin((sqrt((args0[POS_X]-x0)**2 + (args0[POS_Y]-y0)**2))/(2*Re))
        #tmp_range = Re*arccos( 1. - 1./2*((args0[POS_X]-x0)/Re)**2 - 1./2*((args0[POS_Y]-y0)/Re)**2 )
        #tmp_range = args0[POS_X]
        if tmp_range > down_range:
            down_range = tmp_range
        
        t = t + h
        args0 = RK4(t,args0)
        
    return trajectory

fig1 = plt.figure()
fig1, ((ax1,ax2),(ax4, ax3)) = plt.subplots(2, 2)
ax1_handles = []
ax2_handles = []
ax3_handles = []
ax4_handles = []

max_drag_acceleration = 0. # in g
max_acceleration = 0.
max_altitude = 0.
down_range = 0.
terminal_speed = 0.
curr_acceleration = 0.
curr_drag_acceleration = 0.

ENABLE_DRAG = True
trajectory = get_trajectory(x0, y0, v0, fpa) 
tmp_plot_1, = ax1.plot([x0i[POS_X]/1000. for ti,x0i in trajectory], [x0i[POS_Y]/1000. for ti,x0i in trajectory], color='r', label='fpa ' + str(fpa))
tmp_plot_2, = ax2.plot([ti/60. for ti,x0i in trajectory], [sqrt(x0i[POS_VX]**2 + x0i[POS_VY]**2)/1000. for ti,x0i in trajectory], label='fpa ' + str(fpa))
tmp_plot_3, = ax3.plot([ti/60. for ti,x0i in trajectory], [x0i[TMP_POS_A] for ti,x0i in trajectory], label='max ' + str(round(max_acceleration,1)) + 'g')
tmp_plot_4, = ax4.plot([ti/60. for ti,x0i in trajectory], [(sqrt(x0i[POS_X]**2 + x0i[POS_Y]**2) - Re)/1000. for ti,x0i in trajectory], label='fpa ' + str(fpa))
ax1_handles.append(tmp_plot_1)
ax2_handles.append(tmp_plot_2)
ax3_handles.append(tmp_plot_3)
ax4_handles.append(tmp_plot_4)

earth = ax1.add_artist(plt.Circle((0,0),Re/1000.0,color="b",fill=True,label='Earth'))
ax1.add_artist(plt.Circle((0,0),(Re/1.0e3+25),color="b",alpha=0.1))
ax1.add_artist(plt.Circle((0,0),(Re/1.0e3+50),color="b",alpha=0.05))
ax1.add_artist(plt.Circle((0,0),(Re/1.0e3+90),color="b",alpha=0.04))

  
ax1_handles.append(earth)
ax1.set_xlabel('X (Km)')
ax1.set_ylabel('Y (Km)')
ax2.set_xlabel('Time (min)')
ax2.set_ylabel('Speed Km/s)')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Acceleration (g)')
ax4.set_xlabel('Time (min)')
ax4.set_ylabel('Altitude (Km)')
ax1.legend(handles=ax1_handles, loc='upper right')
ax3.legend(handles=ax3_handles, loc='upper right')
plt.show()


print 'max_drag_acceleration:', round(max_drag_acceleration,1), 'g, max_acceleration:', round(max_acceleration,1), 'g, max_altitude:', round(max_altitude/1000.), 'km, down_range:', round(down_range/1000.), 'km'