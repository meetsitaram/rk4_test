# Simulation of Falcon 9 launch to orbit


import matplotlib.pyplot as plt
from math import cos, sin, pi, sqrt
from numpy import zeros, exp, arcsin

h = .1          # stepsize used in RK4 (sec)

air_density_sl = 1.2  # density of air (kg/m^3)
H_FOR_DRAG = 7000.   # H value used in drag model (m)
ATMOSPHERE_HEIGHT = 90000 # 90 KM - we can ignore air drag beyond this
G = 6.672e-11   # universal gravitation constant
Re = 6372e3     # radius of earch (m)
Me = 5.976e24   # mass of earth (kg)


g_sl = 9.81


draf_coefficient = 0.3    # drag coefficient - temporary
#falcin 8 Full Thrust specs from http://spaceflight101.com/spacerockets/falcon-9-ft/
inert_mass = 18200.     # had to remove 4 tons weight to achieve orbit
#inert_mass = 22200.
propellant_mass = 409500.    # kg
#propellant_mass = 409500.    # kg
merlin_1D_thrust_sl = 756000. # N
merlin_1D_thrust_vac = 825000. # N
thrust_sl = 9.*merlin_1D_thrust_sl # N,
merlin_1D_specific_impulse_sl = 282.   #sec
merlin_1D_specific_impulse_vac = 311.   #sec
burn_time_1 = 100.
burn_time_2_start =  101.
burn_time_2_end = 280.
burn_time_3_start = 390.
burn_time_3_end = 440.  # until fuel is empty
diameter = 3.66
ref_area = diameter**2 * pi / 4.   #  reference cross section area - used for drag (m^2)
flow_rate = 9.*merlin_1D_thrust_sl/(g_sl * merlin_1D_specific_impulse_sl)
mass = inert_mass + propellant_mass
throttle = 1.

second_stage_cut_off = 0.


x0,y0,v0 = 0., Re, 0.     # initial altitude (m), Initial speed (m/s)

POS_X = 0
POS_Y = 1
POS_VX = 2
POS_VY = 3
NUM_VARS = 4
TMP_POS_A_DRAG = 4
TMP_POS_A = 5
TMP_POS_MASS = 6
TMP_POS_THRUST = 7
TMP_POS_FLOW_RATE = 8
TMP_POS_THRUST_ANGLE = 9

ENABLE_DRAG = True

max_drag_acceleration = 0. # in g
max_acceleration = 0. # in g
max_altitude = 0.
down_range = 0.
terminal_speed = 0.
curr_acceleration = 0.
curr_drag_acceleration = 0.
current_thrust = 0.
current_thrust_angle = 0.
current_flow_rate = 0.

engines_on = True

def get_rocket_acceleration(x,y,t, vx, vy):
    a_rocket = [0., 0.]

    global propellant_mass    
    if propellant_mass <= 0:
        return [0., 0.]

    global h, mass, flow_rate
    
    #flow_rate = 0.
            
    global burn_time_1, burn_time_2_start, burn_time_2_end, throttle, second_stage_cut_off
    global engines_on
    
    thrust_total = 0.
    thrust_angle = 0.
    engines_on = False
    if propellant_mass  > h * flow_rate:
        if t >= 0 and t <= burn_time_1:
            engines_on = True
            flow_rate = 9.*throttle*merlin_1D_thrust_sl/(g_sl * merlin_1D_specific_impulse_sl)
            thrust_sl = 9.*throttle*merlin_1D_thrust_sl # N,
            thrust_total = thrust_sl
            thrust_angle = 60.
        elif t >= burn_time_2_start and t <= burn_time_2_end:  
            engines_on = True
            second_stage_cut_off = t
            flow_rate = 3.*throttle*merlin_1D_thrust_vac/(g_sl * merlin_1D_specific_impulse_vac)
            thrust_vac = 3.*throttle*merlin_1D_thrust_vac # N,
            thrust_total = thrust_vac
            
            #print 't:', thrust_total/(mass*g_sl)
            
            desired_ay = 0.4*g_sl
            a_max = thrust_total/mass
            
            if desired_ay < a_max:
                thrust_angle = 180*arcsin(desired_ay/a_max)/pi 
                #print 'thrust_angle', thrust_angle
                
        elif t >= burn_time_3_start and t <= burn_time_3_end:
            engines_on = True
            flow_rate = 3.*throttle*merlin_1D_thrust_vac/(g_sl * merlin_1D_specific_impulse_sl)
            thrust_vac = 3.*throttle*merlin_1D_thrust_vac # N,
            thrust_total = 1.1*thrust_vac 
            thrust_angle = -4
            
    
    global current_thrust, current_thrust_angle, current_flow_rate
    current_thrust = throttle*thrust_total
    current_flow_rate = flow_rate
    current_thrust_angle = thrust_angle
    if engines_on == False:
        current_flow_rate = 0.
        current_thrust = 0.
        current_thrust_angle = 0.    
    
    a_rocket_total = throttle*thrust_total/mass
        

    a_rocket = [a_rocket_total*cos(pi*thrust_angle/180.), a_rocket_total*sin(pi*thrust_angle/180.)]            
        
    return a_rocket
        
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
    
    if v == 0:
        return [0., 0.]
    
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
    
def get_acceleration(x, y, vx, vy, t):
    
    g_sl = G*Me/Re**2
    
    a_drag = get_drag_acceleration(vx, vy, x, y)
    a_gravity = get_gravity_acceleration(x,y)
    a_rocket = get_rocket_acceleration(x,y,t,vx, vy)
    
    a = [a_drag[0]+a_gravity[0]+a_rocket[0], a_drag[1]+a_gravity[1]+a_rocket[1]]
    
    a_magnitude = sqrt(a[0]**2 + a[1]**2)
    global max_acceleration
    if a_magnitude > max_acceleration*g_sl:
        max_acceleration = a_magnitude  / g_sl
        
    global curr_acceleration 
    curr_acceleration = a_magnitude / g_sl    

    return a

#Derivative function   
def f(t,x0):

    ax,ay = get_acceleration(x0[POS_X], x0[POS_Y], x0[POS_VX], x0[POS_VY] , t)
    
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
   
def get_trajectory(x0, y0, v0):

    args0 = zeros(NUM_VARS)
    args0[POS_X] = x0
    args0[POS_Y] = y0
    args0[POS_VX] = 0.
    args0[POS_VY] = 0.
    
    
    trajectory = []
    t = 0.          # sec
    
    global curr_drag_acceleration
    global curr_acceleration
    
    while sqrt(args0[POS_X]**2 + args0[POS_Y]**2) >= Re and t < 9000: 
        tmp_args = list(args0)
        tmp_args.append(curr_drag_acceleration)            
        tmp_args.append(curr_acceleration)
        tmp_args.append(mass)
        tmp_args.append(current_thrust)
        tmp_args.append(current_flow_rate)
        tmp_args.append(current_thrust_angle)
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
        
        global mass, inert_mass,propellant_mass
        
        global burn_time_1, burn_time_2_start, burn_time_2_end
        if propellant_mass > h*flow_rate and engines_on == True:
            propellant_mass = propellant_mass - throttle*flow_rate*h
        
        if propellant_mass <=  0:
            mass = inert_mass 
        else:
            mass = inert_mass + propellant_mass
        
        t = t + h
        
        
        args0 = RK4(t,args0)
        
    return trajectory

fig1 = plt.figure()
fig1, ((ax1,ax5),(ax4, ax7), (ax2, ax6), (ax3, ax8)) = plt.subplots(4, 2)
ax1_handles = []
ax2_handles = []
ax3_handles = []
ax4_handles = []
ax5_handles = []
ax6_handles = []
ax7_handles = []
ax8_handles = []

max_drag_acceleration = 0. # in g
max_acceleration = 0.
max_altitude = 0.
down_range = 0.
terminal_speed = 0.
curr_acceleration = 0.
curr_drag_acceleration = 0.
current_thrust_angle = 0.

ENABLE_DRAG = True
trajectory = get_trajectory(x0, y0, v0) 
tmp_plot_1, = ax1.plot([x0i[POS_X]/1000. for ti,x0i in trajectory], [x0i[POS_Y]/1000. for ti,x0i in trajectory], color='r', label='')
tmp_plot_2x, = ax2.plot([ti/60. for ti,x0i in trajectory], [abs(x0i[POS_VX])/1000. for ti,x0i in trajectory], label='Vx')
tmp_plot_2y, = ax2.plot([ti/60. for ti,x0i in trajectory], [abs(x0i[POS_VY])/1000. for ti,x0i in trajectory], label='Vy')
tmp_plot_2, = ax2.plot([ti/60. for ti,x0i in trajectory], [sqrt(x0i[POS_VX]**2 + x0i[POS_VY]**2)/1000. for ti,x0i in trajectory], label='V')
tmp_plot_3, = ax3.plot([ti/60. for ti,x0i in trajectory if ti < 450], [x0i[TMP_POS_A] for ti,x0i in trajectory  if ti < 450], label='max ' + str(round(max_acceleration,1)) + 'g')
tmp_plot_4, = ax4.plot([ti/60. for ti,x0i in trajectory], [(sqrt(x0i[POS_X]**2 + x0i[POS_Y]**2) - Re)/1000. for ti,x0i in trajectory], label='')
tmp_plot_5, = ax5.plot([ti/60. for ti,x0i in trajectory if ti < 450], [x0i[TMP_POS_MASS] for ti,x0i in trajectory if ti < 450], label='dry_mass=18200*, prop_mass=409500')
tmp_plot_6, = ax6.plot([ti/60. for ti,x0i in trajectory if ti < 450], [x0i[TMP_POS_THRUST]/1000. for ti,x0i in trajectory if ti < 450], label='3 Engines at burn 2 and 3')
tmp_plot_7, = ax7.plot([ti/60. for ti,x0i in trajectory if ti < 450], [x0i[TMP_POS_FLOW_RATE] for ti,x0i in trajectory if ti < 450], label='isp_sl=282, isp_vac=311')
tmp_plot_8, = ax8.plot([ti/60. for ti,x0i in trajectory if ti < 450], [x0i[TMP_POS_THRUST_ANGLE] for ti,x0i in trajectory if ti < 450], label='')
ax1_handles.append(tmp_plot_1)
ax2_handles.append(tmp_plot_2)
ax2_handles.append(tmp_plot_2x)
ax2_handles.append(tmp_plot_2y)
ax3_handles.append(tmp_plot_3)
ax4_handles.append(tmp_plot_4)
ax5_handles.append(tmp_plot_5)
ax6_handles.append(tmp_plot_6)
ax7_handles.append(tmp_plot_7)

earth = ax1.add_artist(plt.Circle((0,0),Re/1000.0,color="b",fill=True,label='Earth'))
ax1.add_artist(plt.Circle((0,0),(Re/1.0e3+25),color="b",alpha=0.1))
ax1.add_artist(plt.Circle((0,0),(Re/1.0e3+50),color="b",alpha=0.05))
ax1.add_artist(plt.Circle((0,0),(Re/1.0e3+90),color="b",alpha=0.04))

  
ax1_handles.append(earth)
ax1.set_xlabel('X (Km)')
ax1.set_ylabel('Y (Km)')
ax2.set_xlabel('Time (min)')
ax2.set_ylabel('Speed Km/s)')
ax3.set_xlabel('Time (min)')
ax3.set_ylabel('Acceleration (g)')
ax4.set_xlabel('Time (min)')
ax4.set_ylabel('Altitude (Km)')
ax5.set_xlabel('Time (min)')
ax5.set_ylabel('Mass (Kg)')
ax6.set_xlabel('Time (min)')
ax6.set_ylabel('Thrust (KN)')
ax7.set_xlabel('Time (min)')
ax7.set_ylabel('Flow Rate (Kg/s)')
ax8.set_xlabel('Time (min)')
ax8.set_ylabel('Thrust Angle (Degrees)')
ax1.legend(handles=ax1_handles, loc='upper right')
ax2.legend(handles=ax2_handles, loc='lower right')
ax3.legend(handles=ax3_handles, loc='upper right')
ax5.legend(handles=ax5_handles, loc='upper right')
ax6.legend(handles=ax6_handles, loc='upper right')
ax7.legend(handles=ax7_handles, loc='upper right')
plt.show()


print 'max_drag_acceleration:', round(max_drag_acceleration,1), 'g, max_acceleration:', round(max_acceleration,1), 'g, max_altitude:', round(max_altitude/1000.), 'km, down_range:', round(down_range/1000.), 'km'
print 'second_stage_cut_off:', second_stage_cut_off