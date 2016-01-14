# Simulation of Falcon 9 launch to orbit


import matplotlib.pyplot as plt
from math import cos, sin, pi, sqrt
from numpy import zeros, exp, arcsin, arctan

h = .1          # stepsize used in RK4 (sec)

air_density_sl = 1.2  # density of air (kg/m^3)
H_FOR_DRAG = 7000.   # H value used in drag model (m)
H_FOR_PRESSURE = 7000. # constant temperature model
ATMOSPHERE_HEIGHT = 90000 # 90 KM - we can ignore air drag beyond this
G = 6.672e-11   # universal gravitation constant
Re = 6372e3     # radius of earch (m)
Me = 5.976e24   # mass of earth (kg)
mu = G*Me

g_sl = 9.81


draf_coefficient = 0.3    # drag coefficient - temporary
#falcin 8 Full Thrust specs from http://spaceflight101.com/spacerockets/falcon-9-ft/
s1_inert_mass = 22200.    # Kg
s1_propellant_mass = 409500.    # kg
s1_merlin_1D_thrust_sl = 756000. # N
s1_merlin_1D_thrust_vac1 = 825000. # N 
s1_merlin_1D_specific_impulse_sl = 282.*g_sl   # m/s
s1_merlin_1D_specific_impulse_vac1 = 311.*g_sl   # m/s
s1_burn_time_end = 145  # s

s2_inert_mass = 4000.    # Kg
s2_propellant_mass = 103500.    # kg
s2_merlin_1D_thrust_vac1 = 934000. # N
s2_merlin_1D_specific_impulse_vac1 = 348.*g_sl   # m/s 
s2_burn_time_start =  155.
s2_burn_time_end = 570.

fairing_mass = 1750. # kg
payload_mass = 2200. # kg 12*176 + ~100kg

orbit_altitude = 620000
v_terminal = sqrt(G*Me/ (Re+orbit_altitude))

inert_mass = s1_inert_mass +  s2_inert_mass + fairing_mass + payload_mass   # Kg
propellant_mass = s1_propellant_mass    # kg
mass = inert_mass + s1_propellant_mass + s2_propellant_mass
thrust_sl = 9.*s1_merlin_1D_thrust_sl # N,

PITCH_KICK_ANGLE = 3. #degrees from vertical
stage_separation = False
fairing_separation = False
lift_off = False

diameter = 3.7
ref_area = diameter**2 * pi / 4.   #  reference cross section area - used for drag (m^2)
s1_m1D_flow_rate = s1_merlin_1D_thrust_sl/s1_merlin_1D_specific_impulse_sl
s2_m1D_flow_rate = s2_merlin_1D_thrust_vac1/s2_merlin_1D_specific_impulse_vac1

flow_rate = 0.
isp = 0.
thrust_total = 0.

omega = 2.0*pi/(23.*3600+56*50+4)  # Earth's angular velocity

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
TMP_POS_ISP = 10
TMP_POS_TWR = 11
TMP_POS_Q = 11

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
current_dynamic_pressure = 0.

engines_on = True

periapsis_radius = 0.
apoapsis_radius = 0.
semimajor_axis = 0.
eccentricity = 0.

#time,speed, altitude, range
actual_stage1_trajectory = [
    [0,0,0,0],
    [5,42,0,],
    [10,98,.1,0],
    [15,162,.3,0],
    [20,228,.6,0],
    [25,304,1,0],
    [30,388,1.5,0],
    [35,477,2,0],
    [40,576,2.8,0],
    [45,686,3.6,0],
    [50,803,4.6,0],
    [55,939,5.9,0],
    [60,1082,7.3,0],
    [65,1221,8.7,0],
    [70,1382,10.5,0],
    [75,1555,12.5,0],   # max Q at 73 sec at 431m/s speed and 12.5km altitude
    [80,1734,14.7,0],
    [85,1943,17.2,2.5],   #
    [90,2175,19.9,0],
    [95,2396,22.5,0],
    [100,2678,25.9,0],
    [105,2959,29.4,0],
    [110,3260,33.2,0],
    [115,3602,37.7,0],
    [120,3949,42.5,11],
    [125,4317,47.6,0],
    [130,4720,53.4,0],
    [135,5168,59.9,0],
    [140,5603,66.5,0],
    [145,6012,74.2,0],  # MECO
]

actual_stage2_trajectory = [
    [155,5700,91.8,0],
    [160,5684,96.9,0],
    [165,5668,104,0],
    [170,5657,111,0], #fairing separation
    [175,5651,118,0],
    [180,5649,126,0],
    [190,5652,140,0],
    [200,5667,154,0],
    [220,5734,182,0],
    [240,5856,210,0],
    [260,6039,238,0],
    [280,6291,267,0],
    [300,6623,295,0],
    [320,7035,323,0],
    [340,7553,351,0],
    [360,8187,379,157], # 157km?
    [380,8930,408,0],
    [400,9820,437,0],
    [420,10875,466,0],
    [440,12123,495,0],  #couldn't get downrange
    [460,13589,523,0],
    [480,15360,549,0],
    [500,17560,573,0],
    [520,20281,593,0],
    [540,23164,609,0],
    [560,25376,618,0],
    [561,25505,618,0],
    [562,25596,618,0],
    [563,25701,619,0],
    [564,25848,619,0],
    [565,25945,619,0],
    [566,25974,619,0],
    [567,25988,619,0],
    [570,25991,619,0],
    [571,25992,619,0], 
    [572,25992,619,0],
    [573,25992,619,0],
    [574,25991,619,0],
    [580,25991,619,0],
    [590,25990,619,0],  # SECO
]

range_values = [
    [0,0,0,0],
    [85,1943,17.2,2.5],
    [120,3949,42.5,11],
    [360,8187,379,157],
]


#press kit - http://www.spacex.com/sites/spacex/files/spacex_orbcomm_press_kit_final2.pdf



def compute_orbit(x, y, vx, vy):
  r = sqrt(x**2+y**2)
  v = sqrt(vx**2+vy**2)  
  E0 = 0.5*v**2 - mu/r
  H0 = sqrt((r*v)**2-(x*vx+y*vy)**2)
  if E0 == 0:
    rpe = H0**2/(2*mu)
    rap = None
    sma = None
    ecc = 1.0
  else:
    rpe = (sqrt(mu**2+2*E0*H0**2)-mu)/(2*E0)
    rap = (sqrt(mu**2+2*E0*H0**2)+mu)/(-2*E0)
    sma = -mu/(2*E0)
    ecc = sqrt(1+(2*E0*H0**2)/mu**2)
  return (rpe, rap, sma, ecc)
  
def get_rocket_acceleration(x,y,t, vx, vy):
    a_rocket = [0., 0.]

    global flow_rate, thrust_total
    flow_rate = 0.
    thrust_total = 0.
    throttle = 1.
        
    global propellant_mass    
    if propellant_mass <= 0:
        return [0., 0.]

    global h, mass
    
            
    global s2_burn_time_start, s2_burn_time_end, s1_burn_time_end, second_stage_cut_off
    global engines_on
    
    thrust_total = 0.
    thrust_angle = 0.
    flow_rate = 0. 
    engines_on = False
    
    r = sqrt(x**2 + y**2)
    altitude = r - Re
    global isp, stage_separation
    if stage_separation == True:
        isp = s2_merlin_1D_specific_impulse_vac1
    else:    
        isp = (s1_merlin_1D_specific_impulse_sl - s1_merlin_1D_specific_impulse_vac1)*exp(-altitude/H_FOR_PRESSURE) + s1_merlin_1D_specific_impulse_vac1
    

    vy_surf = vy - omega*(-x)
    vx_surf = vx - omega*y
            
    if t >= 0 and t < s1_burn_time_end and propellant_mass > h*9.*s1_m1D_flow_rate:
        if t <= 7.5:
            thrust_angle = 90.
            
        elif t<= 55:
            global PITCH_KICK_ANGLE
            PITCH_KICK_ANGLE = 3.
            #thrust_angle = (180./pi)*arctan(vy_surf/vx_surf) - PITCH_KICK_ANGLE*exp((t-7.5)/110.)
            thrust_angle = 90 - PITCH_KICK_ANGLE*exp((t-7.5)/110.)
            
        else:
            if altitude < ATMOSPHERE_HEIGHT:
                thrust_angle = (180./pi)*arctan(vy_surf/vx_surf)
            else:
                thrust_angle =  (180./pi)*arctan(vy/vx)
        
        engines_on = True
        
        flow_rate = 9.*s1_m1D_flow_rate
        
        thrust_total = flow_rate*isp
        
        throttle = 1 - .26*exp((t-145)/300)
        flow_rate = flow_rate*throttle
        thrust_total = flow_rate*isp
        #print 'PITCH_KICK_ANGLE diff', (180./pi)*arctan(vy_surf/ (vx_surf+0.00001)) - thrust_angle
        


            
    elif t >= s2_burn_time_start and t < s2_burn_time_end and propellant_mass > h*1.*s2_m1D_flow_rate:
        engines_on = True
  
        second_stage_cut_off = t

        
        flow_rate = 1.*s2_m1D_flow_rate
        thrust_total = flow_rate*isp

        if t < 400:
            thrust_angle = (180./pi)*arctan(vy/vx) + 25.3
        else:
            thrust_angle = (180./pi)*arctan(vy/vx) - 21.
        
        #thrust_angle = (180./pi)*arctan(vy/vx) + 15 #.*exp((t-155)/390.)
        #print (180./pi)*arctan(vy/vx)  - thrust_angle, (180./pi)*arctan(vy/vx)
        
            
        throttle = 1.
        if t > 200:
            throttle = 1. - .181*exp((t-590)/800.)
        flow_rate = flow_rate*throttle
        #print throttle
        thrust_total = flow_rate*isp        
        
    
    global current_thrust, current_thrust_angle, current_flow_rate
    current_thrust = thrust_total
    current_flow_rate = flow_rate
    current_thrust_angle = thrust_angle
    if engines_on == False:
        current_flow_rate = 0.
        current_thrust = 0.
        current_thrust_angle = 0.    
    
    a_rocket_total = thrust_total/mass
        

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
        
    v_air = omega*r     # velocity of air
    vx_air = v_air*(y/r)
    vy_air = v_air*(-x/r)
    vx1 = vx - vx_air
    vy1 = vy - vy_air
    
        
    v = sqrt(vx1**2 + vy1**2)

    if v == 0:
        return [0., 0.]
    
    air_density = air_density_sl * exp ( - h / H_FOR_DRAG)
    
    global current_dynamic_pressure
    vy_surf = vy - omega*(-x)
    vx_surf = vx - omega*y
    v_surf = sqrt(vx_surf**2 + vy_surf**2)
    current_dynamic_pressure = 0.5*air_density*v_surf**2
    
    f_drag = (1./2) * air_density * draf_coefficient * ref_area * v**2    
    a_drag = f_drag/mass
    
    g_sl = G*Me/Re**2
    
    global max_drag_acceleration
    if a_drag > max_drag_acceleration*g_sl:
        max_drag_acceleration = a_drag / g_sl
        
    global curr_drag_acceleration 
    curr_drag_acceleration = a_drag / g_sl
        
    return [-a_drag*vx1/v, -a_drag*vy1/v]
   
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
    args0[POS_VX] = omega*Re
    args0[POS_VY] = 0.
    
    
    trajectory = []
    t = 0.          # sec
    
    global curr_drag_acceleration
    global curr_acceleration
    
    while sqrt(args0[POS_X]**2 + args0[POS_Y]**2) >= Re and t < 600 : 
        
        global stage_separation, inert_mass, propellant_mass, mass, lift_off
        global s1_burn_time_end, s2_burn_time_start, s2_burn_time_end
        global s1_propellant_mass, s2_propellant_mass
        
        if lift_off == False:
            lift_off = True
            inert_mass = s1_inert_mass + s2_inert_mass + fairing_mass + payload_mass   # Kg
            propellant_mass = s1_propellant_mass    # kg
            mass = inert_mass + s1_propellant_mass + s2_propellant_mass
            
        if t >= s2_burn_time_start and stage_separation == False:    
            stage_separation = True
            inert_mass = s2_inert_mass + payload_mass   # Kg
            propellant_mass = s2_propellant_mass    # kg
            mass = inert_mass + s2_propellant_mass
            
        #print propellant_mass
            
        tmp_args = list(args0)
        tmp_args.append(curr_drag_acceleration)            
        tmp_args.append(curr_acceleration)
        tmp_args.append(propellant_mass)
        tmp_args.append(current_thrust)
        tmp_args.append(current_flow_rate)
        tmp_args.append(current_thrust_angle)
        tmp_args.append(isp)
        tmp_args.append(thrust_total/(mass*g_sl))
        tmp_args.append(current_dynamic_pressure)
        trajectory.append( (t, tmp_args) )        
        
        
        global max_altitude
        altitude = sqrt((args0[POS_X])**2 + (args0[POS_Y])**2) - Re
        if altitude > max_altitude:
            max_altitude = altitude

        global down_range
        #get arc distance
        tmp_range = tmp_args[POS_X]*cos(omega*t) - tmp_args[POS_Y]*sin(omega*t) 
        #tmp_range = 2*Re*arcsin((sqrt((args0[POS_X]-x0)**2 + (args0[POS_Y]-y0)**2))/(2*Re))
        #tmp_range = Re*arccos( 1. - 1./2*((args0[POS_X]-x0)/Re)**2 - 1./2*((args0[POS_Y]-y0)/Re)**2 )
        #tmp_range = args0[POS_X]
        if tmp_range > down_range:
            down_range = tmp_range
        
        if propellant_mass > h*flow_rate and engines_on == True:
            propellant_mass = propellant_mass - flow_rate*h
            if t < s1_burn_time_end:
                s1_propellant_mass = s1_propellant_mass - flow_rate*h
            else:
                s2_propellant_mass = s2_propellant_mass - flow_rate*h
        
        if propellant_mass <=  0:
            mass = inert_mass 
        else:
            mass = inert_mass + propellant_mass
        
        global periapsis_radius, apoapsis_radius, semimajor_axis, s2_burn_time_end, eccentricity
        if semimajor_axis == 0 and t > s2_burn_time_end:
            print 'altitude:', altitude
            periapsis_radius, apoapsis_radius, semimajor_axis,eccentricity = compute_orbit(args0[POS_X], args0[POS_Y], args0[POS_VX], args0[POS_VY])
            print 'periapsis_radius:', periapsis_radius - Re, 'apoapsis_radius:', apoapsis_radius - Re, 'semimajor_axis', semimajor_axis, 'eccentricity:', eccentricity
            
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
tmp_plot_1, = ax1.plot([x0i[POS_X]/1000. for ti,x0i in trajectory], [x0i[POS_Y]/1000. for ti,x0i in trajectory], color='r', label='Trajectory')
#tmp_plot_1, = ax1.plot([ (x0i[POS_X]*cos(omega*ti) - x0i[POS_Y]*sin(omega*ti))/1000. for ti,x0i in trajectory ], [ (sqrt(x0i[POS_X]**2 + x0i[POS_Y]**2) - Re)/1000.  for ti,x0i in trajectory ], color='r', label='Trajectory')
tmp_plot_2x, = ax2.plot([ti/60. for ti,x0i in trajectory], [abs(x0i[POS_VX])/1000. for ti,x0i in trajectory], ls='dashed', alpha=0.4, label='Vx')
tmp_plot_2y, = ax2.plot([ti/60. for ti,x0i in trajectory], [abs(x0i[POS_VY])/1000. for ti,x0i in trajectory],  ls='dashed', alpha=0.4,label='Vy')
tmp_plot_2, = ax2.plot([ti/60. for ti,x0i in trajectory], [sqrt(x0i[POS_VX]**2 + x0i[POS_VY]**2)/1000. for ti,x0i in trajectory], label='V')
tmp_plot_3, = ax3.plot([ti/60. for ti,x0i in trajectory if ti < 450], [x0i[TMP_POS_A] for ti,x0i in trajectory  if ti < 450], label='max ' + str(round(max_acceleration,1)) + 'g')
tmp_plot_4, = ax4.plot([ti/60. for ti,x0i in trajectory], [(sqrt(x0i[POS_X]**2 + x0i[POS_Y]**2) - Re)/1000. for ti,x0i in trajectory], label='')
tmp_plot_5, = ax5.plot([ti/60. for ti,x0i in trajectory if ti < 450], [x0i[TMP_POS_TWR] for ti,x0i in trajectory if ti < 450], label='')
tmp_plot_6, = ax6.plot([ti/60. for ti,x0i in trajectory if ti < 450], [x0i[TMP_POS_THRUST_ANGLE] for ti,x0i in trajectory if ti < 450], label='')
#tmp_plot_7, = ax7.plot([ti/60. for ti,x0i in trajectory if ti < 450], [int(round(x0i[TMP_POS_ISP]/g_sl)) for ti,x0i in trajectory if ti < 450], label='isp_sl=282, isp_vac=311')
tmp_plot_7, = ax7.plot([ti/60. for ti,x0i in trajectory if ti < 450], [x0i[TMP_POS_Q] for ti,x0i in trajectory if ti < 450], label='')
tmp_plot_8, = ax8.plot([ti/60. for ti,x0i in trajectory if ti < 450], [x0i[TMP_POS_A_DRAG] for ti,x0i in trajectory if ti < 450], label='')
ax1_handles.append(tmp_plot_1)
ax2_handles.append(tmp_plot_2)
ax2_handles.append(tmp_plot_2x)
ax2_handles.append(tmp_plot_2y)
ax3_handles.append(tmp_plot_3)
ax4_handles.append(tmp_plot_4)
ax5_handles.append(tmp_plot_5)
ax6_handles.append(tmp_plot_6)
ax7_handles.append(tmp_plot_7)

earth = ax1.add_artist(plt.Circle((0,0),Re/1000.0,color="b",fill=False,label='Earth'))
ax1.add_artist(plt.Circle((0,0),(Re+ATMOSPHERE_HEIGHT)/1000.0,color="r", ls='dashed', fill=False, alpha=0.4))
#ax1.add_artist(plt.Circle((0,0),(Re/1.0e3+25),color="b",alpha=0.1))
#ax1.add_artist(plt.Circle((0,0),(Re/1.0e3+50),color="b",alpha=0.05))
#ax1.add_artist(plt.Circle((0,0),(Re/1.0e3+90),color="b",alpha=0.04))

  
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
ax5.set_ylabel('Thrust-to-weight ratio')
ax6.set_xlabel('Time (min)')
ax6.set_ylabel('Thrust Angle')
ax7.set_xlabel('Time (min)')
#ax7.set_ylabel('ISP (sec)')
ax7.set_ylabel('Q')
ax8.set_xlabel('Time (min)')
ax8.set_ylabel('Drag A')
ax1.legend(handles=ax1_handles, loc='center')
ax2.legend(handles=ax2_handles, loc='lower right')
ax3.legend(handles=ax3_handles, loc='upper right')
ax5.legend(handles=ax5_handles, loc='upper right')
ax6.legend(handles=ax6_handles, loc='upper right')
ax7.legend(handles=ax7_handles, loc='upper right')
plt.show()


print 'max_drag_acceleration:', round(max_drag_acceleration,1), 'g, max_acceleration:', round(max_acceleration,1), 'g, max_altitude:', round(max_altitude/1000.), 'km, down_range:', round(down_range/1000.), 'km'
print 'second_stage_cut_off:', second_stage_cut_off, 'propellant_left s1/s2:', s1_propellant_mass, s2_propellant_mass

fig1 = plt.figure()
fig1, ((ax1,ax2)) = plt.subplots(1, 2)

trajectory_s1, = ax1.plot([i[0] for i in actual_stage1_trajectory], [i[2] for i in actual_stage1_trajectory], label='OG2 S1 Altitude')
trajectory_s2, = ax1.plot([i[0] for i in actual_stage2_trajectory], [i[2] for i in actual_stage2_trajectory], label='OG2 S2 Altitude')
trajectory_d, = ax1.plot([ ti for ti,x0i in trajectory], [ (sqrt(x0i[POS_X]**2 + x0i[POS_Y]**2) - Re)/1000. for ti,x0i in trajectory], color='r', label='Sim Trajectory')
tmp_plot_oa, = ax1.plot([ti for ti,x0i in trajectory], [orbit_altitude/1000. for ti,x0i in trajectory], ls='dashed', alpha=0.4, label='')
#trajectory_s2, = ax1.plot([ ti for ti,x0i in trajectory if ti < 450], [ (x0i[POS_Y]-Re)/1000. for ti,x0i in trajectory if ti < 450], color='r', label='Trajectory')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Altitude (km)')
ax1.legend(handles=[trajectory_s1, trajectory_s2, trajectory_d, tmp_plot_oa], loc='upper left')


trajectory_v1, = ax2.plot([i[0] for i in actual_stage1_trajectory], [i[1]*5./18 for i in actual_stage1_trajectory], label='OG2 S1 Speed')
trajectory_v2, = ax2.plot([i[0] for i in actual_stage2_trajectory], [i[1]*5./18 for i in actual_stage2_trajectory], label='OG2 S2 Speed')
tmp_plot_2, = ax2.plot([ti for ti,x0i in trajectory], [sqrt( (x0i[POS_VX] - omega*x0i[POS_Y])**2 + (x0i[POS_VY] + omega*x0i[POS_X])**2) for ti,x0i in trajectory], label='Sim Speed')
tmp_plot_vt, = ax2.plot([ti for ti,x0i in trajectory], [v_terminal for ti,x0i in trajectory], ls='dashed', alpha=0.4, label='')
tmp_plot_vt_s, = ax2.plot([ti for ti,x0i in trajectory], [v_terminal - omega*Re for ti,x0i in trajectory], ls='dashed', alpha=0.4, label='')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Speed (m/s)')
ax2.legend(handles=[trajectory_v1, trajectory_v2, tmp_plot_2, tmp_plot_vt, tmp_plot_vt_s ], loc='upper left')
plt.show()