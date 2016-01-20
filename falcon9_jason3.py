# Simulation of Falcon 9 launch to orbit


import matplotlib.pyplot as plt
from math import cos, sin, pi, sqrt
from numpy import zeros, exp, arcsin, arctan

h = .1          # stepsize used in RK4 (sec)

air_density_sl = 1.225  # density of air (kg/m^3)
H_FOR_DRAG = 7000.   # H value used in drag model (m)
H_FOR_PRESSURE = 7000. # constant temperature model
ATMOSPHERE_HEIGHT = 90000 # 90 KM - we can ignore air drag beyond this
G = 6.67384e-11   # universal gravitation constant
Re = 6371e3     # radius of earch (m)
Me = 5.9722e24   # mass of earth (kg)
mu = G*Me

g_sl = G*Me/Re**2


draf_coefficient = 0.3    # drag coefficient - temporary
#falcin 8 Full Thrust specs from http://spaceflight101.com/spacerockets/falcon-9-ft/
s1_inert_mass = 23100.    # Kg
s1_propellant_mass = 395700.    # kg
s1_merlin_1D_thrust_sl = 654000. # N
s1_merlin_1D_thrust_vac1 = 716000. # N 
s1_merlin_1D_specific_impulse_sl = 282.*g_sl   # m/s
s1_merlin_1D_specific_impulse_vac1 = 311.*g_sl   # m/s
s1_burn_time_end = 155.  # s

s2_inert_mass = 3900.    # Kg
s2_propellant_mass = 92670.    # kg
s2_merlin_1D_thrust_vac1 = 801000. # N
s2_merlin_1D_specific_impulse_vac1 = 345.*g_sl   # m/s 
s2_burn_time_start =  165.
s2_burn_time_end = 545.

fairing_mass = 1750. # kg
payload_mass = 550. # kg 

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

diameter = 3.66
length= 68.4
ref_area = diameter**2 * pi / 4.   #  reference cross section area - used for drag (m^2)
cross_sec_area = diameter*length
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
TMP_POS_Q = 12
TMP_POS_ANGLE_OF_ATTACK = 13

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
current_angle_of_attack = 0.
drag_losses = 0.
gravity_losses = 0.

engines_on = True

periapsis_radius = 0.
apoapsis_radius = 0.
semimajor_axis = 0.
eccentricity = 0.

#time,speed, altitude, range
actual_stage1_trajectory = [
    [0,0,0,0],
    [5,38,0,0],
    [6,47,.1,0],
    [7,51,.1,0],
    [8,58,.1,0],
    [9,66,.2,0],
    [10,74,.2,0],
    [11,83,.2,0],
    [12,91,.2,0],
    [13,100,.2,0],
    [14,110,.3,0],
    [15,120,.3,0],
    [20,172,.5,0],
    [25,225,.8,0], # pitching down range??
    [30,289,1.1,0],
    [35,361,1.6,0],
    [40,433,2.1,0],
    [45,518,2.8,0],
    [50,612,3.5,0],
    [55,716,4.5,0],
    [60,831,5.5,0],
    [65,954,6.7,1.1],
    [70,1097,8.2,0],
    [75,1240,9.7,0],  
    [80,1384,11.4,0],
    [85,1556,13.3,0], # max Q
    [90,1734,15.4,0],
    [95,1931,17.7,0],
    [100,2149,20.2,0],
    [105,2398,23.1,0],
    [110,2656,26,0],
    [115,2947,29.4,0],
    [120,3259,33.0,0], # vehicle is supersonic ??? looks wrong!!
    [125,3572,36.6,0], # vehicle reached max. aerodynamic pressure?? looks wrong!!
    [130,3954,41.0,0],
    [135,4320,45.2,0],
    [140,4750,50.1,0],
    [145,5235,55.6,0],
    [150,5732,61.1,43],
    [155,6146,66.8,0], # MECO
]

actual_stage2_trajectory = [

    [165,5932,78.4,0], #
    [166,5931,79.5,0],
    [167,5934,80.4,0],
    [170,5950,83.7,0],
    [175,5984,89.1,0],
    [180,6024,94.4,0],
    [185,6068,99.5,0],
    [190,6117,105,0],
    [195,6170,110,0], # fairing separation
    [200,6235,115,0],
    [210,6378,124,0],
    [220,6542,133,0],
    [230,6727,141,154],
    [240,6932,150,0],
    [250,7166,157,0],
    [260,7414,164,0],
    [267,7632,170,0], #boost back started
    [270,7687,171,0], 
    [280,7989,177,0],
    [290,8311,182,0],
    [300,8648,187,0],
    [305,8854,190,0], #boost back shutdown
    [310,9029,191,0],
    [320,9429,195,0],
    [330,9852,198,0],
    [340,10295,200,400],
    [350,10769,202,0],
    [360,11251,203,0],
    [370,11771,204,0],
    [380,12301,205,0],
    [390,12876,205,0],
    [400,13496,205,590],
    [410,14112,205,0],
    [420,14787,204,0],
    [423,15000,204,0], # stage1 engine burn started
    [430,15494,204,0],
    [440,16224,202,0],
    [447,16781,202,0], # stage1 engine shutdown
    [450,17032,201,0],
    [460,17857,200,0],
    [470,18757,199,899],
    [480,19709,197,0],
    [490,20770,196,0],
    [500,21911,195,0], # stage1 is transonic
    [510,23115,194,0],
    [516,24021,194,0], # landing startup stage 1
    [520,24560,194,0],
    [530,26098,194,0],
    [540,27805,195,0],
    [545,28444,195,0], # SECO, s1 landing as well?
]


range_values = [
    [0,0,0,0],
    [65,954,6.7,1.1],
    [150,5732,61.1,43],
    [230,6727,141,154],
    [340,10295,200,400],
    [400,13496,205,590],
    [470,18757,199,899],
]

boost_back_start = 267.
boost_back_end = 305.

reentry_start = 423.
reentry_end = 447.

landing_start = 515.
landing_end = 545.

#typical losses
# http://forum.nasaspaceflight.com/index.php?topic=9959.msg189860#msg189860
# Ariane A-44L: Gravity Loss: 1576 m/s Drag Loss: 135 m/s
# Atlas I: Gravity Loss: 1395 m/s Drag Loss: 110 m/s
# Delta 7925: Gravity Loss: 1150 m/s Drag Loss: 136 m/s
# Shuttle: Gravity Loss: 1222 m/s Drag Loss: 107 m/s
# Saturn V: Gravity Loss: 1534 m/s Drag Loss: 40 m/s (!!)
# Titan IV/Centaur: Gravity Loss: 1442 m/s Drag Loss: 156 m/s

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
    g = G*Me/r**2
    altitude = r - Re
    global isp, stage_separation
    if stage_separation == True:
        isp = s2_merlin_1D_specific_impulse_vac1
    else:    
        isp = (s1_merlin_1D_specific_impulse_sl - s1_merlin_1D_specific_impulse_vac1)*exp(-altitude/H_FOR_PRESSURE) + s1_merlin_1D_specific_impulse_vac1
        #isp = s1_merlin_1D_specific_impulse_sl 
    

    vy_surf = vy #- omega*(-x)
    vx_surf = vx #- omega*y

       
    if t >= 0 and t < s1_burn_time_end and propellant_mass > h*9.*s1_m1D_flow_rate:
        
        
        
        if t <= 25.5:
            thrust_angle = 90.
            
        elif t<= 30:
            thrust_angle = 89.1

        else:
            thrust_angle = (180./pi)*arctan(vy_surf/vx_surf)
            
        
        engines_on = True
        
        flow_rate = 9.*s1_m1D_flow_rate
        
        thrust_total = flow_rate*isp
        
        throttle = 1.
        
        flow_rate = flow_rate*throttle
        thrust_total = flow_rate*isp
        #print 'PITCH_KICK_ANGLE diff', (180./pi)*arctan(vy_surf/ (vx_surf+0.00001)) - thrust_angle        
            
    elif t >= s2_burn_time_start and t < s2_burn_time_end and propellant_mass > h*1.*s2_m1D_flow_rate:
        engines_on = True
  
        second_stage_cut_off = t

        throttle = 1.
        if t > 470:
            throttle = 0.8
        elif t > 420:
            throttle = 0.95
            
        correction_angle = 0.
        if t < 420:
            correction_angle = 7.
        elif t > 420 :
            correction_angle = 8.
        thrust_angle = (180./pi)*arctan(vy_surf/ vx_surf) + correction_angle 

        flow_rate = throttle*1.*s2_m1D_flow_rate
        thrust_total = flow_rate*isp        
        
    
    global current_thrust, current_thrust_angle, current_flow_rate, current_angle_of_attack
    current_thrust = thrust_total
    current_flow_rate = flow_rate
    current_thrust_angle = thrust_angle
    current_angle_of_attack = thrust_angle -  (180./pi)*arctan(vy_surf/ vx_surf)
    
    if engines_on == False:
        current_flow_rate = 0.
        current_thrust = 0.
        current_thrust_angle = 0.   
        current_angle_of_attack = 0.
    
    #print 't, thrust_angle', t, thrust_angle

    a_rocket_total = thrust_total/mass
        

    a_rocket = [a_rocket_total*cos(pi*thrust_angle/180.), a_rocket_total*sin(pi*thrust_angle/180.)]  
   
        
    return a_rocket
        
def get_drag_acceleration(vx, vy, x, y):
    SIMPLE_AIR_MODEL = True
    
    global ENABLE_DRAG
    if not ENABLE_DRAG:
        return [0., 0.]
    
    r = sqrt(x*x+y*y)
    height = r - Re
    if height < 0:
        return [0., 0.] 
        
    if height > ATMOSPHERE_HEIGHT:
        ENABLE_DRAG = False     # TODO remove later.
        return [0., 0.]
        
    #v_air = omega*r     # velocity of air
    #vx_air = v_air*(y/r)
    #vy_air = v_air*(-x/r)
    #vx1 = vx - vx_air
    #vy1 = vy - vy_air
    
        
    #v = sqrt(vx1**2 + vy1**2)
    v = sqrt(vx**2 + vy**2)

    if v == 0:
        return [0., 0.]
    
        
    if SIMPLE_AIR_MODEL:
        air_density = air_density_sl * exp ( - height / H_FOR_DRAG)
    
    else:
        # atmosphere model - https://www.grc.nasa.gov/www/k-12/rocket/atmos.html
            
        temperature = 0.
        pressure = 0.
        if height < 36152*0.3048:
            temperature = 59 - .00356 * height/0.3048
            pressure = 2116 * ((temperature + 459.7)/ 518.6)**5.256        
        elif height < 82345*0.3048:
            temperature = -70
            pressure = 473.1 * exp(1.73 - .000048 * height/0.3048)        
        else:
            temperature =  -205.05 + .00164 * height/0.3048
            pressure =  51.97 * ((temperature + 459.7)/ 389.98)**-11.388
            
        air_density = pressure / (1718 * (temperature + 459.7))
        air_density = 515.4*air_density #convert from slugs per cub ft
        
    #print air_density
    
    global current_dynamic_pressure
    vy_surf = vy #- omega*(-x)
    vx_surf = vx #- omega*y
    v_surf = sqrt(vx_surf**2 + vy_surf**2)
    current_dynamic_pressure = 0.5*air_density*v_surf**2
    
    f_drag = (1./2) * air_density * draf_coefficient * ref_area * v**2    
    a_drag = f_drag/mass
    #print 'a_drag', a_drag    
    
    g = G*Me/r**2
    
    global max_drag_acceleration
    if a_drag > max_drag_acceleration*g:
        max_drag_acceleration = a_drag / g
        
    global curr_drag_acceleration 
    curr_drag_acceleration = a_drag / g
        
    return [-a_drag*vx/v, -a_drag*vy/v]
   
def get_gravity_acceleration(x,y):
    
    r = sqrt(x*x+y*y)
    
    if r == 0:
        return [0., 0.]
    
    g = G*Me/r**2
    
    return [-g*x/r, -g*y/r]    
    
def get_acceleration(x, y, vx, vy, t):
    
    g_sl = G*Me/Re**2
    
    r = sqrt(x**2 + y**2)
    g = G*Me/r**2
    
    a_drag = get_drag_acceleration(vx, vy, x, y)
    a_gravity = get_gravity_acceleration(x,y)
    a_rocket = get_rocket_acceleration(x,y,t,vx, vy)
    
    a = [a_drag[0]+a_gravity[0]+a_rocket[0], a_drag[1]+a_gravity[1]+a_rocket[1]]
    
    a_magnitude = sqrt(a[0]**2 + a[1]**2)
    global max_acceleration
    if a_magnitude > max_acceleration*g:
        max_acceleration = a_magnitude  / g
        
    global curr_acceleration 
    curr_acceleration = a_magnitude / g 

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
    args0[POS_VX] = 0. #omega*Re
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
            inert_mass = s1_inert_mass + s2_inert_mass + s2_propellant_mass + fairing_mass + payload_mass   # Kg
            propellant_mass = s1_propellant_mass    # kg
            mass = inert_mass + s1_propellant_mass 
            
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
        tmp_args.append(current_angle_of_attack)
        trajectory.append( (t, tmp_args) )     
        
        vy_surf = args0[POS_VY]
        vx_surf = args0[POS_VX]
        v_surf = sqrt(vx_surf**2 + vy_surf**2)
        r = sqrt(args0[POS_X]**2 + args0[POS_Y]**2)
        g= G*Me/r**2
        
        #(3900+92670+23100+1750+550)*exp(3000/(311*9.81)) -> 326062 fuel expended    
        global gravity_losses, drag_losses
        if engines_on == True and r > 0 and v_surf > 0:
            gravity_losses += - (h*((vx_surf/v_surf)*g*(-args0[POS_X]/r) + (vy_surf/v_surf)*g*(-args0[POS_Y]/r) ) )
            drag_losses += h * curr_drag_acceleration*g      
        
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
tmp_plot_1, = ax1.plot([x0i[POS_X]/1000. for ti,x0i in trajectory  if ti < 155], [ (x0i[POS_Y] - Re)/1000. for ti,x0i in trajectory  if ti < 155], color='r')
#tmp_plot_1, = ax1.plot([ (x0i[POS_X]*cos(omega*ti) - x0i[POS_Y]*sin(omega*ti))/1000. for ti,x0i in trajectory ], [ (sqrt(x0i[POS_X]**2 + x0i[POS_Y]**2) - Re)/1000.  for ti,x0i in trajectory ], color='r', label='Trajectory')
tmp_plot_2x, = ax2.plot([ti/60. for ti,x0i in trajectory], [abs(x0i[POS_VX])/1000. for ti,x0i in trajectory], ls='dashed', alpha=0.4, label='Vx')
tmp_plot_2y, = ax2.plot([ti/60. for ti,x0i in trajectory], [abs(x0i[POS_VY])/1000. for ti,x0i in trajectory],  ls='dashed', alpha=0.4,label='Vy')
tmp_plot_2, = ax2.plot([ti/60. for ti,x0i in trajectory], [sqrt(x0i[POS_VX]**2 + x0i[POS_VY]**2)/1000. for ti,x0i in trajectory], label='V')
tmp_plot_3, = ax3.plot([ti/60. for ti,x0i in trajectory if ti < 600], [x0i[TMP_POS_A] for ti,x0i in trajectory  if ti < 600], label='max ' + str(round(max_acceleration,1)) + 'g')
tmp_plot_4, = ax4.plot([ti/60. for ti,x0i in trajectory if ti < 155], [(sqrt(x0i[POS_X]**2 + x0i[POS_Y]**2) - Re)/1000. for ti,x0i in trajectory if ti < 155])
tmp_plot_5, = ax5.plot([ti/60. for ti,x0i in trajectory if ti < 600], [x0i[TMP_POS_TWR] for ti,x0i in trajectory if ti < 600])
tmp_plot_6, = ax6.plot([ti/60. for ti,x0i in trajectory if ti < 600], [x0i[TMP_POS_THRUST_ANGLE] for ti,x0i in trajectory if ti < 600])
#tmp_plot_7, = ax7.plot([ti/60. for ti,x0i in trajectory if ti < 450], [int(round(x0i[TMP_POS_ISP]/g_sl)) for ti,x0i in trajectory if ti < 450], label='isp_sl=282, isp_vac=311')
#tmp_plot_7, = ax7.plot([ti/60. for ti,x0i in trajectory if ti < 600], [x0i[TMP_POS_Q] for ti,x0i in trajectory if ti < 600])
tmp_plot_7, = ax7.plot([ti/60. for ti,x0i in trajectory if ti < 155], [x0i[POS_X]/1000. for ti,x0i in trajectory if ti < 155])
tmp_plot_8, = ax8.plot([ti/60. for ti,x0i in trajectory if ti < 155], [x0i[TMP_POS_A_DRAG] for ti,x0i in trajectory if ti < 155])
#ax1_handles.append(tmp_plot_1)
ax2_handles.append(tmp_plot_2)
ax2_handles.append(tmp_plot_2x)
ax2_handles.append(tmp_plot_2y)
ax3_handles.append(tmp_plot_3)
ax4_handles.append(tmp_plot_4)
ax5_handles.append(tmp_plot_5)
ax6_handles.append(tmp_plot_6)
ax7_handles.append(tmp_plot_7)

earth = ax1.add_artist(plt.Circle((0,0),Re/1000.0,color="b",fill=False))
ax1.add_artist(plt.Circle((0,0),(Re+ATMOSPHERE_HEIGHT)/1000.0,color="r", ls='dashed', fill=False, alpha=0.4))
#ax1.add_artist(plt.Circle((0,0),(Re/1.0e3+25),color="b",alpha=0.1))
#ax1.add_artist(plt.Circle((0,0),(Re/1.0e3+50),color="b",alpha=0.05))
#ax1.add_artist(plt.Circle((0,0),(Re/1.0e3+90),color="b",alpha=0.04))

  
#ax1_handles.append(earth)
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
#ax7.set_ylabel('Q')
ax7.set_ylabel('Range')
ax8.set_xlabel('Time (min)')
ax8.set_ylabel('Drag A')
#ax1.legend(handles=ax1_handles, loc='center')
#ax2.legend(handles=ax2_handles, loc='lower right')
#ax3.legend(handles=ax3_handles, loc='upper right')
#ax5.legend(handles=ax5_handles, loc='upper right')
#ax6.legend(handles=ax6_handles, loc='upper right')
#ax7.legend(handles=ax7_handles, loc='upper right')
plt.show()


fig1, ((ax1,ax2)) = plt.subplots(1, 2)

trajectory_s1, = ax1.plot([i[0] for i in actual_stage1_trajectory], [i[2] for i in actual_stage1_trajectory], label='S1')
trajectory_s2, = ax1.plot([i[0] for i in actual_stage2_trajectory] , [i[2] for i in actual_stage2_trajectory ], label='S2')
trajectory_d, = ax1.plot([ ti for ti,x0i in trajectory  if ti < 600], [ (sqrt(x0i[POS_X]**2 + x0i[POS_Y]**2) - Re)/1000. for ti,x0i in trajectory  if ti < 600], color='r', label='Sim')
#tmp_plot_oa, = ax1.plot([ti for ti,x0i in trajectory if ti < 160], [orbit_altitude/1000. for ti,x0i in trajectory  if ti < 160], ls='dashed', alpha=0.4, label='')
#trajectory_s2, = ax1.plot([ ti for ti,x0i in trajectory if ti < 450], [ (x0i[POS_Y]-Re)/1000. for ti,x0i in trajectory if ti < 450], color='r', label='Trajectory')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Altitude (km)')
ax1.legend(handles=[trajectory_s1, trajectory_s2, trajectory_d], loc='upper left')


trajectory_v1, = ax2.plot([i[0] for i in actual_stage1_trajectory ], [i[1]*5./18 for i in actual_stage1_trajectory], label='S1')
trajectory_v2, = ax2.plot([i[0] for i in actual_stage2_trajectory], [i[1]*5./18 for i in actual_stage2_trajectory], label='S2')
tmp_plot_2, = ax2.plot([ti for ti,x0i in trajectory if ti < 600], [sqrt( x0i[POS_VX]**2 + x0i[POS_VY]**2) for ti,x0i in trajectory if ti < 600], label='Sim')
#tmp_plot_vt, = ax2.plot([ti for ti,x0i in trajectory], [v_terminal for ti,x0i in trajectory], ls='dashed', alpha=0.4)
#tmp_plot_vt_s, = ax2.plot([ti for ti,x0i in trajectory], [v_terminal - omega*Re for ti,x0i in trajectory], ls='dashed', alpha=0.4)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Speed (m/s)')
ax2.legend(handles=[trajectory_v1, trajectory_v2, tmp_plot_2], loc='upper left')


#aoa, = ax3.plot([ti for ti,x0i in trajectory], [x0i[TMP_POS_ANGLE_OF_ATTACK] for ti,x0i in trajectory], label='')
#ax3.set_xlabel('Time (s)')
#ax3.set_ylabel('Angle of Attack')
#ax3.legend(handles=[aoa ], loc='upper right')
plt.show()

print 'max_drag_acceleration:', round(max_drag_acceleration,1), 'g, max_acceleration:', round(max_acceleration,1), 'g, max_altitude:', round(max_altitude/1000.), 'km, down_range:', round(down_range/1000.), 'km'
print 'second_stage_cut_off:', second_stage_cut_off, 'propellant_left s1/s2:', s1_propellant_mass, s2_propellant_mass
print 'gravity_losses', gravity_losses, 'drag_losses', drag_losses