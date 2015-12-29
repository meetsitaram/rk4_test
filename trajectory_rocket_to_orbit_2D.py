import matplotlib.pyplot as plt
from math import cos, sin, pi, sqrt
from numpy import arctan

h = .01          # stepsize (sec)
g_sl = 9.81        # m/sec2
G = 6.672*10**-11
Re = 6378*1000.
Me = 5.976*10**24

# Falcon FT, Merlin specs from http://spaceflight101.com/spacerockets/falcon-9-ft/
merlin1D = {
    'isp_sl' : 282,   #sec
    'isp_vac' : 311,  #sec
    'thrust_sl' : 756000.,    # N
    'thrust_vac' : 825000.,   # N  
    'min_throttle': 55, # %       
    'max_throttle': 55, # %        
}
merlin1D['mass_flow_rate'] = merlin1D['thrust_sl']/(g_sl*merlin1D['isp_sl'])

stage1 = {
    'engines' : 9,
    'length': 41.2,     # m
    'interstage_length' : 6.75, # m
    'diameter' : 3.66,  # m
    'inert_mass' : 22200,    # kg
    'propellant_mass': 409500,  # kg  
}

stage2_and_payload = {
    'inert_mass':4000,  # kg
    'propellant_mass':103500,   # kg
    'fairing_mass': 1750,   #kg
    'payload_mass': 5000,   # less than 50% caoability for og2
}

stage2_and_payload['total_mass'] = stage2_and_payload['inert_mass'] + \
                                        stage2_and_payload['propellant_mass'] + \
                                        stage2_and_payload['fairing_mass'] + \
                                        stage2_and_payload['payload_mass']

fairing = {
    'mass': 1750,   # kg
}

#t,v(km/h),h(km), x(km)
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
    [75,1555,12.5,0],   # max Q at 73 sec
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
    [580,25991,619,0],
]



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
