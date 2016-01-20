import matplotlib.pyplot as plt
from numpy import arcsin
from math import pi, cos, sin

g_sl = 9.81

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

actual_stage2_trajectory_2 = [
    [2820,24277,998,0],
#    [,,,],
#    [,,,],
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


fig = plt.figure()
ax1 = fig.add_subplot(111)
trajectory_s1, = ax1.plot([i[0] for i in actual_stage1_trajectory], [i[2] for i in actual_stage1_trajectory], label='OG2 S1 Altitude')
trajectory_s2, = ax1.plot([i[0] for i in actual_stage2_trajectory], [i[2] for i in actual_stage2_trajectory], label='OG2 S2 Altitude')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Altitude (km)')
plt.legend(handles=[trajectory_s1, trajectory_s2], loc='center left')
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(111)
trajectory_s1, = ax1.plot([i[0] for i in actual_stage1_trajectory], [i[1]*5./18 for i in actual_stage1_trajectory], label='OG2 S1 Speed')
trajectory_s2, = ax1.plot([i[0] for i in actual_stage2_trajectory], [i[1]*5./18 for i in actual_stage2_trajectory], label='OG2 S2 Speed')
velocity_vx_s1, = ax1.plot([i[0] for i in actual_stage1_trajectory], [ i[1]*(5./18)*cos(arcsin(i[2]*1000./((i[1]+0.00001)*(5./18)*(i[0]+0.0001)))) for i in actual_stage1_trajectory], label='S1 Vx')
velocity_vy_s1, = ax1.plot([i[0] for i in actual_stage1_trajectory], [ i[1]*(5./18)*sin(arcsin(i[2]*1000./((i[1]+0.00001)*(5./18)*(i[0]+0.0001)))) for i in actual_stage1_trajectory], label='S1 Vy')
velocity_vx_s2, = ax1.plot([i[0] for i in actual_stage2_trajectory], [ i[1]*(5./18)*cos(arcsin(i[2]*1000./(i[1]*(5./18)*(i[0]+0.0001)))) for i in actual_stage2_trajectory], label='Vx')
velocity_vy_s2, = ax1.plot([i[0] for i in actual_stage2_trajectory], [ i[1]*(5./18)*sin(arcsin(i[2]*1000./(i[1]*(5./18)*(i[0]+0.0001)))) for i in actual_stage2_trajectory], label='Vy')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Speed (m/s)')
plt.legend(handles=[trajectory_s1, trajectory_s2, velocity_vx_s1, velocity_vy_s1, velocity_vx_s2, velocity_vy_s2], loc='upper left')
plt.show()


fig = plt.figure()
ax1 = fig.add_subplot(111)
trajectory_both, = ax1.plot([i[3] for i in range_values], [i[1] for i in range_values], label='OG2 S1 Trajectory')
ax1.set_xlabel('Range (km)')
ax1.set_ylabel('Altitude (km)')
plt.legend(handles=[trajectory_both], loc='bottom right')
plt.show()


fig = plt.figure()
ax1 = fig.add_subplot(111)
theta1_plot, = ax1.plot([i[0] for i in actual_stage1_trajectory], [ (180/pi)*arcsin(i[2]*1000/((i[1]+0.001)*(5./18)*(i[0]+0.0001))) for i in actual_stage1_trajectory], label='OG2 S1 Theta')
theta2_plot, = ax1.plot([i[0] for i in actual_stage2_trajectory], [ (180/pi)*arcsin(i[2]*1000/((i[1]+0.001)*(5./18)*(i[0]+0.0001))) for i in actual_stage2_trajectory], label='OG2 S2 Theta')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Theta (degrees)')
plt.legend(handles=[theta1_plot, theta2_plot], loc='bottom right')
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(111)
acc_pts = []
        
theta_plot, = ax1.plot([i[0] for i in actual_stage1_trajectory if i[0]>0], [ (5./18.)*(pt[1] - actual_stage1_trajectory[idx-1][1])/(pt[0]-actual_stage1_trajectory[idx-1][0])/g_sl  for idx, pt in enumerate(actual_stage1_trajectory) if pt[0] > 0], label='OG2 S1 Acc')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Acceleration (g)')
plt.legend(handles=[theta_plot], loc='bottom right')
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(111)
acc_pts = []
        
theta_plot, = ax1.plot([pt[0] for idx,pt in enumerate(actual_stage2_trajectory) if idx>0], [ (5./18.)*(pt[1] - actual_stage2_trajectory[idx-1][1])/(pt[0]-actual_stage2_trajectory[idx-1][0])/g_sl  for idx, pt in enumerate(actual_stage2_trajectory) if idx > 0], label='OG2 S2 Acc')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Acceleration (g)')
plt.legend(handles=[theta_plot], loc='bottom right')
plt.show()