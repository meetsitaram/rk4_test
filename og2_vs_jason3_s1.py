import matplotlib.pyplot as plt
from numpy import arcsin
from math import pi, cos, sin

g_sl = 9.81

#time,speed, altitude, range
og2_s1_trajectory = [
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

jason3_s1_trajectory = [
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

#press kit - http://www.spacex.com/sites/spacex/files/spacex_orbcomm_press_kit_final2.pdf

# LAUNCH AND FIRST-STAGE LANDING
# Hour/Min Events
# 00:01 Max Q (moment of peak mechanical stress on the rocket)
# 00:02:20 1st stage engine shutdown/main engine cutoff (MECO)
# 00:02:24 1st and 2nd stages separate
# 00:02:35 2nd stage engine starts
# 00:03 Fairing deployment
# 00:04 1st stage boostback burn
# 00:08 1st stage re-entry burn
# 00:10 2nd stage engine cutoff (SECO)
# 00:10 1st stage landing
# 00:15 ORBCOMM satellites begin deployment
# 00:20 ORBCOMM satellites end deployment
# 00:26 1st satellite completes antenna & solar array deployment & starts transmitting
# 00:31 All satellites complete antenna & solar array deployment & start transmitting
#

fig = plt.figure()
ax1 = fig.add_subplot(111)
trajectory_s1, = ax1.plot([i[0] for i in og2_s1_trajectory], [i[2] for i in og2_s1_trajectory], label='OG2')
trajectory_s2, = ax1.plot([i[0] for i in jason3_s1_trajectory], [i[2] for i in jason3_s1_trajectory], label='JASON3')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Altitude (km)')
plt.legend(handles=[trajectory_s1, trajectory_s2], loc='center left')
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(111)
trajectory_s1, = ax1.plot([i[0] for i in og2_s1_trajectory], [i[1]*5./18 for i in og2_s1_trajectory], label='OG2')
trajectory_s2, = ax1.plot([i[0] for i in jason3_s1_trajectory], [i[1]*5./18 for i in jason3_s1_trajectory], label='JASON3')
tmp_plot_vt, = ax1.plot([i[0] for i in jason3_s1_trajectory], [6000.*5./18 for i in jason3_s1_trajectory], ls='dashed', alpha=0.4, label='6000 kmph mark')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Speed (m/s)')
plt.legend(handles=[trajectory_s1, trajectory_s2, tmp_plot_vt], loc='center left')
plt.show()
