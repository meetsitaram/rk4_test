import matplotlib.pyplot as plt
from numpy import arcsin
from math import pi, cos, sin



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

range_values = [
    [0,0,0,0],
    [85,1943,17.2,2.5],
    [120,3949,42.5,11],
    [360,8187,379,157],
]

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
theta_plot, = ax1.plot([i[0] for i in actual_stage2_trajectory], [ (180/pi)*arcsin(i[2]*1000/(i[1]*(5./18)*(i[0]+0.0001))) for i in actual_stage2_trajectory], label='OG2 S2 Theta')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Theta (degrees)')
plt.legend(handles=[theta_plot], loc='bottom right')
plt.show()