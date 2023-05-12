#!/home/leo/anaconda3/envs/py39/bin/python3

import rospy
from math import sin,cos,sqrt,atan2,acos,pi
import numpy as np
import gurobipy as gp
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelStates
from px4_mavros import Px4Controller
from gurobipy import GRB
from time import sleep, time

P1,P2,P3,PO,P4,P5,P6,A,b = None,None,None,None,None,None,None,None,None
cmd_vel = Twist()
d_safe = 1.0
m,x = None,None

def odom(msg):
	global P1,P2,P3,P4,P4v,P5,P5v,P6,P6v,PO,A,b
	
	UAV1_index = msg.name.index('iris_0')
	UAV2_index = msg.name.index('iris_1')
	UAV3_index = msg.name.index('iris_2')
	UAV4_index = msg.name.index('solo_3')
	UAV5_index = msg.name.index('solo_4')
	UAV6_index = msg.name.index('solo_5')
	# obs_index = msg.name.index('obstacle')

	P1 = np.array([msg.pose[UAV1_index].position.x, msg.pose[UAV1_index].position.y, msg.pose[UAV1_index].position.z])
	P2 = np.array([msg.pose[UAV2_index].position.x, msg.pose[UAV2_index].position.y, msg.pose[UAV2_index].position.z])
	P3 = np.array([msg.pose[UAV3_index].position.x, msg.pose[UAV3_index].position.y, msg.pose[UAV3_index].position.z])
	P4 = np.array([msg.pose[UAV4_index].position.x, msg.pose[UAV4_index].position.y, msg.pose[UAV4_index].position.z])
	P4v = np.array([msg.twist[UAV4_index].linear.x, msg.twist[UAV4_index].linear.y, msg.twist[UAV4_index].linear.z])
	P5 = np.array([msg.pose[UAV5_index].position.x, msg.pose[UAV5_index].position.y, msg.pose[UAV5_index].position.z])
	P5v = np.array([msg.twist[UAV5_index].linear.x, msg.twist[UAV5_index].linear.y, msg.twist[UAV5_index].linear.z])
	P6 = np.array([msg.pose[UAV6_index].position.x, msg.pose[UAV6_index].position.y, msg.pose[UAV6_index].position.z])
	P6v = np.array([msg.twist[UAV6_index].linear.x, msg.twist[UAV6_index].linear.y, msg.twist[UAV6_index].linear.z])
	# PO = np.array([msg.pose[obs_index].position.x, msg.pose[obs_index].position.y, msg.pose[obs_index].position.z])
	PO = np.array([100, 100, 100])

	A = np.array([ \
				  (-2*(P4-PO)[:2]).tolist() \
				  ])

	b = np.array([ \
				  np.linalg.norm((P4-PO)[:2])**2 - d_safe**2 \
				  ])

def qp_ini():
	global m,x
	
	m = gp.Model("qp")
	m.setParam("NonConvex", 2.0)
	m.setParam("LogToConsole",0)
	x = m.addVars(2,ub=0.5, lb=-0.5, name="x")

def addCons(i):
	global m

	m.addConstr(A[i,0]*x[0] + A[i,1]*x[1] <= b[i], "c"+str(i))

def	controller(time_):

	global cmd_vel
	move_gain = 5
	delay_time = 1e-4
	# tra = [3*cos(t*pi), 3*sin(t*pi)]

	if time_ > rospy.Duration(40.00) and time_ <= rospy.Duration(70.00):

		tra = [P4[0] + 0.00*move_gain, P4[1] + 0.01*move_gain]
		rospy.sleep(rospy.Duration(delay_time))
	elif time_ > rospy.Duration(70.00) and time_ <= rospy.Duration(110.00):

		tra = [P4[0] + 0.05*move_gain, P4[1] - 0.008*move_gain]
		rospy.sleep(rospy.Duration(delay_time))
	elif time_ > rospy.Duration(110.00) and time_ <= rospy.Duration(150.00):

		tra = [P4[0] + 0.007*move_gain, P4[1] - 0.058*move_gain]
		rospy.sleep(rospy.Duration(delay_time))
	else:
		tra = [P4[0], P4[1]]

	u_des = np.array([1*((tra[0] - P4[0]) + 0),\
			1*((tra[1] - P4[1]) + 0),\
			0.5 - P4[2]])

	obj = (x[0] - u_des[0])**2 + (x[1] - u_des[1])**2
	m.setObjective(obj)

	m.remove(m.getConstrs())

	for i in range (b.size):
		addCons(i)

	m.optimize()
	u_opt = m.getVars()

	cmd_vel.linear.x = 0.5*u_des[0]
	cmd_vel.linear.y = 0.5*u_des[1]
	cmd_vel.linear.z = u_des[2]

	px4_3.vel_control(cmd_vel)

if __name__ == '__main__':
	try:
		rospy.init_node('controller_4')
		px4_3 = Px4Controller("uav3")
		rospy.Subscriber('/gazebo/model_states', ModelStates, odom, queue_size=10)
		rate = rospy.Rate(100)
		while P4 is None:
			rate.sleep()

		t = 0
		qp_ini()
		last = rospy.Time.now()

		while not rospy.is_shutdown():

			controller(rospy.Time.now() - last)

			t += 5*1e-4
			rate.sleep()
	except rospy.ROSInterruptException:
		pass
