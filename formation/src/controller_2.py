#!/usr/bin/python

import rospy
from math import sin,cos,sqrt,atan2,acos,pi
import numpy as np
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelStates
from px4_mavros import Px4Controller

P1,P2,P3 = None,None,None
cmd_vel = Twist()

def odom(msg):
	global P1,P2,P3
	
	UAV1_index = msg.name.index('iris_1')
	UAV2_index = msg.name.index('iris_2')
	UAV3_index = msg.name.index('iris_3')

	P1 = np.array([msg.pose[UAV1_index].position.x, msg.pose[UAV1_index].position.y, msg.pose[UAV1_index].position.z])
	P2 = np.array([msg.pose[UAV2_index].position.x, msg.pose[UAV2_index].position.y, msg.pose[UAV2_index].position.z])
	P3 = np.array([msg.pose[UAV3_index].position.x, msg.pose[UAV3_index].position.y, msg.pose[UAV3_index].position.z])
	
def	controller():
	global cmd_vel
	
	cmd_vel.linear.x = 1*((P1[0] - P2[0]) + 2.0 + (P3[0] - P2[0]) + 1.0)
	cmd_vel.linear.y = 1*((P1[1] - P2[1]) + 0.0 + (P3[1] - P2[1]) - sqrt(3))
	cmd_vel.linear.z = 0.5 - P2[2]

	px4_2.vel_control(cmd_vel)

if __name__ == '__main__':
	try:
		rospy.init_node('controller_2')
		px4_2 = Px4Controller("iris_2")
		rospy.Subscriber('/gazebo/model_states', ModelStates, odom, queue_size=10)
		rate = rospy.Rate(100)
		while P3 is None:
			rate.sleep()

		while not rospy.is_shutdown():
			controller()
			rate.sleep()
	except rospy.ROSInterruptException:
		pass
