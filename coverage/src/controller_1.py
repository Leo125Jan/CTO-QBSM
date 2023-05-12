#!/home/leo/anaconda3/envs/py39/bin/python3

import warnings
warnings.simplefilter(action = 'ignore', category = RuntimeWarning)

import sys
import csv
import rospy
import random
import numpy as np
from QBSC import PTZcon
from time import sleep, time
from sensor_msgs.msg import Imu
from pyquaternion import Quaternion
from geometry_msgs.msg import Twist
from px4_mavros import Px4Controller
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Float64MultiArray
from math import sin, cos, tan, sqrt, atan2, acos, pi, exp

class UAV():

	def __init__(self):

		# Variable of Control
		self.P1, self.P1o, self.P2, self.P2o, self.P3, self.P3o = None, None, None, None, None, None
		self.P4, self.P4v, self.P5, self.P5v, self.P6, self.P6v = None, None, None, None, None, None
		self.cmd_vel = Twist()
		self.px4 = Px4Controller("uav0")

		# Publisher & Subscriber
		self.imu_sub = rospy.Subscriber("/uav0/mavros/imu/data", Imu, self.imu_callback, queue_size = 20, buff_size = 52428800)
		self.odom_sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.odom_callback, queue_size = 20, buff_size = 52428800)

		self.fov_sub_1 = rospy.Subscriber("/uav1/fov", Float64MultiArray, self.fov_1_callback, queue_size = 20, buff_size = 52428800)
		self.fov_sub_2 = rospy.Subscriber("/uav2/fov", Float64MultiArray, self.fov_2_callback, queue_size = 20, buff_size = 52428800)
		self.cluster_dist_sub_1 = rospy.Subscriber("/uav1/cluster_dist", Float64MultiArray,\
												self.cluster_dist_1_callback, queue_size = 20, buff_size = 52428800)
		self.cluster_dist_sub_2 = rospy.Subscriber("/uav2/cluster_dist", Float64MultiArray,\
												self.cluster_dist_2_callback, queue_size = 20, buff_size = 52428800)
		self.cluster_teammate_sub_1 = rospy.Subscriber("/uav1/cluster_teammate", Float64MultiArray,\
												self.cluster_teammate_1_callback, queue_size = 20, buff_size = 52428800)
		self.cluster_teammate_sub_2 = rospy.Subscriber("/uav2/cluster_teammate", Float64MultiArray,\
												self.cluster_teammate_2_callback, queue_size = 20, buff_size = 52428800)

		size = int(25/0.1)
		self.current_heading = 0.0
		self.neighor_1_fov = np.zeros(size)
		self.neighor_2_fov = np.zeros(size)
		self.neighor_1_cluster_dist = np.array([100, 100, 100])
		self.neighor_2_cluster_dist = np.array([100, 100, 100])
		self.neighor_1_cluster_teammate = np.array([None, None])
		self.neighor_2_cluster_teammate = np.array([None, None])

	def odom_callback(self, msg):
		
		UAV1_index = msg.name.index('iris_0')
		UAV2_index = msg.name.index('iris_1')
		UAV3_index = msg.name.index('iris_2')
		UAV4_index = msg.name.index('solo_3')
		UAV5_index = msg.name.index('solo_4')
		UAV6_index = msg.name.index('solo_5')

		P1 = np.array([msg.pose[UAV1_index].position.x, msg.pose[UAV1_index].position.y, msg.pose[UAV1_index].position.z])
		P1o = np.array([msg.pose[UAV1_index].orientation.x, msg.pose[UAV1_index].orientation.y,\
						msg.pose[UAV1_index].orientation.z, msg.pose[UAV1_index].orientation.w])
		P2 = np.array([msg.pose[UAV2_index].position.x, msg.pose[UAV2_index].position.y, msg.pose[UAV2_index].position.z])
		P2o = np.array([msg.pose[UAV2_index].orientation.x, msg.pose[UAV2_index].orientation.y,\
						msg.pose[UAV2_index].orientation.z, msg.pose[UAV2_index].orientation.w])
		P3 = np.array([msg.pose[UAV3_index].position.x, msg.pose[UAV3_index].position.y, msg.pose[UAV3_index].position.z])
		P3o = np.array([msg.pose[UAV3_index].orientation.x, msg.pose[UAV3_index].orientation.y,\
						msg.pose[UAV3_index].orientation.z, msg.pose[UAV3_index].orientation.w])

		P4 = np.array([msg.pose[UAV4_index].position.x, msg.pose[UAV4_index].position.y, msg.pose[UAV4_index].position.z])
		P4v = np.array([msg.twist[UAV4_index].linear.x, msg.twist[UAV4_index].linear.y, msg.twist[UAV4_index].linear.z])
		P5 = np.array([msg.pose[UAV5_index].position.x, msg.pose[UAV5_index].position.y, msg.pose[UAV5_index].position.z])
		P5v = np.array([msg.twist[UAV5_index].linear.x, msg.twist[UAV5_index].linear.y, msg.twist[UAV5_index].linear.z])
		P6 = np.array([msg.pose[UAV6_index].position.x, msg.pose[UAV6_index].position.y, msg.pose[UAV6_index].position.z])
		P6v = np.array([msg.twist[UAV6_index].linear.x, msg.twist[UAV6_index].linear.y, msg.twist[UAV6_index].linear.z])

		self.P1, self.P1o, self.P2, self.P2o, self.P3, self.P3o = P1, P1o, P2, P2o, P3, P3o
		self.P4, self.P4v, self.P5, self.P5v, self.P6, self.P6v = P4, P4v, P5, P5v, P6, P6v

	def imu_callback(self, msg):

		self.current_heading = self.q2yaw(msg.orientation)

	def q2yaw(self, q):

		if isinstance(q, Quaternion):

			rotate_z_rad = q.yaw_pitch_roll[0]
		else:

			q_ = Quaternion(q.w, q.x, q.y, q.z)
			rotate_z_rad = q_.yaw_pitch_roll[0]

		return rotate_z_rad 

	def fov_1_callback(self, msg):

		self.neighor_1_fov = msg.data

	def fov_2_callback(self, msg):

		self.neighor_2_fov = msg.data

	def cluster_dist_1_callback(self, msg):

		self.neighor_1_cluster_dist = msg.data

	def cluster_dist_2_callback(self, msg):

		self.neighor_2_cluster_dist = msg.data

	def cluster_teammate_1_callback(self, msg):

		self.neighor_1_cluster_teammate = msg.data

	def cluster_teammate_2_callback(self, msg):

		self.neighor_2_cluster_teammate = msg.data

	def Collect_Data(self):

		# Targets and Self States 
		self.targets = [[(self.P4[0], self.P4[1]), 1, 10], [(self.P5[0], self.P5[1]), 1, 10],\
						[(self.P6[0], self.P6[1]), 1, 10]]

		self.pos = np.array([self.P1[0], self.P1[1]])
		theta = self.current_heading
		self.perspective = np.array([1*cos(theta), 1*sin(theta)])

		self.states = {"Position": self.pos, "Perspective": self.perspective}

		# Neighbors States
		self.neighbors_fov = np.array([self.neighor_1_fov, self.neighor_2_fov])

		self.neighor_1_cluster_dist = np.array(self.neighor_1_cluster_dist)
		self.neighor_2_cluster_dist = np.array(self.neighor_2_cluster_dist)
		self.neighor_1_cluster_teammate = np.array(self.neighor_1_cluster_teammate)
		self.neighor_2_cluster_teammate = np.array(self.neighor_2_cluster_teammate)

		self.neighbors_cluster_dist = np.array(\
		[self.neighor_1_cluster_dist if np.all(self.neighor_1_cluster_dist != None) else [None, None, None],\
		self.neighor_2_cluster_dist if np.all(self.neighor_2_cluster_dist != None) else [None, None, None]])

		self.neighbors_cluster_teammate = np.array(\
		[self.neighor_1_cluster_teammate if np.all(self.neighor_1_cluster_teammate != None) else [None, None, None],\
		self.neighor_2_cluster_teammate if np.all(self.neighor_2_cluster_teammate != None) else [None, None, None]])

		self.neighbors_position = np.array([[self.P2[0], self.P2[1]], [self.P3[0], self.P3[1]]])

		self.neighbors = {"Position": self.neighbors_position,"FoV": self.neighbors_fov,\
			"Cluster_Dist": self.neighbors_cluster_dist, "Cluster_Teammate": self.neighbors_cluster_teammate}

		# print(self.perspective)

	def	controller(self, dx, dp, step):

		'''
		P1, P2, P3, P4, P4v, PO = self.P1, self.P2, self.P3, self.P4, self.P4v, self.PO
		
		u_des = np.array([1.0*( (P3[0] - P1[0]) + 1.0 + (P2[0] - P1[0]) + 2.0 + (P4[0] - P1[0]) + 1 + P4v[0]),\
			1.0*( (P3[1] - P1[1]) - sqrt(3) + (P2[1] - P1[1]) + 0.0 + (P4[1] - P1[1]) - 0.5*sqrt(3) + P4v[1]),\
			0.7 - P1[2]])

		obj = (self.x[0] - u_des[0])**2 + (self.x[1] - u_des[1])**2
		self.m.setObjective(obj)
		self.m.remove(self.m.getConstrs())

		for i in range (self.b.size):

			self.addCons(i)

		self.m.optimize()
		u_opt = self.m.getVars()

		self.cmd_vel.linear.x = u_opt[0].X
		self.cmd_vel.linear.y = u_opt[1].X
		self.cmd_vel.linear.z = u_des[2]
		'''

		self.cmd_vel.linear.x = dx[0]*step
		self.cmd_vel.linear.y = dx[1]*step
		self.cmd_vel.linear.z = 0.5 - self.P1[2]

		persp_t = self.perspective + dp*step
		persp_t /= np.linalg.norm(persp_t)

		axis = np.cross(self.perspective, persp_t)
		axis = axis/np.linalg.norm(axis)
		dot_product = np.dot(self.perspective, persp_t)
		dtheta = np.arccos( dot_product/(np.linalg.norm(self.perspective) * np.linalg.norm(persp_t)) )

		self.cmd_vel.angular.z = dtheta*axis

		self.px4.vel_control(self.cmd_vel)

if __name__ == '__main__':

	try:
		rospy.init_node('controller_1')
		rate = rospy.Rate(100)

		map_size = np.array([20, 20])
		grid_size = np.array([0.1, 0.1])

		camera0 = { 'id'            :  0,
					'ecode'         :  "/uav0",
					'position'      :  np.array([2.0, 0.0]),
					'perspective'   :  np.array([1.0, 0.0]),
					'AngleofView'   :  20,
					'range_limit'   :  4.5,
					'lambda'        :  2,
					'color'         : (200, 0, 0)}

		ptz_1 = PTZcon(camera0, map_size, grid_size)
		uav_1 = UAV()

		while uav_1.P1 is None:

			rate.sleep()

		last = time()

		while not rospy.is_shutdown():

			past = time()
			uav_1.Collect_Data()
			dx, dp, step = ptz_1.UpdateState(uav_1.targets, uav_1.neighbors, uav_1.states,\
												np.round(time() - last, 2))
			uav_1.controller(dx, dp, step)
			print("Calculation Time 1: " + str(time() - past))

			rate.sleep()

	except rospy.ROSInterruptException:
		pass
