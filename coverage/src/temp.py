#!/home/leo/anaconda3/bin/python3

import sys
import csv
import rospy
import random
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from time import sleep, time
from matplotlib.path import Path
from scipy import ndimage, sparse
from shapely.geometry import Point
from geometry_msgs.msg import Twist
from px4_mavros import Px4Controller
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Float64MultiArray
from scipy.stats import multivariate_normal
from shapely.geometry.polygon import Polygon
from scipy.optimize import linear_sum_assignment
from shapely.plotting import plot_polygon, plot_points
from math import sin, cos, tan, sqrt, atan2, acos, pi, exp

class UAV():

	def __init__(self, properties, map_size, grid_size,\
					Kv = 40, Ka = 3, Kp = 3, step = 0.5):

		# Variable of Control
		self.P1, self.P1o, self.P2, self.P2o, self.P3, self.P3o, self.PO = None, None, None, None, None, None, None
		self.P4, self.P4v, self.P5, self.P5v, self.P6, self.P6v = None, None, None, None, None, None
		self.A, self.b = None, None
		self.cmd_vel = Twist()
		self.d_safe = 1.0
		self.m, self.x = None, None
		self.px4 = Px4Controller("uav0")

		# Publisher & Subscriber
		self.fov_pub = rospy.Publisher("/fov", Float64MultiArray, queue_size = 100)
		self.cluster_dist_pub = rospy.Publisher("/cluster_dist", Float64MultiArray, queue_size = 100)
		self.cluster_teammate_pub = rospy.Publisher("/cluster_teammate", Float64MultiArray, queue_size = 100)

		self.odom_sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.odom_callback, queue_size = 10)
		self.fov_sub_1 = rospy.Subscriber("/uav1/fov", Float64MultiArray, self.fov_1_callback, queue_size = 100)
		self.fov_sub_2 = rospy.Subscriber("/uav2/fov", Float64MultiArray, self.fov_2_callback, queue_size = 100)
		self.cluster_dist_sub_1 = rospy.Subscriber("/uav1/cluster_dist", Float64MultiArray,\
												self.cluster_dist_1_callback, queue_size = 100)
		self.cluster_dist_sub_2 = rospy.Subscriber("/uav2/cluster_dist", Float64MultiArray,\
												self.cluster_dist_2_callback, queue_size = 100)
		self.cluster_teammate_sub_1 = rospy.Subscriber("/uav1/cluster_teammate", Float64MultiArray,\
												self.cluster_teammate_1_callback, queue_size = 100)
		self.cluster_teammate_sub_2 = rospy.Subscriber("/uav2/cluster_teammate", Float64MultiArray,\
												self.cluster_teammate_2_callback, queue_size = 100)

		# Variable of PTZ
		self.grid_size = grid_size
		self.map_size = map_size
		self.size = (int(map_size[0]/grid_size[0]), int(map_size[1]/grid_size[1]))
		self.id = properties['id']
		self.pos = properties['position']
		self.perspective = properties['perspective']/self.norm(properties['perspective'])
		self.alpha = properties['AngleofView']/180*np.pi
		self.R = properties['range_limit']
		self.lamb = properties['lambda']
		self.color = properties['color']

		# self.max_speed = properties['max_speed']
		self.translation_force = 0  # dynamics of positional changes
		self.perspective_force = 0  # dynamics of changing perspective direction
		self.stage = 1              # 1: Free player 2: Occupied Player 3: Cooperative player
		self.target = None
		self.target_assigned = -1
		self.step = step

		self.FoV = np.zeros(self.size)
		self.Kv = Kv                # control gain for perspective control law toward voronoi cell
		self.Ka = Ka                # control gain for zoom level control stems from voronoi cell
		self.Kp = Kp                # control gain for positional change toward voronoi cell 
		self.event = np.zeros((self.size[0], self.size[1]))

		self.top = 0
		self.ltop = 0
		self.rtop = 0
		self.r = 0
		self.centroid = None
		self.cluster_count = 0
		self.dist_to_cluster = 0
		self.dist_to_targets = 0
		self.Clsuter_Checklist = None
		self.Cluster_Teammate = np.array([None, None])
		self.dispatch_occpied = False

		self.neighor_1_fov = np.zeros(self.size)
		self.neighor_2_fov = np.zeros(self.size)
		self.neighor_1_cluster_dist = [100, 100, 100]
		self.neighor_2_cluster_dist = [100, 100, 100]
		self.neighor_1_cluster_teammate = np.array([None, None])
		self.neighor_2_cluster_teammate = np.array([None, None])

	def odom_callback(self, msg):
		
		UAV1_index = msg.name.index('iris_0')
		UAV2_index = msg.name.index('iris_1')
		UAV3_index = msg.name.index('iris_2')
		UAV4_index = msg.name.index('solo_3')
		UAV5_index = msg.name.index('solo_4')
		UAV6_index = msg.name.index('solo_5')
		# obs_index = msg.name.index('obstacle')

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
		# PO = np.array([msg.pose[obs_index].position.x, msg.pose[obs_index].position.y, msg.pose[obs_index].position.z])
		PO = np.array([100, 100, 100])

		self.A = np.array([(-2*(P1-PO)[:2]).tolist()])
		self.b = np.array([np.linalg.norm((P1-PO)[:2])**2 - self.d_safe**2])

		self.P1, self.P1o, self.P2, self.P2o, self.P3, self.P3o, self.PO = P1, P1o, P2, P2o, P3, P3o, PO
		self.P4, self.P4v, self.P5, self.P5v, self.P6, self.P6v = P4, P4v, P5, P5v, P6, P6v

		self.targets = [[(self.P4[0], self.P4[1]), 1, 10], [(self.P5[0], self.P5[1]), 1, 10],\
						[(self.P6[0], self.P6[1]), 1, 10]]

		self.pos = np.array([P1[0], P1[1]])
		theta = 2*acos(P1o[3])
		self.perspective = np.array([cos(theta), sin(theta)])

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

	def qp_ini(self):
		
		self.m = gp.Model("qp")
		self.m.setParam("NonConvex", 2.0)
		self.m.setParam("LogToConsole",0)
		self.x = self.m.addVars(2,ub=0.5, lb=-0.5, name="x")

	def addCons(self, i):

		self.m.addConstr(self.A[i,0]*self.x[0] + self.A[i,1]*self.x[1] <= self.b[i], "c"+str(i))

	def	controller(self):

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

		self.cmd_vel.linear.x = self.translational_force[0]*self.step
		self.cmd_vel.linear.y = self.translational_force[1]*self.step
		self.cmd_vel.linear.z = 0
		self.cmd_vel.angular.z = atan2(self.perspective_force[1], self.perspective_force[0])*self.step

		self.px4.vel_control(self.cmd_vel)

	def UpdateState(self, time_):

		self.time = time_

		self.UpdateFoV()
		self.polygon_FOV()
		self.EscapeDensity(time_)
		self.UpdateLocalVoronoi()

		self.Cluster_Formation()
		# self.Cluster_Assignment(targets)
		self.Cluster_Assignment(time_)

		event = np.zeros((self.size[0], self.size[1]))
		self.event = self.event_density(event, self.target, self.grid_size)
		
		self.ComputeCentroidal()
		self.StageAssignment()
		self.FormationControl()
		# self.UpdateOrientation()
		self.UpdateZoomLevel()
		# self.UpdatePosition()
		self.controller()

	def norm(self, arr):

		sum = 0

		for i in range(len(arr)):

			sum += arr[i]**2

		return sqrt(sum)

	def event_density(self, event, target, grid_size):

		x = np.arange(event.shape[0])*grid_size[0]
 
		for y_map in range(0, event.shape[1]):

			y = y_map*grid_size[1]
			density = 0

			for i in range(len(target)):

				density += target[i][2]*np.exp(-target[i][1]*np.linalg.norm(np.array([x,y], dtype=object)\
											-np.array((target[i][0][1],target[i][0][0]))))

			event[:][y_map] = density

		return 0 + event

	def Cluster_Formation(self):

		checklist = np.zeros((len(self.targets), len(self.targets)))
		threshold = 2.3
		self.cluster_count = 0

		for i in range(len(self.targets)):

			for j in range(len(self.targets)):

				if j != i:

					p1 = np.array([self.targets[i][0][0], self.targets[i][0][1]])
					p2 = np.array([self.targets[j][0][0], self.targets[j][0][1]])

					dist = self.norm(p1 - p2)

					if dist <= threshold:

						checklist[i][j] = 1
						self.cluster_count += 1
					else:

						checklist[i][j] = 0

		self.Clsuter_Checklist = checklist

		return

	def Cluster_Assignment(self, time_):

		count = 0
		Cluster = []
		Cluster_pair = []

		if len(self.targets) == 3:

			cluster_count_ref = 6
			AtoT = 3

		for i in range(np.shape(self.Clsuter_Checklist)[0]):

			nonindex = np.nonzero(self.Clsuter_Checklist[i][:])[0]

			if i > 0:

				for j in nonindex:

					if j < i and i in np.nonzero(self.Clsuter_Checklist[j][:])[0]:

						if self.id == 0:

							pass
					else:

						c_x = 0.5*(self.targets[i][0][0] + self.targets[j][0][0])
						c_y = 0.5*(self.targets[i][0][1] + self.targets[j][0][1])

						Cluster.append([(c_x, c_y), 1, 10])
						Cluster_pair.append((i,j))

			elif i == 0:

				for j in nonindex:
		        
					c_x = 0.5*(self.targets[i][0][0] + self.targets[j][0][0])
					c_y = 0.5*(self.targets[i][0][1] + self.targets[j][0][1])

					Cluster.append([(c_x, c_y), 1, 10])
					Cluster_pair.append((i,j))

		# Calculate dist between each cluster for Hungarian Algorithm
		dist_to_cluster = np.array([100.0000, 100.0000, 100.0000], dtype = float)

		for (mem, i) in zip(Cluster, range(len(Cluster))):

			p1 = np.array([self.pos[0], self.pos[1]])
			p2 = np.array([mem[0][0], mem[0][1]])

			dist_to_cluster[i] = self.norm(p1 - p2)

		self.dist_to_cluster = dist_to_cluster

		output_ = Float64MultiArray(data = dist_to_cluster)
		self.cluster_dist_pub.publish(output_)

		if (self.cluster_count == cluster_count_ref):

			x, y = 0, 0
			cert = 0
			score = -np.inf

			for mem in self.targets:

				x += mem[0][0]
				y += mem[0][1]

			for mem in Cluster:

				p1 = np.array([mem[0][0], mem[0][1]])
				p2 = np.array([x/len(self.targets), y/len(self.targets)])

				dist = self.norm(p1 - p2)

				if dist > score and dist > 0:

					cert = np.exp(-0.5*(dist/1.5))

			self.target = [[(x/AtoT, y/AtoT), cert, 10]]

		# Mode Switch Control
		pt = [self.pos, self.ltop, self.top, self.rtop, self.pos]
		polygon = Polygon(pt)
		
		dist_to_targets = np.array([100.00, 100.00, 100.00], dtype = float)

		for (mem, i) in zip(self.targets, range(len(self.targets))):

			gemos = Point(mem[0])
			if polygon.is_valid and polygon.contains(gemos):

				p1 = np.array([self.pos[0], self.pos[1]])
				p2 = np.array([mem[0][0], mem[0][1]])

				dist_to_targets[i] = self.norm(p1 - p2)

		self.dist_to_targets = dist_to_targets

		# Cost function 1-3
		Avg_dist = []
		k1, k2 = self.HW_IT, self.HW_BT

		for i in range(len(self.targets)):

			if i == 2:

				j = 0
			else:

				j = i + 1

			p1 = np.array([self.targets[i][0][0], self.targets[i][0][1]])
			p2 = np.array([self.targets[j][0][0], self.targets[j][0][1]])

			dist = self.norm(p1 - p2)
			Avg_dist.append(dist)

		Avg_dist = np.sum(Avg_dist)/len(self.targets)
		Avg_Sense = np.sum(self.HW_Sensing)/len(self.HW_Sensing)
		C_3 = (1/k1)*(1/Avg_Sense) + k2*Avg_dist

		# Cost Function 1-2
		# if (self.dist_to_cluster == 100.00).all():

		# 	dist = 10
		# 	Avg_Sense = 0.1
		# else:

		# 	switch_index = np.argmin(self.dist_to_cluster)
		# 	p1 = np.array([targets[Cluster_pair[switch_index][0]][0][0], targets[Cluster_pair[switch_index][0]][0][1]])
		# 	p2 = np.array([targets[Cluster_pair[switch_index][1]][0][0], targets[Cluster_pair[switch_index][1]][0][1]])
		# 	dist = self.norm(p1 - p2)
		# 	Avg_dist = dist
		# 	Avg_Sense = (np.sum(self.HW_Sensing[Cluster_pair[switch_index][0]] +\
		# 								self.HW_Sensing[Cluster_pair[switch_index][1]])/2)

			# Avg_dist, Avg_Sense = [], []
			# for i in range(len(Cluster_pair)):

			# 	p1 = np.array([targets[Cluster_pair[i][0]][0][0], targets[Cluster_pair[i][0]][0][1]])
			# 	p2 = np.array([targets[Cluster_pair[i][1]][0][0], targets[Cluster_pair[i][1]][0][1]])
			# 	dist = self.norm(p1 - p2)
			# 	Avg_dist.append(dist)

				# Avg_Sense.append(np.sum(self.HW_Sensing[Cluster_pair[i][0]] +\
				# 						self.HW_Sensing[Cluster_pair[i][1]])/2)
		t_index = [0,1,2]
		t_index = np.delete(t_index, np.argmax(self.dist_to_targets))
		p1 = np.array([self.targets[t_index[0]][0][0], self.targets[t_index[0]][0][1]])
		p2 = np.array([self.targets[t_index[1]][0][0], self.targets[t_index[1]][0][1]])
		Avg_dist = self.norm(p1 - p2)
		Avg_Sense = (np.sum(self.HW_Sensing[t_index[0]] +\
							self.HW_Sensing[t_index[1]])/2)
		C_2 = (1/k1)*(1/Avg_Sense) + k2*Avg_dist

		# Cost Function 1-1
		switch_index = np.argmin(self.dist_to_targets)
		Avg_dist = np.sum(np.where(self.dist_to_targets < 100))/len(np.where(self.dist_to_targets < 100))
		# dist = self.dist_to_targets[switch_index]
		Sense = self.HW_Sensing[switch_index]
		C_1 = (1/k1)*(1/Sense) + k2*Avg_dist

		C_total = [C_1, C_2, C_3]
		min_C = np.argmin(C_total)+1

		print("C1: " + str(C_1))
		print("C2: " + str(C_2))
		print("C3: " + str(C_3))

		C_total.append(time_)
		# filename = "D:/上課資料/IME/實驗室研究/Paper/Coverage Control/Quality based switch mode/Data/"
		# filename = "D:/Leo/IME/Paper Study/Coverage Control/Quality based switch mode/Data/"
		# filename += "Data_" + str(self.id) + ".csv"
		# with open(filename, "a", encoding='UTF8', newline='') as f:

		# 	row = C_total
		# 	writer = csv.writer(f)
		# 	writer.writerow(row)

		neighbors_cluster_dist = []
		neighbors_cluster_dist =\
		[self.neighor_1_cluster_dist if self.neighor_1_cluster_dist != None else None,\
		self.neighor_2_cluster_dist if self.neighor_2_cluster_dist != None else None]

		neighbors_cluster_teammate = []
		neighbors_cluster_teammate =\
		[self.neighor_1_cluster_teammate if (self.neighor_1_cluster_teammate != None).all() else [None, None],\
		self.neighor_2_cluster_teammate if (self.neighor_2_cluster_teammate != None).all() else [None, None]]

		if (len(Cluster) == AtoT):

			if min_C == 3:

				x, y = 0, 0

				for target in self.targets:

					x += target[0][0]
					y += target[0][1]

				self.target = [[(x/AtoT, y/AtoT), 1, 10]]
			elif min_C == 2:

				switch_index = np.argmin(self.dist_to_cluster)
				self.target = [Cluster[switch_index]]

		# if (self.HW_Interior <= self.HW_Boundary):
		if (len(Cluster) < AtoT):

			# if (len(Cluster) == AtoT):
			if min_C == 3:

				cost_matrix = [self.dist_to_cluster]
				# for neighbor in self.neighbors:

				# 	cost_matrix = np.concatenate((cost_matrix, [neighbor.dist_to_cluster]),\
				# 									axis = 0)

				# row_ind, col_ind = linear_sum_assignment(cost_matrix)
				# self.target = [Cluster[col_ind[0]]]
				# switch_index = np.argmin(self.dist_to_cluster)
				# self.target = [Cluster[switch_index]]

			elif min_C == 2:

				if (len(Cluster) == AtoT - 1):

					switch_index = np.argmin(self.dist_to_cluster)
					self.target = [Cluster[switch_index]]
					self.Cluster_Teammate = np.array([2, switch_index])

					output_ = Float64MultiArray(data = self.Cluster_Teammate)
					self.cluster_teammate_pub.publish(output_)

					cost_matrix = [self.dist_to_cluster]
					Teammate_matrix = [self.Cluster_Teammate]
					count = np.ones(len(Cluster))
					count[self.Cluster_Teammate[1]] = 0
					len_ = len(count)

					for i in range(2):

						temp1 = neighbors_cluster_dist[i]
						cost_matrix = np.concatenate((cost_matrix, [temp1]), axis = 0)

						if int(len_) <= 1:

							count = np.array([0])
						else:
							if ((neighbors_cluster_teammate[i] != None).all()) and\
								(neighbors_cluster_teammate[i][0] == 2):

								temp2 = neighbors_cluster_teammate[i]
								Teammate_matrix = np.concatenate((Teammate_matrix, [temp2]), axis = 0)
								count[neighbors_cluster_teammate[i][1]] = 0
							else:
								temp2 = np.array([None, None])
								Teammate_matrix = np.concatenate((Teammate_matrix, [temp2]), axis = 0)

					if (not (count == 0).all() and (Teammate_matrix != None).all()):

						dist_untracked = cost_matrix[:,np.nonzero(count)[0]]
						dispatch_index = np.argmin(dist_untracked)

						if dispatch_index == 0:

							self.target = [Cluster[np.nonzero(count)[0][0]]]
							self.Cluster_Teammate = np.array([2, np.nonzero(count)[0][0]])

					self.last_Cluster_pair = Cluster_pair
				elif (len(Cluster) == AtoT - 2):

					# index = np.delete(self.last_Cluster_pair, Cluster_pair)

					# p1 = np.array([targets[index[0]][0][0], targets[index[0]][0][1]])
					# p2 = np.array([targets[index[1]][0][0], targets[index[1]][0][1]])
					# dist = self.norm(p1 - p2)

					# cost_matrix = [self.dist_to_cluster[0], dist]
					# switch_index = np.argmin(cost_matrix)

					# if switch_index == 0:

					# 	switch_index = np.argmin(self.dist_to_cluster)
					# 	self.target = [Cluster[switch_index]]
					# elif switch_index == 1:

					# 	x = (targets[index[0]][0][0] + targets[index[1]][0][0])/2
					# 	y = (targets[index[0]][0][1] + targets[index[1]][0][1])/2

					# 	self.target = [[(x, y), 1, 10]]

					index = [0,1,2]
					index = np.delete(index, np.argmax(self.dist_to_targets))
					x = (self.targets[index[0]][0][0] + self.targets[index[1]][0][0])/2
					y = (self.targets[index[0]][0][1] + self.targets[index[1]][0][1])/2

					self.target = [[(x, y), 1, 10]]

			# elif (len(Cluster) < AtoT - 1):
			elif min_C == 1:

				print(self.dist_to_targets)

				switch_index = np.argmin(self.dist_to_targets)
				self.target = [self.targets[switch_index]]
				self.Cluster_Teammate = np.array([0, switch_index])

				output_ = Float64MultiArray(data = self.Cluster_Teammate)
				self.cluster_teammate_pub.publish(output_)

				cost_matrix = [self.dist_to_targets]
				Teammate_matrix = [self.Cluster_Teammate]
				count = np.ones(len(self.targets))
				count[self.Cluster_Teammate[1]] = 0

				for i in range(2):

					if (neighbors_cluster_teammate[i][0] == 0) and\
						(neighbors_cluster_teammate[i][1] == self.Cluster_Teammate[1]):

						self.dist_to_targets[self.Cluster_Teammate[1]] = 100
						switch_index = np.argmin(self.dist_to_targets)
						self.target = [self.targets[switch_index]]
						self.Cluster_Teammate = np.array([0, switch_index])

						# temp1 = neighbor.dist_to_targets
						# cost_matrix = np.concatenate((cost_matrix, [temp1]), axis = 0)

						# over_tracked = cost_matrix[:,self.Cluster_Teammate[1]]
						# dispatch_index = np.argmin(over_tracked)

						# if dispatch_index != 0:

						# 	self.dist_to_targets[self.Cluster_Teammate[1]] = 100
						# 	print(self.dist_to_targets)
						# 	switch_index = np.argmin(self.dist_to_targets)
						# 	print(switch_index)
						# 	self.target = [targets[switch_index]]
						# 	self.Cluster_Teammate = np.array([0, switch_index])

				self.dispatch_occpied == False

		print(self.id, "\n")

	def StageAssignment(self):

		# range_max = 0.85*(self.lamb + 1)/(self.lamb)*self.R*cos(self.alpha)
		range_max = self.R*cos(self.alpha)

		if self.centroid is not None:

			range_local_best = (self.norm(np.asarray(self.centroid) - self.pos))
			r = range_max*range_local_best/(range_max+range_local_best)\
				+ range_local_best*range_max/(range_max+range_local_best)

			if self.stage == 1:

				r = max(r, range_max - sqrt(1/(2*self.target[0][1])))
			else:

				r = self.R*cos(self.alpha)

			tmp = 0
			for i in range(len(self.target)):

				dist = self.norm(self.pos - np.asarray(self.target[i][0]))
				if dist <= r and -dist <= tmp:
					tmp = -dist
					self.stage = 2

			self.r = r

	def UpdateFoV(self):

		range_max = (self.lamb + 1)/(self.lamb)*self.R
		quality_map = None
		self.Fov = np.zeros(self.size)

		for y_map in range(max(int((self.pos[1] - range_max)/self.grid_size[1]), 0),\
				min(int((self.pos[1] + range_max)/self.grid_size[1]), self.size[1])):

			x_map = np.arange(max(int((self.pos[0] - range_max)/self.grid_size[0]), 0),
					min(int((self.pos[0] + range_max)/self.grid_size[0]), self.size[0]))
			q_per = self.PerspectiveQuality(x_map*self.grid_size[0], y_map*self.grid_size[1])
			q_res = self.ResolutionQuality(x_map*self.grid_size[0], y_map*self.grid_size[1])
			quality = np.where((q_per > 0) & (q_res > 0), q_per*q_res, 0)

			if quality_map is None:

				quality_map = quality
			else:

				quality_map = np.vstack((quality_map, quality))

		self.FoV[max(int((self.pos[1] - range_max)/self.grid_size[1]), 0):\
				min(int((self.pos[1] + range_max)/self.grid_size[1]), self.size[0]), \
		    	max(int((self.pos[0] - range_max)/self.grid_size[0]), 0):\
		    	min(int((self.pos[0] + range_max)/self.grid_size[0]), self.size[0])]\
		    	= quality_map

		output_ = Float64MultiArray(data = self.FoV)
		self.fov_pub.publish(output_)

		return

	def PerspectiveQuality(self, x, y):

		x_p = np.array([x, y], dtype = object) - self.pos

		return (np.matmul(x_p,self.perspective.transpose())/np.linalg.norm(x_p)\
				- np.cos(self.alpha))/(1 - np.cos(self.alpha))

	def ResolutionQuality(self, x, y):

		x_p = np.array([x, y], dtype = object) - self.pos

		return (((np.linalg.norm(x_p)**self.lamb)*(self.R*np.cos(self.alpha)\
				- self.lamb*( np.linalg.norm(x_p) - self.R*np.cos(self.alpha)) ))\
				/(self.R**(self.lamb+1)))

	def EscapeDensity(self, time_):

		# Environment
		L = self.map_size[0]
		Wi = self.map_size[1]
		x_range = np.arange(0, L+0.1, self.grid_size[0])
		y_range = np.arange(0, L+0.1, self.grid_size[1])
		X, Y = np.meshgrid(x_range, y_range)

		W = np.vstack([X.ravel(), Y.ravel()])
		W = W.transpose()

		# Vertices of Boundary of FoV
		A = np.array([self.pos[0], self.pos[1]])
		B = np.array([self.rtop[0], self.rtop[1]])
		C = np.array([self.ltop[0], self.ltop[1]])
		range_max = (self.lamb + 1)/(self.lamb)*self.R

		# Bivariate Normal Distribution
		F1 = multivariate_normal([self.targets[0][0][0], self.targets[0][0][1]],\
								[[self.targets[0][1], 0.0], [0.0, self.targets[0][1]]])
		F2 = multivariate_normal([self.targets[1][0][0], self.targets[1][0][1]],\
								[[self.targets[1][1], 0.0], [0.0, self.targets[1][1]]])
		F3 = multivariate_normal([self.targets[2][0][0], self.targets[2][0][1]],\
								[[self.targets[2][1], 0.0], [0.0, self.targets[2][1]]])

		# Joint Probability
		# P = lambda d, a, IoO, P0: P0*np.multiply(np.multiply(\
		# 			np.divide(np.power(np.subtract(d, range_max), 2).transpose()[0], range_max**2),\
		# 			np.divide(np.power(np.subtract(abs(a), self.alpha), 2), self.alpha**2)), IoO)

		# P = lambda d, a, IoO, P0: P0*np.multiply(np.multiply(\
		# 			np.exp(-np.divide(np.power(np.subtract(d, self.R*np.cos(self.alpha)), 2).transpose()[0], (2*0.1**2)*(self.R*np.cos(self.alpha)**2))),\
		# 			np.exp(-np.divide(np.power(np.subtract(abs(a), 0), 2), (2*0.2**2)*(self.alpha**2)))), IoO)

		# 0.35 & 0.35
		P_t = lambda d, a, IoO, P0: P0*np.multiply(np.multiply(\
					np.exp(-np.divide(np.power(np.subtract(d, self.R*np.cos(self.alpha)), 2).transpose()[0], (2*0.35**2)*(self.R*np.cos(self.alpha)**2))),\
					np.exp(-np.divide(np.power(np.subtract(abs(a), 0), 2), (2*0.35**2)*(self.alpha**2)))), IoO)

		# P_ = lambda d, a, IoO, P0: P0*np.multiply(np.multiply(\
		# 			np.exp(-np.divide(np.power(np.subtract(d, range_max), 2).transpose()[0], (2*0.125**2)*(range_max**2))),\
		# 			np.exp(-np.divide(np.power(np.subtract(abs(a), self.alpha), 2), (2*0.125**2)*(self.alpha**2)))), IoO)

		# P_ = lambda d, a, IoO, P0: P0*np.multiply(np.multiply(\
		# 			np.exp(-np.divide(np.power(np.subtract(abs(d-0.5*range_max), 0.5*range_max), 2).transpose()[0], (2*0.25**2)*(0.5*range_max**2))),\
		# 			np.exp(-np.divide(np.power(np.subtract(abs(a), self.alpha), 2), (2*0.3**2)*(self.alpha**2)))), IoO)
		
		# 0.25 & 0.25
		P_tt = lambda d, a, IoO, P0: P0*np.multiply(np.multiply(\
					np.exp(-np.divide(np.power(np.subtract(abs(d-0.5*range_max), 0.5*range_max), 2).transpose()[0], (2*0.25**2)*(0.5*range_max**2))),\
					np.exp(-np.divide(np.power(np.subtract(abs(a), self.alpha), 2), (2*0.25**2)*(self.alpha**2)))), IoO)	

		# Points in FoV
		pt = [A+np.array([0, 0.1]), B+np.array([0.1, -0.1]), C+np.array([-0.1, -0.1]), A+np.array([0, 0.1])]
		polygon = Path(pt)
		In_polygon = polygon.contains_points(W) # Boolean

		# Distance and Angle of W with respect to self.pos
		d = np.linalg.norm(np.subtract(W, self.pos), axis = 1)
		d = np.array([d]).transpose()
		a = np.arccos( np.dot(np.divide(np.subtract(W, self.pos),np.concatenate((d,d), axis = 1)), self.perspective) )
		a = np.arccos( np.dot(np.subtract(W, self.pos)/np.concatenate((d,d), axis = 1), self.perspective) )

		# Cost Function
		P0_I = 0.9
		# JP_Interior = P(d, a, In_polygon, P0_I)
		# HW_Interior = np.sum(np.multiply(F1.pdf(W), JP_Interior))\
		# 			+ np.sum(np.multiply(F2.pdf(W), JP_Interior))\
		# 			+ np.sum(np.multiply(F3.pdf(W), JP_Interior))
		# self.HW_Interior = HW_Interior*0.1**2

		JP_Interior = P_t(d, a, In_polygon, P0_I)
		HW_Interior = np.sum(np.multiply(F1.pdf(W), JP_Interior))\
					+ np.sum(np.multiply(F2.pdf(W), JP_Interior))\
					+ np.sum(np.multiply(F3.pdf(W), JP_Interior))
		self.HW_IT = HW_Interior*0.1**2

		P0_B = 0.9
		# JP_Boundary = P_(d, a, In_polygon, P0_B)
		# HW_Boundary = np.sum(np.multiply(F1.pdf(W), JP_Boundary))\
		# 			+ np.sum(np.multiply(F2.pdf(W), JP_Boundary))\
		# 			+ np.sum(np.multiply(F3.pdf(W), JP_Boundary))
		# self.HW_Boundary = HW_Boundary*0.1**2

		JP_Boundary = P_tt(d, a, In_polygon, P0_B)
		HW_Boundary = np.sum(np.multiply(F1.pdf(W), JP_Boundary))\
					+ np.sum(np.multiply(F2.pdf(W), JP_Boundary))\
					+ np.sum(np.multiply(F3.pdf(W), JP_Boundary))

		self.HW_BT = HW_Boundary*0.1**2

		# Spatial Sensing Quality
		Q = lambda W, d, IoO, P0: P0*np.multiply(np.multiply((np.divide(\
					np.dot(np.subtract(W, self.pos),self.perspective), d) - np.cos(self.alpha))/(1 - np.cos(self.alpha)),\
					np.multiply((self.R*np.cos(self.alpha) - self.lamb*(d - self.R*np.cos(self.alpha))),\
					(np.power(d, self.lamb)/(self.R**(self.lamb+1))))), IoO)
		P0_Q = 1.0
		d = d.transpose()[0]
		SQ = Q(W, d, In_polygon, P0_Q)
		HW_Sensing = np.sum(np.multiply(F1.pdf(W), SQ))\
					+ np.sum(np.multiply(F2.pdf(W), SQ))\
					+ np.sum(np.multiply(F3.pdf(W), SQ))

		self.HW_Sensing = [np.sum(np.multiply(F1.pdf(W), SQ)), np.sum(np.multiply(F2.pdf(W), SQ)), np.sum(np.multiply(F3.pdf(W), SQ))]

		if self.id == 4:

			print("I: " + str(self.HW_Interior))
			print("B: " + str(self.HW_Boundary))
			print("S: " + str(np.sum(self.HW_Sensing)*0.1**2/len(self.HW_Sensing)))
			print("H: " + str(self.HW_Interior-self.HW_Boundary), "\n")

		# filename = "D:/上課資料/IME/實驗室研究/Paper/Coverage Control/Quality based switch mode/Data/"
		# filename += "Data_" + str(self.id) + ".csv"
		# with open(filename, "a", encoding='UTF8', newline='') as f:

		# 	row = [self.HW_Interior, self.HW_Boundary, self.HW_Sensing, time_]
		# 	writer = csv.writer(f)
		# 	writer.writerow(row)

	def UpdateLocalVoronoi(self):

		quality_map = self.FoV
		# neighbors_fov = [self.neighor_1_fov, self.neighor_2_fov]

		# for FoV in neighbors_fov:

		# 	quality_map = np.where((quality_map > FoV), quality_map, 0)

		# self.voronoi = np.array(np.where((quality_map != 0) & (self.FoV != 0)))
		self.voronoi = np.array(np.where((self.FoV != 0)))
		# self.voronoi_map = np.where(((quality_map > 0.) & (self.FoV > 0.)), quality_map, 0)
		# self.map_plt = np.array(np.where(quality_map != 0, self.id + 1, 0))

		return

	def ComputeCentroidal(self):

		translational_force = np.array([0.,0.])
		rotational_force = np.array([0.,0.]).reshape(2,1)
		zoom_force = 0
		centroid = None

		if len(self.voronoi[0]) > 0:

			mu_V = 0
			v_V_t = np.array([0, 0], dtype = np.float64)
			delta_V_t = 0
			x_center = 0
			y_center = 0

			# Control law for maximizing resolution and perspective quality
			for i in range(len(self.voronoi[0])):

				x_map = self.voronoi[1][i]
				y_map = self.voronoi[0][i]

				x, y = x_map*self.grid_size[0], y_map*self.grid_size[1]
				x_p = np.array([x,y]) - self.pos
				norm = self.norm(x_p)

				if norm == 0: continue

				mu_V += ((norm**self.lamb)*self.event[x_map,y_map] )/(self.R**self.lamb)
				v_V_t += ((x_p)/norm)*(cos(self.alpha) - \
					( ( self.lamb*norm )/((self.lamb+1)*self.R)))*\
					( (norm**self.lamb)/(self.R**self.lamb) )*self.event[x_map,y_map]
				dist = (1 - (self.lamb*norm)/((self.lamb+1)*self.R))
				dist = dist if dist >= 0 else 0
				delta_V_t += (1 - (((x_p)@self.perspective.T))/norm)\
								*dist*((norm**self.lamb)/(self.R**self.lamb))\
								*self.event[x_map,y_map]
				x_center += x*(((norm**self.lamb)*self.event[x_map,y_map] )/(self.R**self.lamb))
				y_center += y*(((norm**self.lamb)*self.event[x_map,y_map] )/(self.R**self.lamb))
		    
			v_V = v_V_t/mu_V
			delta_V = delta_V_t/mu_V
			delta_V = delta_V if delta_V > 0 else 1e-10
			alpha_v = acos(1-sqrt(delta_V))
			alpha_v = alpha_v if alpha_v > 5/180*np.pi else 5/180*np.pi
		    
			centroid = np.array([x_center/mu_V, y_center/mu_V])
			translational_force += self.Kp*(self.norm(centroid - self.pos)\
											- self.R*cos(self.alpha))*self.perspective
			rotational_force += self.Kv*(np.eye(2) - np.dot(self.perspective[:,None],\
									self.perspective[None,:]))  @  (v_V.reshape(2,1))
			zoom_force += -self.Ka*(self.alpha - alpha_v)

		self.translational_force = translational_force if self.stage != 2 else 0
		self.perspective_force = np.asarray([rotational_force[0][0], rotational_force[1][0]])
		self.zoom_force = zoom_force
		self.centroid = centroid

		return

	def FormationControl(self):

		# Consider which neighbor is sharing the same target and only use them to obtain formation force
		neighbor_force = np.array([0.,0.])
		neighbors_pos = [[self.P2[0], self.P2[1]], [self.P3[0], self.P3[1]]]

		for pos in neighbors_pos:
			neighbor_force += (self.pos - pos)/(self.norm(self.pos - pos))

		neighbor_norm = self.norm(neighbor_force)

		if self.stage == 2:

			target_force = (np.asarray(self.target[0][0]) - self.pos)\
				/(self.norm(np.asarray(self.target[0][0])- self.pos))

			target_norm = self.norm(target_force)

			center_force = (np.asarray(self.target[0][0]) - self.pos)\
				/(self.norm(np.asarray(self.target[0][0]) - self.pos))

			formation_force = (target_force*(neighbor_norm/(target_norm+neighbor_norm))\
							+ neighbor_force*(target_norm/(target_norm+neighbor_norm)))

			formation_force -= (center_force/self.norm(center_force))*(self.r - self.norm\
							(self.pos - np.asarray(self.target[0][0])))

			self.translational_force += formation_force 

			return

		else:

			formation_force = neighbor_force
			self.translational_force += formation_force 

			return

	def UpdateOrientation(self):

		self.perspective += self.perspective_force*self.step
		self.perspective /= self.norm(self.perspective)

		return

	def UpdateZoomLevel(self):

		self.alpha += self.zoom_force*self.step

		return

	def UpdatePosition(self):

		self.pos += self.translational_force*self.step

		return

	def polygon_FOV(self):

		range_max = (self.lamb + 1)/(self.lamb)*self.R*cos(self.alpha)
		R = np.array([[np.cos(self.alpha), -np.sin(self.alpha)]
					,[np.sin(self.alpha), np.cos(self.alpha)]])

		self.top = self.pos + range_max*self.perspective

		self.ltop = self.pos + range_max*np.reshape(R@np.reshape(self.perspective,(2,1)),(1,2))
		self.ltop = self.ltop[0]

		self.rtop = self.pos + range_max*np.reshape(np.linalg.inv(R)@np.reshape(self.perspective,(2,1)),(1,2))
		self.rtop = self.rtop[0]

if __name__ == '__main__':

	try:
		rospy.init_node('controller_1')
		rate = rospy.Rate(100)

		map_size = np.array([25, 25])
		grid_size = np.array([0.1, 0.1])

		camera0 = { 'id'            :  0,
					'position'      :  np.array([1., 8.]),
					'perspective'   :  np.array([0.9,1]),
					'AngleofView'   :  20,
					'range_limit'   :  5,
					'lambda'        :  2,
					'color'         : (200, 0, 0)}

		uav_1 = UAV(camera0, map_size, grid_size)

		while uav_1.b is None:

			rate.sleep()

		uav_1.qp_ini()

		last = time()

		while not rospy.is_shutdown():

			uav_1.UpdateState(np.round(time() - last, 2))
			rate.sleep()

	except rospy.ROSInterruptException:
		pass
