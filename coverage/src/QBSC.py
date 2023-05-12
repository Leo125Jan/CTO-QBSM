#!/home/leo/anaconda3/envs/py39/bin/python3

import sys
import csv
import rospy
import random
import numpy as np
import numexpr as ne
from time import sleep, time
from matplotlib.path import Path
from scipy import ndimage, sparse
from shapely.geometry import Point
from scipy.stats import multivariate_normal
from shapely.geometry.polygon import Polygon
from scipy.optimize import linear_sum_assignment
from shapely.plotting import plot_polygon, plot_points
from math import sin, cos, tan, sqrt, atan2, acos, pi, exp
from std_msgs.msg import Float64MultiArray, MultiArrayLayout, MultiArrayDimension

class PTZcon():

	def __init__(self, properties, map_size, grid_size,\
					Kv = 50, Ka = 3, Kp = 3, step = 0.3):

		# Publisher & Subscriber properties['ecode'] + 
		self.fov_pub = rospy.Publisher(properties['ecode'] + "/fov", Float64MultiArray, queue_size = 20)
		self.cluster_dist_pub = rospy.Publisher(properties['ecode'] + "/cluster_dist", Float64MultiArray, queue_size = 20)
		self.cluster_teammate_pub = rospy.Publisher(properties['ecode'] + "/cluster_teammate", Float64MultiArray, queue_size = 20)

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

		x_range = np.arange(0, self.map_size[0], self.grid_size[0])
		y_range = np.arange(0, self.map_size[1], self.grid_size[1])
		X, Y = np.meshgrid(x_range, y_range)

		W = np.vstack([X.ravel(), Y.ravel()])
		self.W = W.transpose()

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
		self.centroid = np.array([0,0])
		self.R_max = (self.lamb + 1)/(self.lamb)*self.R
		self.cluster_count = 0
		self.dist_to_cluster = 0
		self.dist_to_targets = 0
		self.Clsuter_Checklist = None
		self.Cluster_Teammate = np.array([None, None, None])

	def UpdateState(self, targets, neighbors, states, time_):

		self.time = time_
		self.targets = targets
		self.neighbors = neighbors
		self.pos = states["Position"]
		self.perspective = states["Perspective"]

		self.UpdateFoV()
		self.polygon_FOV()
		self.EscapeDensity(time_)
		self.UpdateLocalVoronoi()

		self.Cluster_Formation()
		self.Cluster_Assignment(time_)
		self.publish_message()

		# event = np.zeros((self.size[0], self.size[1]))
		# self.event = self.event_density(event, self.target, self.grid_size)
		
		self.ComputeCentroidal()
		self.StageAssignment()
		self.FormationControl()
		self.UpdateOrientation()
		self.UpdateZoomLevel()
		self.UpdatePosition()

		return self.translational_force, self.perspective_force, self.step

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
		# Cluster_pair = []

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
						# Cluster_pair.append((i,j))

			elif i == 0:

				for j in nonindex:
		        
					c_x = 0.5*(self.targets[i][0][0] + self.targets[j][0][0])
					c_y = 0.5*(self.targets[i][0][1] + self.targets[j][0][1])

					Cluster.append([(c_x, c_y), 1, 10])
					# Cluster_pair.append((i,j))

		# Calculate dist between each cluster for Hungarian Algorithm
		dist_to_cluster = [100.0000, 100.0000, 100.0000]

		for (mem, i) in zip(Cluster, range(len(Cluster))):

			p1 = np.array([self.pos[0], self.pos[1]])
			p2 = np.array([mem[0][0], mem[0][1]])

			dist_to_cluster[i] = self.norm(p1 - p2)

		self.dist_to_cluster = dist_to_cluster

		# output_ = Float64MultiArray(data = dist_to_cluster)
		# self.cluster_dist_pub.publish(output_)

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
			# if polygon.is_valid and polygon.contains(gemos):
			if polygon.is_valid:

				p1 = np.array([self.pos[0], self.pos[1]])
				p2 = np.array([mem[0][0], mem[0][1]])

				dist_to_targets[i] = self.norm(p1 - p2)

		self.dist_to_targets = dist_to_targets

		# Configuration of calculation cost function
		Avg_distance = 0.0
		k1, k2 = self.HW_IT, self.HW_BT
		sweet_spot = self.pos + self.R*np.cos(self.alpha)*self.perspective

		t_index = [0,1,2]
		t_index = np.delete(t_index, np.argmax(self.dist_to_targets))
		p1 = np.array([self.targets[t_index[0]][0][0], self.targets[t_index[0]][0][1]])
		p2 = np.array([self.targets[t_index[1]][0][0], self.targets[t_index[1]][0][1]])
		mid_point = np.array([(p1[0]+p2[0])/2, (p1[1]+p2[1])/2])
		decision_line = (mid_point - self.pos)/np.linalg.norm(mid_point - self.pos)
		dl_1 = (p1 - self.pos)/np.linalg.norm(p1 - self.pos)
		dl_2 = (p2 - self.pos)/np.linalg.norm(p2 - self.pos)

		if np.cross(dl_1, decision_line) < 0:

			p_l = np.array([self.targets[t_index[0]][0][0], self.targets[t_index[0]][0][1]])
			p_r = np.array([self.targets[t_index[1]][0][0], self.targets[t_index[1]][0][1]])
		else:

			p_r = np.array([self.targets[t_index[0]][0][0], self.targets[t_index[0]][0][1]])
			p_l = np.array([self.targets[t_index[1]][0][0], self.targets[t_index[1]][0][1]])

		# fail_ = 0
		# for i in range(0, 3):

		# 	if i == 0:

		# 		v_1 = np.array([self.targets[1][0][0], self.targets[1][0][1]])\
		# 			- np.array([self.targets[0][0][0], self.targets[0][0][1]]); v_1 = v_1/np.linalg.norm(v_1)
		# 		v_2 = np.array([self.targets[2][0][0], self.targets[2][0][1]])\
		# 			- np.array([self.targets[0][0][0], self.targets[0][0][1]]); v_2 = v_2/np.linalg.norm(v_2)
		# 		v_3 = np.array([self.pos[0], self.pos[1]])\
		# 			- np.array([self.targets[0][0][0], self.targets[0][0][1]]); v_3 = v_3/np.linalg.norm(v_3)

		# 		if (np.sign(np.cross(v_1, v_2)) == np.sign(np.cross(v_1, v_3))) and\
		# 			(np.sign(np.cross(v_2, v_1)) == np.sign(np.cross(v_2, v_3))):

		# 			p1 = np.array([self.targets[1][0][0], self.targets[1][0][1]])
		# 			p2 = np.array([self.targets[2][0][0], self.targets[2][0][1]])
		# 		else:

		# 			fail_ += 1
		# 	elif i == 1:

		# 		v_1 = np.array([self.targets[0][0][0], self.targets[0][0][1]])\
		# 			- np.array([self.targets[1][0][0], self.targets[1][0][1]]); v_1 = v_1/np.linalg.norm(v_1)
		# 		v_2 = np.array([self.targets[2][0][0], self.targets[2][0][1]])\
		# 			- np.array([self.targets[1][0][0], self.targets[1][0][1]]); v_2 = v_2/np.linalg.norm(v_2)
		# 		v_3 = np.array([self.pos[0], self.pos[1]])\
		# 			- np.array([self.targets[1][0][0], self.targets[1][0][1]]); v_3 = v_3/np.linalg.norm(v_3)

		# 		if (np.sign(np.cross(v_1, v_2)) == np.sign(np.cross(v_1, v_3))) and\
		# 			(np.sign(np.cross(v_2, v_1)) == np.sign(np.cross(v_2, v_3))):

		# 			p1 = np.array([self.targets[0][0][0], self.targets[0][0][1]])
		# 			p2 = np.array([self.targets[2][0][0], self.targets[2][0][1]])
		# 		else:

		# 			fail_ += 1
		# 	elif i == 2:

		# 		v_1 = np.array([self.targets[0][0][0], self.targets[0][0][1]])\
		# 			- np.array([self.targets[2][0][0], self.targets[2][0][1]]); v_1 = v_1/np.linalg.norm(v_1)
		# 		v_2 = np.array([self.targets[1][0][0], self.targets[1][0][1]])\
		# 			- np.array([self.targets[2][0][0], self.targets[2][0][1]]); v_2 = v_2/np.linalg.norm(v_2)
		# 		v_3 = np.array([self.pos[0], self.pos[1]])\
		# 			- np.array([self.targets[2][0][0], self.targets[2][0][1]]); v_3 = v_3/np.linalg.norm(v_3)

		# 		if (np.sign(np.cross(v_1, v_2)) == np.sign(np.cross(v_1, v_3))) and\
		# 			(np.sign(np.cross(v_2, v_1)) == np.sign(np.cross(v_2, v_3))):

		# 			p1 = np.array([self.targets[0][0][0], self.targets[0][0][1]])
		# 			p2 = np.array([self.targets[1][0][0], self.targets[1][0][1]])
		# 		else:

		# 			fail_ += 1

		# 	if fail_ == 3:

		# 		t_index = np.argmin(self.dist_to_targets)

		# 		if t_index == 0:

		# 			v_1 = np.array([self.targets[0][0][0], self.targets[0][0][1]])\
		# 				- np.array([self.targets[1][0][0], self.targets[1][0][1]]); v_1 = v_1/np.linalg.norm(v_1)
		# 			v_2 = np.array([self.targets[0][0][0], self.targets[0][0][1]])\
		# 				- np.array([self.targets[2][0][0], self.targets[2][0][1]]); v_2 = v_2/np.linalg.norm(v_2)
		# 			v_3 = np.array([self.pos[0], self.pos[1]])\
		# 				- np.array([self.targets[0][0][0], self.targets[0][0][1]]); v_3 = v_3/np.linalg.norm(v_3)

		# 			if np.arccos(np.dot(v_1, v_3)) < np.arccos(np.dot(v_2, v_3)):

		# 				p1 = np.array([self.targets[t_index][0][0], self.targets[t_index][0][1]])
		# 				p2 = np.array([self.targets[2][0][0], self.targets[2][0][1]])
		# 			else:

		# 				p1 = np.array([self.targets[t_index][0][0], self.targets[t_index][0][1]])
		# 				p2 = np.array([self.targets[1][0][0], self.targets[1][0][1]])
		# 		elif t_index == 1:

		# 			v_1 = np.array([self.targets[1][0][0], self.targets[1][0][1]])\
		# 				- np.array([self.targets[0][0][0], self.targets[0][0][1]]); v_1 = v_1/np.linalg.norm(v_1)
		# 			v_2 = np.array([self.targets[1][0][0], self.targets[1][0][1]])\
		# 				- np.array([self.targets[2][0][0], self.targets[2][0][1]]); v_2 = v_2/np.linalg.norm(v_2)
		# 			v_3 = np.array([self.pos[0], self.pos[1]])\
		# 				- np.array([self.targets[1][0][0], self.targets[1][0][1]]); v_3 = v_3/np.linalg.norm(v_3)

		# 			if np.arccos(np.dot(v_1, v_3)) < np.arccos(np.dot(v_2, v_3)):

		# 				p1 = np.array([self.targets[t_index][0][0], self.targets[t_index][0][1]])
		# 				p2 = np.array([self.targets[2][0][0], self.targets[2][0][1]])
		# 			else:

		# 				p1 = np.array([self.targets[t_index][0][0], self.targets[t_index][0][1]])
		# 				p2 = np.array([self.targets[0][0][0], self.targets[0][0][1]])
		# 		elif t_index == 2:

		# 			v_1 = np.array([self.targets[2][0][0], self.targets[2][0][1]])\
		# 				- np.array([self.targets[0][0][0], self.targets[0][0][1]]); v_1 = v_1/np.linalg.norm(v_1)
		# 			v_2 = np.array([self.targets[2][0][0], self.targets[2][0][1]])\
		# 				- np.array([self.targets[1][0][0], self.targets[1][0][1]]); v_2 = v_2/np.linalg.norm(v_2)
		# 			v_3 = np.array([self.pos[0], self.pos[1]])\
		# 				- np.array([self.targets[2][0][0], self.targets[2][0][1]]); v_3 = v_3/np.linalg.norm(v_3)

		# 			if np.arccos(np.dot(v_1, v_3)) < np.arccos(np.dot(v_2, v_3)):

		# 				p1 = np.array([self.targets[t_index][0][0], self.targets[t_index][0][1]])
		# 				p2 = np.array([self.targets[1][0][0], self.targets[1][0][1]])
		# 			else:

		# 				p1 = np.array([self.targets[t_index][0][0], self.targets[t_index][0][1]])
		# 				p2 = np.array([self.targets[0][0][0], self.targets[0][0][1]])

		# mid_point = np.array([(p1[0]+p2[0])/2, (p1[1]+p2[1])/2])
		# decision_line = (mid_point - self.pos)/np.linalg.norm(mid_point - self.pos)
		# dl_1 = (p1 - self.pos)/np.linalg.norm(p1 - self.pos)
		# dl_2 = (p2 - self.pos)/np.linalg.norm(p2 - self.pos)

		# if np.sign(np.cross(dl_1, decision_line)) < 0:

		# 	p_l = np.array([p1[0], p1[1]])
		# 	p_r = np.array([p2[0], p2[1]])
		# else:

		# 	p_r = np.array([p1[0], p1[1]])
		# 	p_l = np.array([p2[0], p2[1]])

		# Angle Compensation
		base_v = (p_r - p_l)/np.linalg.norm(p_l - p_r); base_v = np.array([base_v[0], base_v[1], 0])
		v_p = (self.pos - p_l)/np.linalg.norm(self.pos - p_l); v_p = np.array([v_p[0], v_p[1], 0])
		z_ = np.cross(base_v, v_p)
		ct_line = np.cross(z_, base_v); ct_line = np.array([ct_line[0], ct_line[1]]); ct_line /= np.linalg.norm(ct_line)
		theta = np.arccos(np.dot(-self.perspective, ct_line))
		direction = np.cross(-self.perspective, ct_line)
		R = np.array([[np.cos(direction*theta), -np.sin(direction*theta)],[np.sin(direction*theta), np.cos(direction*theta)]])

		pos_v = self.pos - mid_point;
		pos = mid_point + np.matmul(R, pos_v)
		perspective = -np.matmul(R, -self.perspective)

		# Cost function 1-3

		# Calculation of height of trianlge
		base_length = np.linalg.norm(p_l - p_r)

		coords = [(self.targets[0][0][0], self.targets[0][0][1]),\
					(self.targets[1][0][0], self.targets[1][0][1]),\
					(self.targets[2][0][0], self.targets[2][0][1])]

		polygon = Polygon(coords)
		area = polygon.area
		height = (2*area)/base_length
		height_n = 0.5*abs(self.R_max - self.R*cos(self.alpha))

		l_line_v = p_l - pos; l_line_v = l_line_v/np.linalg.norm(l_line_v)
		r_line_v = p_r - pos; r_line_v = r_line_v/np.linalg.norm(r_line_v)
		theta = np.arccos(np.dot(l_line_v, r_line_v))

		Avg_distance = base_length*(height/height_n)*\
						np.exp(1*(theta - 0.5*self.alpha)/0.5*self.alpha)
		Avg_Sense = np.sum(self.HW_Sensing)/len(self.HW_Sensing)
		C_3 = (1/k1)*(1/Avg_Sense) + k2*Avg_distance*height

		# print("dist to target: ", end = "")
		# print(self.dist_to_targets)

		# Cost Function 1-2
		Avg_distance = np.linalg.norm(p_l - p_r)
		C_2 = (1/k1)*(1/Avg_Sense) + k2*Avg_distance*height

		# print("L: " + str(Avg_distance))

		# Cost Function 1-1
		top_side_l_v = sweet_spot - p_l; top_side_l_v = top_side_l_v/np.linalg.norm(top_side_l_v)
		top_side_r_v = sweet_spot - p_r; top_side_r_v = top_side_r_v/np.linalg.norm(top_side_r_v)
		base_v = (p_l - p_r)/base_length
		delta_l = np.dot(top_side_l_v, -base_v)
		delta_r = np.dot(top_side_r_v, +base_v)

		top_side_l = np.linalg.norm(sweet_spot - p_l)*delta_l
		top_side_r = np.linalg.norm(sweet_spot - p_r)*delta_r
		theta_l = np.arccos(np.dot(l_line_v, perspective))
		theta_r = np.arccos(np.dot(r_line_v, perspective))

		Avg_distance = top_side_l*np.exp(0.6*self.alpha - theta_l) + top_side_r*np.exp(0.6*self.alpha - theta_r)
		C_1 = (1/k1)*(1/Avg_Sense) + k2*Avg_distance*height

		# print("L1: " + str(top_side_l), "L2: " + str(top_side_r), end = "")
		# print(" theta1: " + str(theta_l), "theta2: " + str(theta_r))
		# print("alpha: " + str(self.alpha))
		# print("l1 + l2: " + str(Avg_distance))

		C_total = [C_1, C_2, C_3]
		min_C = np.argmin(C_total)+1

		if self.id == 1:

			print("Cost: ", end = "")
			print(C_total)

		# Mode Switch Control
		neighbors_cluster_teammate = self.neighbors["Cluster_Teammate"]
		if (len(Cluster) == AtoT):

			if min_C == 3:

				x, y = 0, 0

				for target in self.targets:

					x += target[0][0]
					y += target[0][1]

				self.target = [[(x/AtoT, y/AtoT), 1, 10]]
				self.Cluster_Teammate = [float(self.id), float(min_C), float(-1)]
			elif min_C == 2:

				switch_index = np.argmin(self.dist_to_cluster)
				self.Cluster_Teammate = [float(self.id), float(2.3), float(switch_index)]
				
				for ct in neighbors_cluster_teammate:

					if (ct[1] == self.Cluster_Teammate[1]) and\
						(ct[2] == self.Cluster_Teammate[2]):

						self.dist_to_cluster[int(self.Cluster_Teammate[2])] == 100
						self.Cluster_Teammate[2] = float(np.argmin(self.dist_to_cluster))

				self.target = [Cluster[int(self.Cluster_Teammate[2])]]
			elif min_C == 1:

				switch_index = np.argmin(self.dist_to_targets)
				self.Cluster_Teammate = [float(self.id), float(min_C), float(switch_index)]

				for ct in neighbors_cluster_teammate:

					if (ct[1] == self.Cluster_Teammate[1]) and\
						(ct[2] == self.Cluster_Teammate[2]):

						self.dist_to_targets[int(self.Cluster_Teammate[2])] == 100
						self.Cluster_Teammate[2] = float(np.argmin(self.dist_to_targets))

				self.target = [self.targets[int(self.Cluster_Teammate[2])]]

			# output_ = Float64MultiArray(data = self.Cluster_Teammate)
			# self.cluster_teammate_pub.publish(output_)

		if (len(Cluster) < AtoT):

			if min_C == 3:

				x = 0
			elif min_C == 2:

				if (len(Cluster) == AtoT - 1):

					switch_index = np.argmin(self.dist_to_cluster)
					self.Cluster_Teammate = [float(self.id), float(2.2), float(switch_index)]

					registration_form = np.ones(len(Cluster))
					sign_form = []

					for ct in neighbors_cluster_teammate:

						sign_form.append(ct[2])

					sign_form = np.array(sign_form)
					# print(sign_form)

					if (sign_form == [1,2]).all() or (sign_form == [2,1]).all():

						registration_form[0], registration_form[1] = 0, 0
					elif (sign_form == [0,2]).all() or (sign_form == [2,0]).all():

						registration_form[0], registration_form[1] = 0, 0
					elif (sign_form == [0,1]).all() or (sign_form == [1,0]).all():

						registration_form[0], registration_form[1] = 0, 0
					elif (sign_form == [0,0]).all():

						registration_form[0], registration_form[1] = 0, 0

					if (registration_form == 0).all():

						self.target = [Cluster[int(self.Cluster_Teammate[2])]]
					else:

						untracked_index = np.nonzero(registration_form)[0][0]
						self.Cluster_Teammate[2] = float(untracked_index)
						self.target = [Cluster[int(self.Cluster_Teammate[2])]]

					# self.last_Cluster_pair = Cluster_pair
				elif (len(Cluster) == AtoT - 2):

					escape_index, num_count = 0, 0

					for i in range(np.shape(self.Clsuter_Checklist)[0]):

						if (self.Clsuter_Checklist[i,:] == 0).all():

							escape_index = i

							break

					if int(np.argmin(self.dist_to_targets)) == escape_index:

						self.Cluster_Teammate = [float(self.id), float(2.1), float(escape_index)]

						for ct in neighbors_cluster_teammate:

							if (ct[1] == self.Cluster_Teammate[1]) and\
								(ct[2] == self.Cluster_Teammate[2]):

								num_count += 1

						if num_count == 0:

							self.target = [self.targets[int(self.Cluster_Teammate[2])]]
					else:
						num_count += 1

					if num_count != 0:

						index = [0,1,2]
						index = np.delete(index, np.argmax(self.dist_to_targets))
						x = (self.targets[index[0]][0][0] + self.targets[index[1]][0][0])/2
						y = (self.targets[index[0]][0][1] + self.targets[index[1]][0][1])/2

						self.target = [[(x, y), 1, 10]]
						self.Cluster_Teammate = [float(self.id), float(2.1), float(-1)]
			elif min_C == 1:

				# print(self.dist_to_targets)

				switch_index = np.argmin(self.dist_to_targets)
				self.Cluster_Teammate = [float(self.id), float(min_C), float(switch_index)]

				for ct in neighbors_cluster_teammate:

					if (ct[1] == self.Cluster_Teammate[1]) and\
						(ct[2] == self.Cluster_Teammate[2]):

						self.dist_to_targets[int(self.Cluster_Teammate[2])] = 100
						self.Cluster_Teammate[2] = float(np.argmin(self.dist_to_targets))

				self.target = [self.targets[int(self.Cluster_Teammate[2])]]

			# output_ = Float64MultiArray(data = self.Cluster_Teammate)
			# self.cluster_teammate_pub.publish(output_)

		# print("id: " + str(self.id), "\n")

	def publish_message(self):

		h_dim = MultiArrayDimension(label = "height", size = 1, stride = 1*1*3)
		w_dim = MultiArrayDimension(label = "width",  size = 1, stride = 1*1)
		c_dim = MultiArrayDimension(label = "channel", size = 3, stride = 3)
		layout = MultiArrayLayout(dim = [h_dim, w_dim, c_dim], data_offset = 0)

		output_ = Float64MultiArray(data = self.dist_to_cluster, layout = layout)
		self.cluster_dist_pub.publish(output_)

		output_ = Float64MultiArray(data = self.Cluster_Teammate, layout = layout)
		self.cluster_teammate_pub.publish(output_)

		if self.id == 1:

			print("targets: ", end='')
			print(self.targets)
			print("self target: ", end='')
			print(self.target)
			print("\n")

	def StageAssignment(self):

		range_max = 0.85*(self.lamb + 1)/(self.lamb)*self.R*cos(self.alpha)
		# range_max = self.R*cos(self.alpha)

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

		W = self.W; pos = self.pos; perspective = self.perspective; alpha = self.alpha; R = self.R
		lamb = self.lamb; R_ = R**(lamb+1)

		out = np.empty_like(self.W)
		ne.evaluate("W - pos", out = out)

		x = out[:,0]; y = out[:,1]
		d = np.empty_like(x)
		ne.evaluate("sqrt(x**2 + y**2)", out = d)

		# d = np.linalg.norm(np.subtract(self.W, self.pos), axis = 1)
		d = np.array([d]).transpose()
		d = d.transpose()[0]

		q_per = self.PerspectiveQuality(d, W, pos, perspective, alpha)
		q_res = self.ResolutionQuality(d, W, pos, perspective, alpha, R, lamb)
		Q = np.multiply(q_per, q_res)

		quality_map = ne.evaluate("where((q_per > 0) & (q_res > 0), Q, 0)")
		self.FoV = quality_map
		
		return

	def PerspectiveQuality(self, d, W, pos, perspective, alpha):

		out = np.empty_like(d)
		ne.evaluate("sum((W - pos)*perspective, axis = 1)", out = out)
		ne.evaluate("(out/d - cos(alpha))/(1 - cos(alpha) )", out = out)

		# return (np.divide(np.dot(np.subtract(self.W, self.pos), self.perspective), d) - np.cos(self.alpha))\
		# 		/(1 - np.cos(self.alpha))
		return out

	def ResolutionQuality(self, d, W, pos, perspective, alpha, R, lamb):

		R_ = R**(lamb+1)

		out = np.empty_like(d)
		ne.evaluate("(R*cos(alpha) - lamb*(d - R*cos(alpha)))*(d**lamb)/R_", out = out)

		# return np.multiply((self.R*np.cos(self.alpha) - self.lamb*(d - self.R*np.cos(self.alpha))),\
		# 					(np.power(d, self.lamb)/(self.R**(self.lamb+1))))
		return out

	def UpdateLocalVoronoi(self):

		id_ = self.id
		quality_map = self.FoV

		# for neighbor in self.neighbors:

		# 	FoV = neighbor.FoV
		# 	quality_map = ne.evaluate("where((quality_map >= FoV), quality_map, 0)")

		# self.voronoi = np.array(np.where((quality_map != 0) & (self.FoV != 0)))
		self.voronoi = np.array(np.where((self.FoV > 0)))
		# self.map_plt = np.array(ne.evaluate("where(quality_map != 0, id_ + 1, 0)"))

		return

	def ComputeCentroidal(self):

		translational_force = np.array([0.,0.])
		rotational_force = np.array([0.,0.]).reshape(2,1)
		zoom_force = 0
		centroid = None

		W = self.W[np.where(self.FoV > 0)]; pos = self.pos; lamb = self.lamb; R = self.R; R_ = R**lamb
		alpha = self.alpha; perspective = self.perspective
		
		out = np.empty_like(W)
		ne.evaluate("W - pos", out = out)
		x = out[:,0]; y = out[:,1]
		d = np.empty_like(x)
		ne.evaluate("sqrt(x**2 + y**2)", out = d)

		# d = np.linalg.norm(np.subtract(W, self.pos), axis = 1)
		d = np.array([d]).transpose()
		d[np.where(d == 0)] = 1

		F = multivariate_normal([self.target[0][0][0], self.target[0][0][1]],\
								[[self.target[0][1], 0.0], [0.0, self.target[0][1]]])

		x, y = self.map_size[0]*self.grid_size[0], self.map_size[1]*self.grid_size[1]

		if len(self.voronoi[0]) > 0:

			mu_V = np.empty_like([0.0], dtype = np.float64)
			v_V_t = np.empty_like([0, 0], dtype = np.float64)
			delta_V_t = np.empty_like([0.0], dtype = np.float64)
			x_center = np.empty_like([0.0], dtype = np.float64)
			y_center = np.empty_like([0.0], dtype = np.float64)

			# mu_V = np.sum(np.multiply(\
			# 		np.multiply(np.power(d, self.lamb).transpose()[0], F.pdf(W))/(self.R**self.lamb),\
			# 		self.In_polygon))
			# mu_V = np.sum(\
			# 		np.multiply(np.power(d, self.lamb).transpose()[0], F.pdf(W))/(self.R**self.lamb)
			# 		)

			out = np.empty_like(d); F_ = F.pdf(W)
			ne.evaluate("d**lamb", out = out)
			out = out.transpose()[0]
			ne.evaluate("sum((out*F_)/R_)", out = mu_V)
			mu_V = mu_V[0]

			# temp = np.multiply(np.multiply(np.multiply(\
			# 	np.cos(self.alpha) - (self.lamb/self.R/(self.lamb+1))*d.transpose()[0],\
			# 	d.transpose()[0]**(self.lamb)/(self.R**self.lamb)),\
			# 	F.pdf(self.W)),\
			# 	self.In_polygon)

			# temp = np.multiply(np.multiply(\
			# 	np.cos(self.alpha) - (self.lamb/self.R/(self.lamb+1))*d.transpose()[0],\
			# 	d.transpose()[0]**(self.lamb)/(self.R**self.lamb)),\
			# 	F.pdf(W))
			# temp = np.array([temp]).transpose()

			# v_V_t =  np.sum(np.multiply(\
			# 		(np.subtract(W, self.pos)/np.concatenate((d,d), axis = 1)),\
			# 		temp), axis = 0)

			d_ = d.transpose()[0]; temp = np.empty_like(d_); F_ = F.pdf(W);
			ne.evaluate("(cos(alpha) - (lamb/R/(lamb+1))*d_)*(d_**lamb/R_)*F_", out = temp)
			temp = np.array([temp]).transpose()

			d_ = np.concatenate((d,d), axis = 1)
			ne.evaluate("sum(((W - pos)/d_)*temp, axis = 0)", out = v_V_t)

			# delta_V_t = np.sum(np.multiply(np.multiply(np.multiply(np.multiply(\
			# 			(1 - np.dot(np.divide(np.subtract(W, self.pos),np.concatenate((d,d), axis = 1)), self.perspective)),\
			# 			(1 - (self.lamb/self.R/(self.lamb+1))*d.transpose()[0])),\
			# 			d.transpose()[0]**(self.lamb)/(self.R**self.lamb)),\
			# 			F.pdf(W)),\
			# 			self.In_polygon))
			# delta_V_t = np.sum(np.multiply(np.multiply(np.multiply(\
			# 			(1 - np.dot(np.divide(np.subtract(W, self.pos),np.concatenate((d,d), axis = 1)), self.perspective)),\
			# 			(1 - (self.lamb/self.R/(self.lamb+1))*d.transpose()[0])),\
			# 			d.transpose()[0]**(self.lamb)/(self.R**self.lamb)),\
			# 			F.pdf(W)))

			d_ = np.concatenate((d,d), axis = 1); F_ = F.pdf(W); out = np.empty_like(F_)
			ne.evaluate("sum(((W - pos)/d_)*perspective, axis = 1)", out = out)
			d_ = d.transpose()[0];
			ne.evaluate("sum((1 - out)*(1 - (lamb/R/(lamb+1))*d_)*(d_**lamb/R_)*F_, axis = 0)", out = delta_V_t)
			delta_V_t = delta_V_t[0]

			v_V = v_V_t/mu_V
			delta_V = delta_V_t/mu_V
			delta_V = delta_V if delta_V > 0 else 1e-10
			alpha_v = acos(1-sqrt(delta_V))
			alpha_v = alpha_v if alpha_v > 5/180*np.pi else 5/180*np.pi

			# x_center = np.sum(np.multiply(np.multiply(\
			# 	W[:,0],\
			# 	(np.multiply(d.transpose()[0]**(self.lamb),F.pdf(W))/(self.R**self.lamb))),\
			# 	self.In_polygon))/mu_V

			# y_center = np.sum(np.multiply(np.multiply(\
			# 	W[:,1],\
			# 	(np.multiply(d.transpose()[0]**(self.lamb),F.pdf(W))/(self.R**self.lamb))),\
			# 	self.In_polygon))/mu_V

			# x_center = np.sum(np.multiply(\
			# 	W[:,0],\
			# 	(np.multiply(d.transpose()[0]**(self.lamb),F.pdf(W))/(self.R**self.lamb))))/mu_V

			# y_center = np.sum(np.multiply(\
			# 	W[:,1],\
			# 	(np.multiply(d.transpose()[0]**(self.lamb),F.pdf(W))/(self.R**self.lamb))))/mu_V

			W_x = W[:,0]; W_y = W[:,1]; d_ = d.transpose()[0]; F_ = F.pdf(W);
			ne.evaluate("sum(W_x*(d_**lamb/R_)*F_)", out = x_center); x_center = x_center[0]/mu_V
			ne.evaluate("sum(W_y*(d_**lamb/R_)*F_)", out = y_center); y_center = y_center[0]/mu_V
		    
			centroid = np.array([x_center, y_center])
			translational_force += self.Kp*(np.linalg.norm(centroid - self.pos)\
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
		neighbors_pos = self.neighbors["Position"]

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

	def P(self, d, a, In_polygon, P0_I, R, alpha):

		out = np.empty_like(d)
		ne.evaluate("(d - R*cos(alpha))**2", out = out); out = out.transpose()[0];
		out_ = np.empty_like(d.transpose()[0]);
		ne.evaluate("P0_I*exp(-(out/((2*0.2**2)*(R*cos(alpha)**2))))*exp(-((abs(a)-0)**2)/((2*0.2**2)*(alpha**2)))",\
					out = out_)

		return out_

	def P_(self, d, a, In_polygon, P0_B, R, alpha, R_max):

		out = np.empty_like(d)
		ne.evaluate("(abs(d-0.5*R_max)-0.5*R_max)**2", out = out); out = out.transpose()[0];
		out_1 = np.empty_like(d.transpose()[0]);
		ne.evaluate("P0_B*exp(-(out/((2*0.25**2)*(0.5*R_max**2))))*exp(-((abs(a)-alpha)**2)/((2*0.35**2)*(alpha**2)))",\
					out = out_1)

		out = np.empty_like(d)
		ne.evaluate("(d - 0.5*R)**2", out = out); out = out.transpose()[0];
		out_2 = np.empty_like(d.transpose()[0]);
		ne.evaluate("out_1 + P0_B*exp(-(out/((2*0.3**2)*(0.5*R**2))))*exp(-((abs(a)-0)**2)/((2*0.5**2)*(alpha**2)))",\
					out = out_2)

		return out_2

	def EscapeDensity(self, time_):

		# Environment
		# L = 25
		# Wi = 25
		# x_range = np.arange(0, L, 0.1)
		# y_range = np.arange(0, L, 0.1)

		# L = self.map_size[0]
		# Wi = self.map_size[1]
		# x_range = np.arange(0, L, self.grid_size[0])
		# y_range = np.arange(0, L, self.grid_size[1])
		# X, Y = np.meshgrid(x_range, y_range)

		# W = np.vstack([X.ravel(), Y.ravel()])
		# W = W.transpose()

		W = self.W[np.where(self.FoV > 0)]
		pos = self.pos; perspective = self.perspective; alpha = self.alpha; R = self.R
		lamb = self.lamb;

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

		# Interior
		# P = lambda d, a, IoO, P0: P0*np.multiply(np.multiply(\
		# 			np.exp(-np.divide(np.power(np.subtract(d, self.R*np.cos(self.alpha)), 2).transpose()[0], (2*0.2**2)*(self.R*np.cos(self.alpha)**2))),\
		# 			np.exp(-np.divide(np.power(np.subtract(abs(a), 0), 2), (2*0.2**2)*(self.alpha**2)))), IoO)
		P = lambda d, a, IoO, P0: P0*np.multiply(\
					np.exp(-np.divide(np.power(np.subtract(d, self.R*np.cos(self.alpha)), 2).transpose()[0], (2*0.2**2)*(self.R*np.cos(self.alpha)**2))),\
					np.exp(-np.divide(np.power(np.subtract(abs(a), 0), 2), (2*0.2**2)*(self.alpha**2))))

		# Boundary
		# P_ = lambda d, a, IoO, P0: P0*np.multiply(np.add(np.add(\
		# 			np.exp(-np.divide(np.power(np.subtract(abs(d-0.5*range_max), 0.5*range_max), 2).transpose()[0], (2*0.25**2)*(0.5*range_max**2))),\
		# 			np.exp(-np.divide(np.power(np.subtract(abs(a), self.alpha), 2), (2*0.35**2)*(self.alpha**2)))),\
		# 			P0*np.multiply(np.multiply(\
		# 			np.exp(-np.divide(np.power(np.subtract(d, 0.5*self.R), 2).transpose()[0], (2*0.3**2)*(0.5*self.R**2))),\
		# 			np.exp(-np.divide(np.power(np.subtract(abs(a), 0), 2), (2*0.5**2)*(self.alpha**2)))), IoO)
		# 			), IoO)
		P_ = lambda d, a, IoO, P0: P0*np.add(np.add(\
					np.exp(-np.divide(np.power(np.subtract(abs(d-0.5*range_max), 0.5*range_max), 2).transpose()[0], (2*0.25**2)*(0.5*range_max**2))),\
					np.exp(-np.divide(np.power(np.subtract(abs(a), self.alpha), 2), (2*0.35**2)*(self.alpha**2)))),\
					P0*np.multiply(\
					np.exp(-np.divide(np.power(np.subtract(d, 0.5*self.R), 2).transpose()[0], (2*0.3**2)*(0.5*self.R**2))),\
					np.exp(-np.divide(np.power(np.subtract(abs(a), 0), 2), (2*0.5**2)*(self.alpha**2)))))

		# Spatial Sensing Quality
		# Q = lambda W, d, IoO, P0: P0*np.multiply(np.multiply((np.divide(\
		# 			np.dot(np.subtract(W, self.pos),self.perspective), d) - np.cos(self.alpha))/(1 - np.cos(self.alpha)),\
		# 			np.multiply((self.R*np.cos(self.alpha) - self.lamb*(d - self.R*np.cos(self.alpha))),\
		# 			(np.power(d, self.lamb)/(self.R**(self.lamb+1))))), IoO)
		Q = lambda W, d, IoO, P0: P0*np.multiply((np.divide(\
					np.dot(np.subtract(W, self.pos),self.perspective), d) - np.cos(self.alpha))/(1 - np.cos(self.alpha)),\
					np.multiply((self.R*np.cos(self.alpha) - self.lamb*(d - self.R*np.cos(self.alpha))),\
					(np.power(d, self.lamb)/(self.R**(self.lamb+1)))))

		# Points in FoV
		pt = [A+np.array([0, 0.1]), B+np.array([0.1, -0.1]), C+np.array([-0.1, -0.1]), A+np.array([0, 0.1])]
		polygon = Path(pt)
		In_polygon = polygon.contains_points(self.W) # Boolean

		# Distance and Angle of W with respect to self.pos
		# d = np.linalg.norm(np.subtract(W, self.pos), axis = 1)
		out = np.empty_like(W)
		ne.evaluate("W - pos", out = out)
		x = out[:,0]; y = out[:,1]
		d = np.empty_like(x)
		ne.evaluate("sqrt(x**2 + y**2)", out = d)
		d = np.array([d]).transpose()

		# a = np.arccos( np.dot(np.divide(np.subtract(W, self.pos),np.concatenate((d,d), axis = 1)), self.perspective) )
		# a = np.arccos( np.dot(np.subtract(W, self.pos)/np.concatenate((d,d), axis = 1), self.perspective) )
		d_ = np.concatenate((d,d), axis = 1); a = np.empty_like(d.transpose()[0])
		ne.evaluate("sum(((W - pos)/d_)*perspective, axis = 1)", out = a)
		ne.evaluate("arccos(a)", out = a)

		# Cost Function
		P0_I = 0.9
		# JP_Interior = P(d, a, In_polygon, P0_I)
		# HW_Interior = np.sum(np.multiply(F1.pdf(W), JP_Interior))\
		# 			+ np.sum(np.multiply(F2.pdf(W), JP_Interior))\
		# 			+ np.sum(np.multiply(F3.pdf(W), JP_Interior))
		JP_Interior = self.P(d, a, In_polygon, P0_I, R, alpha)
		F1_ = F1.pdf(W); F2_ = F2.pdf(W); F3_ = F3.pdf(W)
		HW_Interior_1 = ne.evaluate("sum(F1_*JP_Interior)")
		HW_Interior_2 = ne.evaluate("sum(F2_*JP_Interior)")
		HW_Interior_3 = ne.evaluate("sum(F3_*JP_Interior)")
		HW_Interior = ne.evaluate("HW_Interior_1 + HW_Interior_2 + HW_Interior_3")

		P0_B = 0.9
		# JP_Boundary = P_(d, a, In_polygon, P0_B)
		# HW_Boundary = np.sum(np.multiply(F1.pdf(W), JP_Boundary))\
		# 			+ np.sum(np.multiply(F2.pdf(W), JP_Boundary))\
		# 			+ np.sum(np.multiply(F3.pdf(W), JP_Boundary))
		JP_Boundary = self.P_(d, a, In_polygon, P0_B, R, alpha, range_max)
		HW_Boundary_1 = ne.evaluate("sum(F1_*JP_Boundary)")
		HW_Boundary_2 = ne.evaluate("sum(F2_*JP_Boundary)")
		HW_Boundary_3 = ne.evaluate("sum(F3_*JP_Boundary)")
		HW_Boundary = ne.evaluate("HW_Boundary_1 + HW_Boundary_2 + HW_Boundary_3")

		# Sensing Quality
		P0_Q = 1.0
		d = d.transpose()[0]
		# SQ = Q(W, d, In_polygon, P0_Q)
		# HW_Sensing = np.sum(np.multiply(F1.pdf(W), SQ))\
		# 			+ np.sum(np.multiply(F2.pdf(W), SQ))\
		# 			+ np.sum(np.multiply(F3.pdf(W), SQ))
		q_per = self.PerspectiveQuality(d, W, pos, perspective, alpha)
		q_res = self.ResolutionQuality(d, W, pos, perspective, alpha, R, lamb)
		SQ = ne.evaluate("P0_Q*q_per*q_res")
		HW_SQ = [ne.evaluate("sum(F1_*SQ)"), ne.evaluate("sum(F2_*SQ)"), ne.evaluate("sum(F3_*SQ)")]

		# self.HW_IT = HW_Interior*0.1**2
		# self.HW_BT = HW_Boundary*0.1**2
		# self.HW_Sensing = [np.sum(np.multiply(F1.pdf(W), SQ)), np.sum(np.multiply(F2.pdf(W), SQ)), np.sum(np.multiply(F3.pdf(W), SQ))]
		# self.In_polygon = In_polygon
		self.HW_IT = ne.evaluate("HW_Interior*0.1**2")
		self.HW_BT = ne.evaluate("HW_Boundary*0.1**2")
		self.HW_Sensing = HW_SQ
		self.In_polygon = In_polygon

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

# if __name__ == '__main__':

# 	try:
# 		rospy.init_node('controller_1')
# 		rate = rospy.Rate(100)

# 		map_size = np.array([25, 25])
# 		grid_size = np.array([0.1, 0.1])

# 		camera0 = { 'id'            :  0,
# 					'position'      :  np.array([1., 8.]),
# 					'perspective'   :  np.array([0.9,1]),
# 					'AngleofView'   :  20,
# 					'range_limit'   :  5,
# 					'lambda'        :  2,
# 					'color'         : (200, 0, 0)}

# 		uav_1 = UAV(camera0, map_size, grid_size)

# 		while uav_1.b is None:

# 			rate.sleep()

# 		uav_1.qp_ini()

# 		last = time()

# 		while not rospy.is_shutdown():

# 			uav_1.UpdateState(np.round(time() - last, 2))
# 			rate.sleep()

# 	except rospy.ROSInterruptException:
# 		pass
