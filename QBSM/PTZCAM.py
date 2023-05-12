import warnings
warnings.simplefilter(action = 'ignore', category = RuntimeWarning)

import sys
import csv
import random
import numpy as np
import numexpr as ne
from time import sleep, time
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy import ndimage, sparse
from shapely.geometry import Point
from math import cos, acos, sqrt, exp, sin
from scipy.stats import multivariate_normal
from shapely.geometry.polygon import Polygon
from scipy.optimize import linear_sum_assignment
from shapely.plotting import plot_polygon, plot_points

class PTZcon():

	def __init__(self, properties, map_size, grid_size,\
					Kv = 50, Ka = 3, Kp = 2, step = 0.1):

		# Environment
		self.grid_size = grid_size
		self.map_size = map_size
		self.size = (int(map_size[0]/grid_size[0]), int(map_size[1]/grid_size[1]))

		x_range = np.arange(0, self.map_size[0], self.grid_size[0])
		y_range = np.arange(0, self.map_size[1], self.grid_size[1])
		X, Y = np.meshgrid(x_range, y_range)

		W = np.vstack([X.ravel(), Y.ravel()])
		self.W = W.transpose()
		
		# Properties of UAV
		self.id = properties['id']
		self.pos = properties['position']
		self.perspective = properties['perspective']/np.linalg.norm(properties['perspective'])
		self.alpha = properties['AngleofView']/180*np.pi
		self.R = properties['range_limit']
		self.lamb = properties['lambda']
		self.color = properties['color']
		self.R_max = (self.lamb + 1)/(self.lamb)*self.R
		self.r = 0
		self.top = 0
		self.ltop = 0
		self.rtop = 0
		self.centroid = None

		# Tracking Configuration
		self.cluster_count = 0
		self.dist_to_cluster = 0
		self.dist_to_targets = 0
		self.Clsuter_Checklist = None
		self.state_machine = {"self": None, "mode": None, "target": None}
		self.attract_center = [3, None, 2, None, 1, None] # "0": 3, "1": 0, "2": 2, "3": 0, "4",: 1, "5": 0

		# Relative Control Law
		self.translation_force = 0  # dynamics of positional changes
		self.perspective_force = 0  # dynamics of changing perspective direction
		self.stage = 1              # 1: Tracker, 2: Formation Cooperative
		self.target = None
		self.virtual_target = self.R*cos(self.alpha)*self.perspective
		self.target_assigned = -1
		self.step = step
		self.FoV = np.zeros(np.shape(self.W)[0])
		self.Kv = Kv                # control gain for perspective control law toward voronoi cell
		self.Ka = Ka                # control gain for zoom level control stems from voronoi cell
		self.Kp = Kp                # control gain for positional change toward voronoi cell 
		self.event = np.zeros((self.size[0], self.size[1]))

	def UpdateState(self, targets, neighbors, time_):

		self.neighbors = neighbors
		self.time = time_

		self.UpdateFoV()
		self.polygon_FOV()
		self.EscapeDensity(targets, time_)
		self.UpdateLocalVoronoi()

		# self.Cluster_Formation(targets)
		# self.Cluster_Assignment(targets, time_)
		self.Gradient_Descent(targets, time_)

		# event = np.zeros((self.size[0], self.size[1]))
		# self.event = self.event_density(event, self.target, self.grid_size)
		
		self.ComputeCentroidal(time_)
		self.StageAssignment()
		self.FormationControl()
		self.UpdateOrientation()
		self.UpdateZoomLevel()
		self.UpdatePosition()
		
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

	def Gaussian_Normal_1D(self, x, mu, sigma):

		return (1/(sigma * np.sqrt(2*np.pi))) * np.exp(-(x-mu)**2/(2*sigma**2))

	def Cluster_Formation(self, targets, d):

		checklist = np.zeros((len(targets), len(targets)))
		threshold = d
		self.cluster_count = 0

		for i in range(len(targets)):

			for j in range(len(targets)):

				if j != i:

					p1 = np.array([targets[i][0][0], targets[i][0][1]])
					p2 = np.array([targets[j][0][0], targets[j][0][1]])

					dist = np.linalg.norm(p1 - p2)

					if dist <= threshold:

						checklist[i][j] = 1
						self.cluster_count += 1
					else:

						checklist[i][j] = 0

		self.Clsuter_Checklist = checklist

		return

	def Cluster_Assignment(self, targets, time_):

		self.Cluster_Formation(targets, 2.3)

		count = 0
		Cluster = []

		if len(targets) == 3:

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

						c_x = 0.5*(targets[i][0][0] + targets[j][0][0])
						c_y = 0.5*(targets[i][0][1] + targets[j][0][1])

						Cluster.append([(c_x, c_y), 1, 10])
			elif i == 0:

				for j in nonindex:
		        
					c_x = 0.5*(targets[i][0][0] + targets[j][0][0])
					c_y = 0.5*(targets[i][0][1] + targets[j][0][1])

					Cluster.append([(c_x, c_y), 1, 10])

		# Calculate dist between each cluster for Hungarian Algorithm
		dist_to_cluster = np.array([100.0000, 100.0000, 100.0000], dtype = float)

		for (mem, i) in zip(Cluster, range(len(Cluster))):

			p1 = np.array([self.pos[0], self.pos[1]])
			p2 = np.array([mem[0][0], mem[0][1]])

			dist_to_cluster[i] = np.linalg.norm(p1 - p2)

		self.dist_to_cluster = dist_to_cluster

		if (self.cluster_count == cluster_count_ref):

			x, y = 0, 0
			cert = 0
			score = -np.inf

			for mem in targets:

				x += mem[0][0]
				y += mem[0][1]

			for mem in Cluster:

				p1 = np.array([mem[0][0], mem[0][1]])
				p2 = np.array([x/len(targets), y/len(targets)])

				dist = np.linalg.norm(p1 - p2)

				if dist > score and dist > 0:

					cert = np.exp(-0.5*(dist/1.5))

			self.target = [[(x/AtoT, y/AtoT), cert, 10]]

		# Mode Switch Control
		# cluster_quality = 0
		# Best_quality_ref = 2200

		# for (target, i) in zip(targets, range(0,len(targets))):

		# 	F = multivariate_normal([target[0][0], target[0][1]],\
		# 						[[target[1], 0.0], [0.0, target[0][1]]])

		# 	# event = np.zeros((self.size[0], self.size[1]))
		# 	# event1 = self.event_density(event, [target], self.grid_size)
		# 	# event1 = np.transpose(event1)
		# 	cluster_quality += np.sum(np.multiply(F.pdf(self.W), self.FoV))

		# Calculate dist between each target for Hungarian Algorithm
		pt = [self.pos, self.ltop, self.top, self.rtop, self.pos]
		polygon = Polygon(pt)
		
		dist_to_targets = np.array([100.00, 100.00, 100.00], dtype = float)

		for (mem, i) in zip(targets, range(len(targets))):

			gemos = Point(mem[0])
			# if polygon.is_valid and polygon.contains(gemos):
			if polygon.is_valid:

				p1 = np.array([self.pos[0], self.pos[1]])
				p2 = np.array([mem[0][0], mem[0][1]])

				dist_to_targets[i] = np.linalg.norm(p1 - p2)

		self.dist_to_targets = dist_to_targets

		# Configuration of calculation cost function
		Avg_distance = 0.0
		k1, k2 = self.HW_IT, self.HW_BT
		sweet_spot = self.pos + self.R*np.cos(self.alpha)*self.perspective

		t_index = [0,1,2]
		t_index = np.delete(t_index, np.argmax(self.dist_to_targets))
		p1 = np.array([targets[t_index[0]][0][0], targets[t_index[0]][0][1]])
		p2 = np.array([targets[t_index[1]][0][0], targets[t_index[1]][0][1]])
		mid_point = np.array([(p1[0]+p2[0])/2, (p1[1]+p2[1])/2])
		decision_line = (mid_point - self.pos)/np.linalg.norm(mid_point - self.pos)
		dl_1 = (p1 - self.pos)/np.linalg.norm(p1 - self.pos)
		dl_2 = (p2 - self.pos)/np.linalg.norm(p2 - self.pos)

		if np.cross(dl_1, decision_line) < 0:

			p_l = np.array([targets[t_index[0]][0][0], targets[t_index[0]][0][1]])
			p_r = np.array([targets[t_index[1]][0][0], targets[t_index[1]][0][1]])
		else:

			p_r = np.array([targets[t_index[0]][0][0], targets[t_index[0]][0][1]])
			p_l = np.array([targets[t_index[1]][0][0], targets[t_index[1]][0][1]])

		# fail_ = 0
		# for i in range(0, 3):

		# 	if i == 0:

		# 		v_1 = np.array([targets[1][0][0], targets[1][0][1]])\
		# 			- np.array([targets[0][0][0], targets[0][0][1]]); v_1 = v_1/np.linalg.norm(v_1)
		# 		v_2 = np.array([targets[2][0][0], targets[2][0][1]])\
		# 			- np.array([targets[0][0][0], targets[0][0][1]]); v_2 = v_2/np.linalg.norm(v_2)
		# 		v_3 = np.array([self.pos[0], self.pos[1]])\
		# 			- np.array([targets[0][0][0], targets[0][0][1]]); v_3 = v_3/np.linalg.norm(v_3)

		# 		if (np.sign(np.cross(v_1, v_2)) == np.sign(np.cross(v_1, v_3))) and\
		# 			(np.sign(np.cross(v_2, v_1)) == np.sign(np.cross(v_2, v_3))):

		# 			p1 = np.array([targets[1][0][0], targets[1][0][1]])
		# 			p2 = np.array([targets[2][0][0], targets[2][0][1]])
		# 		else:

		# 			fail_ += 1
		# 	elif i == 1:

		# 		v_1 = np.array([targets[0][0][0], targets[0][0][1]])\
		# 			- np.array([targets[1][0][0], targets[1][0][1]]); v_1 = v_1/np.linalg.norm(v_1)
		# 		v_2 = np.array([targets[2][0][0], targets[2][0][1]])\
		# 			- np.array([targets[1][0][0], targets[1][0][1]]); v_2 = v_2/np.linalg.norm(v_2)
		# 		v_3 = np.array([self.pos[0], self.pos[1]])\
		# 			- np.array([targets[1][0][0], targets[1][0][1]]); v_3 = v_3/np.linalg.norm(v_3)

		# 		if (np.sign(np.cross(v_1, v_2)) == np.sign(np.cross(v_1, v_3))) and\
		# 			(np.sign(np.cross(v_2, v_1)) == np.sign(np.cross(v_2, v_3))):

		# 			p1 = np.array([targets[0][0][0], targets[0][0][1]])
		# 			p2 = np.array([targets[2][0][0], targets[2][0][1]])
		# 		else:

		# 			fail_ += 1
		# 	elif i == 2:

		# 		v_1 = np.array([targets[0][0][0], targets[0][0][1]])\
		# 			- np.array([targets[2][0][0], targets[2][0][1]]); v_1 = v_1/np.linalg.norm(v_1)
		# 		v_2 = np.array([targets[1][0][0], targets[1][0][1]])\
		# 			- np.array([targets[2][0][0], targets[2][0][1]]); v_2 = v_2/np.linalg.norm(v_2)
		# 		v_3 = np.array([self.pos[0], self.pos[1]])\
		# 			- np.array([targets[2][0][0], targets[2][0][1]]); v_3 = v_3/np.linalg.norm(v_3)

		# 		if (np.sign(np.cross(v_1, v_2)) == np.sign(np.cross(v_1, v_3))) and\
		# 			(np.sign(np.cross(v_2, v_1)) == np.sign(np.cross(v_2, v_3))):

		# 			p1 = np.array([targets[0][0][0], targets[0][0][1]])
		# 			p2 = np.array([targets[1][0][0], targets[1][0][1]])
		# 		else:

		# 			fail_ += 1

		# 	if fail_ == 3:

		# 		t_index = np.argmin(self.dist_to_targets)

		# 		if t_index == 0:

		# 			v_1 = np.array([targets[0][0][0], targets[0][0][1]])\
		# 				- np.array([targets[1][0][0], targets[1][0][1]]); v_1 = v_1/np.linalg.norm(v_1)
		# 			v_2 = np.array([targets[0][0][0], targets[0][0][1]])\
		# 				- np.array([targets[2][0][0], targets[2][0][1]]); v_2 = v_2/np.linalg.norm(v_2)
		# 			v_3 = np.array([self.pos[0], self.pos[1]])\
		# 				- np.array([targets[0][0][0], targets[0][0][1]]); v_3 = v_3/np.linalg.norm(v_3)

		# 			if np.arccos(np.dot(v_1, v_3)) < np.arccos(np.dot(v_2, v_3)):

		# 				p1 = np.array([targets[t_index][0][0], targets[t_index][0][1]])
		# 				p2 = np.array([targets[2][0][0], targets[2][0][1]])
		# 			else:

		# 				p1 = np.array([targets[t_index][0][0], targets[t_index][0][1]])
		# 				p2 = np.array([targets[1][0][0], targets[1][0][1]])
		# 		elif t_index == 1:

		# 			v_1 = np.array([targets[1][0][0], targets[1][0][1]])\
		# 				- np.array([targets[0][0][0], targets[0][0][1]]); v_1 = v_1/np.linalg.norm(v_1)
		# 			v_2 = np.array([targets[1][0][0], targets[1][0][1]])\
		# 				- np.array([targets[2][0][0], targets[2][0][1]]); v_2 = v_2/np.linalg.norm(v_2)
		# 			v_3 = np.array([self.pos[0], self.pos[1]])\
		# 				- np.array([targets[1][0][0], targets[1][0][1]]); v_3 = v_3/np.linalg.norm(v_3)

		# 			if np.arccos(np.dot(v_1, v_3)) < np.arccos(np.dot(v_2, v_3)):

		# 				p1 = np.array([targets[t_index][0][0], targets[t_index][0][1]])
		# 				p2 = np.array([targets[2][0][0], targets[2][0][1]])
		# 			else:

		# 				p1 = np.array([targets[t_index][0][0], targets[t_index][0][1]])
		# 				p2 = np.array([targets[0][0][0], targets[0][0][1]])
		# 		elif t_index == 2:

		# 			v_1 = np.array([targets[2][0][0], targets[2][0][1]])\
		# 				- np.array([targets[0][0][0], targets[0][0][1]]); v_1 = v_1/np.linalg.norm(v_1)
		# 			v_2 = np.array([targets[2][0][0], targets[2][0][1]])\
		# 				- np.array([targets[1][0][0], targets[1][0][1]]); v_2 = v_2/np.linalg.norm(v_2)
		# 			v_3 = np.array([self.pos[0], self.pos[1]])\
		# 				- np.array([targets[2][0][0], targets[2][0][1]]); v_3 = v_3/np.linalg.norm(v_3)

		# 			if np.arccos(np.dot(v_1, v_3)) < np.arccos(np.dot(v_2, v_3)):

		# 				p1 = np.array([targets[t_index][0][0], targets[t_index][0][1]])
		# 				p2 = np.array([targets[1][0][0], targets[1][0][1]])
		# 			else:

		# 				p1 = np.array([targets[t_index][0][0], targets[t_index][0][1]])
		# 				p2 = np.array([targets[0][0][0], targets[0][0][1]])

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
		# pos = self.pos
		# perspective = self.perspective

		# Cost function 1-3

		# Calculation of height of trianlge
		base_length = np.linalg.norm(p_l - p_r)

		coords = [(targets[0][0][0], targets[0][0][1]),\
					(targets[1][0][0], targets[1][0][1]),\
					(targets[2][0][0], targets[2][0][1])]

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

		if time_ >= 35:

			C_3 = min((1/k1)*(1/Avg_Sense) + k2*Avg_distance*height, 20)
		else:

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

		# print("C1: " + str(C_1))
		# print("C2: " + str(C_2))
		# print("C3: " + str(C_3))

		# C_total.append(time_)
		# # filename = "D:/上課資料/IME/實驗室研究/Paper/Coverage Control/Quality based switch mode/Data/"
		# filename = "/home/leo/mts/src/QBSM/Data/"
		# filename += "Data_" + str(self.id) + ".csv"
		# with open(filename, "a", encoding='UTF8', newline='') as f:

		# 	row = C_total
		# 	writer = csv.writer(f)
		# 	writer.writerow(row)

		# Mode Switch Control
		if (len(Cluster) == AtoT):

			if min_C == 3:

				x, y = 0, 0

				for target in targets:

					x += target[0][0]
					y += target[0][1]

				self.target = [[(x/AtoT, y/AtoT), 1, 10]]
				self.state_machine["self"] = self.id
				self.state_machine["mode"] = min_C
				self.state_machine["target"] = -1
			elif min_C == 2:

				switch_index = np.argmin(self.dist_to_cluster)

				self.state_machine["self"] = self.id
				self.state_machine["mode"] = 2.3
				self.state_machine["target"] = int(switch_index)
				
				for neighbor in self.neighbors:

					if (neighbor.state_machine["mode"] == self.state_machine["mode"]) and\
						(neighbor.state_machine["target"] == self.state_machine["target"]):

						self.dist_to_cluster[self.state_machine["target"]] == 100
						self.state_machine["target"] = int(np.argmin(self.dist_to_cluster))

				self.target = [Cluster[self.state_machine["target"]]]
			elif min_C == 1:

				switch_index = np.argmin(self.dist_to_targets)

				self.state_machine["self"] = self.id
				self.state_machine["mode"] = min_C
				self.state_machine["target"] = int(switch_index)

				for neighbor in self.neighbors:

					if (neighbor.state_machine["mode"] == self.state_machine["mode"]) and\
						(neighbor.state_machine["target"] == self.state_machine["target"]):

						self.dist_to_targets[self.state_machine["target"]] == 100
						self.state_machine["target"] = int(np.argmin(self.dist_to_targets))

				self.target = [targets[self.state_machine["target"]]]

		if (len(Cluster) < AtoT):

			if min_C == 3:

				x = 0
			elif min_C == 2:

				if (len(Cluster) == AtoT - 1):

					switch_index = np.argmin(self.dist_to_cluster)

					self.state_machine["self"] = self.id
					self.state_machine["mode"] = 2.2
					self.state_machine["target"] = int(switch_index)

					registration_form = np.ones(len(Cluster))
					sign_form = []

					for neighbor in self.neighbors:

						sign_form.append(neighbor.state_machine["target"])

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

						self.target = [Cluster[self.state_machine["target"]]]
					else:

						untracked_index = np.nonzero(registration_form)[0][0]
						self.state_machine["target"] = int(untracked_index)
						self.target = [Cluster[self.state_machine["target"]]]

					# self.last_Cluster_pair = Cluster_pair
				elif (len(Cluster) == AtoT - 2):

					escape_index, num_count = 0, 0

					for i in range(np.shape(self.Clsuter_Checklist)[0]):

						if (self.Clsuter_Checklist[i,:] == 0).all():

							escape_index = i

							break
					
					if np.argmin(self.dist_to_targets) == escape_index:

						self.state_machine["self"] = self.id
						self.state_machine["mode"] = 2.1
						self.state_machine["target"] = escape_index

						for neighbor in self.neighbors:

							if (neighbor.state_machine["mode"] == self.state_machine["mode"]) and\
								(neighbor.state_machine["target"] == self.state_machine["target"]):

								num_count += 1

						if num_count == 0:

							self.target = [targets[self.state_machine["target"]]]
					else:
						num_count += 1

					if num_count != 0:

						index = [0,1,2]
						index = np.delete(index, np.argmax(self.dist_to_targets))
						x = (targets[index[0]][0][0] + targets[index[1]][0][0])/2
						y = (targets[index[0]][0][1] + targets[index[1]][0][1])/2

						self.target = [[(x, y), 1, 10]]
						self.state_machine["self"] = self.id
						self.state_machine["mode"] = 2.1
						self.state_machine["target"] = -1
			elif min_C == 1:

				# print(self.dist_to_targets)

				switch_index = np.argmin(self.dist_to_targets)

				self.state_machine["self"] = self.id
				self.state_machine["mode"] = min_C
				self.state_machine["target"] = int(switch_index)

				for neighbor in self.neighbors:

					if (neighbor.state_machine["mode"] == self.state_machine["mode"]) and\
						(neighbor.state_machine["target"] == self.state_machine["target"]):

						self.dist_to_targets[self.state_machine["target"]] = 100
						self.state_machine["target"] = int(np.argmin(self.dist_to_targets))

				self.target = [targets[self.state_machine["target"]]]

		print("id: " + str(self.id), "\n")

	def Gradient_Descent(self, targets, time_):

		self.Cluster_Formation(targets, 30)

		count = 0
		Cluster = []
		Cluster_pair = []

		if len(targets) == 3:

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

						c_x = 0.5*(targets[i][0][0] + targets[j][0][0])
						c_y = 0.5*(targets[i][0][1] + targets[j][0][1])

						Cluster.append([(c_x, c_y), 1, 10])
						Cluster_pair.append((i,j))
			elif i == 0:

				for j in nonindex:
		        
					c_x = 0.5*(targets[i][0][0] + targets[j][0][0])
					c_y = 0.5*(targets[i][0][1] + targets[j][0][1])

					Cluster.append([(c_x, c_y), 1, 10])
					Cluster_pair.append((i,j))

		# Decide Geometry Center
		if (self.cluster_count == cluster_count_ref):

			x, y = 0, 0
			cert = 0
			score = -np.inf

			for mem in targets:

				x += mem[0][0]
				y += mem[0][1]

			for mem in Cluster:

				p1 = np.array([mem[0][0], mem[0][1]])
				p2 = np.array([x/len(targets), y/len(targets)])

				dist = np.linalg.norm(p1 - p2)

				if dist > score and dist > 0:

					cert = np.exp(-0.5*(dist/1.5))

			Gc = [[(x/AtoT, y/AtoT), cert, 10]]
			self.attract_center[1] = 0
		else:

			Gc = [[self.pos, 1, 10]]

		# Decide Side Center
		dist_to_cluster = np.array([100.0000, 100.0000, 100.0000], dtype = float)

		for (mem, i) in zip(Cluster, range(len(Cluster))):

			p1 = np.array([self.pos[0], self.pos[1]])
			p2 = np.array([mem[0][0], mem[0][1]])

			dist_to_cluster[i] = np.linalg.norm(p1 - p2)

		if (len(Cluster) == AtoT):

			Sc_index = np.argmin(dist_to_cluster)
			
			for neighbor in self.neighbors:

				if neighbor.attract_center[3] == Sc_index:

					dist_to_cluster[Sc_index] = 100
					Sc_index = np.argmin(dist_to_cluster)

			self.attract_center[3] = Sc_index
			t_index = Cluster_pair[Sc_index]

		# elif (len(Cluster) == AtoT-1):

		# 	registration_form = np.ones(len(Cluster))

		# 	Sc_index = np.argmin(dist_to_cluster)
			
		# 	for neighbor in self.neighbors:

		# 		if neighbor.attract_center[3] == Sc_index:

		# 			registration_form[neighbor.attract_center[3]] = 0

		# 	if (registration_form == 0).all():

		# 		Sc_index = np.argmin(dist_to_cluster)
		# 	else:

		# 		untracked_index = np.nonzero(registration_form)[0][0]
		# 		Sc_index = int(untracked_index)

		# 	self.attract_center[3] = Sc_index
		# 	t_index = Cluster_pair[Sc_index]

		# elif (len(Cluster) == AtoT-2):

		# 	Sc_index = 0
		# 	self.attract_center[3] = Sc_index

		# Decide One Target
		pt = [self.pos, self.ltop, self.top, self.rtop, self.pos]
		polygon = Polygon(pt)
		
		dist_to_targets = np.array([100.00, 100.00, 100.00], dtype = float)

		for (mem, i) in zip(targets, range(len(targets))):

			gemos = Point(mem[0])
			# if polygon.is_valid and polygon.contains(gemos):
			if polygon.is_valid:

				p1 = np.array([self.pos[0], self.pos[1]])
				p2 = np.array([mem[0][0], mem[0][1]])

				dist_to_targets[i] = np.linalg.norm(p1 - p2)

		No_index = np.argmin(dist_to_targets)
		
		for neighbor in self.neighbors:

			if neighbor.attract_center[5] == No_index:

				dist_to_targets[No_index] == 100
				No_index = np.argmin(dist_to_targets)

		self.attract_center[5] = No_index

		# Configuration of calculation cost function
		Avg_distance = 0.0
		k1, k2 = self.HW_IT, self.HW_BT
		sweet_spot = self.pos + self.R*np.cos(self.alpha)*self.perspective

		# t_index = [0,1,2]
		# t_index = np.delete(t_index, np.argmax(self.dist_to_targets))
		p1 = np.array([targets[t_index[0]][0][0], targets[t_index[0]][0][1]])
		p2 = np.array([targets[t_index[1]][0][0], targets[t_index[1]][0][1]])
		mid_point = np.array([(p1[0]+p2[0])/2, (p1[1]+p2[1])/2])
		decision_line = (mid_point - self.pos)/np.linalg.norm(mid_point - self.pos)
		dl_1 = (p1 - self.pos)/np.linalg.norm(p1 - self.pos)
		dl_2 = (p2 - self.pos)/np.linalg.norm(p2 - self.pos)

		if np.cross(dl_1, decision_line) < 0:

			p_l = np.array([targets[t_index[0]][0][0], targets[t_index[0]][0][1]])
			p_r = np.array([targets[t_index[1]][0][0], targets[t_index[1]][0][1]])
		else:

			p_r = np.array([targets[t_index[0]][0][0], targets[t_index[0]][0][1]])
			p_l = np.array([targets[t_index[1]][0][0], targets[t_index[1]][0][1]])

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

		coords = [(targets[0][0][0], targets[0][0][1]),\
					(targets[1][0][0], targets[1][0][1]),\
					(targets[2][0][0], targets[2][0][1])]

		polygon = Polygon(coords)
		area = polygon.area
		height = (2*area)/base_length
		height_n = abs(self.R_max - self.R*cos(self.alpha))

		l_line_v = p_l - pos; l_line_v = l_line_v/np.linalg.norm(l_line_v)
		r_line_v = p_r - pos; r_line_v = r_line_v/np.linalg.norm(r_line_v)
		theta = np.arccos(np.dot(l_line_v, r_line_v))

		Avg_distance = base_length*(height/height_n)*\
						np.exp(1*(theta - 0.5*self.alpha)/0.5*self.alpha)
		Avg_Sense = np.sum(self.HW_Sensing)/len(self.HW_Sensing)

		# p1 = np.array([self.pos[0], self.pos[1]])
		# Gc = np.array([self.target[0][0][0], self.target[0][0][1]])
		d_3 = np.linalg.norm(self.virtual_target - Gc[0][0])

		Coe_3 = np.exp( -( (height/height_n)*(1/(2*0.5**2)) ) )*\
				np.exp( -( ((theta - 0.0*self.alpha)/(1*self.alpha))*(1/(2*0.5**2)) ) )
		Cost_3 = Coe_3*d_3

		# Cost Function 1-2
		l = np.linalg.norm(p_l - p_r)
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

		# p1 = np.array([self.pos[0], self.pos[1]])
		Sc = mid_point
		d_2 = np.linalg.norm(self.virtual_target - Sc)

		Coe_2 = np.exp( -( (abs(height-1.5*height_n)/height_n)*(1/(2*0.5**2)) ) )*\
				np.exp( -( (abs(theta-0.6*self.alpha)/(self.alpha))*(1/(2*0.5**2)) ) )*\
				np.exp( -( ((l-Avg_distance)/l)*(1/(2*0.5**2)) ) )
		Cost_2 = Coe_2*d_2

		# Cost Function 1-1
		# p1 = np.array([self.pos[0], self.pos[1]])
		# switch_index = np.argmin(self.dist_to_targets)
		Ot = np.array([targets[self.attract_center[5]][0]])
		d_1 = np.linalg.norm(self.virtual_target - Ot)

		Coe_1 = np.exp( -( ((1.2*height_n)/height)*(1/(2*0.5**2)) ) )*\
				np.exp( -( (0.6*self.alpha/theta)*(1/(2*0.5**2)) ) )*\
				np.exp( -( ((Avg_distance-l)/l)*(1/(2*0.5**2)) ) )
		Cost_1 = Coe_1*d_1

		T_Cost = Cost_1 + Cost_2 + Cost_3
		T_Cost = [T_Cost, time_]

		# Gradient Desent
		dx_3 = -Coe_3*(-2)*np.array([(Gc[0][0][0]-self.virtual_target[0]), (Gc[0][0][1]-self.virtual_target[1])])
		dx_2 = -Coe_2*(-2)*np.array([(Sc[0]-self.virtual_target[0]), (Sc[1]-self.virtual_target[1])])
		dx_1 = -Coe_1*(-2)*np.array([(Ot[0][0]-self.virtual_target[0]), (Ot[0][1]-self.virtual_target[1])])

		dx = 1*dx_1 + 1*dx_2 + 1*dx_3

		self.virtual_target += 1*dx
		self.target = [[self.virtual_target, 1, 10]]

		# # filename = "D:/上課資料/IME/實驗室研究/Paper/Coverage Control/Quality based switch mode/Data/"
		# filename = "/home/leo/mts/src/QBSM/Data/"
		# filename += "Cost_" + str(self.id) + ".csv"
		# with open(filename, "a", encoding='UTF8', newline='') as f:

		# 	row = T_Cost
		# 	writer = csv.writer(f)
		# 	writer.writerow(row)

		# print("Cost 1: " + str(Cost_1), "Cost 2: " + str(Cost_2), "Cost 3: " + str(Cost_3))
		# print("Total Cost: " + str(T_Cost))
		# print("Virtual Target: ", end="")
		# print(self.virtual_target)
		# print(Sc)
		
		# print("id: " + str(self.id), "\n")

	def StageAssignment(self):

		range_max = 0.85*(self.lamb + 1)/(self.lamb)*self.R*cos(self.alpha)
		# range_max = self.R*cos(self.alpha)

		if self.centroid is not None:

			range_local_best = (np.linalg.norm(np.asarray(self.centroid) - self.pos))
			r = range_max*range_local_best/(range_max+range_local_best)\
				+ range_local_best*range_max/(range_max+range_local_best)

			if self.stage == 1:

				r = max(r, range_max - sqrt(1/(2*self.target[0][1])))
			else:

				r = self.R*cos(self.alpha)

			tmp = 0
			for i in range(len(self.target)):

				dist = np.linalg.norm(self.pos - np.asarray(self.target[i][0]))
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

		for neighbor in self.neighbors:

			FoV = neighbor.FoV
			quality_map = ne.evaluate("where((quality_map >= FoV), quality_map, 0)")

		# self.voronoi = np.array(np.where((quality_map != 0) & (self.FoV != 0)))
		self.voronoi = np.array(np.where((self.FoV > 0)))
		self.map_plt = np.array(ne.evaluate("where(quality_map != 0, id_ + 1, 0)"))

		return

	def ComputeCentroidal(self, time_):

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

			if time_ >= 30.00:

				self.Kv = 10

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

		for neighbor in self.neighbors:
			neighbor_force += (self.pos - neighbor.pos)/(np.linalg.norm(self.pos - neighbor.pos))

		neighbor_norm = np.linalg.norm(neighbor_force)

		if self.stage == 2:

			target_force = (np.asarray(self.target[0][0]) - self.pos)\
				/(np.linalg.norm(np.asarray(self.target[0][0])- self.pos))

			target_norm = np.linalg.norm(target_force)

			center_force = (np.asarray(self.target[0][0]) - self.pos)\
				/(np.linalg.norm(np.asarray(self.target[0][0]) - self.pos))

			formation_force = (target_force*(neighbor_norm/(target_norm+neighbor_norm))\
							+ neighbor_force*(target_norm/(target_norm+neighbor_norm)))

			formation_force -= (center_force/np.linalg.norm(center_force))*(self.r - self.norm\
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

	def EscapeDensity(self, targets, time_):

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
		F1 = multivariate_normal([targets[0][0][0], targets[0][0][1]],\
								[[targets[0][1], 0.0], [0.0, targets[0][1]]])
		F2 = multivariate_normal([targets[1][0][0], targets[1][0][1]],\
								[[targets[1][1], 0.0], [0.0, targets[1][1]]])
		F3 = multivariate_normal([targets[2][0][0], targets[2][0][1]],\
								[[targets[2][1], 0.0], [0.0, targets[2][1]]])

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

		# if self.id == 0:

		# 	print(self.HW_Interior)
		# 	print(self.HW_Boundary, "\n")

	def UpdateOrientation(self):

		self.perspective += self.perspective_force*self.step
		self.perspective /= np.linalg.norm(self.perspective)

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