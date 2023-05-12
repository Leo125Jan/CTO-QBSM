import warnings
warnings.simplefilter(action = 'ignore', category = RuntimeWarning)

import sys
import csv
import random
import numpy as np
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
					Kv = 40, Ka = 3, Kp = 3, step = 0.1):

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
		self.centroid = np.array([0, 0])
		self.cluster_count = 0
		self.dist_to_cluster = 0
		self.dist_to_targets = 0
		self.Clsuter_Checklist = None
		self.Cluster_Teammate = np.array([None, None])
		self.dispatch_occpied = False
		self.state_machine = {"self": None, "mode": None, "target": None}

	def UpdateState(self, targets, neighbors, time_):

		self.neighbors = neighbors
		self.time = time_

		self.UpdateFoV()
		self.polygon_FOV()
		self.EscapeDensity(targets, time_)
		self.UpdateLocalVoronoi()

		self.Cluster_Formation(targets)
		self.Cluster_Assignment(targets, time_)

		event = np.zeros((self.size[0], self.size[1]))
		self.event = self.event_density(event, self.target, self.grid_size)
		
		self.ComputeCentroidal()
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

	def Cluster_Formation(self, targets):

		checklist = np.zeros((len(targets), len(targets)))
		threshold = 2.3
		self.cluster_count = 0

		for i in range(len(targets)):

			for j in range(len(targets)):

				if j != i:

					p1 = np.array([targets[i][0][0], targets[i][0][1]])
					p2 = np.array([targets[j][0][0], targets[j][0][1]])

					dist = self.norm(p1 - p2)

					if dist <= threshold:

						checklist[i][j] = 1
						self.cluster_count += 1
					else:

						checklist[i][j] = 0

		self.Clsuter_Checklist = checklist

		return
	
	def Cluster_Assignment(self, targets, time_):

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

		# Calculate dist between each cluster for Hungarian Algorithm
		dist_to_cluster = np.array([100.0000, 100.0000, 100.0000], dtype = float)

		for (mem, i) in zip(Cluster, range(len(Cluster))):

			p1 = np.array([self.pos[0], self.pos[1]])
			p2 = np.array([mem[0][0], mem[0][1]])

			dist_to_cluster[i] = self.norm(p1 - p2)

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

				dist = self.norm(p1 - p2)

				if dist > score and dist > 0:

					cert = np.exp(-0.5*(dist/1.5))

			self.target = [[(x/AtoT, y/AtoT), cert, 10]]

		# Mode Switch Control
		pt = [self.pos, self.ltop, self.top, self.rtop, self.pos]
		polygon = Polygon(pt)
		
		dist_to_targets = np.array([100.00, 100.00, 100.00], dtype = float)

		for (mem, i) in zip(targets, range(len(targets))):

			gemos = Point(mem[0])
			# if polygon.is_valid and polygon.contains(gemos):
			if polygon.is_valid:

				p1 = np.array([self.pos[0], self.pos[1]])
				p2 = np.array([mem[0][0], mem[0][1]])

				dist_to_targets[i] = self.norm(p1 - p2)

		self.dist_to_targets = dist_to_targets

		# Cost function 1-3
		Avg_dist = []
		k1, k2 = self.HW_IT, self.HW_BT
		sweet_spot = self.pos + self.R*np.cos(self.alpha)*self.perspective

		# for i in range(len(targets)):

		# 	if i == 2:

		# 		j = 0
		# 	else:

		# 		j = i + 1

		# 	p1 = np.array([targets[i][0][0], targets[i][0][1]])
		# 	p2 = np.array([targets[j][0][0], targets[j][0][1]])

		# 	dist = self.norm(p1 - p2)
		# 	Avg_dist.append(dist)

		# Avg_dist = np.sum(Avg_dist)/len(targets)
		# Avg_Sense = np.sum(self.HW_Sensing)/len(self.HW_Sensing)
		# C_3 = (1/k1)*(1/Avg_Sense) + k2*Avg_dist

		t_index = [0,1,2]
		t_index = np.delete(t_index, np.argmax(self.dist_to_targets))
		p1 = np.array([targets[t_index[0]][0][0], targets[t_index[0]][0][1]])
		p2 = np.array([targets[t_index[1]][0][0], targets[t_index[1]][0][1]])
		mid_point = np.array([(p1[0]+p2[0])/2, (p1[1]+p2[1])/2])
		decision_line = (mid_point - self.pos)/np.linalg.norm(mid_point - self.pos)
		dl_1 = (p1 - self.pos)/np.linalg.norm(p1 - self.pos)
		dl_2 = (p2 - self.pos)/np.linalg.norm(p2 - self.pos)

		if np.cross(dl_1, decision_line) < 0:

			p1 = np.array([targets[t_index[0]][0][0], targets[t_index[0]][0][1]])
			p2 = np.array([targets[t_index[1]][0][0], targets[t_index[1]][0][1]])
		else:

			p2 = np.array([targets[t_index[0]][0][0], targets[t_index[0]][0][1]])
			p1 = np.array([targets[t_index[1]][0][0], targets[t_index[1]][0][1]])

		base = self.norm(p1 - p2)

		coords = [(targets[0][0][0], targets[0][0][1]),\
					(targets[1][0][0], targets[1][0][1]),\
					(targets[2][0][0], targets[2][0][1])]

		polygon = Polygon(coords)
		area = polygon.area
		height = (2*area)/base

		line_1 = p1 - self.pos; line_1 = line_1/self.norm(line_1)
		line_2 = p2 - self.pos; line_2 = line_2/self.norm(line_2)
		theta = np.arccos(np.dot(line_1, line_2))

		dist = height*np.exp(3*(theta - 0.5*self.alpha)/0.5*self.alpha)
		Avg_Sense = np.sum(self.HW_Sensing)/len(self.HW_Sensing)
		C_3 = (1/k1)*(1/Avg_Sense) + k2*dist

		print("dist to target: ", end = "")
		print(self.dist_to_targets)

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
		Avg_dist = self.norm(p1 - p2)
		print("L: " + str(Avg_dist))
		# Avg_Sense = (np.sum(self.HW_Sensing[t_index[0]] +\
		# 					self.HW_Sensing[t_index[1]])/2)
		C_2 = (1/k1)*(1/Avg_Sense) + k2*Avg_dist

		# Cost Function 1-1
		# switch_index = np.argmin(self.dist_to_targets)
		# Avg_dist = np.sum(np.where(self.dist_to_targets < 100))/len(np.where(self.dist_to_targets < 100))
		# # dist = self.dist_to_targets[switch_index]
		# Sense = self.HW_Sensing[switch_index]
		# C_1 = (1/k1)*(1/Sense) + k2*Avg_dist

		switch_index = np.argmin(self.dist_to_targets)
		t_index = [0,1,2]
		t_index = np.delete(t_index, np.argmax(self.dist_to_targets))
		# mid_point = np.array([(p1[0]+p2[0])/2, (p1[1]+p2[1])/2])

		line_1 = p1 - self.pos; line_1 = line_1/self.norm(line_1)
		line_2 = p2 - self.pos; line_2 = line_2/self.norm(line_2)
		L_1 = self.norm(sweet_spot - p1)
		L_2 = self.norm(sweet_spot - p2)
		theta_1 = np.arccos(np.dot(line_1, self.perspective))
		theta_2 = np.arccos(np.dot(line_2, self.perspective))

		lenth_1 = sweet_spot - p1; lenth_1 = lenth_1/self.norm(lenth_1)
		lenth_2 = sweet_spot - p2; lenth_2 = lenth_2/self.norm(lenth_2)
		base_line = (p1 - p2)/base
		delta_1 = np.dot(lenth_1, -base_line)
		delta_2 = np.dot(lenth_2, +base_line)
		L_1 *= delta_1
		L_2 *= delta_2

		print("L1: " + str(L_1), "L2: " + str(L_2), "theta 1: " + str(theta_1), "theta 2: " + str(theta_2))
		print("alpha: " + str(self.alpha))

		Avg_dist = L_1*np.exp(0.6*self.alpha - theta_1) + L_2*np.exp(0.6*self.alpha - theta_2)
		# Avg_dist = (L_1 + L_2)*(np.exp(theta_1 + theta_2 - self.alpha))
		print("l1 + l2: " + str(Avg_dist))
		# Avg_Sense = (np.sum(self.HW_Sensing[t_index[0]] + self.HW_Sensing[t_index[1]])/2)
		# Sense = self.HW_Sensing[switch_index]

		C_1 = (1/k1)*(1/Avg_Sense) + k2*Avg_dist

		C_total = [C_1, C_2, C_3]
		min_C = np.argmin(C_total)+1

		print("C1: " + str(C_1))
		print("C2: " + str(C_2))
		print("C3: " + str(C_3))

		C_total.append(time_)
		# filename = "D:/上課資料/IME/實驗室研究/Paper/Coverage Control/Quality based switch mode/Data/"
		# # filename = "D:/Leo/IME/Paper Study/Coverage Control/Quality based switch mode/Data/"
		# filename += "Data_" + str(self.id) + ".csv"
		# with open(filename, "a", encoding='UTF8', newline='') as f:

		# 	row = C_total
		# 	writer = csv.writer(f)
		# 	writer.writerow(row)

		# if time_ <= 40:

		# 	x, y = 0, 0

		# 	for target in targets:

		# 		x += target[0][0]
		# 		y += target[0][1]

		# 	self.target = [[(x/3, y/3), 1, 10]]

		
		if (len(Cluster) == AtoT):

			if min_C == 3:

				x, y = 0, 0

				for target in targets:

					x += target[0][0]
					y += target[0][1]

				self.target = [[(x/AtoT, y/AtoT), 1, 10]]
				self.state_machine["self"] = self.id
				self.state_machine["mode"] = min_C
				self.state_machine["target"] = 0
			elif min_C == 2:

				switch_index = np.argmin(self.dist_to_cluster)

				self.state_machine["self"] = self.id
				self.state_machine["mode"] = min_C
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

						self.dist_to_targets[self.state_machine["target"]] = 100
						self.state_machine["target"] = int(np.argmin(self.dist_to_targets))

				self.target = [targets[self.state_machine["target"]]]

		# if (self.HW_Interior <= self.HW_Boundary):
		if (len(Cluster) < AtoT):

			# if (len(Cluster) == AtoT):
			if min_C == 3:

				# for neighbor in self.neighbors:

				# 	cost_matrix = np.concatenate((cost_matrix, [neighbor.dist_to_cluster]),\
				# 									axis = 0)

				# row_ind, col_ind = linear_sum_assignment(cost_matrix)
				# self.target = [Cluster[col_ind[0]]]
				# switch_index = np.argmin(self.dist_to_cluster)
				# self.target = [Cluster[switch_index]]

			# elif min_C == 2:

			# 	if (len(Cluster) == AtoT - 1):

			# 		switch_index = np.argmin(self.dist_to_cluster)
			# 		self.target = [Cluster[switch_index]]
			# 		self.Cluster_Teammate = np.array([2, switch_index])

			# 		cost_matrix = [self.dist_to_cluster]
			# 		Teammate_matrix = [self.Cluster_Teammate]
			# 		count = np.ones(len(Cluster))
			# 		count[self.Cluster_Teammate[1]] = 0
			# 		len_ = len(count)

			# 		for neighbor in self.neighbors:

			# 			temp1 = neighbor.dist_to_cluster
			# 			cost_matrix = np.concatenate((cost_matrix, [temp1]), axis = 0)

			# 			if int(len_) <= 1:

			# 				count = np.array([0])
			# 			else:
			# 				if ((neighbor.Cluster_Teammate != None).all()) and\
			# 					(neighbor.Cluster_Teammate[0] == 2):

			# 					temp2 = neighbor.Cluster_Teammate
			# 					Teammate_matrix = np.concatenate((Teammate_matrix, [temp2]), axis = 0)
			# 					count[neighbor.Cluster_Teammate[1]] = 0
			# 				else:
			# 					temp2 = np.array([None, None])
			# 					Teammate_matrix = np.concatenate((Teammate_matrix, [temp2]), axis = 0)

			# 		if (not (count == 0).all() and (Teammate_matrix != None).all()):

			# 			dist_untracked = cost_matrix[:,np.nonzero(count)[0]]
			# 			dispatch_index = np.argmin(dist_untracked)

			# 			if dispatch_index == 0:

			# 				self.target = [Cluster[np.nonzero(count)[0][0]]]
			# 				self.Cluster_Teammate = np.array([2, np.nonzero(count)[0][0]])

			# 		self.last_Cluster_pair = Cluster_pair
			# 	elif (len(Cluster) == AtoT - 2):

			# 		# index = np.delete(self.last_Cluster_pair, Cluster_pair)

			# 		# p1 = np.array([targets[index[0]][0][0], targets[index[0]][0][1]])
			# 		# p2 = np.array([targets[index[1]][0][0], targets[index[1]][0][1]])
			# 		# dist = self.norm(p1 - p2)

			# 		# cost_matrix = [self.dist_to_cluster[0], dist]
			# 		# switch_index = np.argmin(cost_matrix)

			# 		# if switch_index == 0:

			# 		# 	switch_index = np.argmin(self.dist_to_cluster)
			# 		# 	self.target = [Cluster[switch_index]]
			# 		# elif switch_index == 1:

			# 		# 	x = (targets[index[0]][0][0] + targets[index[1]][0][0])/2
			# 		# 	y = (targets[index[0]][0][1] + targets[index[1]][0][1])/2

			# 		# 	self.target = [[(x, y), 1, 10]]

			# 		index = [0,1,2]
			# 		index = np.delete(index, np.argmax(self.dist_to_targets))
			# 		x = (targets[index[0]][0][0] + targets[index[1]][0][0])/2
			# 		y = (targets[index[0]][0][1] + targets[index[1]][0][1])/2

			# 		self.target = [[(x, y), 1, 10]]
			elif min_C == 2:

				if (len(Cluster) == AtoT - 1):

					switch_index = np.argmin(self.dist_to_cluster)

					self.state_machine["self"] = self.id
					self.state_machine["mode"] = min_C
					self.state_machine["target"] = int(switch_index)

					registration_form = np.ones(len(Cluster))
					sign_form = []

					for neighbor in self.neighbors:

						sign_form.append(neighbor.state_machine["target"])

					sign_form = np.array(sign_form)

					if (sign_form == [1,2]).all() or (sign_form == [2,1]).all():

						registration_form[0], registration_form[1] = 0, 0
					if (sign_form == [0,2]).all() or (sign_form == [2,0]).all():

						registration_form[0], registration_form[1] = 0, 0
					if (sign_form == [0,1]).all() or (sign_form == [1,0]).all():

						registration_form[0], registration_form[1] = 0, 0

					if (registration_form == 0).all():

						self.target = [Cluster[self.state_machine["target"]]]
					else:

						untracked_index = np.nonzero(registration_form)[0][0]
						self.state_machine["target"] = int(untracked_index)
						self.target = [Cluster[self.state_machine["target"]]]

					# self.last_Cluster_pair = Cluster_pair
				elif (len(Cluster) == AtoT - 2):

					index = [0,1,2]
					index = np.delete(index, np.argmax(self.dist_to_targets))
					x = (targets[index[0]][0][0] + targets[index[1]][0][0])/2
					y = (targets[index[0]][0][1] + targets[index[1]][0][1])/2

					self.target = [[(x, y), 1, 10]]

			# elif (len(Cluster) < AtoT - 1):
			elif min_C == 1:

				print(self.dist_to_targets)

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

				# self.target = [targets[switch_index]]
				# self.Cluster_Teammate = np.array([0, switch_index])

				# cost_matrix = [self.dist_to_targets]
				# Teammate_matrix = [self.Cluster_Teammate]
				# count = np.ones(len(targets))
				# count[self.Cluster_Teammate[1]] = 0

				# for neighbor in self.neighbors:

				# 	if (neighbor.Cluster_Teammate[0] == 0) and\
				# 		(neighbor.Cluster_Teammate[1] == self.Cluster_Teammate[1]):

				# 		self.dist_to_targets[self.Cluster_Teammate[1]] = 100
				# 		switch_index = np.argmin(self.dist_to_targets)
				# 		self.target = [targets[switch_index]]
				# 		self.Cluster_Teammate = np.array([0, switch_index])

				# 		# temp1 = neighbor.dist_to_targets
				# 		# cost_matrix = np.concatenate((cost_matrix, [temp1]), axis = 0)

				# 		# over_tracked = cost_matrix[:,self.Cluster_Teammate[1]]
				# 		# dispatch_index = np.argmin(over_tracked)

				# 		# if dispatch_index != 0:

				# 		# 	self.dist_to_targets[self.Cluster_Teammate[1]] = 100
				# 		# 	print(self.dist_to_targets)
				# 		# 	switch_index = np.argmin(self.dist_to_targets)
				# 		# 	print(switch_index)
				# 		# 	self.target = [targets[switch_index]]
				# 		# 	self.Cluster_Teammate = np.array([0, switch_index])

				# self.dispatch_occpied == False
		
		print("id: " + str(self.id), "\n")
	
	'''
	def Cluster_Assignment(self, targets):

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

			dist_to_cluster[i] = self.norm(p1 - p2)

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

				dist = self.norm(p1 - p2)

				if dist > score and dist > 0:

					cert = np.exp(-0.5*(dist/1.5))

			self.target = [[(x/AtoT, y/AtoT), cert, 10]]

		# Mode Switch Control

		if self.HW_Sensing >= 0.1:

			pt = [self.pos, self.ltop, self.top, self.rtop, self.pos]
			polygon = Polygon(pt)
			
			dist_to_targets = np.array([100.00, 100.00, 100.00], dtype = float)

			for (mem, i) in zip(targets, range(len(targets))):

				gemos = Point(mem[0])
				if polygon.is_valid and polygon.contains(gemos):

					p1 = np.array([self.pos[0], self.pos[1]])
					p2 = np.array([mem[0][0], mem[0][1]])

					dist_to_targets[i] = self.norm(p1 - p2)

			self.dist_to_targets = dist_to_targets

		if (len(Cluster) == AtoT):

			x, y = 0, 0

			for target in targets:

				x += target[0][0]
				y += target[0][1]

			self.target = [[(x/AtoT, y/AtoT), 1, 10]]
		# elif (cluster_quality < 2.35*Best_quality_ref) and \
		# 		(cluster_quality >= 1.6*Best_quality_ref):
		if (self.HW_Interior <= self.HW_Boundary):

			if (len(Cluster) == AtoT):

				cost_matrix = [self.dist_to_cluster]
				for neighbor in self.neighbors:

					cost_matrix = np.concatenate((cost_matrix, [neighbor.dist_to_cluster]),\
													axis = 0)

				row_ind, col_ind = linear_sum_assignment(cost_matrix)
				self.target = [Cluster[col_ind[0]]]

			elif (len(Cluster) == AtoT - 1):

				switch_index = np.argmin(self.dist_to_cluster)
				self.target = [Cluster[switch_index]]
				self.Cluster_Teammate = np.array([2, switch_index])

				cost_matrix = [self.dist_to_cluster]
				Teammate_matrix = [self.Cluster_Teammate]
				count = np.ones(len(Cluster))
				count[self.Cluster_Teammate[1]] = 0

				for neighbor in self.neighbors:

					temp1 = neighbor.dist_to_cluster
					cost_matrix = np.concatenate((cost_matrix, [temp1]), axis = 0)

					if ((neighbor.Cluster_Teammate != None).all()) and\
						(neighbor.Cluster_Teammate[0] == 2):

						temp2 = neighbor.Cluster_Teammate
						Teammate_matrix = np.concatenate((Teammate_matrix, [temp2]), axis = 0)
						count[neighbor.Cluster_Teammate[1]] = 0
					else:
						temp2 = np.array([None, None])
						Teammate_matrix = np.concatenate((Teammate_matrix, [temp2]), axis = 0)

				if not (count == 0).all() and (Teammate_matrix != None).all():

					dist_untracked = cost_matrix[:,np.nonzero(count)[0]]
					dispatch_index = np.argmin(dist_untracked)

					if dispatch_index == 0:

						self.target = [Cluster[np.nonzero(count)[0][0]]]
						self.Cluster_Teammate = np.array([2, np.nonzero(count)[0][0]])

		# elif (cluster_quality < 1.6*Best_quality_ref) and\
		# 		(self.cluster_count != cluster_count_ref):

			elif (len(Cluster) < AtoT - 1):

				print(self.dist_to_targets)

				switch_index = np.argmin(self.dist_to_targets)
				self.target = [targets[switch_index]]
				self.Cluster_Teammate = np.array([0, switch_index])

				cost_matrix = [self.dist_to_targets]
				Teammate_matrix = [self.Cluster_Teammate]
				count = np.ones(len(targets))
				count[self.Cluster_Teammate[1]] = 0

				for neighbor in self.neighbors:

					if (neighbor.Cluster_Teammate[0] == 0) and\
						(neighbor.Cluster_Teammate[1] == self.Cluster_Teammate[1]):

						temp1 = neighbor.dist_to_targets
						cost_matrix = np.concatenate((cost_matrix, [temp1]), axis = 0)

						over_tracked = cost_matrix[:,self.Cluster_Teammate[1]]
						dispatch_index = np.argmin(over_tracked)

						if dispatch_index != 0:

							self.dist_to_targets[self.Cluster_Teammate[1]] = 100
							switch_index = np.argmin(self.dist_to_targets)
							self.target = [targets[switch_index]]
							self.Cluster_Teammate = np.array([0, switch_index])

				self.dispatch_occpied == False
	
	def Cluster_Assignment(self, targets):

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

			dist_to_cluster[i] = self.norm(p1 - p2)

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

				dist = self.norm(p1 - p2)

				if dist > score and dist > 0:

					cert = np.exp(-0.5*(dist/1.5))

			self.target = [[(x/AtoT, y/AtoT), cert, 10]]

		# Mode Switch Control
		cluster_quality = 0
		Best_quality_ref = 2200

		for (target, i) in zip(targets, range(0,len(targets))):

			event = np.zeros((self.size[0], self.size[1]))
			event1 = self.event_density(event, [target], self.grid_size)
			event1 = np.transpose(event1)
			cluster_quality += np.sum(self.FoV*event1)

		# print(cluster_quality)
		# Calculate dist between each target for Hungarian Algorithm
		if cluster_quality >= 10:

			pt = [self.pos, self.ltop, self.top, self.rtop, self.pos]
			polygon = Polygon(pt)
			
			dist_to_targets = np.array([100.00, 100.00, 100.00], dtype = float)

			for (mem, i) in zip(targets, range(len(targets))):

				gemos = Point(mem[0])
				if polygon.is_valid and polygon.contains(gemos):

					p1 = np.array([self.pos[0], self.pos[1]])
					p2 = np.array([mem[0][0], mem[0][1]])

					dist_to_targets[i] = self.norm(p1 - p2)

			self.dist_to_targets = dist_to_targets

		if (cluster_quality >= 2.35*Best_quality_ref):

			x, y = 0, 0

			for target in targets:

				x += target[0][0]
				y += target[0][1]

			self.target = [[(x/AtoT, y/AtoT), 1, 10]]
		elif (cluster_quality < 2.35*Best_quality_ref) and \
				(cluster_quality >= 1.6*Best_quality_ref):

			if (len(Cluster) == AtoT):

				cost_matrix = [self.dist_to_cluster]
				for neighbor in self.neighbors:

					cost_matrix = np.concatenate((cost_matrix, [neighbor.dist_to_cluster]),\
													axis = 0)

				row_ind, col_ind = linear_sum_assignment(cost_matrix)
				self.target = [Cluster[col_ind[0]]]

			if (len(Cluster) == AtoT - 1):

				switch_index = np.argmin(self.dist_to_cluster)
				self.target = [Cluster[switch_index]]
				self.Cluster_Teammate = np.array([2, switch_index])

				cost_matrix = [self.dist_to_cluster]
				Teammate_matrix = [self.Cluster_Teammate]
				count = np.ones(len(Cluster))
				count[self.Cluster_Teammate[1]] = 0

				for neighbor in self.neighbors:

					temp1 = neighbor.dist_to_cluster
					cost_matrix = np.concatenate((cost_matrix, [temp1]), axis = 0)

					if ((neighbor.Cluster_Teammate != None).all()) and\
						(neighbor.Cluster_Teammate[0] == 2):

						temp2 = neighbor.Cluster_Teammate
						Teammate_matrix = np.concatenate((Teammate_matrix, [temp2]), axis = 0)
						count[neighbor.Cluster_Teammate[1]] = 0
					else:
						temp2 = np.array([None, None])
						Teammate_matrix = np.concatenate((Teammate_matrix, [temp2]), axis = 0)

				if not (count == 0).all() and (Teammate_matrix != None).all():

					dist_untracked = cost_matrix[:,np.nonzero(count)[0]]
					dispatch_index = np.argmin(dist_untracked)

					if dispatch_index == 0:

						self.target = [Cluster[np.nonzero(count)[0][0]]]
						self.Cluster_Teammate = np.array([2, np.nonzero(count)[0][0]])

			# if (len(Cluster) <= AtoT - 2 and len(Cluster) > 0):

			# 	for i in range(np.shape(self.Clsuter_Checklist)[0]):

			# 		temp = self.Clsuter_Checklist[i,:]

			# 		if (temp == 0).all():

			# 			p1 = np.array([self.pos[0], self.pos[1]])
			# 			p2 = np.array([targets[i][0][0], targets[i][0][1]])
			# 			dist = self.norm(p1 - p2)
			# 			escape_index = i
			# 			Cluster.append(targets[i])

			# 	self.dist_to_cluster[1] = dist
			# 	switch_index = np.argmin(self.dist_to_cluster)
			# 	self.target = [Cluster[switch_index]]
			# 	self.Cluster_Teammate = np.array([1, switch_index])
			# 	cost_matrix = [self.dist_to_cluster]
			# 	print(self.Cluster_Teammate)

			# 	if self.dispatch_occpied:

			# 		self.target = [Cluster[0]]
			# 		self.Cluster_Teammate = np.array([1, 0])
			# 	elif (self.Cluster_Teammate[1] == 1) and not self.dispatch_occpied:

			# 		for neighbor in self.neighbors:

			# 			if (neighbor.Cluster_Teammate[1] == 1):

			# 				temp1 = neighbor.dist_to_cluster
			# 				cost_matrix = np.concatenate((cost_matrix, [temp1]), axis = 0)

			# 				over_tracked = cost_matrix[:,0]
			# 				dispatch_index = np.argmin(over_tracked)

			# 				if dispatch_index == 0:

			# 					self.target = [Cluster[0]]
			# 					self.Cluster_Teammate = np.array([1, 0])
			# 					self.dispatch_occpied = True

		elif (cluster_quality < 1.6*Best_quality_ref) and\
				(self.cluster_count != cluster_count_ref):

			switch_index = np.argmin(self.dist_to_targets)
			self.target = [targets[switch_index]]
			self.Cluster_Teammate = np.array([0, switch_index])

			cost_matrix = [self.dist_to_targets]
			Teammate_matrix = [self.Cluster_Teammate]
			count = np.ones(len(targets))
			count[self.Cluster_Teammate[1]] = 0

			for neighbor in self.neighbors:

				if (neighbor.Cluster_Teammate[0] == 0) and\
					(neighbor.Cluster_Teammate[1] == self.Cluster_Teammate[1]):

					temp1 = neighbor.dist_to_targets
					cost_matrix = np.concatenate((cost_matrix, [temp1]), axis = 0)
					# temp2 = neighbor.Cluster_Teammate
					# Teammate_matrix = np.concatenate((Teammate_matrix, [temp2]), axis = 0)
					# count[neighbor.Cluster_Teammate[1]] = 0

					over_tracked = cost_matrix[:,self.Cluster_Teammate[1]]
					dispatch_index = np.argmin(over_tracked)

					if dispatch_index != 0:

						self.dist_to_targets[self.Cluster_Teammate[1]] = 100
						switch_index = np.argmin(self.dist_to_targets)
						self.target = [targets[switch_index]]
						self.Cluster_Teammate = np.array([0, switch_index])
			# 	else:
			# 		temp2 = np.array([None, None])
			# 		Teammate_matrix = np.concatenate((Teammate_matrix, [temp2]), axis = 0)

			# if not (count == 0).all() and (Teammate_matrix != None).all():

			# 	dist_untracked = cost_matrix[:,np.nonzero(count)[0]]
			# 	dispatch_index = np.argmin(dist_untracked)

			# 	if dispatch_index == 0:

			# 		self.target = [targets[np.nonzero(count)[0][0]]]

			self.dispatch_occpied == False
	
	def Cluster_Assignment(self, targets, time_):

		if time_ <= 1:

			x = targets[0][0][0] + targets[1][0][0] + targets[2][0][0]
			y = targets[0][0][1] + targets[1][0][1] + targets[2][0][1]

			self.target = [[(x/3, y/3), 1, 10]]

		count = 0
		Cluster = []
		Cluster_pair = []
		Avg_dist = []

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

						p1 = np.array([targets[i][0][0], targets[i][0][1]])
						p2 = np.array([targets[j][0][0], targets[j][0][1]])

						dist = self.norm(p1 - p2)
						Avg_dist.append(dist)
			elif i == 0:

				for j in nonindex:
		        
					c_x = 0.5*(targets[i][0][0] + targets[j][0][0])
					c_y = 0.5*(targets[i][0][1] + targets[j][0][1])

					Cluster.append([(c_x, c_y), 1, 10])
					Cluster_pair.append((i,j))

					p1 = np.array([targets[i][0][0], targets[i][0][1]])
					p2 = np.array([targets[j][0][0], targets[j][0][1]])

					dist = self.norm(p1 - p2)
					Avg_dist.append(dist)

		# Calculate dist between each cluster for Hungarian Algorithm
		dist_to_cluster = np.array([100.0000, 100.0000, 100.0000], dtype = float)

		for (mem, i) in zip(Cluster, range(len(Cluster))):

			p1 = np.array([self.pos[0], self.pos[1]])
			p2 = np.array([mem[0][0], mem[0][1]])

			dist_to_cluster[i] = self.norm(p1 - p2)

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

				dist = self.norm(p1 - p2)

				if dist > score and dist > 0:

					cert = np.exp(-0.5*(dist/1.5))

			self.target = [[(x/AtoT, y/AtoT), cert, 10]]

		if self.HW_Sensing[0] >= 0.1:

			pt = [self.pos, self.ltop, self.top, self.rtop, self.pos]
			polygon = Polygon(pt)
			
			dist_to_targets = np.array([100.00, 100.00, 100.00], dtype = float)

			for (mem, i) in zip(targets, range(len(targets))):

				gemos = Point(mem[0])
				if polygon.is_valid and polygon.contains(gemos):

					p1 = np.array([self.pos[0], self.pos[1]])
					p2 = np.array([mem[0][0], mem[0][1]])

					dist_to_targets[i] = self.norm(p1 - p2)

			self.dist_to_targets = dist_to_targets

			# Mode Switch Control

			# Cost function 1-3
			# p1 = np.array([self.target[0][0][0], self.target[0][0][1]])
			# p2 = np.array([self.pos[0], self.pos[1]])
			# dist = self.norm(p1 - p2)
			dist = np.sum(Avg_dist)/len(targets)
			Avg_Sense = np.sum(self.HW_Sensing)/len(self.HW_Sensing)
			k1, k2 = (1/self.HW_Interior), self.HW_Boundary
			C_3 = k1*(1/Avg_Sense) + k2*dist

			# Cost Function 1-2
			if (self.dist_to_cluster == 100.00).all():

				dist = 10
				Avg_Sense = 0.1
			else:

				switch_index = np.argmin(self.dist_to_cluster)
				p1 = np.array([targets[Cluster_pair[switch_index][0]][0][0], targets[Cluster_pair[switch_index][0]][0][1]])
				p2 = np.array([targets[Cluster_pair[switch_index][1]][0][0], targets[Cluster_pair[switch_index][1]][0][1]])
				dist = self.norm(p1 - p2)
				# dist = np.sum(np.where(self.dist_to_cluster < 100))/len(np.where(self.dist_to_cluster < 100))

				Avg_Sense = np.sum(self.HW_Sensing[Cluster_pair[switch_index][0]] +\
									self.HW_Sensing[Cluster_pair[switch_index][1]])/2
			C_2 = (1/self.HW_Interior)*(1/Avg_Sense) + self.HW_Boundary*dist

			# Cost Function 1-1
			switch_index = np.argmin(self.dist_to_targets)
			# dist = self.dist_to_targets[switch_index]
			dist = np.sum(np.where(self.dist_to_targets < 100))/len(np.where(self.dist_to_targets < 100))
			Sense = self.HW_Sensing[switch_index]
			C_1 = (1/self.HW_Interior)*(1/Sense) + self.HW_Boundary*dist

			if self.id == 0:

				print("C1: " + str(C_1))
				print("C2: " + str(C_2))
				print("C3: " + str(C_3))
	'''
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

	def EscapeDensity(self, targets, time_):

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
		F1 = multivariate_normal([targets[0][0][0], targets[0][0][1]],\
								[[targets[0][1], 0.0], [0.0, targets[0][1]]])
		F2 = multivariate_normal([targets[1][0][0], targets[1][0][1]],\
								[[targets[1][1], 0.0], [0.0, targets[1][1]]])
		F3 = multivariate_normal([targets[2][0][0], targets[2][0][1]],\
								[[targets[2][1], 0.0], [0.0, targets[2][1]]])

		# Joint Probability
		# P = lambda d, a, IoO, P0: P0*np.multiply(np.multiply(\
		# 			np.divide(np.power(np.subtract(d, range_max), 2).transpose()[0], range_max**2),\
		# 			np.divide(np.power(np.subtract(abs(a), self.alpha), 2), self.alpha**2)), IoO)

		# P = lambda d, a, IoO, P0: P0*np.multiply(np.multiply(\
		# 			np.exp(-np.divide(np.power(np.subtract(d, self.R*np.cos(self.alpha)), 2).transpose()[0], (2*0.1**2)*(self.R*np.cos(self.alpha)**2))),\
		# 			np.exp(-np.divide(np.power(np.subtract(abs(a), 0), 2), (2*0.2**2)*(self.alpha**2)))), IoO)

		# 0.35 & 0.35
		P_t = lambda d, a, IoO, P0: P0*np.multiply(np.multiply(\
					np.exp(-np.divide(np.power(np.subtract(d, self.R*np.cos(self.alpha)), 2).transpose()[0], (2*0.2**2)*(self.R*np.cos(self.alpha)**2))),\
					np.exp(-np.divide(np.power(np.subtract(abs(a), 0), 2), (2*0.2**2)*(self.alpha**2)))), IoO)

		# P_ = lambda d, a, IoO, P0: P0*np.multiply(np.multiply(\
		# 			np.exp(-np.divide(np.power(np.subtract(d, range_max), 2).transpose()[0], (2*0.125**2)*(range_max**2))),\
		# 			np.exp(-np.divide(np.power(np.subtract(abs(a), self.alpha), 2), (2*0.125**2)*(self.alpha**2)))), IoO)

		# P_ = lambda d, a, IoO, P0: P0*np.multiply(np.multiply(\
		# 			np.exp(-np.divide(np.power(np.subtract(abs(d-0.5*range_max), 0.5*range_max), 2).transpose()[0], (2*0.25**2)*(0.5*range_max**2))),\
		# 			np.exp(-np.divide(np.power(np.subtract(abs(a), self.alpha), 2), (2*0.3**2)*(self.alpha**2)))), IoO)
		
		# 0.25 & 0.25
		# P_tt = lambda d, a, IoO, P0: P0*np.multiply(np.add(\
		# 			np.exp(-np.divide(np.power(np.subtract(abs(d-0.5*range_max), 0.5*range_max), 2).transpose()[0], (2*0.25**2)*(0.5*range_max**2))),\
		# 			np.exp(-np.divide(np.power(np.subtract(abs(a), self.alpha), 2), (2*0.35**2)*(self.alpha**2)))), IoO)

		P_tt = lambda d, a, IoO, P0: P0*np.multiply(np.add(np.add(\
					np.exp(-np.divide(np.power(np.subtract(abs(d-0.5*range_max), 0.5*range_max), 2).transpose()[0], (2*0.25**2)*(0.5*range_max**2))),\
					np.exp(-np.divide(np.power(np.subtract(abs(a), self.alpha), 2), (2*0.35**2)*(self.alpha**2)))),\
					P0*np.multiply(np.multiply(\
					np.exp(-np.divide(np.power(np.subtract(d, 0.5*self.R), 2).transpose()[0], (2*0.3**2)*(0.5*self.R**2))),\
					np.exp(-np.divide(np.power(np.subtract(abs(a), 0), 2), (2*0.5**2)*(self.alpha**2)))), IoO)
					), IoO)

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
		JP_Boundary = (np.subtract(JP_Boundary, min(JP_Boundary)))/(np.subtract(max(JP_Boundary), min(JP_Boundary)))
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

		for neighbor in self.neighbors:

			quality_map = np.where((quality_map > neighbor.FoV), quality_map, 0)

		# self.voronoi = np.array(np.where((quality_map != 0) & (self.FoV != 0)))
		self.voronoi = np.array(np.where((self.FoV != 0)))
		self.voronoi_map = np.where(((quality_map > 0.) & (self.FoV > 0.)), quality_map, 0)
		self.map_plt = np.array(np.where(quality_map != 0, self.id + 1, 0))

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

		for neighbor in self.neighbors:
			neighbor_force += (self.pos - neighbor.pos)/(self.norm(self.pos - neighbor.pos))

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
