import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import sys
import csv
import pygame
import random
import numpy as np
from time import sleep, time
from PTZcamera import PTZcon
from math import cos, acos, sqrt, exp, sin
from scipy.stats import multivariate_normal

initialized = False

class UAVs():

    def __init__(self, map_size, resolution):
        
        self.members = []
        self.map_size = map_size
        self.grid_size = resolution

    def AddMember(self, ptz_info):
        
        ptz = PTZcon(ptz_info, self.map_size, self.grid_size)
        self.members.append(ptz)
        
        return

    # inefficient way, might come up with other data structure to manage the swarm 
    def DeleteMember(self, id): 
        
        for i in range(len(self.members)):
            if self.members.id == id:
                del self.members[i]
                break
        return

class Visualize():

	def __init__(self, map_size, grid_size):

		self.size = (np.array(map_size)/np.array(grid_size)).astype(np.int64)
		self.grid_size = grid_size
		self.window_size = np.array(self.size)*4
		self.display = pygame.display.set_mode(self.window_size)
		self.display.fill((0,0,0))
		self.blockSize = int(self.window_size[0]/self.size[0]) #Set the size of the grid block

		for x in range(0, self.window_size[0], self.blockSize):

		    for y in range(0, self.window_size[1], self.blockSize):

		        rect = pygame.Rect(x, y, self.blockSize, self.blockSize)
		        pygame.draw.rect(self.display, (125,125,125), rect, 1)

		pygame.display.update()

	
	def Visualize2D(self, cameras, event_plt, targets):

		map_plt = np.zeros((event_plt.shape)) - 1

		for i in range(len(cameras)):

			if i == 1:

				for (row, j) in zip(map_plt, range(np.shape(map_plt)[0])):

					for col in range(np.shape(row)[0]):

						if map_plt[j][col] == 0 and cameras[i].map_plt[j][col] > 0:

							cameras[i].map_plt[j][col] = 1

			map_plt = cameras[i].map_plt + map_plt

		x_map = 0
		for x in range(0, self.window_size[0], self.blockSize):

			y_map = 0
			for y in range(0, self.window_size[1], self.blockSize):

				dense = event_plt[x_map][y_map]
				w = 0.6
				id = int(map_plt[y_map][x_map])

				if id == -1:
					gray = (1-w)*125 + w*dense
					rect = pygame.Rect(x, y, self.blockSize, self.blockSize)
					pygame.draw.rect(self.display, (gray, gray, gray), rect, 0)
				elif id not in range(len(cameras)):

					if id == 3:# N: id-2 -> Green of head of blue
						color = ((1-w)*cameras[id-1].color[0] + w*dense,\
								(1-w)*cameras[id-1].color[1] + w*dense,\
								(1-w)*cameras[id-1].color[2] + w*dense)
						rect = pygame.Rect(x, y, self.blockSize, self.blockSize)
						pygame.draw.rect(self.display, color, rect, 0)
					elif id == 4:
						color = ((1-w)*cameras[id-4].color[0] + w*dense,\
								(1-w)*cameras[id-4].color[1] + w*dense,\
								(1-w)*cameras[id-4].color[2] + w*dense)
						rect = pygame.Rect(x, y, self.blockSize, self.blockSize)
						pygame.draw.rect(self.display, color, rect, 0)
					elif id == 5:
						color = ((1-w)*cameras[id-5].color[0] + w*dense,\
								(1-w)*cameras[id-5].color[1] + w*dense,\
								(1-w)*cameras[id-5].color[2] + w*dense)
						rect = pygame.Rect(x, y, self.blockSize, self.blockSize)
						pygame.draw.rect(self.display, color, rect, 0)
				else:
					color = ((1-w)*cameras[id].color[0] + w*dense,\
							(1-w)*cameras[id].color[1] + w*dense,\
							(1-w)*cameras[id].color[2] + w*dense)
					rect = pygame.Rect(x, y, self.blockSize, self.blockSize)
					pygame.draw.rect(self.display, color, rect, 0)
				y_map += 1
			x_map += 1

		for camera in cameras:

			color = (camera.color[0], camera.color[1], camera.color[2])
			center = camera.pos/self.grid_size*self.blockSize

			R = camera.R*cos(camera.alpha)/self.grid_size[0]*self.blockSize
			pygame.draw.line(self.display, color, center, center + camera.perspective*R, 3)
			pygame.draw.circle(self.display, color, camera.pos/self.grid_size*self.blockSize, 10)

		for camera in cameras:

			color = (camera.color[0]*0.5, camera.color[1]*0.5, camera.color[2]*0.5)
			pygame.draw.polygon(self.display, color, [camera.pos/self.grid_size*self.blockSize, \
			                                            camera.ltop/self.grid_size*self.blockSize, \
			                                            camera.top/self.grid_size*self.blockSize, \
			                                            camera.rtop/self.grid_size*self.blockSize], 2)

		for target in targets:

			pygame.draw.circle(self.display, (0,0,0), np.asarray(target[0])/self.grid_size\
			                    *self.blockSize, 6)

		for camera in cameras:

			color = (camera.color[0]*0.7, camera.color[1]*0.7, camera.color[2]*0.7)
			pygame.draw.circle(self.display, color, np.asarray(camera.target[0][0])/self.grid_size\
			                    *self.blockSize, 3)

		pygame.draw.rect(self.display, (0, 0, 0), (0, 0, map_size[0]/grid_size[0]*self.blockSize, \
                                                        map_size[1]/grid_size[1]*self.blockSize), width = 3)
		pygame.display.flip()
	
def norm(arr):

	sum = 0

	for i in range(len(arr)):

	    sum += arr[i]**2

	return sqrt(sum)

def event_density(event, target, grid_size):

	x = np.arange(event.shape[0])*grid_size[0]

	for y_map in range(0, event.shape[1]):

	    y = y_map*grid_size[1]
	    density = 0

	    for i in range(len(target)):

	        density += target[i][2]*np.exp(-target[i][1]*np.linalg.norm(np.array([x,y], dtype=object)\
	                        -np.array((target[i][0][1],target[i][0][0]))))
	    event[:][y_map] = density

	return 0 + event


def TargetDynamic(x, y):
    dx = np.random.uniform(-0.5, 0.5, 1)
    dy = np.random.uniform(-0.5, 0.5, 1)

    return (x, y)
    #(np.round(float(np.clip(dx/2 + x, 0, 24)),1), np.round(float(np.clip(dy/2 + y, 0, 24)),1))

if __name__ == "__main__":

	pygame.init()

	map_size = np.array([25, 25])
	grid_size = np.array([0.1, 0.1])

	cameras = []

	camera0 = { 'id'            :  0,
				'position'      :  np.array([2.0, 0.0]),
				'perspective'   :  np.array([1.0, 0.0]),
				'AngleofView'   :  20,
				'range_limit'   :  5,
				'lambda'        :  2,
				'color'         : (200, 0, 0)}
	cameras.append(camera0)

	camera1 = { 'id'            :  1,
				'position'      :  np.array([0.0, 0.0]),
				'perspective'   :  np.array([1.0, 0.0]),
				'AngleofView'   :  20,
				'range_limit'   :  5,
				'lambda'        :  2,
				'color'         : (0, 200, 0)}
	cameras.append(camera1)

	camera2 = { 'id'            :  2,
				'position'      :  np.array([0.0, 2.0]),
				'perspective'   :  np.array([1.0, 0.0]),
				'AngleofView'   :  20,
				'range_limit'   :  5,
				'lambda'        :  2,
				'color'         : (0, 0, 200)}
	cameras.append(camera2)

	# for i in range(len(cameras)):

	# 	filename = "D:/上課資料/IME/實驗室研究/Paper/Coverage Control/Quality based switch mode/Data/"
	# 	# filename = "D:/Leo/IME/Paper Study/Coverage Control/Quality based switch mode/Data/"
	# 	filename += "Data_" + str(i) + ".csv"

	# 	f = open(filename, "w+")
	# 	f.close()

	# Initialize UAV team with PTZ cameras
	uav_team = UAVs(map_size, grid_size)

	for camera in cameras:
		uav_team.AddMember(camera)

	# initialize environment with targets
	size = (map_size/grid_size).astype(np.int64)
	event = np.zeros((size[0], size[1]))

	# target's [position, certainty, weight, velocity]
	targets = [[(6.5, 19), 1, 10], [(6.0, 18.0), 1, 10], [(7.0, 18.0), 1, 10]]
	# event1 = event_density(event, targets, grid_size)

	# Start Simulation
	Done = False
	vis = Visualize(map_size, grid_size)
	last = time()

	while not Done:

		for op in pygame.event.get():

			if op.type == pygame.QUIT:

				Done = True

		if np.round(time() - last, 2) > 40.00 and np.round(time() - last, 2) < 80.00:

			targets[0][0] = (targets[0][0][0] + 0.00, targets[0][0][1] + 0.01)
			targets[1][0] = (targets[1][0][0] - 0.01, targets[1][0][1] - 0.02)
			targets[2][0] = (targets[2][0][0] + 0.03, targets[2][0][1] - 0.04)
			# targets[3][0] = (targets[3][0][0] + 0.02, targets[3][0][1] - 0.03)

			sleep(0.001)
		elif np.round(time() - last, 2) > 70.00 and np.round(time() - last, 2) < 130:

			targets[0][0] = (targets[0][0][0] + 0.06, targets[0][0][1] - 0.01)
			targets[1][0] = (targets[1][0][0] + 0.008, targets[1][0][1] - 0.05)
			targets[2][0] = (targets[2][0][0] + 0.03, targets[2][0][1] - 0.03)
			# targets[3][0] = (targets[3][0][0] + 0.05, targets[3][0][1] - 0.008)

			sleep(0.001)

		elif np.round(time() - last, 2) > 130.00 and np.round(time() - last, 2) < 180:

			# targets[0][0] = (targets[0][0][0] + 0.008, targets[0][0][1] - 0.07)
			targets[1][0] = (targets[1][0][0] + 0.065, targets[1][0][1] - 0.008)
			targets[2][0] = (targets[2][0][0] + 0.015, targets[2][0][1] - 0.015)
			# targets[3][0] = (targets[3][0][0] + 0.004, targets[3][0][1] - 0.035)

			sleep(0.001)

		event1 = event_density(event, targets, grid_size)
		event_plt1 = ((event - event1.min()) * (1/(event1.max() - event1.min()) * 255)).astype('uint8')

		for i in range(len(uav_team.members)):

			neighbors = [uav_team.members[j] for j in range(len(uav_team.members)) if j != i]
			uav_team.members[i].UpdateState(targets, neighbors, np.round(time() - last, 2))

		vis.Visualize2D(uav_team.members, event_plt1, targets)

		if np.round(time() - last, 2) > 200.00:

			sys.exit()

	pygame.quit()