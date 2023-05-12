import pygame
import numpy as np
from time import time, sleep
from ptz import PTZCamera_
from math import cos, acos, sqrt, exp, sin
from pygame_recorder import ScreenRecorder


initialized = False

class UAVs():
    def __init__(self, map_size, resolution):
        
        self.members = []
        self.map_size = map_size
        self.grid_size = resolution

    def AddMember(self, ptz_info):
        
        ptz = PTZCamera(ptz_info, self.map_size, self.grid_size)
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
        self.size = (np.array(map_size) / np.array(grid_size)).astype(np.int64)
        self.grid_size = grid_size
        self.window_size = np.array(self.size)*4
        self.display = pygame.display.set_mode(self.window_size)
        self.display.fill((0,0,0))
        self.blockSize = int(self.window_size[0]/self.size[0]) #Set the size of the grid block
        self.recorder = ScreenRecorder(1024, 1024, 60)

        for x in range(0, self.window_size[0], self.blockSize):
            for y in range(0, self.window_size[1], self.blockSize):
                rect = pygame.Rect(x, y, self.blockSize, self.blockSize)
                pygame.draw.rect(self.display, (125,125,125), rect, 1)

        pygame.display.update()
        self.recorder.capture_frame(self.display)

    def Visualize2D(self, cameras, global_voronoi, event_plt, targets, centroids, sub_centroids, sub_global_voronoi):

        map_plt = np.zeros((event_plt.shape)) - 1

        sum, sum1 = 0, 0
        for i in range(len(cameras)):
            map_plt = cameras[i].map_plt + map_plt
            sum += cameras[i].intercept_quality
            sum1 += cameras[i].coverage_quality

        print("Interception Quality: ",sum, "Coverage Quality: ",sum1)

        x_map = 0
        for x in range(0, self.window_size[0], self.blockSize):

            y_map = 0

            for y in range(0, self.window_size[1], self.blockSize):

                dense = event_plt[x_map][y_map]
                w = 0.6

                id = int(map_plt[y_map][x_map])

                if id not in range(len(cameras)):
                    color = [0,0,0]
                    # color[0] = (1-w)*(50+cameras[int(global_voronoi[x_map][y_map])].color[0]) + w*dense
                    # color[1] = (1-w)*(50+cameras[int(global_voronoi[x_map][y_map])].color[1]) + w*dense
                    # color[2] = (1-w)*(50+cameras[int(global_voronoi[x_map][y_map])].color[2]) + w*dense
                    color[0] = (1-w)*(50+cameras[int(sub_global_voronoi[x_map][y_map])].color[0]) + w*dense
                    color[1] = (1-w)*(50+cameras[int(sub_global_voronoi[x_map][y_map])].color[1]) + w*dense
                    color[2] = (1-w)*(50+cameras[int(sub_global_voronoi[x_map][y_map])].color[2]) + w*dense
                    rect = pygame.Rect(x, y, self.blockSize, self.blockSize)
                    pygame.draw.rect(self.display, color, rect, 0)
                else:
                    color = ((1-w)*cameras[id].color[0] + w*dense, \
                                (1-w)*cameras[id].color[1] + w*dense,\
                                    (1-w)*cameras[id].color[2] + w*dense)
                    rect = pygame.Rect(x, y, self.blockSize, self.blockSize)
                    pygame.draw.rect(self.display, color, rect, 0)

                y_map += 1
            x_map += 1 

        for camera in cameras:
            color = (camera.color[0], camera.color[1], camera.color[2])
            center = camera.last_pos/self.grid_size*self.blockSize
            R = camera.R*cos(camera.alpha)/self.grid_size[0]*self.blockSize
            pygame.draw.line(self.display, color, center, center + camera.perspective*R, 3)
            pygame.draw.circle(self.display, color, camera.last_pos/self.grid_size*self.blockSize, 10)
            
            if camera.role == "Tracker":
                pygame.draw.circle(self.display, color, camera.last_pos/self.grid_size*self.blockSize,
                                    (camera.lamb + 1)/camera.lamb*R, 1)

            elif camera.role == "Interceptor":

                pygame.draw.circle(self.display, color, camera.last_pos/self.grid_size*self.blockSize,
                                    camera.max_speed*self.blockSize/self.grid_size[0], 1)

            camera.last_pos = camera.pos


        # for i in range(len(cameras)):

        #     if cameras[i].role == "Interceptor":

        #         centroid = sub_centroids[i][0]*self.blockSize
        #         p1 = (centroid[0] + 10, centroid[1] + 10)
        #         p2 = (centroid[0] - 10, centroid[1] + 10)
        #         p3 = (centroid[0], centroid[1] - 10)
        #         pygame.draw.polygon(self.display, cameras[i].color,(p1, p2, p3)) 
             
        for target in targets:

            pos = np.asarray(target[0])/self.grid_size*self.blockSize
            pygame.draw.circle(self.display, (0,0,0), pos, 4)
            pygame.draw.line(self.display, (0,0,0), pos, pos + target[3].reshape(1, 2)[0]/2\
                                /self.grid_size*self.blockSize, 2)
        
        pygame.draw.rect(self.display, (0, 0, 0), (0, 0, map_size[0]/grid_size[0]*self.blockSize, \
                                                    map_size[1]/grid_size[1]*self.blockSize), width = 3)
        
        pygame.display.flip()
        self.recorder.capture_frame(self.display)

def ComputeGlobalVoronoi(PTZs, event):
    global initialized

    global_voronoi = np.zeros(event.shape)
    sub_global_voronoi = np.zeros(event.shape)
    weighted_sum = [[0 for i in range(len(PTZs))], [0 for i in range(len(PTZs))]]
    voronoi_sum = [[0 for i in range(len(PTZs))], [0 for i in range(len(PTZs))]]
    weighted_sum_sub = [0 for i in range(len(PTZs))]
    voronoi_sum_sub = [0 for i in range(len(PTZs))]

    for x in range(event.shape[0]):
        for y in range(event.shape[1]):

            distance = np.inf
            distance_sub = np.inf

            for i in range(len(PTZs)):
                    
                pos = PTZs[i].pos + PTZs[i].R*cos(PTZs[i].alpha)*PTZs[i].perspective
                pos_self = PTZs[i].pos

                if False:

                    r = PTZs[i].R*cos(PTZs[i].alpha)/grid_size[0] if PTZs[i].role == "Tracker" else 1/grid_size[0]
                    tmp = ((norm(pos_self/grid_size - np.array([x,y])))**2 - (r**2))*event[x, y]  
                
                else:
                    
                    r = PTZs[i].R*cos(PTZs[i].alpha)/grid_size[0]
                    r_sub = PTZs[i].max_speed/grid_size[0]
                    tmp = ((norm(pos_self/grid_size - np.array([x,y])))**2 - r**2)*event[x, y]
                    tmp_sub = ((norm(pos_self/grid_size - np.array([x,y])))**2 - r_sub**2)*event[x, y]
                
                if tmp < distance and PTZs[i].role == "Tracker":

                    distance = tmp 
                    global_voronoi[x, y] = i 

                if tmp_sub < distance_sub:        
                    
                    distance_sub = tmp_sub 
                    sub_global_voronoi[x, y] = i  
                    

            id =  global_voronoi[x, y].astype(np.int64)
            id_sub = sub_global_voronoi[x, y].astype(np.int64)

            weighted_sum[0][id] += np.array([x,y])*event[x,y]
            voronoi_sum[0][id] += event[x,y]
            weighted_sum[1][id] += np.array([x,y])
            voronoi_sum[1][id] += 1
            weighted_sum_sub[id_sub] += np.array([x,y])*event[x,y]
            voronoi_sum_sub[id_sub] += event[x,y]

    centroids = [[0, 0] for i in range(len(PTZs))]
    sub_centroids = [[0, 0] for i in range(len(PTZs))]
    geo_centers = [[0, 0] for i in range(len(PTZs))]

    for i in range(len(PTZs)):

        if PTZs[i].role == "Tracker":
            centroids[i] = [(weighted_sum[0][i]/voronoi_sum[0][i]).astype(np.int64)]
            geo_centers[i] = [(weighted_sum[1][i]/voronoi_sum[1][i]).astype(np.int64)]


        sub_centroids[i] = [(weighted_sum_sub[i]/voronoi_sum_sub[i]).astype(np.int64)]

    # weighted_sum = [0 for i in range(len(PTZs))]
    # voronoi_sum = [0 for i in range(len(PTZs))]
    # for x in range(event.shape[0]):
    #     for y in range(event.shape[1]):  

    #         id =  global_voronoi[x, y].astype(np.int64)

    #         x_p = centroids[id] - np.array([x,y])
    #         x_q = (centroids[id] - PTZs[id].pos/grid_size)
    #         R = PTZs[id].R/PTZs[id].grid_size[0]

    #         qr = (((norm(x_p)**PTZs[id].lamb)*(R*cos(PTZs[id].alpha)- PTZs[id].lamb\
    #                 *(norm(x_p) - R*cos(PTZs[id].alpha))))/(R**(PTZs[id].lamb+1)))
    #         # qp = (np.matmul(x_p,x_q.transpose())/np.linalg.norm(x_p)\
    #         #         - np.cos(PTZs[id].alpha))/(1 - np.cos(PTZs[id].alpha))

    #         d = norm(np.array([x, y]) - PTZs[i].pos/PTZs[i].grid_size)

    #         qr = 0. if qr < 0 else qr
            
    #         g = qr
            
    #         weighted_sum[id] += np.array([x,y])*(g)
    #         voronoi_sum[id] += (g)

    # centroids = [(weighted_sum[i]/voronoi_sum[i]).astype(np.int64) for i in range(len(weighted_sum))]      
    
    initialized = True
    return global_voronoi, centroids, geo_centers, sub_centroids, sub_global_voronoi

def norm(arr):
    sum = 0
    for i in range(len(arr)):
        sum += arr[i]**2

    return sqrt(sum)

def ComputeEventDensity(event, target, grid_size):
    x = np.arange(event.shape[0])*grid_size[0]
    for y_map in range(0, event.shape[1]):
        y = y_map*grid_size[1]
        density = 0
        for i in range(len(target)):
            density += target[i][2]*np.exp(-target[i][1]*np.linalg.norm(np.array([x,y], dtype=object)\
                            -np.array((target[i][0][1],target[i][0][0]))))
        event[:][y_map] = density

    return 0 + event 

def TargetDynamics(x, y, v, res):
    turn = np.random.randint(-30, 30)/180*np.pi
    rot = np.array([[cos(turn), -sin(turn)],
                    [sin(turn), cos(turn)]])
    v = rot@v.reshape(2,1)
    vx = v[0] if v[0]*res + x > 0 and v[0]*res + x < 24 else -v[0]
    vy = v[1] if v[1]*res + y > 0 and v[1]*res + y < 24 else -v[1]

    return (x,y), np.asarray([[0],[0]])
    #return (np.round(float(np.clip(v[0]*res + x, 0, 24)),1), np.round(float(np.clip(v[1]*res + y, 0, 24)),1)),\
    #            np.round(np.array([[vx],[vy]]), len(str(res).split(".")[1]))

def RandomUnitVector():
    v = np.asarray([np.random.normal() for i in range(2)])
    return v/norm(v)

if __name__ == "__main__":
    pygame.init()

    map_size = np.array([24, 24])    
    grid_size = np.array([0.1, 0.1])

    cameras = []
    camera0 = { 'id'            :  0,
                'position'      :  np.array([6.,3.]),
                'perspective'   :  np.array([0.9,1]),
                'AngleofView'   :  10,
                'range_limit'   :  3,
                'lambda'        :  2,
                'color'         : (200, 0, 0),
                'max_speed'     :  1,
                'intensity'     :  0.5}

    cameras.append(camera0)

    camera1 = { 'id'            :  1,
                'position'      :  np.array([3.,4.]),
                'perspective'   :  np.array([0.7,1]),
                'AngleofView'   :  10,
                'range_limit'   :  3,
                'lambda'        :  2,
                'color'         : (0, 200, 0),
                'max_speed'     :  1,
                'intensity'     :  0.5}

    cameras.append(camera1)

    camera2 = { 'id'            :  2,
                'position'      :  np.array([4.,5.]),
                'perspective'   :  np.array([0.7,1]),
                'AngleofView'   :  10,
                'range_limit'   :  3,
                'lambda'        :  2,
                'color'         : (50, 50, 200),
                'max_speed'     :  1,
                'intensity'     :  0.5}

    cameras.append(camera2)

    camera3 = { 'id'            :  3,
                'position'      :  np.array([7.4,5.]),
                'perspective'   :  np.array([0.7,1]),
                'AngleofView'   :  10,
                'range_limit'   :  2,
                'lambda'        :  2,
                'color'         : (150, 100, 200),
                'max_speed'     :  2,
                'intensity'     :  0.5}

    cameras.append(camera3)
    
    # camera4 = { 'id'            :  4,
    #             'position'      :  np.array([7.1,5.]),
    #             'perspective'   :  np.array([0.7,1]),
    #             'AngleofView'   :  10,
    #             'range_limit'   :  2,
    #             'lambda'        :  2,
    #             'color'         : (100, 150, 200)}

    # cameras.append(camera4)

    # camera5 = { 'id'            :  5,
    #             'position'      :  np.array([7.4,5.6]),
    #             'perspective'   :  np.array([0.7,1]),
    #             'AngleofView'   :  10,
    #             'range_limit'   :  2,
    #             'lambda'        :  2,
    #             'color'         : (100, 150, 75)}

    # cameras.append(camera5)

    # camera6 = { 'id'            :  6,
    #             'position'      :  np.array([6.4,5.]),
    #             'perspective'   :  np.array([0.7,1]),
    #             'AngleofView'   :  10,
    #             'range_limit'   :  2,
    #             'lambda'        :  2,
    #             'color'         : (200, 100, 200)}

    # cameras.append(camera6)


    # Initialize UAV team with PTZ cameras
    uav_team = UAVs(map_size, grid_size)

    for camera in cameras:
        uav_team.AddMember(camera)

    # initialize environment with targets
    size = (map_size/grid_size).astype(np.int64)
    event = np.zeros((size[0], size[1]))

    # target's [position, certainty, weight, velocity]
    targets = [[(12, 11.5), 3, 10, RandomUnitVector()], \
                [(12.5, 12.5), 3, 10,RandomUnitVector()], \
                    [(11.5, 12.5), 3, 10, RandomUnitVector()]]
    event1 = ComputeEventDensity(event, targets, grid_size)

    # Start Simulation
    Done = False
    vis = Visualize(map_size, grid_size)
    clock = pygame.time.Clock()

    while not Done:
        for op in pygame.event.get():
            if op.type == pygame.QUIT:
                Done = True 

        for i in range(len(targets)):
            pos, vel = TargetDynamics(targets[i][0][0], targets[i][0][1], targets[i][3], grid_size[0])
            targets[i][0] = pos
            targets[i][3] = vel

        event1 = ComputeEventDensity(event, targets, grid_size)
        event_plt1 = ((event - event1.min()) * (1/(event1.max() - event1.min()) * 255)).astype('uint8')

        # Update Global and Local Voronoi 
        global_voronoi, centroids, geo_centers, sub_centroids, sub_global_voronoi = \
            ComputeGlobalVoronoi(uav_team.members, event1) # Will be integrated into each member to achieve distributive computation in the future

        for i in range(len(uav_team.members)):

            neighbors = [uav_team.members[j] for j in range(len(uav_team.members)) if j != i]

            if uav_team.members[i].role == "Tracker":
                uav_team.members[i].Update(targets, neighbors, centroids[i], geo_centers[i])
            elif uav_team.members[i].role == "Interceptor":
                uav_team.members[i].Update(targets, neighbors, sub_centroids[i], geo_centers[i])

        vis.Visualize2D(uav_team.members, global_voronoi, event_plt1, targets, centroids, \
                            sub_centroids, sub_global_voronoi)
        clock.tick(60)

    vis.recorder.end_recording()
        

