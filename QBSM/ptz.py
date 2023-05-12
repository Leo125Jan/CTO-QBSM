import numpy as np
from math import cos, acos, sqrt, exp, sin
from time import time, sleep

class PTZCamera_():
    def __init__(self, properties, map_size, grid_size,
                    Kv = 60, Ka = 5, Kp = 1, step = 0.1):

        self.grid_size = grid_size
        self.size = (int(map_size[0]/grid_size[0]), int(map_size[1]/grid_size[1]))
        self.id = properties['id']
        self.pos = properties['position']
        self.perspective = properties['perspective']/self.Norm(properties['perspective'])
        self.alpha = properties['AngleofView']/180*np.pi
        self.R = properties['range_limit']
        self.lamb = properties['lambda']
        self.color = properties['color']
        self.max_speed = properties['max_speed']
        self.sigma = properties['intensity']
        self.perspective_force = 0
        self.zoom_force = 0
        self.positional_force = np.array([0.,0.])
        self.targets = []
        self.FoV = np.zeros(self.size)
        self.global_event = np.zeros(self.size)
        self.global_event_plt = np.zeros(self.size)
        self.global_voronoi = np.zeros(self.size)
        self.local_Voronoi = []
        self.Kv = Kv
        self.Ka = Ka
        self.Kp = Kp
        self.step = step
        self.neighbors = []
        self.map_plt = np.zeros((self.size))
        self.intercept_quality = 0
        self.coverage_quality = 0
        self.last_pos = self.pos
        self.role = "Interceptor"
        
    def Update(self, targets, neighbors, centroid_tmp, geo_center_tmp):

        self.neighbors = neighbors
        self.targets = targets
        self.centroid = centroid_tmp
        self.geo_center = geo_center_tmp

        self.global_event = self.ComputeEventDensity(targets = self.targets) 
        self.global_event_plt = ((self.global_event - self.global_event.min()) * (1/(self.global_event.max()
                                    - self.global_event.min()) * 255)).astype('uint8')
        
        self.FoV = np.zeros(self.size)
        self.UpdateFoV()

        # if self.role == "Tracker":
        #     self.UpdateGlobalVoronoi()
        # elif self.role == "Interceptor":
        #     self.UpdateSubGlobalVoronoi()

        self.UpdateLocalVoronoi()

        self.ComputeLocalCentroidal()
        self.UpdateOrientation()
        self.UpdateZoomLevel()

        self.UpdateRole()
        self.UpdatePosition()

        return

    def UpdateOrientation(self):

        self.perspective += self.perspective_force*self.step
        self.perspective /= self.Norm(self.perspective)

        return

    def UpdateZoomLevel(self):

        self.alpha += self.zoom_force*self.step

        return

    def UpdatePosition(self):
        
        # Tracker Control Law (Move to Sweetspot)

        if self.role == "Tracker":
            
            centroid_force = (self.centroid*self.grid_size - self.pos) * (1 - self.R*cos(self.alpha)
                                /self.Norm((self.centroid*self.grid_size - self.pos)[0]))
            rot = np.array([[cos(np.pi/2), -sin(np.pi/2)],
                        [sin(np.pi/2), cos(np.pi/2)]])
            v = rot@self.perspective.reshape(2,1)
            allign_force = ((self.geo_center*self.grid_size - self.pos)@v)*v

            self.positional_force = centroid_force[0]/2 + np.array([allign_force[0][0], allign_force[1][0]])/2

        # Interceptor Control Law (Move to Centroid)
        elif self.role == "Interceptor":

            self.positional_force = (self.centroid[0]*self.grid_size - self.pos)

        self.pos += self.Kp*self.positional_force*self.step

        return

    def UpdateGlobalVoronoi(self): # Not Finished
        return

    def UpdateSubGlobalVoronoi(self): # Not Finished
        return

    def UpdateFoV(self):

        range_max = (self.lamb + 1)/(self.lamb)*self.R
        quality_map = None
        quality_int_map = None
        intercept_map = np.zeros(self.FoV.shape)
        self.intercept_quality = 0

        for y_map in range(max(int((self.pos[1] - range_max)/self.grid_size[1]), 0),\
                            min(int((self.pos[1] + range_max)/self.grid_size[1]), self.size[1])):

            x_map = np.arange(max(int((self.pos[0] - range_max)/self.grid_size[0]), 0),\
                            min(int((self.pos[0] + range_max)/self.grid_size[0]), self.size[0]))
            
            q_per = self.ComputePerspectiveQuality(x_map*self.grid_size[0], y_map*self.grid_size[1])
            q_res = self.ComputeResolutionQuality(x_map*self.grid_size[0], y_map*self.grid_size[1])
            q_int = self.ComputeInterceptionQuality(x_map*self.grid_size[0], y_map*self.grid_size[1])

            quality = np.where((q_per > 0) & (q_res > 0), q_per*q_res, 0)
            q_int = np.where(q_int >= 0, q_int, 0)
            
            if quality_map is None:
                quality_map = quality
                quality_int_map = q_int
            
            else:
                quality_map = np.vstack((quality_map, quality))
                quality_int_map = np.vstack((quality_int_map, q_int))

        intercept_map[max(int((self.pos[1] - range_max)/self.grid_size[1]), 0):\
                                min(int((self.pos[1] + range_max)/self.grid_size[1]), self.size[0]),\
                                    max(int((self.pos[0] - range_max)/self.grid_size[0]), 0):\
                                        min(int((self.pos[0] + range_max)/self.grid_size[0]), self.size[0])]\
                                            = quality_int_map

        self.intercept_quality = np.sum(intercept_map*np.transpose(self.global_event))

        self.FoV[max(int((self.pos[1] - range_max)/self.grid_size[1]), 0):\
                    min(int((self.pos[1] + range_max)/self.grid_size[1]), self.size[0]),\
                        max(int((self.pos[0] - range_max)/self.grid_size[0]), 0):\
                            min(int((self.pos[0] + range_max)/self.grid_size[0]), self.size[0])]\
                                = quality_map
        
        return 

    def UpdateLocalVoronoi(self):
        
        quality_map = self.FoV
        for neighbor in self.neighbors:
            quality_map = np.where((quality_map > neighbor.FoV), quality_map, 0)

        self.coverage_quality = np.sum(quality_map*np.transpose(self.global_event))
        self.local_voronoi = np.array(np.where((quality_map != 0) & (self.FoV != 0))) #np.where(self.FoV != 0) 
        self.local_voronoi_map = np.where(((quality_map > 0.) & (self.FoV > 0.)), quality_map, 0)
        self.overlap = np.array(np.where((quality_map == 0) & (self.FoV != 0)))
        self.map_plt = np.array(np.where(quality_map != 0, self.id + 1, 0))

        return

    def UpdateRole(self):

        self.role = "Interceptor"

        for target in self.targets:
            pos = (np.asarray(target[0])/self.grid_size).astype(np.int64)

            if self.local_voronoi_map[pos[1]-1, pos[0]-1] > 0:
                self.role = "Tracker"
        # if self.coverage_quality >= self.intercept_quality:
        #     self.role = "Tracker"

        print(self.id, "=>", self.role, "=>", self.intercept_quality, "=>", self.coverage_quality)

        return

    def ComputeLocalCentroidal(self):

        rotational_force = np.array([0.,0.]).reshape(2,1)
        zoom_force = 0

        if len(self.local_voronoi[0]) > 0:
            mu_V = 0
            v_V_t = np.array([0, 0], dtype=np.float64)
            delta_V_t = 0

            # Control law for maximizing local resolution and perspective quality
            for i in range(len(self.local_voronoi[0])):
                x_map = self.local_voronoi[1][i]
                y_map = self.local_voronoi[0][i]

                x, y = x_map*self.grid_size[0], y_map*self.grid_size[1]
                x_p = np.array([x,y]) - self.pos
                norm = self.Norm(x_p)

                if norm == 0: continue

                mu_V += ((norm**self.lamb)*self.global_event[x_map,y_map] )/(self.R**self.lamb)
                v_V_t += ((x_p)/norm)*(cos(self.alpha) - \
                                ( ( self.lamb*norm )/((self.lamb+1)*self.R)))*\
                                    ( (norm**self.lamb)/(self.R**self.lamb) )*self.global_event[x_map,y_map]
                dist = (1 - (self.lamb*norm)/((self.lamb+1)*self.R))
                dist = dist if dist >= 0 else 0
                delta_V_t += (1 - (((x_p)@self.perspective.T))/norm)\
                                *dist*((norm**self.lamb)/(self.R**self.lamb))\
                                    *self.global_event[x_map,y_map]
            
            v_V = v_V_t/mu_V
            delta_V = delta_V_t/mu_V
            delta_V = delta_V if delta_V > 0 else 1e-10
            alpha_v = acos(1-sqrt(delta_V))
            alpha_v = alpha_v if alpha_v > 5/180*np.pi else 5/180*np.pi
            
            rotational_force += self.Kv*(np.eye(2) - np.dot(self.perspective[:,None],\
                                            self.perspective[None,:]))  @  (v_V.reshape(2,1))
            zoom_force -= self.Ka*(self.alpha - alpha_v)

        self.perspective_force = np.asarray([rotational_force[0][0], rotational_force[1][0]])
        self.zoom_force = zoom_force

        return

    def ComputePerspectiveQuality(self, x, y):

        x_p = np.array([x,y], dtype=object) - self.pos

        return (np.matmul(x_p,self.perspective.transpose())/np.linalg.norm(x_p)
            - np.cos(self.alpha))/(1 - np.cos(self.alpha))

    def ComputeResolutionQuality(self, x, y):

        x_p = np.array([x, y], dtype=object) - self.pos

        return (((np.linalg.norm(x_p)**self.lamb)*(self.R*np.cos(self.alpha)
            - self.lamb*( np.linalg.norm(x_p) - self.R*np.cos(self.alpha)) ))
                / (self.R**(self.lamb+1)))    

    def ComputeInterceptionQuality(self, x, y):
        x_p = np.linalg.norm(np.array([x, y], dtype=object) - self.pos)
        quality = np.exp(-x_p**2/(2*self.sigma**2))*np.sign(1 - x_p)

        return quality

    def ComputeEventDensity(self, targets):

        x = np.arange(self.global_event.shape[0])*self.grid_size[0]
        event = np.zeros(self.global_event.shape)
        for y_map in range(0, self.global_event.shape[1]):
            y = y_map*self.grid_size[1]
            density = 0
            for i in range(len(targets)):
                density += targets[i][2]*np.exp(-targets[i][1]*np.linalg.norm(np.array([x,y], dtype=object)\
                                -np.array((targets[i][0][1],targets[i][0][0]))))
            event[:][y_map] = density

        return 0 + event 

    def ComputeMaxCapacity(self): # Not Finished
        return

    def PublishInfo(self):
        pass

    def Norm(self, arr):

        sum = 0

        for i in range(len(arr)):
            sum += arr[i]**2

        return sqrt(sum)
    
    # self.positional_force[0] = np.sign((cos(self.alpha)/(self.R**self.lamb))*(self.pos[0] - centroid[0]*self.grid_size[0])\
    #                             * self.Norm(self.pos - centroid)**(self.lamb-2)*(self.lamb**2 - self.lamb)\
    #                             - (self.lamb/(self.R**(self.lamb+1)))*((self.lamb-1)*(self.pos[0] - centroid[0]*self.grid_size[0])\
    #                                 * self.Norm(self.pos-centroid*self.grid_size)**(self.lamb-3)))

    # self.positional_force[1] = np.sign((cos(self.alpha)/(self.R**self.lamb))*(self.pos[1] - centroid[1]*self.grid_size[0])\
    #                             * self.Norm(self.pos - centroid)**(self.lamb-2)*(self.lamb**2 - self.lamb)\
    #                             - (self.lamb/(self.R**(self.lamb+1)))*((self.lamb-1)*(self.pos[1] - centroid[1]*self.grid_size[0])\
    #                                 * self.Norm(self.pos-centroid*self.grid_size)**(self.lamb-3)))