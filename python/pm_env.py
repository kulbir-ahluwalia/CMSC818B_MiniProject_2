import pygame
import numpy as np
import cv2

import gym
from gym import spaces

from actors import Player, Obstacle, Drone
from custom_algorithms import *

try:
    from google.colab import output
    from google.colab.patches import cv2_imshow
except ImportError:
    pass

N_DISCRETE_ACTIONS = 4

class Canvas():
    def __init__(self, screenHeight , screenWidth):
        self.height = screenHeight
        self.width = screenWidth
        
        self.grid = np.zeros((self.height, self.width),dtype=int)
        
        
        pygame.init()

        # Set the height and width of the screen
        size = [self.width, self.height]
        self.screen = pygame.display.set_mode(size)

        pygame.display.set_caption("Bouncing Balls")

    
    def update(self):
        temp_surf = pygame.Surface(self.grid.shape[::-1], pygame.SRCALPHA)
        # pygame.surfarray.array_to_surface(temp_surf, np.transpose(np.tile(self.grid,(3,1,1)),(1,2,0)))
        pygame.surfarray.array_to_surface(temp_surf, np.transpose(np.tile(self.grid,(3,1,1)),(2,1,0)))
        self.screen.blit(temp_surf, (0, 0))
    



def process_into_image(canvas, playerList, obstacleList, droneList):
    grid = canvas.grid.copy()
    player_area = np.zeros(canvas.grid.shape, dtype=int)
    obstacle_area = np.zeros(canvas.grid.shape, dtype=int)
    drone_area =  np.zeros(canvas.grid.shape, dtype=int)

    for player in playerList:
        radius = player.size
        center = player.pos

        xv, yv = np.meshgrid(range(-radius, radius+1), range(-radius, radius+1), sparse=False, indexing='ij')
        valid_array = (xv**2 + yv**2 <= radius**2).astype(int)
#         print(valid_array.shape)
        
        pos_array =  np.stack([xv, yv], axis=-1) + center
#         valid_mask = (pos_array[:,:,0] < 0) 
#         valid_array[pos_array[:,:,0] < 0] = 0
#         valid_array[pos_array[:,:,1] < 0] = 0
#         valid_array[pos_array[:,:,0] >= canvas.width-1] = 0
#         valid_array[pos_array[:,:,1] >= canvas.height-1] = 0

        r_ind_min = max(0, -(center[0]-radius))
        c_ind_min = max(0, -(center[1]-radius))
        if center[0]+radius+1 <= canvas.width:
            r_ind_max = 2*radius+1
        else:
            r_ind_max = -(center[0]+radius -canvas.width)-1
        
        if center[1]+radius+1 <= canvas.width:
            c_ind_max = 2*radius+1
        else:
            c_ind_max = -(center[1]+radius -canvas.height)-1
        
#         print('limits: ', r_ind_min, r_ind_max, c_ind_min, c_ind_max)
#         print('player limits: ', center[0]-radius , center[0]+radius+1, center[1]-radius, center[1]+radius+1)
        
#         player_area[center[0]-radius : center[0]+radius+1, center[1]-radius : center[1]+radius+1] = valid_array
        player_area[max(0, center[0]-radius) : min(center[0]+radius+1, canvas.width), 
                    max(0, center[1]-radius) : min(center[1]+radius+1, canvas.height)] = valid_array[r_ind_min:r_ind_max, c_ind_min:c_ind_max]
        
        
        
    for obstacle in obstacleList:
        corner = obstacle.pos
        dimensions = obstacle.size
        
        obstacle_area[corner[0]:corner[0]+dimensions[0], corner[1]-dimensions[1]:corner[1]] = 1

    for drone in droneList:
        corner = drone.pos
        dimensions = drone.size

        center = corner - dimensions//2
        
        drone_area[center[0]:center[0] + dimensions, center[1] - dimensions :center[1]] = 1

    return grid*0 + player_area*1 + obstacle_area * 2 + drone_area *3


def process_img(coverage, img_bgr):
    drone_cover = img_bgr[:,:,0]
    
    coverage[coverage > 0] -= 5
    coverage[coverage < 0] = 0
    coverage[img_bgr[:,:,1] == 255] = 255

    obstacle = img_bgr[:,:,2]
    

    data_img = np.stack([drone_cover, coverage, obstacle], axis=2)
    return data_img, coverage



def get_coverage(img_bgr):
    return img_bgr[:,:,1].astype(int)

def get_obstacles(img_bgr):
    return (img_bgr[:,:,2] > 0).astype(int)

def get_droneview2(drone_map, img_bgr, latency_factor):
    prev_map = drone_map.astype(int)
    drone_map = np.clip(prev_map-latency_factor, 0, 255).astype(np.uint8)
    drone_mask = img_bgr[:,:,0] > 0
    drone_map[:,:,1][drone_mask] = img_bgr[:,:,1][drone_mask]
    drone_map[:,:,2][drone_mask] = img_bgr[:,:,2][drone_mask]

    return drone_map

def get_droneview(drone_map, img_bgr, latency_factor, droneList):
    prev_map = drone_map.astype(int)
    drone_map = np.clip(prev_map-latency_factor, 0, 255).astype(np.uint8)
    
    for drone in droneList:
        drone_map[drone.pos[0]:drone.pos[0]+drone.size, drone.pos[1]:drone.pos[1]+drone.size,1:] = img_bgr[drone.pos[0]:drone.pos[0]+drone.size, drone.pos[1]:drone.pos[1]+drone.size,1:]

   
    return drone_map


def update_all(canvas, playerList, obstacleList, droneList):
    ## Update the environment
    patch = canvas.update()
    
    ## Update the drone
    for drone in droneList:
        patch = drone.update(canvas)
    
    ## Update the robot locations
    for player in playerList:
        patch = player.update(canvas)

    ## Update the obstacles
    for obstacle in obstacleList:
        patch = obstacle.update(canvas)

     

def process_screen(my_coverage, my_canvas):
    ### Create the images
    pygame.display.flip()
    # data_img = process_into_image(canvas, playerList, obstacleList, droneList)
    
    #convert image so it can be displayed in OpenCV
    view = pygame.surfarray.array3d(my_canvas.screen)

    #  convert from (width, height, channel) to (height, width, channel)
    view = view.transpose([1, 0, 2])

    #  convert from rgb to bgr
    img_bgr = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
    # Convert from x-y format to row-column format and get images as numpy array
    xv, yv = np.meshgrid(range(img_bgr.shape[0]), range(img_bgr.shape[1]), indexing='ij')
    data_img, my_coverage = process_img(my_coverage, img_bgr[yv, xv])
    
     ## Clip the values to void overflow
    data_img = np.clip(img_bgr[yv, xv].astype(int) + data_img.astype(int), a_min=0, a_max=255)
    
    ## Update pygame environemnt
    surf = pygame.surfarray.make_surface(data_img)
    my_canvas.screen.blit(surf, (0, 0))
    
    ## Convert data type from int to bytes
    data_img = data_img.astype(np.uint8)


    return my_coverage, data_img


def init_env(canvas, playerList, obstacleList, droneList):
    covergae = np.zeros(canvas.grid.shape, dtype=np.uint8)
    data_img = np.zeros(canvas.grid.shape+(3,), dtype=np.uint8)

    patch = canvas.update()
    ## Update the drone
    for drone in droneList:
        patch = drone.update(canvas)
    covergae, temp_img = process_screen(covergae, canvas)

    data_img[:,:,0] = temp_img[:,:,0]

    patch = canvas.update()
    ## Update the robot locations
    for player in playerList:
        patch = player.update(canvas)
    covergae, temp_img = process_screen(covergae, canvas)

    data_img[:,:,1] = temp_img[:,:,1]


    patch = canvas.update()
    ## Update the obstacles
    for obstacle in obstacleList:
        patch = obstacle.update(canvas)
    covergae, temp_img = process_screen(covergae, canvas)

    data_img[:,:,2] = temp_img[:,:,2]
    data_img[temp_img[:,:,2] > 0, 1 ] = 0
    # cv2_imshow(data_img)


    return data_img


class PMGridEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, height, width, robot_info, drone_info, num_obstacles, drone_latency):
    super(PMGridEnv, self).__init__()
    # Define action and observation space
    self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
    # using image as input (can be channel-first or channel-last):
    self.observation_space = spaces.Box(low=0, high=255,
                                        shape=(height, width, 3), dtype=np.uint8)

    # To convert integer actions into string
    self.action_list = ['up', 'down', 'left', 'right']

    ## Canvas is the grid we are going to use
    self.canvas = Canvas(height, width)

    self.latency_factor = drone_latency

    ## Create the robots
    self.playerList = []
    for i in range(len(robot_info)//3):
        x_loc, y_loc, size = robot_info[3*i : 3*(i+1)]
        self.playerList.append(Player(pos=[x_loc, y_loc], color='g', size=size))    
    
    ### Add drone to the environememnt
    self.droneList = []
    for i in range(len(drone_info)//3):
        x_loc, y_loc, size = drone_info[3*i : 3*(i+1)]
        self.droneList.append(Drone(pos=[x_loc, y_loc], color='b',  size=size))

    ## Create the obstacle at random locations
    self.n_obj = num_obstacles # number of objects
    self.obstacleList = []
    r_coords = np.random.randint(0, self.canvas.height, (self.n_obj)) # random rows
    c_coords = np.random.randint(0, self.canvas.width, (self.n_obj)) # random columns
    # Width and height would be chosen from 10,15,20,25,30 randomly
    for i in range(len(r_coords)):
        scale_r = self.canvas.height/300.
        scale_c = self.canvas.width/300.
        length = int(scale_r * np.random.choice([10,15,20,25,30]))
        breadth = int(scale_c * np.random.choice([10,15,20,25,30]))
        
        self.obstacleList.append(Obstacle(pos=[r_coords[i], c_coords[i]], 
                                    size=[length, breadth]))

    ### Environement image as a numpy array
    self.env_img   = np.zeros(self.canvas.grid.shape+(3,), dtype=np.uint8)
    ### Coverage information
    self.coverage  = np.zeros(self.canvas.grid.shape, dtype=np.uint8)
    ### Obstacle map information
    self.obstacle_map = np.zeros(self.canvas.grid.shape, dtype=np.uint8)
    #### Drone Map
    self.drone_map = np.zeros(self.canvas.grid.shape+(3,), dtype=np.uint8)
    #### Saving initial state for resets
    self.inital_state = [self.playerList.copy(), self.droneList.copy(), self.obstacleList.copy()]
    #### Initializing locaal info for each robot
    for player in self.playerList:
        player.info = self.env_img.copy()
    ### Update entites in screen
    self.update_all() 
    self.process_screen()

  def step(self, action):
    ### Update robots' locations
    for player in self.playerList:
        # player.pos = lawn_mover(player, data_img)
        player.pos = greedy_algorithm(player, self.env_img)
        # player.move(np.random.choice(['up', 'down', 'left', 'right']), player.size, canvas)

    ### Update drones' locations
    for drone in self.droneList:
         drone.move(self.action_list[action], drone.step_size, self.canvas)

    ### Update graphics
    self.update_all() 
    
    ### Get updated coverage and observations
    # self.covergae, self.env_img = self.process_screen(my_coverage=self.coverage, my_canvas=self.canvas)
    self.process_screen()
    '''
    ## Get obstacle info
    self.obstacle_map = self.get_obstacles(self.env_img)
    '''
    ### Get drone's view
    self.drone_map = self.get_droneview(self.drone_map, self.env_img)
    
    ### Reward coverage from the drone's view
    drone_coverage = self.get_coverage(self.drone_map)
    drone_coverage = drone_coverage/255.0
    reward = np.sum(drone_coverage)

    # Persistent monitoring never stops
    done = False

    # No info as of now
    info = None

    # # observation is same as the environment image
    # observation = self.env_img

    # observation is same as the drone image
    observation = self.drone_map


    return observation, reward, done, info

  def reset(self):
    self.playerList, self.droneList, self.obstacleList = self.inital_state
    
    self.process_screen()

    return self.env_img  # reward, done, info can't be included

  def render(self, mode='human'):
    output.clear()
    vertical_var = np.full((self.env_img.shape[0],10,3), 128, dtype=np.uint8)
    # cv2_imshow(data_img) #img_bgr[yv, xv])
    print('\t\t Environment \t\t\t Drone View ')
    cv2_imshow(np.hstack([self.env_img, vertical_var, self.drone_map]))
    # print('\t\t Coverage \t\t\t Obstacles ')
    # cv2_imshow(np.hstack([self.coverage, 255*self.obstacle_map]))
    print('\t\t Coverage \t\t\t Drone Coverage ')
    cv2_imshow(np.hstack([self.coverage, vertical_var[:,:,0], self.drone_map[:,:,1]]))


  def close (self):
    pygame.close()

  def get_action_space(self):
      return self.action_space

  def update_all(self):
    ## Update the environment
    patch = self.canvas.update()
    
    ## Update the drone
    for drone in self.droneList:
        patch = drone.update(self.canvas)
    
    ## Update the robot locations
    for player in self.playerList:
        patch = player.update(self.canvas)

    ## Update the obstacles
    for obstacle in self.obstacleList:
        patch = obstacle.update(self.canvas)  


  def process_into_image(self):
    grid = self.canvas.grid.copy()
    player_area = np.zeros(self.canvas.grid.shape, dtype=int)
    obstacle_area = np.zeros(self.canvas.grid.shape, dtype=int)
    drone_area =  np.zeros(self.canvas.grid.shape, dtype=int)

    for player in self.playerList:
        radius = player.size
        center = player.pos

        xv, yv = np.meshgrid(range(-radius, radius+1), range(-radius, radius+1), sparse=False, indexing='ij')
        valid_array = (xv**2 + yv**2 <= radius**2).astype(int)
        
        pos_array =  np.stack([xv, yv], axis=-1) + center

        r_ind_min = max(0, -(center[0]-radius))
        c_ind_min = max(0, -(center[1]-radius))

        if center[0]+radius+1 <= self.canvas.height:
            r_ind_max = 2*radius+1
        else:
            r_ind_max = -(center[0]+radius -self.canvas.height)-1
        
        if center[1]+radius+1 <= self.canvas.width:
            c_ind_max = 2*radius+1
        else:
            c_ind_max = -(center[1]+radius - self.canvas.width)-1
        
        player_area[max(0, center[0]-radius) : min(center[0]+radius+1, self.canvas.height), 
                    max(0, center[1]-radius) : min(center[1]+radius+1, self.canvas.width)] = valid_array[r_ind_min:r_ind_max, c_ind_min:c_ind_max]
        
        
    for obstacle in self.obstacleList:
        corner = obstacle.pos
        dimensions = obstacle.size
        
        obstacle_area[corner[0]:corner[0]+dimensions[0], corner[1]-dimensions[1]:corner[1]] = 1

    for drone in self.droneList:
        corner = drone.pos
        dimensions = drone.size

        center = corner - dimensions//2
        
        drone_area[center[0]:center[0] + dimensions, center[1] - dimensions :center[1]] = 1

    return grid*0 + player_area*1 + obstacle_area * 2 + drone_area *3

  def process_img(self, coverage, img_bgr):
    drone_cover = img_bgr[:,:,0]
    
    coverage[coverage > 0] -= 5
    coverage[coverage < 0] = 0
    coverage[img_bgr[:,:,1] == 255] = 255

    obstacle = img_bgr[:,:,2]

    data_img = np.stack([drone_cover, coverage, obstacle], axis=2)
    

    return data_img, coverage

  def process_screen(self):
    ### Create the images
    pygame.display.flip()
    
    #convert image so it can be displayed in OpenCV
    view = pygame.surfarray.array3d(self.canvas.screen)
    #  convert from (width, height, channel) to (height, width, channel)
    
    view = view.transpose([1, 0, 2])

    #  convert from rgb to bgr
    img_bgr = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
    
    # Convert from x-y format to row-column format and get images as numpy array
    xv, yv = np.meshgrid(range(img_bgr.shape[1]), range(img_bgr.shape[0]), indexing='ij')
    # self.env_img, self.coverage = self.process_img(self.coverage, img_bgr[yv, xv])
    self.env_img, self.coverage = self.process_img(self.coverage, img_bgr)
    
    ## Clip the values to void overflow
    # self.env_img = np.clip(img_bgr[yv, xv].astype(int) + self.env_img.astype(int), a_min=0, a_max=255)
    self.env_img = np.clip(img_bgr.astype(int) + self.env_img.astype(int), a_min=0, a_max=255)

    ## Convert data type from int to bytes
    self.env_img = self.env_img.astype(np.uint8)

    ## Get teh coverage info so far
    local_coverage = self.coverage.copy()
    ## Remove older info to avoid large connected components
    local_coverage[local_coverage < 128] = 0 
    ## Find connected components
    num_labels, labels_im = cv2.connectedComponents(local_coverage)
    for player in self.playerList:
        ## Latency
        player.info = np.clip(player.info.astype(int)-2, 0, 255).astype(int)
        ## Get labels for this conneted component
        mask = (labels_im == labels_im[player.pos[0], player.pos[1]])
        ## Include into coverage
        player.info[:,:,1][mask] = local_coverage[mask]
        
        ## Get information from drone is in view
        for drone in self.droneList:
            if ( (drone.pos[0] <= player.pos[0] <= drone.pos[0]+drone.size) and 
                 (drone.pos[1] <= player.pos[1] <= drone.pos[1]+drone.size)):
                player.info = np.maximum(player.info, self.drone_map)

    

    ## Update pygame environemnt
    surf = pygame.surfarray.make_surface(self.env_img.transpose((1,0,2)))
    self.canvas.screen.blit(surf, (0, 0))
    
    

  def get_coverage(self, img_bgr):
    return img_bgr[:,:,1].astype(int)

  def get_obstacles(self, img_bgr):
    return (img_bgr[:,:,2] > 0).astype(int)

  def get_droneview(self, drone_map, img_bgr):
    prev_map = drone_map.astype(int)
    drone_map = np.clip(prev_map-self.latency_factor, 0, 255).astype(np.uint8)
    
    for drone in self.droneList:
        drone_map[drone.pos[0]:drone.pos[0]+drone.size, drone.pos[1]:drone.pos[1]+drone.size,1:] = img_bgr[drone.pos[0]:drone.pos[0]+drone.size, drone.pos[1]:drone.pos[1]+drone.size,1:]
   
    return drone_map

