import pygame
import numpy as np


class Canvas():
    def __init__(self, screenWidth , screenHeight):
        self.width = screenWidth
        self.height = screenHeight
        
        self.grid = np.zeros((self.width , self.height),dtype=int)
        
        
        pygame.init()

        # Set the height and width of the screen
        size = [self.width, self.height]
        self.screen = pygame.display.set_mode(size)

        pygame.display.set_caption("Bouncing Balls")

    
    def update(self):
        temp_surf = pygame.Surface(self.grid.shape)
        pygame.surfarray.array_to_surface(temp_surf, np.transpose(np.tile(self.grid,(3,1,1)),(1,2,0)))
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
    # data_img[data_img[:,:,1] > 0, 1] -= 2 
    # data_img[img_bgr[:,:,1] == 255, 1] = 255
    
    # data_img[img_bgr[:,:,2] == 255, 1] = 0
    # data_img[img_bgr[:,:,2] == 255, 2] = 255
    
    # coverage[data_img[:,:,1] > 0] = 255
    
    

    return data_img, coverage
