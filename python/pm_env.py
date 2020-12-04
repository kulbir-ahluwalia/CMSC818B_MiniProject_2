import pygame
import numpy as np
import cv2

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
        temp_surf = pygame.Surface(self.grid.shape, pygame.SRCALPHA)
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

