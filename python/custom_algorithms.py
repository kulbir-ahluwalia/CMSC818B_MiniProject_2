import numpy as np
import cv2


## Step size. How many grids would the robot move at each step
step_size = 10

## the dictionary defniing how many to update the location
move_dict = {
    'left' : np.array([0, -step_size]),
    'up' : np.array([-step_size,0]),
    'right' : np.array([0, step_size]),
    'down' : np.array([step_size,0])
}

## Function to check if a location is valid (within grid)
def is_valid(pos, size, limits):
    if ((pos[0] < size) or (pos[1] < size) or (pos[0] >= limits[0]-size-1) or (pos[1] >= limits[1]-size-1)):
        return False
    else:
        return True

## Greedy algorithm (distributed)
def greedy_algorithm(player, data_img):
    pos = player.pos #get player location
    obstacle = data_img[:,:,2].astype(int)/255.# get grid indicating where objects are
    covered = (data_img[:,:,1]).astype(int)/255. # get grid indicating which areas have been covered and how much latency is there
    # covered[data_img[:,:,1] == 255] = 0
    cost_grid = obstacle*10 + covered # give high cost to obstacles

    # if the last action is not recorded, move to left or right as per the row
    action_list = ['right', 'down', 'left', 'up']
    if player.prev_action == None:
        player.prev_action = ['right', 'left'][pos[0]%2]


    #######
    ## Looks into future move for all 4 directions, move whereever robot would have least cost
    temp_action_list = []
    temp_cost_list = []
    for i in range(4):
        new_action = action_list[ (action_list.index(player.prev_action) + i) % 4 ]
        move = move_dict[ new_action ]
        
        temp_pos = pos + move
        # print('CHECK: ', pos, temp_pos, new_action)
        if not is_valid(temp_pos, player.size, data_img.shape[:2]):
            # print('NOT Valid: ', pos, temp_pos)
            temp_action_list.append(new_action)
            temp_cost_list.append(50000)
            # continue
        else:
            # if (covered[temp_pos[0], temp_pos[1]] == 1) or (obstacle[temp_pos[0], temp_pos[1]] == 1):
            temp_action_list.append(new_action)
            temp_cost = np.sum(cost_grid[max(0,temp_pos[0]-player.size):min(temp_pos[0]+player.size, data_img.shape[0]), 
                                         max(0, temp_pos[1]-player.size): min(temp_pos[1]+player.size,data_img.shape[1])])
            temp_cost_list.append(temp_cost)


    # Choose next action based on where the cost would be least
    new_action = temp_action_list[np.argmin(temp_cost_list)]
    temp_pos = pos + move_dict[ new_action ]

    # update last action
    player.prev_action = new_action
    
    # if the move is not valid, do not move
    if not is_valid(temp_pos, player.size, data_img.shape[:2]):
        temp_pos = pos
        
    return temp_pos
