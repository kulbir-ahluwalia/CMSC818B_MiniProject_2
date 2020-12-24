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



def drone_lawnmower(drone, canvas):
    row_pos, col_pos = drone.pos
    height, width = canvas.height, canvas.width

    num_r = int(np.ceil((height) / drone.step_size))
    num_c = int(np.ceil((width) / drone.step_size))-1


    env_action_list = ['up', 'down', 'left', 'right']
    action_list = []
    for i in range(num_r):
        for j in range(num_c):
            if i%2:
                action_list.append(2)
            else:
                action_list.append(3)
        action_list.append(1)

    action_list = action_list[:-1]
    action_list.extend([0]*(num_r - 1))
    if num_r%2 == 1:
        action_list.extend([2]*(num_c - 0))

    start_idx = row_pos//drone.step_size
    if row_pos%2:
        start_idx += (width - col_pos)//drone.step_size
    else:
        start_idx += col_pos//drone.step_size
    
    return start_idx, action_list



class Player_Lawnmower:
    def __init__(self, playerList, canvas):
        self.player_size = playerList[0].size
        self.num_players = len(playerList)

        height, width = canvas.height, canvas.width
        cand_rows = [int(i*height/self.num_players) for i in range(self.num_players)]
        # print(cand_rows)
        self.lm_move_dict = {}

        for id, player in enumerate(playerList):
            start_idx, action_list = self.player_action_maker(player, 
                                                        cand_rows[id], 
                                                        (height//self.num_players, width))
            self.lm_move_dict[id] = (start_idx, action_list)
        
    def player_action_maker(self, player, start_row, limits):
        row_pos, col_pos = player.pos
        row_pos -= start_row
        height, width = limits

        # print(row_pos, col_pos, height, width)

        num_r = int(np.ceil((height - player.size) / player.step_size))
        num_c = int(np.ceil((width - player.size) / player.step_size))

        # print(num_r, num_c)

        action_list = []
        for i in range(num_r):
            for j in range(num_c):
                if i%2:
                    action_list.append('left')
                else:
                    action_list.append('right')
            action_list.append('down')

        action_list.extend(['up']*(num_r - 0))
        action_list.extend(['left']*(num_c - 0))

        start_idx = height*row_pos//player.step_size
        if row_pos%2:
            start_idx += (width - col_pos)//player.step_size
        else:
            start_idx += col_pos//player.step_size
        print(num_r, num_c, start_idx)
        return start_idx, action_list

    def get_generator_single(self, player_id):
        start_loc = self.lm_move_dict[player_id][0]
        action_list = self.lm_move_dict[player_id][1]
        total_act = len(action_list)
        step = 0
        while True:
            yield action_list[(start_loc + step)%total_act]
            step += 1
        
    def get_all_generators(self):
        self.gen_list = []
        for i in range(self.num_players):
            self.gen_list.append(self.get_generator_single(i))

        return self.gen_list

def info_greedy_1(player, data_img):
    center = player.pos
    min_size = (player.size**2)/2
    height, width, _ = data_img.shape

    m1 = (0 - center[1])/(0 - center[0])
    m2 = (width - center[1])/(0 - center[0])
    m3 = (0 - center[1])/(height - center[0])
    m4 = (width - center[1])/(height - center[0])

    line1 = lambda x:  x[1] - m1*(x[0] - center[0]) - center[1]
    line2 = lambda x:  x[1] - m2*(x[0] - center[0]) - center[1]
    line3 = lambda x:  x[1] - m3*(x[0] - center[0]) - center[1]
    line4 = lambda x:  x[1] - m4*(x[0] - center[0]) - center[1]

    xv, yv = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    mapping1 = np.apply_along_axis(line1, 2, np.stack([xv, yv], axis=-1))
    mapping2 = np.apply_along_axis(line2, 2, np.stack([xv, yv], axis=-1))
    mapping3 = np.apply_along_axis(line3, 2, np.stack([xv, yv], axis=-1))
    mapping4 = np.apply_along_axis(line4, 2, np.stack([xv, yv], axis=-1))


    mask1 = (mapping1 >= 0)
    mask2 = (mapping2 >= 0)
    mask3 = (mapping3 >= 0)
    mask4 = (mapping4 >= 0)


    up = mask1 & ~mask2
    down = ~mask4 & mask3
    left = ~mask3 & ~mask1
    right = mask2 & mask4

    mask_list = [up, right, down, left]

    coverage = data_img[:,:,1]/255. + data_img[:,:,2]/255.
    action_list = ['up', 'right', 'down', 'left']
    cost_list = np.array([0.,0.,0.,0.])
    size_list = np.array([0, 0, 0, 0])
    for i in range(4):
        # cost_list[i] = np.mean(coverage[mask_list[i]])
        cost_list[i] = np.sum(coverage[mask_list[i]] == 0)
        size_list[i] = coverage[mask_list[i]].shape[0]
        
    cost_list[size_list <= min_size] = cost_list.max()

    min_cost_idx = np.argmin(cost_list)
    
    return action_list[min_cost_idx]

def info_greedy_2(player, data_img):
    center = player.pos
    min_size = (player.size**2)/2
    height, width, _ = data_img.shape

    line1 = lambda x:  x[1] - x[0] - (center[1] - center[0])
    line2 = lambda x:  x[1] + x[0] - (center[1] + center[0])

    xv, yv = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    mapping1 = np.apply_along_axis(line1, 2, np.stack([xv, yv], axis=-1))
    mapping2 = np.apply_along_axis(line2, 2, np.stack([xv, yv], axis=-1))


    mask1 = (mapping1 >= 0)
    mask2 = (mapping2 >= 0)


    right = mask1 & mask2
    down = ~mask1 & mask2
    up =  mask1 & ~mask2
    left = ~mask1 & ~mask2

    mask_list = [up, right, down, left]

    coverage = data_img[:,:,1]/255. + data_img[:,:,2]/255.

    action_list = ['up', 'right', 'down', 'left']
    cost_list = np.array([0.,0.,0.,0.])
    size_list = np.array([0, 0, 0, 0])
    for i in range(4):
        size_list[i] = coverage[mask_list[i]].shape[0]+1
        cost_list[i] = np.sum(coverage[mask_list[i]] == 0)/size_list[i]

    
    cost_list[size_list <= min_size+20] = cost_list.min()

    #min_cost_idx = np.argmin(cost_list)
    min_cost_idx = np.argmax(cost_list)

    return action_list[min_cost_idx]


def info_greedy_drone_2(drone, data_img, old_img):
    center = drone.pos + drone.size//2
    #min_size = (player.size**2)
    height, width, _ = data_img.shape

    line1 = lambda x:  x[1] - x[0] - (center[1] - center[0])
    line2 = lambda x:  x[1] + x[0] - (center[1] + center[0])

    xv, yv = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    mapping1 = np.apply_along_axis(line1, 2, np.stack([xv, yv], axis=-1))
    mapping2 = np.apply_along_axis(line2, 2, np.stack([xv, yv], axis=-1))


    mask1 = (mapping1 >= 0)
    mask2 = (mapping2 >= 0)


    right = mask1 & mask2
    down = ~mask1 & mask2
    up =  mask1 & ~mask2
    left = ~mask1 & ~mask2

    mask_list = [up, right, down, left]

    coverage = (data_img[:,:,1].astype(int) - old_img[:,:,1].astype(int))/255. 
    
    action_list = ['up', 'right', 'down', 'left']
    cost_list = np.array([0.,0.,0.,0.])
    size_list = np.array([0, 0, 0, 0])
    for i in range(4):
        size_list[i] = coverage[mask_list[i]].shape[0]+1
        cost_list[i] = np.sum(coverage[mask_list[i]])
        
    #cost_list[size_list <= min_size] = cost_list.min()
    min_cost_idx = np.argmax(cost_list)
    
    if len(np.unique(cost_list)) == 1:
        min_cost_idx = np.random.randint(0,4)

    return action_list[min_cost_idx], up, down, left, right

def info_greddy_PF(drone, heatmap, env_action_list, grid_size=10):
    pos_in_heatmap = (drone.pos +  drone.size//2 ) // grid_size

    costs = np.zeros((5,)) 
    costs[0] = np.sum(heatmap[:pos_in_heatmap[0], pos_in_heatmap[1]])
    costs[1] = np.sum(heatmap[pos_in_heatmap[0]+1:, pos_in_heatmap[1]])
    costs[2] = np.sum(heatmap[pos_in_heatmap[0], :pos_in_heatmap[1]])
    costs[3] = np.sum(heatmap[pos_in_heatmap[0], pos_in_heatmap[1]+1 :])
    # costs[4] = heatmap[pos_in_heatmap[0], pos_in_heatmap[1]]

    if len(np.unique(costs[:4])) == 1:
        action = np.random.choice(env_action_list)
    else:
        action = env_action_list[np.argmax(costs)]

    return action
