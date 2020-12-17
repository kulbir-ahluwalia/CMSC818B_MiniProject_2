import numpy as np

# check if point in map
def check_inbound(point, env):
  bound_x= env.env_img.shape[0] - 1
  bound_y= env.env_img.shape[1] - 1

  new_point= np.array([np.clip(point[0], 0, bound_x), np.clip(point[1], 0, bound_y)])

  return (new_point == point).all(), new_point

# in obstacle
def check_inObs(point, env):
  return env.env_img[:,:,2][point[0], point[1]] == 255

# iun drone view
def check_inView(point, env):
  num_drones= len(env.droneList)

  ret= False

  for i in range(num_drones):
    drone_x= env.droneList[i].pos[0]
    drone_y= env.droneList[i].pos[1]
    drone_size= env.droneList[i].size
    if point[0] in range(drone_x, drone_x + drone_size + 1) and point[1] in range(drone_y, drone_y + drone_size + 1):
      ret= True 

  
    
  return ret

# whicl player is in view
def which_inView(env):
  in_view= [] 
  for i in range(num_agents):
    agent= env.playerList[i]
    if check_inView(agent.pos, env):
      in_view += [i]

  return in_view 

# Particle filter 
def update_belief(belief,env):
  new_belief= np.zeros_like(belief)
  for agent in env.playerList:# in range(num_agents):
    #agent = env.playerList[i]
    if not check_inView(agent.pos, env):
      
      #update weight by 1: 
      agent.weight += 1

      for p in range(agent.particles.shape[0]):
        delta_x= np.random.randint(-1, 2) * agent.step_size
        delta_y= 0
        if delta_x == 0:
          delta_y = np.random.randint(-1, 2) * agent.step_size
        
        agent.particles[p] = check_inbound(np.array([delta_x, delta_y]) + agent.particles[p], env)[1]  
        #new_belief[agent.particles[p][0], agent.particles[p][1]] = 255

      
    else: # when agent is in fov of drone
    #   detect_agent(i)
        detect_agent(agent, env)
      #for p in range(agent.particles.shape[0]):
        #new_belief[agent.particles[p][0], agent.particles[p][1]] = 255

  resample_particles(env)

#   for i in range(num_agents):
  for agent in env.playerList:
    # agent = env.playerList[i]
    for p in range(agent.particles.shape[0]):
        new_belief[agent.particles[p][0], agent.particles[p][1]] = 255

  return new_belief
        

## we need to resample when partciles are visible but not the robot
def resample_particles(env):
  height= env.env_img.shape[0]
  width=  env.env_img.shape[1]

#   for a in range(num_agents):
  for agent in env.playerList:
    # agent= env.playerList[a]
    t= [check_inView(p, env) for p in agent.particles] # which partcle can teh drone see
    in_view= [i for i, x in enumerate(t) if x]
    out_view= [i for i, x in enumerate(t) if not x]
    if len(in_view) == env.num_beliefs: # if al partcles visible but not the robot
      if check_inView(agent.pos, env):
        break
      x= np.random.randint(0, height)
      y= np.random.randint(0, width)
      while check_inView([x,y], env):
        x= np.random.randint(0, height)
        y= np.random.randint(0, width)
      
      agent.particles= gen_particles(np.array([x,y]), env.num_beliefs)
    else:
      if len(in_view) > 0:
        for j in in_view:
          new_p= np.random.choice(out_view)
          agent.particles[j] = agent.particles[new_p]


## upull all particles at robot location. latency = 0 becauise drone can see the player (for a single player)
def detect_agent(player, env):
#   env.playerList[player].particles= gen_particles(env.playerList[player].pos, env.num_beliefs)
#   env.playerList[player].weight= 0
  player.particles = gen_particles(player.pos, env.num_beliefs)
  player.weight = 0


## initiallzie belief (all players visible to drone initially)
def init_belief(env):
#   for i in range(len(env.playerList)):
    # detect_agent(i, env)

    for player in env.playerList:
        detect_agent(player, env)


## generating partcile aroud a point
def gen_particles(point, num):
  #noise= np.array([np.random.normal(0, 10, (2,)) for i in range(num)])
  noise= np.array([[0,0] for i in range(num)])
  noise= noise.astype('int')
  
  particles= np.copy(noise)
  for i in range(noise.shape[0]):
    #particles[i] = check_inbound(noise[i] + point, env)[1]
    particles[i] = noise[i] + point
    
  
  return particles

## smaller grid size. gives heatmap
def discretize_env(env, grid_size = 30):
  num_grid= env.belief.shape[0] // grid_size
  heatmap= np.zeros((num_grid, num_grid))

  for agent in env.playerList:#range(num_agents):
    # agent = env.playerList[i]
    for p in range(agent.particles.shape[0]):
      x= agent.particles[p][0] // grid_size
      y= agent.particles[p][1] // grid_size
      heatmap[x, y] += agent.weight//2 #// 10 


  drone= env.droneList[0]
  drone_x= (drone.pos[0] + drone.size //2) // grid_size
  drone_y= (drone.pos[1] + drone.size //2) // grid_size
  heatmap[drone_x, drone_y] = -100


  return heatmap # input to PPO
  
# Reward fuction. Number of agents + their weight
def reward(env):
  res= 0 
  for agent in env.playerList:
    # agent= env.playerList[i]
    # if check_inView(agent.pos, env):
    res += agent.weight
    #   detect_agent(i)

  return res  

