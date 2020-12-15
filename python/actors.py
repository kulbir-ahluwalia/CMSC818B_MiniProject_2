import pygame
import pygame.gfxdraw
import numpy as np

class Player():
    def __init__(self, pos, color='g', size=5, step_size=1, num_particles= 10):
        self.pos = np.array(pos)
        self.color = color 
        self.size = size
        self.prev_action = None
        self.step_size = step_size
        
        self.particles= np.zeros((num_particles,2)).astype('int')
        
        self.info = None

        self.COLOR = None
        if self.color == 'r':
            self.COLOR = (255,0,0)
        elif self.color == 'g':
            self.COLOR = (0, 255, 0)
        elif self.color == 'b':
            self.COLOR = (0, 0, 255)
        else:
            self.COLOR = (255, 255, 255)


    def move(self, action, step_size=None, canvas=None):
        if step_size is None:
            step_size = self.step_size
        else:
            self.step_size = step_size

        if action == 'up':
            motion = np.array([-step_size,0])
        elif action == 'down':
            motion = np.array([step_size, 0])
        elif action == 'left':
            motion = np.array([0, -step_size])
        else:
            motion = np.array([0, step_size])

        self.pos += motion

        if self.pos[0] <= self.size:
            self.pos[0] = 0+self.size
        if self.pos[0]+self.size > canvas.height-1:
            self.pos[0] = min(self.pos[0], canvas.height-self.size-1)
        if self.pos[1] <= self.size:
            self.pos[1] = 0+self.size
        if self.pos[1]+self.size > canvas.width-1:
            self.pos[1] = min(self.pos[1], canvas.width-self.size-1)
        # if self.pos[0] < 0:
        #     self.pos[0] = 0 
        # if self.pos[0] >= canvas.width-1:
        #     self.pos[0] = canvas.width - 1
        # if self.pos[1] < 0:
        #     self.pos[1] = 0 
        # if self.pos[1] >= canvas.height-1:
        #     self.pos[1] = canvas.height - 1
         

    def update(self, canvas):
        pygame.draw.circle(canvas.screen, self.COLOR, self.pos[::-1], self.size)
        pygame.draw.circle(canvas.screen, (0,0,0), self.pos[::-1], min(1, self.size//5))
        
        

class Obstacle():
    def __init__ (self, pos, color='r', size=[10,20], step_size=0):
        self.pos = pos
        self.size = size
        self.color = color
        self.step_size = step_size

        self.COLOR = None
        if self.color == 'r':
            self.COLOR = (255,0,0)
        elif self.color == 'g':
            self.COLOR = (0, 255, 0)
        elif self.color == 'b':
            self.COLOR = (0, 0, 255)
        else:
            self.COLOR = (255, 255, 255)

#         self.rect = patches.Rectangle(pos, self.size[0], self.size[1], linewidth=0,edgecolor='blue',facecolor='blue')
        

    def move(self, action, step_size=None, canvas=None):
        if step_size is None:
            step_size = self.step_size
        else:
            self.step_size = step_size
        
        if action == 'up':
            motion = np.array([-step_size,0])
        elif action == 'down':
            motion = np.array([step_size, 0])
        elif action == 'left':
            motion = np.array([0, -step_size])
        else:
            motion = np.array([0, step_size])

        self.pos += motion

        if self.pos[0]-self.size[0] < 0:
            self.pos[0] = 0+self.size[0]
        if self.pos[0]+self.size[0] > canvas.height-1:
            self.pos[0] = min(self.pos[0], canvas.height-self.size[0]-1)
        if self.pos[1]-self.size[1] < 0:
            self.pos[1] = 0+self.size[1]
        if self.pos[1]+self.size[1] > canvas.width-1:
            self.pos[1] = min(self.pos[1], canvas.width-self.size[1]-1)

    def update(self, canvas):
#         self.rect.set_xy(self.pos)
#         ax.add_patch(self.rect)

        pygame.draw.rect(canvas.screen, self.COLOR, [self.pos[1], self.pos[0], self.size[1], self.size[0]])
        
        
class Drone():
    def __init__(self, pos, color='b', size=30, step_size=20):
        self.pos = np.array(pos)
        self.color = color
        self.size = size
        self.step_size = step_size

        self.COLOR = None
        self.view_COLOR = None
        if self.color == 'r':
            self.COLOR = (255,0,0)
            self.view_COLOR = (192,0,0)
        elif self.color == 'g':
            self.COLOR = (0, 255, 0)
            self.view_COLOR = (0,192,0)
        elif self.color == 'b':
            self.COLOR = (0, 0, 255)
            self.view_COLOR = (0,0,192)
        else:
            self.COLOR = (255, 255, 255)
            self.view_COLOR = (192,192,192)

        # self.view =  patches.Rectangle(self.pos, self.size, self.size, linewidth=0, edgecolor=self.color, facecolor=self.color)
        # self.view.set_alpha(0.3)

        # self.circle = patches.Circle(self.pos, self.size//10, fc='black')
        # self.circle.set_alpha(1.0)

    def move(self, action, step_size=2, canvas=None):
        if step_size is None:
            step_size = self.step_size
        else:
            self.step_size = step_size
        
        if action == 'up':
            motion = np.array([-step_size,0])
        elif action == 'down':
            motion = np.array([step_size, 0])
        elif action == 'left':
            motion = np.array([0, -step_size])
        else:
            motion = np.array([0, step_size])

        self.pos += motion

        if self.pos[0] < 0:
            self.pos[0] = 0 
        if self.pos[0] + self.size > canvas.height-1:
            self.pos[0] = min(self.pos[0], canvas.height - self.size- 1)
        if self.pos[1] < 0:
            self.pos[1] = 0 
        if self.pos[1] + self.size > canvas.width-1:
            self.pos[1] = min(self.pos[1], canvas.width - self.size - 1)
            
        # if self.pos[0] < 0:
        #     self.pos[0] = 0 
        # if self.pos[0] >= canvas.width-1:
        #     self.pos[0] = canvas.width - 1
        # if self.pos[1] < 0:
        #     self.pos[1] = 0 
        # if self.pos[1] >= canvas.height-1:
        #     self.pos[1] = canvas.height - 1

         

    def update(self, canvas):
        pygame.draw.rect(canvas.screen, self.view_COLOR, [self.pos[1], self.pos[0], self.size, self.size])
        # pygame.gfxdraw.box(canvas.screen, self.COLOR+(128,), [self.pos[0], self.pos[1], self.size, self.size])
        pygame.draw.circle(canvas.screen, self.COLOR, self.pos[::-1], min(1,self.size//5))
        
