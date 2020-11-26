import pygame
import numpy as np

class Player():
    def __init__(self, pos, color, size):
        self.pos = np.array(pos)
        self.color = color 
        self.size = size
        self.prev_action = None

    def move(self, action, step_size=1):
        global canvas
        if action == 'up':
            motion = np.array([-step_size,0])
        elif action == 'down':
            motion = np.array([step_size, 0])
        elif action == 'left':
            motion = np.array([0, -step_size])
        else:
            motion = np.array([0, step_size])

        self.pos += motion

        if self.pos[0]-self.size < 0:
            self.pos[0] = 0+self.size
        if self.pos[0]+self.size >= canvas.width-1:
            self.pos[0] = canvas.width-self.size-1
        if self.pos[1]-self.size < 0:
            self.pos[1] = 0+self.size
        if self.pos[1]+self.size >= canvas.height-1:
            self.pos[1] = canvas.height-self.size-1
        # if self.pos[0] < 0:
        #     self.pos[0] = 0 
        # if self.pos[0] >= canvas.width-1:
        #     self.pos[0] = canvas.width - 1
        # if self.pos[1] < 0:
        #     self.pos[1] = 0 
        # if self.pos[1] >= canvas.height-1:
        #     self.pos[1] = canvas.height - 1
         

    def update(self, canvas):
        if self.color == 'r':
            COLOR = (255,0,0)
        elif self.color == 'g':
            COLOR = (0, 255, 0)
        elif self.color == 'b':
            COLOR = (0, 0, 255)
        else:
            COLOR = (255, 255, 255)
        
        
        pygame.draw.circle(canvas.screen, COLOR, self.pos, self.size)
        pygame.draw.circle(canvas.screen, (0,0,0), self.pos, min(1, self.size//5))
        
        

class Obstacle():
    def __init__ (self, pos, size):
        self.pos = pos
        self.size = size

#         self.rect = patches.Rectangle(pos, self.size[0], self.size[1], linewidth=0,edgecolor='blue',facecolor='blue')
        

    def move(self, action, step_size=0):
        global canvas
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
        if self.pos[0]+self.size[0] >= canvas.width-1:
            self.pos[0] = canvas.width-self.size[0]-1
        if self.pos[1]-self.size[1] < 0:
            self.pos[1] = 0+self.size[1]
        if self.pos[1]+self.size[1] >= canvas.height-1:
            self.pos[1] = canvas.height-self.size[1]-1

    def update(self, canvas):
#         self.rect.set_xy(self.pos)
#         ax.add_patch(self.rect)

        pygame.draw.rect(canvas.screen, (255,0,0), [self.pos[0], self.pos[1], self.size[0], self.size[1]])
        
        
class Drone():
    def __init__(self, pos, color, size):
        self.pos = np.array(pos)
        self.color = color 
        self.size = size

        self.view =  patches.Rectangle(self.pos, self.size, self.size, linewidth=0, edgecolor=self.color, facecolor=self.color)
        self.view.set_alpha(0.3)

        self.circle = patches.Circle(self.pos, self.size//10, fc='black')
        self.circle.set_alpha(1.0)

    def move(self, action, step_size=2):
        global canvas
        if action == 'up':
            motion = np.array([-step_size,0])
        elif action == 'down':
            motion = np.array([step_size, 0])
        elif action == 'left':
            motion = np.array([0, -step_size])
        else:
            motion = np.array([0, step_size])

        self.pos += motion

        if self.pos[0]-self.size < 0:
            self.pos[0] = 0 + self.size//2
        if self.pos[0] + self.size//2 >= canvas.width-1:
            self.pos[0] = canvas.width - self.size//2 - 1
        if self.pos[1] - self.size//2 < 0:
            self.pos[1] = 0 + self.size//2
        if self.pos[1] + self.size//2 >= canvas.height-1:
            self.pos[1] = canvas.height - self.size//2 - 1
        
        # if self.pos[0] < 0:
        #     self.pos[0] = 0 
        # if self.pos[0] >= canvas.width-1:
        #     self.pos[0] = canvas.width - 1
        # if self.pos[1] < 0:
        #     self.pos[1] = 0 
        # if self.pos[1] >= canvas.height-1:
        #     self.pos[1] = canvas.height - 1

         

    def update(self, canvas):
        self.circle.center = self.pos
        self.view.set_xy((self.pos[0] - self.size//2, self.pos[1] - self.size//2))
        

        ax.add_patch(self.view)
        ax.add_patch(self.circle)

        return self.circle

    
