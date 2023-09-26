#Environment adapated (and modified to one player game and gym architecture) from github repository https://gist.github.com/vinothpandian/4337527

import gym
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
numpy.random.Generator = numpy.random._Generator
np.random.Generator = np.random._Generator
import random
import pygame, sys
from pygame.locals import *
#colors
WHITE = (255,255,255)
BLACK = (0,0,0)

class Custom_Pong(Env):
    
    def __init__(self):
        self.action_space = Discrete(3)
        
        #Initialising variables / Reset
        self.WIDTH = 160
        self.HEIGHT = 210      
        self.BALL_RADIUS = 5
        
        self.PAD_WIDTH = 4
        self.PAD_HEIGHT = 42
        self.HALF_PAD_WIDTH = self.PAD_WIDTH // 2
        self.HALF_PAD_HEIGHT = self.PAD_HEIGHT // 2
        
        self.ball_pos = [0,0]
        self.ball_vel = [0,0]
        self.paddle1_pos = [self.HALF_PAD_WIDTH - 1,self.HEIGHT//2]
        self.paddle2_pos = [self.WIDTH +1 - self.HALF_PAD_WIDTH,self.HEIGHT//2]
        self.paddle1_vel = 0
        self.paddle2_vel = 0
        self.score = 0
        
        high = np.zeros((210,160,3))+255
        low = np.zeros((210,160,3))
        self.observation_space = Box(low, high, dtype=np.float32)
        self.state = None
        pass
    
    
    def step(self, action):
        done=False
        # Apply action
        if action== 1:
            self.paddle2_vel = -4
        elif action == 2:
            self.paddle2_vel = 4
            
            
        # update paddle's vertical position, keep paddle on the screen
        if self.paddle1_pos[1] > self.HALF_PAD_HEIGHT and self.paddle1_pos[1] < self.HEIGHT - self.HALF_PAD_HEIGHT:
            self.paddle1_pos[1] += self.paddle1_vel
        elif self.paddle1_pos[1] == self.HALF_PAD_HEIGHT and self.paddle1_vel > 0:
            self.paddle1_pos[1] += self.paddle1_vel
        elif self.paddle1_pos[1] == self.HEIGHT - self.HALF_PAD_HEIGHT and self.paddle1_vel < 0:
            self.paddle1_pos[1] += self.paddle1_vel

        if self.paddle2_pos[1] > self.HALF_PAD_HEIGHT and self.paddle2_pos[1] < self.HEIGHT - self.HALF_PAD_HEIGHT:
            self.paddle2_pos[1] += self.paddle2_vel
        elif self.paddle2_pos[1] == self.HALF_PAD_HEIGHT and self.paddle2_vel > 0:
            self.paddle2_pos[1] += self.paddle2_vel
        elif self.paddle2_pos[1] == self.HEIGHT - self.HALF_PAD_HEIGHT and self.paddle2_vel < 0:
            self.paddle2_pos[1] += self.paddle2_vel

        #update ball
        self.ball_pos[0] += int(self.ball_vel[0])
        self.ball_pos[1] += int(self.ball_vel[1])

        # Calculate reward
        #ball collision check on top and bottom walls
        if int(self.ball_pos[1]) <= self.BALL_RADIUS:
            self.ball_vel[1] = - self.ball_vel[1]
        if int(self.ball_pos[1]) >= self.HEIGHT + 1 - self.BALL_RADIUS:
            self.ball_vel[1] = -self.ball_vel[1]

        #ball collison check on gutters or paddles
        if int(self.ball_pos[0]) <= self.BALL_RADIUS + self.PAD_WIDTH and int(self.ball_pos[1]) in range(self.paddle1_pos[1] - 16*self.HALF_PAD_HEIGHT,self.paddle1_pos[1] + 16*self.HALF_PAD_HEIGHT,1):
            self.ball_vel[0] = -self.ball_vel[0]

        if int(self.ball_pos[0]) >= self.WIDTH + 1 - self.BALL_RADIUS - self.PAD_WIDTH and int(self.ball_pos[1]) in range(self.paddle2_pos[1] - self.HALF_PAD_HEIGHT,self.paddle2_pos[1] + self.HALF_PAD_HEIGHT,1):
            self.ball_vel[0] = -self.ball_vel[0]
            self.score += 1
        elif int(self.ball_pos[0]) >= self.WIDTH + 1 - self.BALL_RADIUS - self.PAD_WIDTH:
            done=True
    
        self.state=self.observation_space
        # Check if shower is done
        info = {}
        # Return step information
        if(self.score>100):
            done=True
        return np.array(self.state, dtype=np.float32), self.score, done, info
    
    def render(self, mode='human'):
        
        pygame.init()
        pygame.display.set_caption('Ping-Pong')
        fps = pygame.time.Clock()
        
        canvas = pygame.display.set_mode((self.WIDTH, self.HEIGHT), 0, 32)
        canvas.fill(BLACK)
        #draw paddles and ball
        pygame.draw.circle(canvas, WHITE, self.ball_pos, 5, 0)
        pygame.draw.polygon(canvas, WHITE, [[self.paddle1_pos[0] - self.HALF_PAD_WIDTH, self.paddle1_pos[1] - 8*self.HALF_PAD_HEIGHT], [self.paddle1_pos[0] - self.HALF_PAD_WIDTH, self.paddle1_pos[1] + 8*self.HALF_PAD_HEIGHT], [self.paddle1_pos[0] + self.HALF_PAD_WIDTH, self.paddle1_pos[1] + 8*self.HALF_PAD_HEIGHT], [self.paddle1_pos[0] + self.HALF_PAD_WIDTH, self.paddle1_pos[1] - 8*self.HALF_PAD_HEIGHT]], 0)
        pygame.draw.polygon(canvas, WHITE, [[self.paddle2_pos[0] - self.HALF_PAD_WIDTH, self.paddle2_pos[1] - self.HALF_PAD_HEIGHT], [self.paddle2_pos[0] - self.HALF_PAD_WIDTH, self.paddle2_pos[1] + self.HALF_PAD_HEIGHT], [self.paddle2_pos[0] + self.HALF_PAD_WIDTH, self.paddle2_pos[1] + self.HALF_PAD_HEIGHT], [self.paddle2_pos[0] + self.HALF_PAD_WIDTH, self.paddle2_pos[1] - self.HALF_PAD_HEIGHT]], 0)
        myfont1 = pygame.font.SysFont("Segoe UI", 20)
        label1 = myfont1.render("Dynamic Reward "+str(self.score), 1, (255,255,0))
        canvas.blit(label1, (50,20))
        pygame.display.update()
        fps.tick(60)
        self.observation_space = np.reshape(pygame.surfarray.array3d(canvas), (210,160,3), order='C')
        pass
    
    def reset(self):
        #Initialising variables / Reset
        self.half_pad_width = self.PAD_WIDTH // 2
        self.half_pad_heigth = self.PAD_HEIGHT // 2
        self.ball_pos = [0,0]
        self.ball_vel = [0,0]
        
        self.paddle1_pos = [self.HALF_PAD_WIDTH - 1,self.HEIGHT//2]
        self.paddle2_pos = [self.WIDTH +1 - self.HALF_PAD_WIDTH, self.HEIGHT//2]
        self.paddle1_vel = 0
        self.paddle2_vel = 0
        self.score = 0
        
        #Throwing ball forward
        self.ball_pos = [self.WIDTH//2,self.HEIGHT//2]
        horz = random.randrange(2,4)
        vert = random.randrange(1,3)
        self.ball_vel = [-horz,-vert]
        
        pass
    
    def close(self):
        pygame.quit()