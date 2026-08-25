import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GridEvacEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.grid = np.array([
                [1,1,1,1,1,1,1],
                [1,0,0,0,0,0,1],
                [1,0,1,1,1,0,1],
                [1,0,0,0,0,0,1],
                [1,0,1,1,1,0,1],
                [1,0,0,0,0,2,1],
                [1,1,1,1,1,1,1]
            ])
        self.start_pos = (1,1)
        self.action_space= spaces.Discrete(4)
        self.max_step = 1000
        rows, col = self.grid.shape
        self.observation_space= spaces.MultiDiscrete([rows,col])
    def reset(self, seed=None,options=None):
        super().reset(seed=seed)
        self.pedestrian_pos = self.start_pos
        self.current_step = 0
        obs = np.array(self.pedestrian_pos)
        info = {}
        return (obs,info)
    def step(self,action):
        assert self.action_space.contains(action)
        terminated = False
        truncated =  False
        row, col = self.pedestrian_pos

        # Moving pedestrian
        if action == 0:
            new_pos = (row-1,col)
        elif action ==1:
            new_pos = (row,col+1)
        elif action == 2:
            new_pos = (row+1,col)
        else:
            new_pos = (row,col-1)

        # Valid move check
        rowBorder,colBorder = self.grid.shape
        if new_pos[0] < 0 or new_pos[0]>=rowBorder or new_pos[1] <0 or new_pos[1]>=colBorder:
            # invalid,out of border, but should not be checked for this grid cuz its wall surround border
            new_pos = self.pedestrian_pos
        elif self.grid[new_pos] == 1:
            # hit wall
            new_pos = self.pedestrian_pos
        elif self.grid[new_pos] == 2:
            self.pedestrian_pos = new_pos
            terminated = True

        else:
            # valid move
            self.pedestrian_pos = new_pos
        self.current_step +=1
        truncated = self.current_step >= self.max_step
        reward = -1
        obs= np.array(self.pedestrian_pos)
        info = {}
        return obs,reward,terminated,truncated,info
        

        