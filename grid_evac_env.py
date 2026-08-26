import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque

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
        self.start_positions = [
            (1,1),
            (1,2),
            (1,3)
        ]
        
        self.action_space= spaces.Discrete(4)
        self.max_step = 1000
        rows, cols = self.grid.shape
        self.observation_space= spaces.MultiDiscrete([rows,cols])
        self.build_distance_map()

    def build_distance_map(self):
        rows, cols = self.grid.shape
        self.distance_map = np.full((rows,cols),np.inf)
        build_queue = deque([(rows-2,cols-2,0)])
        self.distance_map[rows-2,cols-2] = 0
        while len(build_queue) != 0:
            row,col,dist = build_queue.pop()

            # If left neighbor cell is not wall and is not visited
            if (self.grid[row,col-1] != 1) and (self.distance_map[row,col-1] ==np.inf):
                build_queue.appendleft((row,col-1,dist+1))
                self.distance_map[row,col-1] = dist+1

            # If right neighbor cell is not wall and is not visited 
            if (self.grid[row,col+1] != 1) and (self.distance_map[row,col+1] ==np.inf):
                build_queue.appendleft((row,col+1,dist+1))
                self.distance_map[row,col+1] = dist+1

            # If up neighbor cell is not wall and is not visited
            if (self.grid[row-1,col] != 1) and (self.distance_map[row-1,col] ==np.inf):
                build_queue.appendleft((row-1,col,dist+1))
                self.distance_map[row-1,col] = dist+1

            # If down neighbor cell is not wall and is not visited
            if (self.grid[row+1,col] != 1) and (self.distance_map[row+1,col] ==np.inf):
                build_queue.appendleft((row+1,col,dist+1))
                self.distance_map[row+1,col] = dist+1


    def choose_pedestrian_move(self,position):
        row,col = position
        chosen_position = position

        #  Check left neighbor
        if (self.grid[row,col-1] != 1) and (row,col-1) not in self.pedestrian_positions:
            if self.distance_map[row,col-1] < self.distance_map[chosen_position]:
                chosen_position = (row,col-1)
        # check right neighbor
        if (self.grid[row,col+1] != 1) and (row,col+1) not in self.pedestrian_positions:
            if self.distance_map[row,col+1] < self.distance_map[chosen_position]:
                chosen_position = (row,col+1)
        # check up neighbor
        if (self.grid[row-1,col] != 1) and (row-1,col) not in self.pedestrian_positions:
            if self.distance_map[row-1,col] < self.distance_map[chosen_position]:
                chosen_position = (row-1,col)
        # check down neighbor
        if (self.grid[row+1,col] != 1) and (row+1,col) not in self.pedestrian_positions:
            if self.distance_map[row+1,col] < self.distance_map[chosen_position]:
                chosen_position = (row+1,col)
        return chosen_position
        


    
    def reset(self, seed=None,options=None):
        super().reset(seed=seed)
        self.pedestrian_positions = self.start_positions.copy()
        self.current_step = 0
        obs = np.array(self.pedestrian_positions)
        info = {}
        return (obs,info)
    def step(self,action):
        terminated = False
        truncated =  False
        order = list(range(len(self.pedestrian_positions)))
        self.np_random.shuffle(order)
        for i in order:
            if self.pedestrian_positions[i] is None:
                continue
            new_pos = self.choose_pedestrian_move(self.pedestrian_positions[i])
            if self.grid[new_pos] == 2:
                self.pedestrian_positions[i] = None
            else:
                self.pedestrian_positions[i] = new_pos

        self.pedestrian_positions = [pos for pos in self.pedestrian_positions if pos is not None]


        self.current_step +=1
        terminated = len(self.pedestrian_positions) ==0
        truncated = self.current_step >= self.max_step
        reward = -1
        obs= self.pedestrian_positions
        info = {}
        return obs,reward,terminated,truncated,info
        
        
