import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque

class GridEvacEnv(gym.Env):
    def __init__(self,start_positions=None,max_step=1000):
        super().__init__()
        self.grid = np.array([
            [1,1,1,1,1,1,1,1,1],
            [1,0,0,0,0,0,0,0,1],
            [1,0,1,1,0,1,1,0,1],
            [1,0,0,0,0,0,0,0,1],
            [1,1,1,1,0,1,1,1,1],
            [1,0,0,0,0,0,0,0,1],
            [1,0,1,0,1,0,1,0,1],
            [2,0,0,0,1,0,0,0,2],
            [1,1,1,1,1,1,1,1,1]
        ])

        if start_positions is None:
            self.start_positions = [   
                (1,1), 
                (1,2), 
                (1,3)  
            ]  
        else:  
            self.start_positions = start_positions.copy() 
        rows, cols = self.grid.shape
        assert len(self.start_positions) == len(set(self.start_positions)) , "pedestrians cannot start in the same cell"
        assert all(0<=row < rows and 0<= col <cols for row,col in self.start_positions), "starting positions must be inside grid"
        assert not( 1 in [self.grid[pos] for pos in self.start_positions]), "Pedestrians cannot start on walls"

        self.action_space= spaces.Discrete(2)
        self.max_step = max_step
        self.junction = (4,4)
        self.left_exit = (7,0)
        self.right_exit = (7,8)
        self.observation_space= spaces.MultiDiscrete([len(self.start_positions)+1]*3)
        self.distance_map_junction = self.build_distance_map((4,4))
        self.distance_map_left_exit = self.build_distance_map(self.left_exit)
        self.distance_map_right_exit = self.build_distance_map(self.right_exit)



    def build_distance_map(self,exitPos):
        rows, cols = self.grid.shape
        distance_map = np.full((rows,cols),np.inf)
        exitRow, exitCol = exitPos
        build_queue = deque([(exitRow,exitCol,0)])
        distance_map[exitRow,exitCol] = 0
        while len(build_queue) != 0:
            row,col,dist = build_queue.pop()

            # If left neighbor cell is not wall and is not visited and not out of bounds
            if (0 <= col-1 < cols) and (self.grid[row,col-1] != 1) and (distance_map[row,col-1] ==np.inf):
                build_queue.appendleft((row,col-1,dist+1))
                distance_map[row,col-1] = dist+1

            # If right neighbor cell is not wall and is not visited 
            if  (0 <= col+1 < cols) and (self.grid[row,col+1] != 1) and (distance_map[row,col+1] ==np.inf):
                build_queue.appendleft((row,col+1,dist+1))
                distance_map[row,col+1] = dist+1

            # If up neighbor cell is not wall and is not visited
            if (0 <= row-1 < rows) and (self.grid[row-1,col] != 1) and (distance_map[row-1,col] ==np.inf):
                build_queue.appendleft((row-1,col,dist+1))
                distance_map[row-1,col] = dist+1

            # If down neighbor cell is not wall and is not visited
            if (0 <= row+1 < rows) and (self.grid[row+1,col] != 1) and (distance_map[row+1,col] ==np.inf ):
                build_queue.appendleft((row+1,col,dist+1))
                distance_map[row+1,col] = dist+1
        return distance_map

    def choose_pedestrian_move(self,position,distance_map):
        row,col = position
        candidates=[position]
        #  Check left neighbor
        if (self.grid[row,col-1] != 1) and (row,col-1) not in self.pedestrian_positions and (distance_map[row,col-1]<distance_map[position]):
            candidates.append((row,col-1))
        # check right neighbor
        if (self.grid[row,col+1] != 1) and (row,col+1) not in self.pedestrian_positions and (distance_map[row,col+1]<distance_map[position]):
            candidates.append((row,col+1))
        # check up neighbor
        if (self.grid[row-1,col] != 1) and (row-1,col) not in self.pedestrian_positions and (distance_map[row-1,col]<distance_map[position]):
            candidates.append((row-1,col))
        # check down neighbor
        if (self.grid[row+1,col] != 1) and (row+1,col) not in self.pedestrian_positions and (distance_map[row+1,col]<distance_map[position]):
            candidates.append((row+1,col))


        min_distance = min(distance_map[pos] for pos in candidates)
        best_candidates = [pos for pos in candidates if distance_map[pos] == min_distance]
        chosen_index = self.np_random.integers(len(best_candidates))
        return best_candidates[chosen_index]    

        


    
    def reset(self, seed=None,options=None):
        super().reset(seed=seed)
        self.pedestrian_positions = self.start_positions.copy()
        self.guidance_states = [None for _ in range(len(self.pedestrian_positions))]

        self.current_step = 0
        obs= np.array([self.guidance_states.count(None),self.guidance_states.count(0),self.guidance_states.count(1)])
        info = {}
        return (obs,info)
    def step(self,action):
        assert self.action_space.contains(action)
        terminated = False
        truncated =  False
        order = list(range(len(self.pedestrian_positions)))
        self.np_random.shuffle(order)
        escaped_left =0
        escaped_right = 0
        for i in order:
            if self.pedestrian_positions[i] is None:
                continue
            guide_state = self.guidance_states[i]
            if guide_state == None:
                distance_map = self.distance_map_junction
            elif guide_state == 0:
                distance_map = self.distance_map_left_exit
            else:
                distance_map = self.distance_map_right_exit

            new_pos = self.choose_pedestrian_move(self.pedestrian_positions[i],distance_map)
            if new_pos == self.junction and self.guidance_states[i] is None:
                self.guidance_states[i] = action

            if self.grid[new_pos] == 2:
                self.pedestrian_positions[i] = None
                if self.guidance_states[i] == 0:
                    escaped_left +=1
                elif self.guidance_states[i] == 1:
                    escaped_right +=1
            else:
                self.pedestrian_positions[i] = new_pos
    
        self.guidance_states = [self.guidance_states[i] for i in range(len(self.pedestrian_positions)) if self.pedestrian_positions[i] is not None]
        self.pedestrian_positions = [pos for pos in self.pedestrian_positions if pos is not None]

        self.current_step +=1
        terminated = len(self.pedestrian_positions) ==0
        truncated = self.current_step >= self.max_step
        reward = -1

        obs= np.array([self.guidance_states.count(None),self.guidance_states.count(0),self.guidance_states.count(1)])
        info = {"escaped_left": escaped_left,"escaped_right":escaped_right }
        return obs,reward,terminated,truncated,info
        
        
