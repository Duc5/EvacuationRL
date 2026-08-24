import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TwoExitEvacEnv(gym.Env):
    def __init__(self,
                 initial_population=20,
                 pLeft=0.5,
                 pRight=0.5,
                 maxStep=100):
        super().__init__()
        self.initial_population = initial_population
        self.pLeft = pLeft
        self.pRight = pRight
        self.maxStep = maxStep
        self.action_space= spaces.Discrete(2)
        self.observation_space= spaces.MultiDiscrete([self.initial_population+1]*3)
    def reset(self, seed=None,options=None):
        super().reset(seed=seed)
        self.input_people = self.initial_population
        self.left_people = 0
        self.right_people = 0
        self.current_step = 0
        obs = np.array([self.input_people,self.left_people,self.right_people])
        info = {}
        return (obs,info)
    def step(self,action):
        assert self.action_space.contains(action)
        if self.input_people >0:
            self.input_people -=1
            if action == 0:
                self.left_people +=1
            else:
                self.right_people +=1
        if self.left_people > 0:
            rand = self.np_random.random()
            if rand < self.pLeft:
                self.left_people -=1
        if self.right_people >0:
            rand = self.np_random.random()
            if rand < self.pRight:
                self.right_people -=1
        self.current_step +=1
        obs = np.array([self.input_people,self.left_people,self.right_people])
        reward = (self.input_people + self.left_people + self.right_people)*-1        
        terminated = (self.input_people ==0 and self.right_people==0 and self.left_people ==0)
        truncated  = self.current_step >= self.maxStep
        info = {}
        return obs,reward,terminated,truncated,info


            


