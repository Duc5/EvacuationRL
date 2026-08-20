import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TwoExitEvacEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.initial_population = 20
        self.pLeft = 0.5
        self.pRight = 0.5
        self.maxStep = 100
        self.action_space= spaces.Discrete(2)
        self.observation_space= spaces.MultiDiscrete([self.initial_population+1]*3)
        obs, info = self.reset()
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
        

            


env = TwoExitEvacEnv()
obs,info = env.reset()
print(obs)
print(env.observation_space.contains(obs))
