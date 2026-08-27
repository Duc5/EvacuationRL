import numpy as np
from grid_evac_env import GridEvacEnv

env = GridEvacEnv()


q_table = np.load("q_tableV0.4.3seed3.npy")
epsilon = 0.1
alpha = 0.1
gamma = 0.09
episode_rewards =[]
episode_steps = []
for i in range(10000):
    state,info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0
    while not(terminated or truncated):
            
        state_tuple = tuple(state)
        action = np.argmax(q_table[state_tuple])

        next_state,reward,terminated,truncated,info = env.step(action)
        total_reward += reward
        state = next_state
    episode_rewards.append(total_reward)
    episode_steps.append(env.current_step)
    print(f"Ep {i}: average reward {np.mean(episode_rewards)} average steps {np.mean(episode_steps)}")
