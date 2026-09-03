import numpy as np
from v0_prototype.evac_env import TwoExitEvacEnv
import os
env = TwoExitEvacEnv()

if os.path.exists("q_tableV0.2seed3.npy"):
    q_table = np.load("q_tableV0.2seed3.npy")
    print("Loaded existing Q table")
else:
    q_table = np.zeros((21,21,21,2))
    print("Created new Q table")
epsilon = 0.1
alpha = 0.1
gamma = 0.99
episode_rewards =[]
episode_steps = []
for i in range(100000):
    state,info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0
    while not(terminated or truncated):
            
        state_tuple = tuple(state)
        if np.random.random() <epsilon:
            action = env.action_space.sample()

        else:
            action = np.argmax(q_table[state_tuple])

        next_state,reward,terminated,truncated,info = env.step(action)
        total_reward += reward

        next_state_tuple = tuple(next_state)
        state_action_tuple = state_tuple + (action,)
        # q_update

        q_table[state_action_tuple] += alpha*(reward + gamma*(np.max(q_table[next_state_tuple])) - q_table[state_action_tuple])
        state = next_state
    episode_rewards.append(total_reward)
    episode_steps.append(env.current_step)
    print(f"Ep {i}: average reward {np.mean(episode_rewards)} average steps {np.mean(episode_steps)}")
np.save("q_tableV0.2seed3.npy",q_table)
