import gymnasium as gym


env = gym.make("FrozenLake-v1", is_slippery=False)

state, info = env.reset(seed=42)

print("Initial state:", state)

for step in range(10):
    action = env.action_space.sample()

    next_state, reward, terminated, truncated, info = env.step(action)

    print(
        "state:", state,
        "action:", action,
        "reward:", reward,
        "next state:", next_state
    )

    state = next_state

    if terminated or truncated:
        break

env.close()
