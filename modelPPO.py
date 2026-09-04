from stable_baselines3 import PPO
from jupedsim_evac_env import JuPedSimEvacEnv

env = JuPedSimEvacEnv(record=False)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    seed=0
)

model.learn(
    total_timesteps=10_000
)

eval_env = JuPedSimEvacEnv(record=False)

obs, info = eval_env.reset(seed=100)

terminated = False
truncated = False

while not (terminated or truncated):

    action, _ = model.predict(
        obs,
        deterministic=True
    )

    obs, reward, terminated, truncated, info = eval_env.step(action)

    print(
        f"time={info['elapsed_time']:.2f}",
        f"action={action}"
    )

print("Evacuation:", info["elapsed_time"])