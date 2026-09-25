import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from scenarios.building_v2 import BuildingV2Scenario
from envs.evacuation_env import EvacuationEnv
from pathlib import Path

MODEL_PATH = Path("models/ppo_incident_v1.zip")

SEED = 0

INITIAL_COUNTS = {"A": 40,"B": 20,"C": 20,"D": 20,}

SURGE_ROOM = "C"
SURGE_TIME = 10.0
SURGE_COUNT = 40

INCIDENTS = ["B_connector","C_exit",]

CONTROL_INTERVAL = 10.0
MAX_TIME = 180.0

TOTAL_TIMESTEPS = 8192


def make_env():
    scenario = BuildingV2Scenario(
        room_counts=dict(INITIAL_COUNTS)
    )

    env = EvacuationEnv(
        scenario=scenario,
        control_interval=CONTROL_INTERVAL,
        max_time=MAX_TIME,
        dynamic_surge_rooms=[SURGE_ROOM],
        dynamic_surge_times=[SURGE_TIME],
        dynamic_surge_count=SURGE_COUNT,
        incidents=INCIDENTS,
    )

    return Monitor(env)


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    env = make_env()

    if MODEL_PATH.exists():
        print(f"Loading existing model: {MODEL_PATH}")

        model = PPO.load(
            MODEL_PATH,
            env=env,
            device="cpu",
        )
        print(f"Current timesteps: {model.num_timesteps}")
        print(f"n_steps: {model.n_steps}")
        print(f"batch_size: {model.batch_size}")

    else:
        print("No existing model found. Creating new PPO model.")

        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            seed=SEED,
            device="cpu",
            n_steps=256,
            batch_size=64,
        )
    checkpoint_callback = CheckpointCallback(
        save_freq=2048,
        save_path=f"models/incident_ppo_checkpoints",
        name_prefix="ppo_incident",
        verbose=2
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        reset_num_timesteps=False
    )

    model.save(MODEL_PATH)

    env.close()
    print(
        f"Saved {MODEL_PATH} at "
        f"{model.num_timesteps} timesteps"
    )

if __name__ == "__main__":
    main()