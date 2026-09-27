import torch
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from scenarios.building_v2 import BuildingV2Scenario
from envs.evacuation_env import EvacuationEnv


SEED = 0

INITIAL_COUNTS = {
    "A": 40,
    "B": 20,
    "C": 20,
    "D": 20,
}

SURGE_ROOM = "C"
SURGE_TIME = 10.0
SURGE_COUNT = 40

INCIDENTS = [
    "C_exit",
    "B_route_08",
    "bottom_loop_left",
]

CONTROL_INTERVAL = 10.0
MAX_TIME = 180.0

N_ENVS = 4

# 4 envs × 64 steps = 256 rollout samples,
# matching the old 1 env × 256 steps.
N_STEPS = 64
BATCH_SIZE = 64

TOTAL_TIMESTEPS = 16384

MODEL_PATH = Path(
    "models/ppo_threeway_incident_v3b_4env.zip"
)

CHECKPOINT_DIR = Path(
    "models/incident_threeway_ppo_4env_checkpoints"
)


def make_env(rank):
    def _init():
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

        # Give each worker a distinct deterministic RNG stream.
        env.reset(seed=SEED + rank)

        return Monitor(env)

    return _init


def main():
    # Keep neural-network computation single-threaded.
    # The parallelism comes from 4 independent JuPedSim processes.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    env = SubprocVecEnv(
        [
            make_env(rank)
            for rank in range(N_ENVS)
        ],
        start_method="forkserver",
    )

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

    # Callback is invoked once per vectorised step.
    # Each vectorised step contributes N_ENVS timesteps.
    #
    # 2048 // 4 = 512 callback calls
    # -> checkpoint every 2048 total environment timesteps.
    checkpoint_callback = CheckpointCallback(
        save_freq=2048 // N_ENVS,
        save_path=str(CHECKPOINT_DIR),
        name_prefix="ppo_incident_4env",
        verbose=2,
    )

    print("===== V3-B PARALLEL PPO TRAINING =====")
    print(f"Workers: {N_ENVS}")
    print(
        f"Rollout size: "
        f"{N_ENVS} × {N_STEPS} = "
        f"{N_ENVS * N_STEPS}"
    )
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Total timesteps: {TOTAL_TIMESTEPS}")
    print()

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_callback,
        )

        model.save(MODEL_PATH)

        print()
        print(
            f"Saved {MODEL_PATH} at "
            f"{model.num_timesteps} timesteps"
        )

    finally:
        env.close()


if __name__ == "__main__":
    main()