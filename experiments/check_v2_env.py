from gymnasium.utils.env_checker import check_env

from scenarios.building_v2 import BuildingV2Scenario
from envs.evacuation_env import EvacuationEnv


POPULATION_SCENARIOS = {
    "balanced": {"A": 25, "B": 25, "C": 25, "D": 25},
    "A_extreme": {"A": 49, "B": 17, "C": 17, "D": 17},
    "B_extreme": {"A": 19, "B": 42, "C": 20, "D": 19},
    "C_extreme": {"A": 19, "B": 20, "C": 42, "D": 19},
    "D_extreme": {"A": 22, "B": 21, "C": 21, "D": 36},
}


def main():
    scenario = BuildingV2Scenario(
        room_counts={
            "A": 25,
            "B": 25,
            "C": 25,
            "D": 25,
        }
    )

    env = EvacuationEnv(
        scenario=scenario,
        control_interval=5.0,
        max_time=180.0,
    )

    try:
        # ----------------------------------------------------------
        # 1. Gymnasium API validation
        # ----------------------------------------------------------

        print("===== CHECK_ENV =====")
        check_env(env, skip_render_check=True)
        print("check_env PASSED")

        # ----------------------------------------------------------
        # 2. Inspect spaces
        # ----------------------------------------------------------

        print("\n===== SPACES =====")
        print("Action space:", env.action_space)
        print("Observation space:", env.observation_space)
        print("Number of plans:", len(env.guidance_plans))
        for i, plan in enumerate(env.guidance_plans):
            print(i, plan)
        # ----------------------------------------------------------
        # 3. Reset and inspect observation
        # ----------------------------------------------------------

        observation, info = env.reset(seed=0)

        print("\n===== RESET =====")
        print("Population:", info["population_name"])
        print("Elapsed time:", info["elapsed_time"])
        print("Remaining:", info["remaining_agents"])

        observation_names = [
            "room_A",
            "room_B",
            "room_C",
            "room_D",
            "target_exit_A",
            "target_exit_B",
            "target_exit_C",
            "target_J1",
            "target_J2",
            "target_J3",
            "near_exit_A",
            "near_exit_B",
            "near_exit_C",
            "time_fraction",
            "congestion_J1",
            "congestion_J2",
            "congestion_J3",
            "congestion_upper_corridor",
            "congestion_lower_corridor",
            "congestion_left_connector",
            "congestion_right_connector",
        ]
        print("\nObservation:")

        for name, value in zip(observation_names, observation):
            print(f"  {name:<15}: {value:.3f}")

        print("\nShape:", observation.shape)
        print(
            "Valid observation:",
            env.observation_space.contains(observation)
        )

        # ----------------------------------------------------------
        # 4. Take several random actions
        # ----------------------------------------------------------

        print("\n===== RANDOM STEPS =====")

        env.action_space.seed(0)

        for step in range(5):
            action = env.action_space.sample()
            guidance = env._decode_action(action)

            observation, reward, terminated, truncated, info = env.step(action)

            print(
                f"Step {step + 1}: "
                f"action={action} "
                f"guidance={guidance} "
                f"reward={reward:.3f} "
                f"time={info['elapsed_time']:.2f} "
                f"remaining={info['remaining_agents']}"
                f"rooms={observation[0:4]} "
                f"exits={observation[4:7]} "
                f"junctions={observation[7:10]} "
                f"near={observation[10:13]}"
                f"congestion={observation[14:20]}"
            )

            if terminated or truncated:
                print(
                    f"Episode ended: "
                    f"terminated={terminated}, "
                    f"truncated={truncated}"
                )
                break

    finally:
        env.close()


if __name__ == "__main__":
    main()