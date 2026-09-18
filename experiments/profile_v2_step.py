import time

from scenarios.building_v2 import BuildingV2Scenario
from envs.evacuation_env import EvacuationEnv


ROOM_COUNTS = {
    "A": 25,
    "B": 25,
    "C": 25,
    "D": 25,
}


def main():
    scenario = BuildingV2Scenario(room_counts=ROOM_COUNTS)

    env = EvacuationEnv(
        scenario=scenario,
        control_interval=5.0,
        max_time=180.0,
    )

    observation, info = env.reset(seed=0)

    # Fixed valid action. Exact action does not matter much for profiling.
    action = env.action_space.sample()

    # Warm-up step first.
    env.step(action)

    print("===== PROFILE =====")

    for i in range(5):
        start = time.perf_counter()

        observation, reward, terminated, truncated, info = env.step(action)

        env_step_time = time.perf_counter() - start
        profile = env.backend.last_profile

        print(f"\nStep {i + 1}")
        print(f"Iterations:       {profile['iterations']}")
        print(f"JuPedSim iterate: {profile['iterate_time']:.4f} s")
        print(f"Routing checks:   {profile['routing_time']:.4f} s")
        print(f"get_state():      {profile['state_time']:.4f} s")
        print(f"Backend total:    {profile['total_time']:.4f} s")
        print(f"Full env.step():  {env_step_time:.4f} s")

        iterate_pct = 100 * profile["iterate_time"] / profile["total_time"]
        routing_pct = 100 * profile["routing_time"] / profile["total_time"]
        state_pct = 100 * profile["state_time"] / profile["total_time"]

        print(
            f"Breakdown: iterate={iterate_pct:.1f}% | "
            f"routing={routing_pct:.1f}% | state={state_pct:.1f}%"
        )

        if terminated or truncated:
            break

    env.close()


if __name__ == "__main__":
    main()