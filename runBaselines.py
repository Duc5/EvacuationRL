from scenarios.two_exit_v1 import TwoExitV1Scenario
from envs.evacuation_env import EvacuationEnv


BASELINES = {
    0: "LL",
    1: "LR",
    2: "RL",
    3: "RR",
}


def run_baseline(action, seed=0):
    scenario = TwoExitV1Scenario()

    env = EvacuationEnv(
        scenario=scenario,
        record=False,
    )

    obs, info = env.reset(seed=seed)

    terminated = False
    truncated = False

    total_reward = 0.0
    steps = 0

    while not terminated and not truncated:

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1

    result = {
        "action": action,
        "policy": BASELINES[action],
        "evacuation_time": info["elapsed_time"],
        "total_reward": total_reward,
        "steps": steps,
        "remaining_agents": info["remaining_agents"],
        "terminated": terminated,
        "truncated": truncated,
    }

    env.close()

    return result


def main():

    results = []

    for action in BASELINES:
        result = run_baseline(
            action=action,
            seed=0,
        )

        results.append(result)

    print("\n===== BASELINE RESULTS =====\n")

    for result in results:

        print(
            f"{result['policy']} "
            f"(action {result['action']})"
        )

        print(
            f"  Evacuation time : "
            f"{result['evacuation_time']:.2f} s"
        )

        print(
            f"  Total reward    : "
            f"{result['total_reward']:.2f}"
        )

        print(
            f"  Gym steps       : "
            f"{result['steps']}"
        )

        print(
            f"  Remaining agents: "
            f"{result['remaining_agents']}"
        )

        print(
            f"  Terminated      : "
            f"{result['terminated']}"
        )

        print(
            f"  Truncated       : "
            f"{result['truncated']}"
        )

        print()

    print("============================")


if __name__ == "__main__":
    main()