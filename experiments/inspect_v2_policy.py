import numpy as np
import torch

from stable_baselines3 import PPO

from scenarios.building_v2 import BuildingV2Scenario
from envs.evacuation_env import EvacuationEnv


POPULATION_SCENARIOS = {
    "balanced": {"A": 25, "B": 25, "C": 25, "D": 25},
    "A_extreme": {"A": 49, "B": 17, "C": 17, "D": 17},
    "B_extreme": {"A": 19, "B": 42, "C": 20, "D": 19},
    "C_extreme": {"A": 19, "B": 20, "C": 42, "D": 19},
    "D_extreme": {"A": 22, "B": 21, "C": 21, "D": 36},
}


def print_policy_distribution(model, env, observation):
    obs_tensor, _ = model.policy.obs_to_tensor(observation)

    with torch.no_grad():
        distribution = model.policy.get_distribution(obs_tensor)

    categorical_distributions = distribution.distribution

    total_entropy = 0.0

    for junction, categorical in zip(
        env.junction_names, categorical_distributions
    ):
        probs = categorical.probs.cpu().numpy()[0]
        choices = env.scenario.guidance_choices[junction]

        entropy = -np.sum(
            probs * np.log(probs + 1e-12)
        )

        total_entropy += entropy

        print(f"{junction}:")

        for choice, probability in zip(choices, probs):
            print(f"  {choice:<3} {probability:.4f}")

        print(f"  entropy = {entropy:.4f}")
        print()

    print(f"Total entropy: {total_entropy:.4f}")
    print(f"Maximum entropy: {np.log(48):.4f}")


def main():
    scenario = BuildingV2Scenario(
        room_counts=POPULATION_SCENARIOS["balanced"]
    )

    env = EvacuationEnv(
        scenario=scenario,
        control_interval=5.0,
        max_time=180.0,
    )

    model = PPO.load(
        "models/ppo_v2_1k5sInterval",
        env=env,
        device="cpu",
    )

    observation, info = env.reset(seed=0)

    print("===== INITIAL POLICY =====")
    print_policy_distribution(model, env, observation)

    print("\n===== DETERMINISTIC EPISODE =====")

    terminated = False
    truncated = False

    while not terminated and not truncated:
        print(f"\nTime: {info['elapsed_time']:.2f}")

        print_policy_distribution(
            model, env, observation
        )

        action, _ = model.predict(
            observation,
            deterministic=True,
        )

        guidance = env._decode_action(action)

        print("Chosen:", guidance)

        observation, reward, terminated, truncated, info = env.step(action)

    print()
    print("Final evacuation time:", info["elapsed_time"])

    env.close()


if __name__ == "__main__":
    main()