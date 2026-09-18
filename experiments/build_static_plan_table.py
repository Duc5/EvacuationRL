import json
from pathlib import Path

from envs.static_plan_env import generate_acyclic_policies, policy_key
from scenarios.building_v2 import BuildingV2Scenario
from simulation.jupedsim_backend import JuPedSimBackend


POPULATION_SCENARIOS = {
    "balanced": {"A": 25, "B": 25, "C": 25, "D": 25},
    "A_extreme": {"A": 49, "B": 17, "C": 17, "D": 17},
    "B_extreme": {"A": 19, "B": 42, "C": 20, "D": 19},
    "C_extreme": {"A": 19, "B": 20, "C": 42, "D": 19},
    "D_extreme": {"A": 22, "B": 21, "C": 21, "D": 36},
}

SEED = 0
MAX_TIME = 180.0
OUTPUT_PATH = Path("results/static_plan_seed0.json")


def run_policy(room_counts, policy):
    scenario = BuildingV2Scenario(room_counts=dict(room_counts))

    backend = JuPedSimBackend(
        scenario=scenario,
        record=False,
    )

    try:
        backend.reset(seed=SEED)
        backend.apply_guidance(policy)
        state = backend.advance(MAX_TIME)

        return {
            "elapsed_time": float(state.elapsed_time),
            "remaining_agents": int(state.remaining_agents),
        }

    finally:
        backend.close()


def main():
    # Only used to obtain the routing graph and generate all valid plans.
    template = BuildingV2Scenario(
        room_counts=POPULATION_SCENARIOS["balanced"]
    )

    policies = generate_acyclic_policies(template)
    junction_names = list(template.junctions.keys())

    print(f"Total acyclic policies: {len(policies)}")
    print(
        f"Total simulations: "
        f"{len(policies) * len(POPULATION_SCENARIOS)}"
    )

    results = {}

    for population_name, room_counts in POPULATION_SCENARIOS.items():
        print(f"\n===== {population_name} =====")

        results[population_name] = {}

        for i, policy in enumerate(policies, start=1):
            result = run_policy(room_counts, policy)

            key = policy_key(
                policy,
                junction_names,
            )

            results[population_name][key] = result

            print(
                f"[{i:02}/{len(policies)}] "
                f"{key} | "
                f"time={result['elapsed_time']:.2f}s | "
                f"remaining={result['remaining_agents']}"
            )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(OUTPUT_PATH, "w") as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    print(f"\nSaved results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()