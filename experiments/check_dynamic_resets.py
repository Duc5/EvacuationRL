from scenarios.building_v2 import BuildingV2Scenario
from envs.evacuation_env import EvacuationEnv


INITIAL_COUNTS = {
    "A": 15,
    "B": 15,
    "C": 15,
    "D": 15,
}


def make_env():
    scenario = BuildingV2Scenario(room_counts=dict(INITIAL_COUNTS))

    return EvacuationEnv(
        scenario=scenario,
        control_interval=10.0,
        max_time=180.0,
        dynamic_surge_rooms=["A", "B", "C", "D"],
        dynamic_surge_times=[20.0, 30.0],
        dynamic_surge_count=40,
    )


def main():
    env = make_env()

    print("===== DIFFERENT SEEDS =====")

    seen = set()

    for seed in range(25):
        obs, info = env.reset(seed=seed)

        condition = (
            info["surge_room"],
            info["surge_time"],
        )

        seen.add(condition)

        print(
            f"seed={seed:2d} | "
            f"surge={info['surge_room']}@{info['surge_time']:.0f}"
        )

    print("\nConditions seen:")

    for room, time in sorted(seen):
        print(f"{room}@{time:.0f}")

    print(
        f"\nSeen {len(seen)}/8 possible conditions"
    )

    print("\n===== SAME-SEED REPRODUCIBILITY =====")

    obs1, info1 = env.reset(seed=123)
    obs2, info2 = env.reset(seed=123)

    print(
        f"first:  "
        f"{info1['surge_room']}@{info1['surge_time']:.0f}"
    )

    print(
        f"second: "
        f"{info2['surge_room']}@{info2['surge_time']:.0f}"
    )

    assert info1["surge_room"] == info2["surge_room"]
    assert info1["surge_time"] == info2["surge_time"]

    print("PASS: same seed gives same dynamic event")

    env.close()


if __name__ == "__main__":
    main()