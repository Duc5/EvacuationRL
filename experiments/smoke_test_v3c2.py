from scenarios.building_v2 import BuildingV2Scenario
from envs.evacuation_env import EvacuationEnv


INITIAL_COUNTS = {
    "A": 40,
    "B": 20,
    "C": 20,
    "D": 20,
}

INCIDENTS = [
    "C_exit",
    "B_route_08",
    "bottom_loop_left",
]

SURGE_ROOMS = ["A", "B", "C", "D"]
SURGE_COUNTS = [20, 30, 40]

SURGE_TIME = 10.0
CONTROL_INTERVAL = 10.0
MAX_TIME = 180.0

# Two seeds is enough for a smoke test.
SEEDS = [0, 1]

# Use a known sensible policy so failures are less likely to
# just be caused by terrible random routing.
TEST_POLICY = "ACA"


def short_policy(policy):
    return "".join(
        policy[j]
        for j in ["J1", "J2", "J3"]
    )


def make_env(incident, surge_room, surge_count):
    scenario = BuildingV2Scenario(
        room_counts=dict(INITIAL_COUNTS)
    )

    return EvacuationEnv(
        scenario=scenario,
        control_interval=CONTROL_INTERVAL,
        max_time=MAX_TIME,

        # Singleton lists force this exact test condition.
        dynamic_surge_rooms=[surge_room],
        dynamic_surge_times=[SURGE_TIME],
        dynamic_surge_counts=[surge_count],

        incidents=INCIDENTS,
        forced_incident=incident,
    )


def get_action():
    env = make_env(
        incident="C_exit",
        surge_room="A",
        surge_count=20,
    )

    labels = [
        short_policy(policy)
        for policy in env.guidance_plans
    ]

    env.close()

    return labels.index(TEST_POLICY)


def run_episode(
    incident,
    surge_room,
    surge_count,
    seed,
    action,
):
    env = make_env(
        incident=incident,
        surge_room=surge_room,
        surge_count=surge_count,
    )

    try:
        obs, info = env.reset(seed=seed)

        # Check reset chose exactly what we asked for.
        if info["surge_room"] != surge_room:
            raise RuntimeError(
                f"Expected surge room {surge_room}, "
                f"got {info['surge_room']}"
            )

        if info["surge_count"] != surge_count:
            raise RuntimeError(
                f"Expected surge count {surge_count}, "
                f"got {info['surge_count']}"
            )

        terminated = False
        truncated = False
        episode_reward = 0.0

        while not terminated and not truncated:
            obs, reward, terminated, truncated, info = (
                env.step(action)
            )

            episode_reward += reward

        return {
            "incident": incident,
            "surge_room": surge_room,
            "surge_count": surge_count,
            "seed": seed,
            "time": info["elapsed_time"],
            "reward": episode_reward,
            "remaining": info["remaining_agents"],
            "pending": info["geometry_switch_pending"],
            "delay": info["geometry_switch_delay"],
            "error": "",
        }

    except Exception as error:
        return {
            "incident": incident,
            "surge_room": surge_room,
            "surge_count": surge_count,
            "seed": seed,
            "time": MAX_TIME,
            "reward": 0.0,
            "remaining": -1,
            "pending": False,
            "delay": 0.0,
            "error": str(error),
        }

    finally:
        env.close()


def main():
    action = get_action()

    print("===== V3-C2 SMOKE TEST =====")
    print(f"Policy: {TEST_POLICY}")
    print(
        f"Conditions: "
        f"{len(INCIDENTS)} × "
        f"{len(SURGE_ROOMS)} × "
        f"{len(SURGE_COUNTS)} × "
        f"{len(SEEDS)}"
    )
    print()

    rows = []
    run_number = 0

    total = (
        len(INCIDENTS)
        * len(SURGE_ROOMS)
        * len(SURGE_COUNTS)
        * len(SEEDS)
    )

    for incident in INCIDENTS:
        for surge_room in SURGE_ROOMS:
            for surge_count in SURGE_COUNTS:
                for seed in SEEDS:
                    run_number += 1

                    row = run_episode(
                        incident=incident,
                        surge_room=surge_room,
                        surge_count=surge_count,
                        seed=seed,
                        action=action,
                    )

                    rows.append(row)

                    print(
                        f"[{run_number:03}/{total}] "
                        f"{incident:<18} | "
                        f"surge={surge_room}+{surge_count:<2} | "
                        f"seed={seed:2} | "
                        f"time={row['time']:6.2f} | "
                        f"remaining={row['remaining']:3} | "
                        f"delay={row['delay']:.3f}"
                    )

                    if row["error"]:
                        print(
                            f"    ERROR: {row['error']}"
                        )

    errors = [
        row for row in rows
        if row["error"]
    ]

    timeouts = [
        row for row in rows
        if not row["error"]
        and row["remaining"] > 0
    ]

    pending = [
        row for row in rows
        if not row["error"]
        and row["pending"]
    ]

    print("\n===== SUMMARY =====")
    print(f"Runs:     {len(rows)}")
    print(f"Errors:   {len(errors)}")
    print(f"Timeouts: {len(timeouts)}")
    print(f"Pending:  {len(pending)}")

    if errors:
        print("\nErrors:")
        for row in errors:
            print(
                f"{row['incident']} | "
                f"{row['surge_room']}+"
                f"{row['surge_count']} | "
                f"seed={row['seed']} | "
                f"{row['error']}"
            )

    if timeouts:
        print("\nTimeouts:")
        for row in timeouts:
            print(
                f"{row['incident']} | "
                f"{row['surge_room']}+"
                f"{row['surge_count']} | "
                f"seed={row['seed']}"
            )


if __name__ == "__main__":
    main()