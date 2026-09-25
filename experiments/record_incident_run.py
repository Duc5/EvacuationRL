from pathlib import Path

from shapely.geometry import Point, box

from scenarios.building_v2 import BuildingV2Scenario
from envs.evacuation_env import EvacuationEnv


INITIAL_COUNTS = {"A": 40, "B": 20, "C": 20, "D": 20}

INCIDENT = "top_loop"
SURGE_TIME = 10.0
SURGE_ROOM = "C"
SURGE_COUNT = 40

PRE_POLICY = "CCA"
POST_POLICY = "CCA"

# Small interval ONLY for this diagnostic so we do not miss crossings.
CONTROL_INTERVAL = 0.1
MAX_TIME = 180.0
SEED = 0

TRAJECTORY_PATH = Path(
    f"trajectories/debug_{INCIDENT}.sqlite"
)

# Entire Room D <-> Room B connector.
B_CONNECTOR_REGION = box(4.5, 8.0, 6.5, 10.0)


def short_policy(policy):
    return "".join(policy[j] for j in ("J1", "J2", "J3"))


def main():
    scenario = BuildingV2Scenario(
        room_counts=dict(INITIAL_COUNTS),
        dynamic_events=[{
            "time": SURGE_TIME,
            "room": SURGE_ROOM,
            "count": SURGE_COUNT,
            "incident": INCIDENT,
        }],
    )

    env = EvacuationEnv(
        scenario=scenario,
        control_interval=CONTROL_INTERVAL,
        max_time=MAX_TIME,
        record=True,
        trajectory_path=TRAJECTORY_PATH,
    )

    labels = [
        short_policy(policy)
        for policy in env.guidance_plans
    ]

    pre_action = labels.index(PRE_POLICY)
    post_action = labels.index(POST_POLICY)

    obs, info = env.reset(seed=SEED)

    print(f"Recording to: {TRAJECTORY_PATH}")
    print(f"Incident: {INCIDENT}")
    print(f"Pre-policy:  {PRE_POLICY}")
    print(f"Post-policy: {POST_POLICY}")
    print()

    terminated = False
    truncated = False

    agents_seen_in_connector = set()
    max_connector_occupancy = 0
    ep_reward =0
    while not terminated and not truncated:
        # Keep CCA until the incident time, then switch to CJ3A.
        action = (
            pre_action
            if info["elapsed_time"] < SURGE_TIME
            else post_action
        )

        obs, reward, terminated, truncated, info = env.step(action)
        
        state = env.backend.get_state()

        agents_in_connector = []

        for agent in state.agents:
            if B_CONNECTOR_REGION.covers(Point(agent.position)):
                agents_in_connector.append(agent.id)
                agents_seen_in_connector.add(agent.id)

        max_connector_occupancy = max(
            max_connector_occupancy,
            len(agents_in_connector),
        )
        ep_reward+=reward

        # Only print when somebody is actually there.
        if agents_in_connector:
            print(
                f"t={info['elapsed_time']:6.2f} | "
                f"connector occupancy={len(agents_in_connector):2d} | "
                f"agents={agents_in_connector}"
            )


    print()
    print(f"Finished at t={info['elapsed_time']:.2f}s")
    print(f"Ep reward:{ep_reward}")
    print(f"Remaining: {info['remaining_agents']}")

    env.close()


if __name__ == "__main__":
    main()