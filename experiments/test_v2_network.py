from scenarios.building_v2 import BuildingV2Scenario
from simulation.jupedsim_backend import JuPedSimBackend


ROOM_COUNTS = {
    "A": 40,
    "B": 30,
    "C": 30,
    "D": 30,
}


scenario = BuildingV2Scenario(
    room_counts=ROOM_COUNTS
)

sim = JuPedSimBackend(
    scenario=scenario,
    record=True,
    trajectory_path="v2_crowd_test.sqlite",
)

state = sim.reset(seed=0)

print(
    "Initial population:",
    state.initial_population
)

print(
    "Room populations:",
    ROOM_COUNTS
)


# --------------------------------------------------
# Simple sensible routing policy
# --------------------------------------------------
#
# Room A:
#     A -> J1 -> Exit A
#
# Room B:
#     B -> J3 -> Exit B
#
# Room C:
#     C -> J2 -> Exit C
#
# Room D:
#     D -> Exit B directly
#

guidance = {
    "J1": "A",
    "J2": "C",
    "J3": "A",
}

sim.apply_guidance(
    guidance
)


while (
    not sim.is_evacuated
    and sim.elapsed_time < 60.0
):

    sim.advance(0.5)

    print(
        f"time={sim.elapsed_time:5.2f}",
        f"remaining={sim.remaining_agents}",
    )


print()
print(
    "Final time:",
    sim.elapsed_time,
)

print(
    "Evacuated:",
    sim.is_evacuated,
)

print(
    "Remaining:",
    sim.remaining_agents,
)

sim.close()