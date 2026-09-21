from scenarios.building_v2 import BuildingV2Scenario

scenario = BuildingV2Scenario(
    room_counts={"A": 10, "B": 10, "C": 10, "D": 10},
)

for room in ["A", "B", "C", "D"]:
    candidates = scenario.generate_room_candidates(room)
    print(room, len(candidates))

    