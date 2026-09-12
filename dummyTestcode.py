from scenarios.building_v2 import BuildingV2Scenario
import os

scenario = BuildingV2Scenario()

print(scenario.junction_waypoints)
print(scenario.guidance_choices)
print(scenario.initial_targets)
print(os.cpu_count())