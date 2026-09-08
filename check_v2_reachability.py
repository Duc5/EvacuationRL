import jupedsim as jps

from scenarios.building_v2 import BuildingV2Scenario


MAX_TIME = 60.0


def test_route(scenario, room_name, exit_name):

    simulation = jps.Simulation(
        model=jps.CollisionFreeSpeedModel(),
        geometry=scenario.geometry,
    )

    # --------------------------------------------------
    # Create chosen exit
    # --------------------------------------------------

    exit_stage_id = simulation.add_exit_stage(
        scenario.exits[exit_name]
    )

    journey_id = simulation.add_journey(
        jps.JourneyDescription([
            exit_stage_id
        ])
    )

    # --------------------------------------------------
    # Add one pedestrian in chosen room
    # --------------------------------------------------

    start_position = scenario.test_positions[room_name]

    parameters = jps.CollisionFreeSpeedModelAgentParameters(
        position=start_position,
        journey_id=journey_id,
        stage_id=exit_stage_id,
    )

    simulation.add_agent(parameters)

    # --------------------------------------------------
    # Run simulation
    # --------------------------------------------------

    while (
        simulation.agent_count() > 0
        and simulation.elapsed_time() < MAX_TIME
    ):
        simulation.iterate()

    evacuated = simulation.agent_count() == 0

    return {
        "room": room_name,
        "exit": exit_name,
        "evacuated": evacuated,
        "time": simulation.elapsed_time(),
    }


def main():

    scenario = BuildingV2Scenario()

    results = []

    for room_name in scenario.rooms:
        for exit_name in scenario.exits:

            result = test_route(
                scenario,
                room_name,
                exit_name,
            )

            results.append(result)

    print("\n===== V2 REACHABILITY TEST =====\n")

    for result in results:

        status = (
            "PASS"
            if result["evacuated"]
            else "FAIL"
        )

        print(
            f"Room {result['room']} "
            f"→ Exit {result['exit']} "
            f"| {status} "
            f"| {result['time']:.2f} s"
        )

    print("\n===============================\n")


if __name__ == "__main__":
    main()