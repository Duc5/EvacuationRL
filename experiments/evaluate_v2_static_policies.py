import csv
import itertools
import os

from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)
from pathlib import Path


from scenarios.building_v2 import BuildingV2Scenario
from simulation.jupedsim_backend import JuPedSimBackend


# ============================================================
# Experiment configuration
# ============================================================

ROOM_COUNTS = {
    "A": 40,
    "B": 35,
    "C": 25,
    "D": 20,
}

SEED = 0

# Give slow-but-valid policies enough time to finish.
MAX_TIME = 180.0

CONTROL_INTERVAL = 0.5

# Do not automatically consume every core on a shared university
# machine. Increase this only if you know you have the cores available.
MAX_WORKERS = min(
    11,
    os.cpu_count() or 1,
)


# ============================================================
# Policy generation
# ============================================================

def generate_all_policies(scenario):

    junction_names = [
        "J1",
        "J2",
        "J3",
    ]

    choices = [
        scenario.guidance_choices[junction]
        for junction in junction_names
    ]

    for combination in itertools.product(*choices):

        yield dict(
            zip(
                junction_names,
                combination,
            )
        )


# ============================================================
# Cycle analysis
# ============================================================

def trace_from_junction(
    start_junction,
    policy,
):
    """
    Follow a fixed routing policy starting from one junction.

    Returns:
        {
            "reaches_exit": bool,
            "cycle": tuple | None
        }

    Example:

        J1 -> J3
        J3 -> J1

    gives:

        cycle = ("J1", "J3")
    """

    visited_order = []
    visited_set = set()

    current = start_junction

    while current in policy:

        if current in visited_set:

            cycle_start = (
                visited_order.index(current)
            )

            cycle = tuple(
                visited_order[cycle_start:]
            )

            return {
                "reaches_exit": False,
                "cycle": cycle,
            }

        visited_order.append(current)
        visited_set.add(current)

        current = policy[current]

    # If the target is no longer a junction, it must be an exit
    # for one of our valid generated policies.
    return {
        "reaches_exit": True,
        "cycle": None,
    }


def analyse_cycles(
    policy,
    room_counts,
    initial_targets,
):
    """
    Determine whether the fixed policy traps any currently occupied
    room populations in a routing cycle.

    Example:

        Room A -> J1
        J1 -> J3
        J3 -> J1

    means every Room A pedestrian can never reach an exit.
    """

    trapped_rooms = []
    trapped_agents = 0
    cycles = set()

    for room_name, count in room_counts.items():

        if count == 0:
            continue

        initial_target = (
            initial_targets[room_name]
        )

        # Room D, for example, may go directly to Exit B and therefore
        # never enter the controlled junction network.
        if initial_target not in policy: #initial target is exit
            continue

        
        trace = trace_from_junction(
            initial_target,
            policy,
        )

        if not trace["reaches_exit"]:

            trapped_rooms.append(
                room_name
            )

            trapped_agents += count

            # Canonicalise the cycle only for reporting.
            cycle = trace["cycle"]

            cycles.add(
                tuple(sorted(cycle))
            )

    return {
        "cyclic": trapped_agents > 0,
        "trapped_rooms": tuple(trapped_rooms),
        "trapped_agents": trapped_agents,
        "cycles": tuple(sorted(cycles)),
    }


# ============================================================
# One JuPedSim run
# ============================================================

def run_policy(
    policy,
    room_counts,
    seed,
    max_time,
    control_interval,
):
    """
    Run one fixed guidance policy.

    Important:
    This function is top-level so ProcessPoolExecutor can execute it
    in a separate Python process.
    """

    scenario = BuildingV2Scenario(
        room_counts=room_counts
    )

    sim = JuPedSimBackend(
        scenario=scenario,
        record=False,
    )

    try:

        sim.reset(seed=seed)

        sim.apply_guidance(
            policy
        )

        while (
            not sim.is_evacuated
            and sim.elapsed_time < max_time
        ):

            sim.advance(
                control_interval
            )

        return {
            "J1": policy["J1"],
            "J2": policy["J2"],
            "J3": policy["J3"],

            "status": (
                "PASS"
                if sim.is_evacuated
                else "TIMEOUT"
            ),

            "evacuation_time": sim.elapsed_time,
            "remaining_agents": sim.remaining_agents,

            "cyclic": False,
            "trapped_agents": 0,
            "trapped_rooms": "",
            "cycles": "",
        }

    finally:

        sim.close()


# ============================================================
# Formatting
# ============================================================

def policy_string(result):

    return (
        f"J1={result['J1']:<2} "
        f"J2={result['J2']:<2} "
        f"J3={result['J3']:<2}"
    )


def result_sort_key(result):
    """
    Sorting order:

        1. Successful evacuations, fastest first
        2. Timeouts, fewest remaining agents first
        3. Structurally cyclic policies
    """

    if result["status"] == "PASS":

        return (
            0,
            result["evacuation_time"],
            0,
        )

    if result["status"] == "TIMEOUT":

        return (
            1,
            result["remaining_agents"],
            result["evacuation_time"],
        )

    return (
        2,
        result["remaining_agents"],
        0,
    )


# ============================================================
# Main experiment
# ============================================================

def main():

    scenario = BuildingV2Scenario(
        room_counts=ROOM_COUNTS
    )

    policies = list(
        generate_all_policies(
            scenario
        )
    )

    print()
    print("===== V2 STATIC POLICY SWEEP =====")
    print()

    print(
        f"Policies:       {len(policies)}"
    )

    print(
        f"Population:     "
        f"{sum(ROOM_COUNTS.values())}"
    )

    print(
        f"Room counts:    {ROOM_COUNTS}"
    )

    print(
        f"Seed:           {SEED}"
    )

    print(
        f"Max sim time:   {MAX_TIME:.0f} s"
    )

    print(
        f"Worker processes: {MAX_WORKERS}"
    )

    print()

    # --------------------------------------------------------
    # Pre-screen policies for cycles
    # --------------------------------------------------------

    results = []

    policies_to_simulate = []

    for index, policy in enumerate(
        policies,
        start=1,
    ):

        cycle_info = analyse_cycles(
            policy=policy,
            room_counts=ROOM_COUNTS,
            initial_targets=scenario.initial_targets,
        )

        if cycle_info["cyclic"]:

            result = {
                "J1": policy["J1"],
                "J2": policy["J2"],
                "J3": policy["J3"],

                "status": "CYCLIC",

                # No JuPedSim run occurred.
                "evacuation_time": None,

                # Under a deterministic fixed policy these populations
                # can never reach an exit.
                "remaining_agents": (
                    cycle_info[
                        "trapped_agents"
                    ]
                ),

                "cyclic": True,

                "trapped_agents": (
                    cycle_info[
                        "trapped_agents"
                    ]
                ),

                "trapped_rooms": ",".join(
                    cycle_info[
                        "trapped_rooms"
                    ]
                ),

                "cycles": str(
                    cycle_info["cycles"]
                ),
            }

            results.append(
                result
            )

            print(
                f"[SKIP {index:02d}] "
                f"{policy_string(result)} "
                f"| CYCLIC "
                f"| trapped="
                f"{result['trapped_agents']}"
            )

        else:

            policies_to_simulate.append(
                (
                    index,
                    policy,
                )
            )

    print()

    print(
        f"Cyclic policies skipped: "
        f"{len(policies) - len(policies_to_simulate)}"
    )

    print(
        f"Policies to simulate:     "
        f"{len(policies_to_simulate)}"
    )

    print()

    # --------------------------------------------------------
    # Parallel JuPedSim execution
    # --------------------------------------------------------

    completed = 0

    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        future_to_index = {}

        for (original_index,policy) in policies_to_simulate:

            future = executor.submit(
                run_policy,
                policy,
                ROOM_COUNTS,
                SEED,
                MAX_TIME,
                CONTROL_INTERVAL,
            )

            future_to_index[future] = original_index

        for future in as_completed(future_to_index):

            original_index = (future_to_index[future])

            result = future.result()

            results.append(result)

            completed += 1

            print(
                f"[DONE "
                f"{completed:02d}/"
                f"{len(policies_to_simulate):02d}] "
                f"{policy_string(result)} "
                f"| "
                f"{result['evacuation_time']:6.2f} s "
                f"| remaining="
                f"{result['remaining_agents']:3d} "
                f"| {result['status']}"
            )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    results.sort(
        key=result_sort_key
    )

    successful = [
        result
        for result in results
        if result["status"] == "PASS"
    ]

    timeouts = [
        result
        for result in results
        if result["status"] == "TIMEOUT"
    ]

    cyclic = [
        result
        for result in results
        if result["status"] == "CYCLIC"
    ]

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print(
        "===== SWEEP SUMMARY ====="
    )
    print()

    print(
        f"Successful: {len(successful)}"
    )

    print(
        f"Timeouts:   {len(timeouts)}"
    )

    print(
        f"Cyclic:     {len(cyclic)}"
    )

    # --------------------------------------------------------
    # Best policies
    # --------------------------------------------------------

    print()
    print(
        "===== BEST STATIC POLICIES ====="
    )
    print()

    for rank, result in enumerate(
        successful[:10],
        start=1,
    ):

        print(
            f"{rank:2d}. "
            f"{policy_string(result)} "
            f"| "
            f"{result['evacuation_time']:6.2f} s"
        )

    # --------------------------------------------------------
    # Slowest successful policies
    # --------------------------------------------------------

    if successful:

        print()
        print(
            "===== SLOWEST SUCCESSFUL POLICIES ====="
        )
        print()

        for result in successful[-10:]:

            print(
                f"{policy_string(result)} "
                f"| "
                f"{result['evacuation_time']:6.2f} s"
            )

    # --------------------------------------------------------
    # Timeouts
    # --------------------------------------------------------

    if timeouts:

        print()
        print(
            "===== NON-CYCLIC TIMEOUTS ====="
        )
        print()

        for result in timeouts:

            print(
                f"{policy_string(result)} "
                f"| "
                f"{result['evacuation_time']:6.2f} s "
                f"| remaining="
                f"{result['remaining_agents']}"
            )

    # --------------------------------------------------------
    # Cycles
    # --------------------------------------------------------

    if cyclic:

        print()
        print(
            "===== CYCLIC POLICIES ====="
        )
        print()

        for result in cyclic:

            print(
                f"{policy_string(result)} "
                f"| trapped="
                f"{result['trapped_agents']:3d} "
                f"| rooms="
                f"{result['trapped_rooms']} "
                f"| cycles="
                f"{result['cycles']}"
            )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    output_dir = Path(
        "results"
    )

    output_dir.mkdir(
        exist_ok=True
    )

    output_path = (
        output_dir
        / "v2_static_policies_seed0_parallel.csv"
    )

    with output_path.open(
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "J1",
                "J2",
                "J3",
                "status",
                "evacuation_time",
                "remaining_agents",
                "cyclic",
                "trapped_agents",
                "trapped_rooms",
                "cycles",
            ],
        )

        writer.writeheader()
        writer.writerows(
            results
        )

    print()
    print(
        f"Saved complete results to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()