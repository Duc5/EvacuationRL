from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
import csv
import time

from scenarios.building_v2 import BuildingV2Scenario
from envs.evacuation_env import EvacuationEnv


INITIAL_COUNTS = {"A": 40, "B": 20, "C": 20, "D": 20}

SURGE_ROOM = "C"
SURGE_TIME = 10.0
SURGE_COUNT = 40

INCIDENTS = [
    "C_exit",
    "B_route_08",
    "bottom_loop_left",
]

CONTROL_INTERVAL = 10.0
MAX_TIME = 180.0

# 3 incidents × 35 actions × 50 seeds = 5250 cases
SEEDS = range(50)
MAX_WORKERS = 11

# If a geometry cannot close within this much extra simulated time,
# consider it unsafe.
MAX_DEFER_SECONDS = 5.0

# We call the backend in small chunks while waiting.
# The backend itself retries every JuPedSim iteration.
POLL_SECONDS = 0.1

RESULTS_DIR = Path("results")
CSV_PATH = RESULTS_DIR / "v3b_incident_switch_safety_deferred.csv"
SUMMARY_PATH = RESULTS_DIR / "v3b_incident_switch_safety_deferred_summary.txt"


def policy_name(plan):
    return "".join(plan[j] for j in ("J1", "J2", "J3"))


def run_case(incident, action, seed):
    env = None

    result = {
        "incident": incident,
        "seed": seed,
        "action": action,
        "policy": "",
        "status": "",
        "switch_mode": "",
        "initial_blocking_agents": "",
        "switch_attempts": 0,
        "geometry_applied_time": "",
        "scheduled_delay": "",
        "deferral_wait": "",
        "final_pending": False,
        "error_type": "",
        "error_message": "",
    }

    try:
        scenario = BuildingV2Scenario(
            room_counts=dict(INITIAL_COUNTS)
        )

        # One incident only so this exact incident is forced.
        env = EvacuationEnv(
            scenario=scenario,
            control_interval=CONTROL_INTERVAL,
            max_time=MAX_TIME,
            dynamic_surge_rooms=[SURGE_ROOM],
            dynamic_surge_times=[SURGE_TIME],
            dynamic_surge_count=SURGE_COUNT,
            incidents=[incident],
        )

        result["policy"] = policy_name(
            env.guidance_plans[action]
        )

        obs, info = env.reset(seed=seed)

        if not env.observation_space.contains(obs):
            raise RuntimeError(
                f"Initial observation outside observation_space: {obs}"
            )

        if info["active_incident"] is not None:
            raise RuntimeError(
                f"Incident already active immediately after reset: "
                f"{info['active_incident']}"
            )

        # ----------------------------------------------------------
        # t=0 -> t=10
        #
        # The selected action controls the pre-incident period.
        # The surge + incident should trigger during this step.
        # ----------------------------------------------------------

        obs, reward, terminated, truncated, info = env.step(action)

        if not env.observation_space.contains(obs):
            raise RuntimeError(
                f"Post-event observation outside observation_space: {obs}"
            )

        if info["active_incident"] != incident:
            raise RuntimeError(
                f"Expected active incident {incident}, "
                f"got {info['active_incident']}"
            )

        backend = env.backend

        if len(backend.pending_events) != 1:
            raise RuntimeError(
                f"Expected exactly one dynamic event, "
                f"found {len(backend.pending_events)}"
            )

        event = backend.pending_events[0]

        if not event["triggered"]:
            raise RuntimeError(
                f"Dynamic event was not triggered after first "
                f"{CONTROL_INTERVAL}s control step."
            )

        # ----------------------------------------------------------
        # Case 1: geometry switched immediately
        # ----------------------------------------------------------

        if event["geometry_applied"]:
            result["status"] = "PASS"
            result["switch_mode"] = "IMMEDIATE"
            result["switch_attempts"] = (
                backend.geometry_switch_attempts
            )
            result["geometry_applied_time"] = (
                event["geometry_applied_time"]
            )
            result["scheduled_delay"] = (
                backend.geometry_switch_delay
            )
            result["deferral_wait"] = 0.0
            result["final_pending"] = False

            return result

        # ----------------------------------------------------------
        # Case 2: geometry switch was blocked at t≈10.
        #
        # Save the agents responsible for the first failed attempt.
        # ----------------------------------------------------------

        result["initial_blocking_agents"] = ",".join(
            str(agent_id)
            for agent_id in backend.geometry_switch_blocking_agents
        )

        pending_started_at = backend.elapsed_time
        deadline = SURGE_TIME + MAX_DEFER_SECONDS

        # ----------------------------------------------------------
        # Keep the physical simulation moving.
        #
        # _trigger_due_events() runs after every JuPedSim iteration,
        # so the backend retries the pending geometry automatically.
        # ----------------------------------------------------------

        while (
            backend.geometry_switch_pending
            and backend.elapsed_time < deadline
        ):
            remaining = deadline - backend.elapsed_time
            advance_by = min(POLL_SECONDS, remaining)

            if advance_by <= 0:
                break

            backend.advance(advance_by)

        result["switch_attempts"] = (
            backend.geometry_switch_attempts
        )
        result["final_pending"] = (
            backend.geometry_switch_pending
        )

        # ----------------------------------------------------------
        # Deferred switch eventually succeeded.
        # ----------------------------------------------------------

        if event["geometry_applied"]:
            applied_time = event["geometry_applied_time"]

            result["status"] = "PASS"
            result["switch_mode"] = "DEFERRED"
            result["geometry_applied_time"] = applied_time
            result["scheduled_delay"] = (
                backend.geometry_switch_delay
            )
            result["deferral_wait"] = (
                applied_time - pending_started_at
            )
            result["final_pending"] = False

        # ----------------------------------------------------------
        # Still cannot close after MAX_DEFER_SECONDS.
        # ----------------------------------------------------------

        else:
            result["status"] = "FAIL"
            result["switch_mode"] = "PENDING_TIMEOUT"
            result["deferral_wait"] = (
                backend.elapsed_time - pending_started_at
            )
            result["error_type"] = "GeometrySwitchTimeout"
            result["error_message"] = (
                f"Geometry still pending at "
                f"t={backend.elapsed_time:.4f}s after waiting up to "
                f"{MAX_DEFER_SECONDS:.2f}s."
            )

    except Exception as exc:
        result["status"] = "FAIL"

        if not result["switch_mode"]:
            result["switch_mode"] = "ERROR"

        result["error_type"] = type(exc).__name__
        result["error_message"] = str(exc).replace("\n", " | ")

    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    return result


def save_csv(rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "incident",
        "seed",
        "action",
        "policy",
        "status",
        "switch_mode",
        "initial_blocking_agents",
        "switch_attempts",
        "geometry_applied_time",
        "scheduled_delay",
        "deferral_wait",
        "final_pending",
        "error_type",
        "error_message",
    ]

    with CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def fmt_mean(values):
    return f"{mean(values):.4f}" if values else "-"


def fmt_max(values):
    return f"{max(values):.4f}" if values else "-"


def make_summary(rows):
    lines = []

    total = len(rows)

    passes = [
        r for r in rows
        if r["status"] == "PASS"
    ]

    failures = [
        r for r in rows
        if r["status"] == "FAIL"
    ]

    immediate = [
        r for r in rows
        if r["switch_mode"] == "IMMEDIATE"
    ]

    deferred = [
        r for r in rows
        if r["switch_mode"] == "DEFERRED"
    ]

    pending_timeouts = [
        r for r in rows
        if r["switch_mode"] == "PENDING_TIMEOUT"
    ]

    errors = [
        r for r in rows
        if r["switch_mode"] == "ERROR"
    ]

    deferred_waits = [
        float(r["deferral_wait"])
        for r in deferred
    ]

    deferred_scheduled_delays = [
        float(r["scheduled_delay"])
        for r in deferred
    ]

    lines.append(
        "===== V3-B INCIDENT GEOMETRY SWITCH SAFETY AUDIT ====="
    )
    lines.append("")
    lines.append(f"Seeds tested: {len(SEEDS)}")
    lines.append("Actions tested: 35")
    lines.append(f"Incidents tested: {len(INCIDENTS)}")
    lines.append(f"Total cases: {total}")
    lines.append("")
    lines.append(f"Passes: {len(passes)}")
    lines.append(f"Failures: {len(failures)}")
    lines.append(f"Immediate switches: {len(immediate)}")
    lines.append(f"Deferred switches: {len(deferred)}")
    lines.append(
        f"Pending > {MAX_DEFER_SECONDS:.1f}s: "
        f"{len(pending_timeouts)}"
    )
    lines.append(f"Other errors: {len(errors)}")
    lines.append("")

    lines.append("===== DEFERRED SWITCH TIMING =====")

    if deferred:
        lines.append(
            f"Mean additional wait: "
            f"{fmt_mean(deferred_waits)}s"
        )
        lines.append(
            f"Max additional wait:  "
            f"{fmt_max(deferred_waits)}s"
        )
        lines.append(
            f"Mean delay from scheduled t=10: "
            f"{fmt_mean(deferred_scheduled_delays)}s"
        )
        lines.append(
            f"Max delay from scheduled t=10:  "
            f"{fmt_max(deferred_scheduled_delays)}s"
        )
    else:
        lines.append("No deferred switches.")

    lines.append("")
    lines.append("===== PER-INCIDENT RESULTS =====")

    for incident in INCIDENTS:
        subset = [
            r for r in rows
            if r["incident"] == incident
        ]

        inc_pass = [
            r for r in subset
            if r["status"] == "PASS"
        ]

        inc_fail = [
            r for r in subset
            if r["status"] == "FAIL"
        ]

        inc_immediate = [
            r for r in subset
            if r["switch_mode"] == "IMMEDIATE"
        ]

        inc_deferred = [
            r for r in subset
            if r["switch_mode"] == "DEFERRED"
        ]

        inc_timeouts = [
            r for r in subset
            if r["switch_mode"] == "PENDING_TIMEOUT"
        ]

        waits = [
            float(r["deferral_wait"])
            for r in inc_deferred
        ]

        lines.append(
            f"{incident:18s} | "
            f"pass={len(inc_pass):4d}/{len(subset):4d} | "
            f"immediate={len(inc_immediate):4d} | "
            f"deferred={len(inc_deferred):3d} | "
            f"fail={len(inc_fail):3d} | "
            f"pending={len(inc_timeouts):3d} | "
            f"mean_wait={fmt_mean(waits):>7s}s | "
            f"max_wait={fmt_max(waits):>7s}s"
        )

    if deferred:
        lines.append("")
        lines.append("===== DEFERRED SWITCHES =====")

        for r in sorted(
            deferred,
            key=lambda x: (
                x["incident"],
                x["action"],
                x["seed"],
            ),
        ):
            lines.append(
                f"{r['incident']:18s} | "
                f"seed={r['seed']:2d} | "
                f"action={r['action']:2d} | "
                f"policy={r['policy']:6s} | "
                f"wait={float(r['deferral_wait']):.4f}s | "
                f"scheduled_delay="
                f"{float(r['scheduled_delay']):.4f}s | "
                f"attempts={r['switch_attempts']:3d} | "
                f"initial_agents="
                f"{r['initial_blocking_agents'] or '-'}"
            )

        lines.append("")
        lines.append(
            "===== DEFERRED SWITCHES GROUPED BY ACTION ====="
        )

        for incident in INCIDENTS:
            incident_rows = [
                r for r in deferred
                if r["incident"] == incident
            ]

            if not incident_rows:
                continue

            lines.append("")
            lines.append(incident)

            actions = sorted(set(
                (r["action"], r["policy"])
                for r in incident_rows
            ))

            for action, policy in actions:
                matching = [
                    r for r in incident_rows
                    if r["action"] == action
                ]

                seeds = ",".join(
                    str(r["seed"])
                    for r in matching
                )

                waits = [
                    float(r["deferral_wait"])
                    for r in matching
                ]

                lines.append(
                    f"  action={action:2d} "
                    f"policy={policy:6s} "
                    f"deferred={len(matching):2d} "
                    f"mean_wait={mean(waits):.4f}s "
                    f"max_wait={max(waits):.4f}s "
                    f"seeds=[{seeds}]"
                )

    if failures:
        lines.append("")
        lines.append("===== FAILURES =====")

        for r in sorted(
            failures,
            key=lambda x: (
                x["incident"],
                x["action"],
                x["seed"],
            ),
        ):
            lines.append(
                f"{r['incident']:18s} | "
                f"seed={r['seed']:2d} | "
                f"action={r['action']:2d} | "
                f"policy={r['policy']:6s} | "
                f"mode={r['switch_mode']} | "
                f"{r['error_type']}: "
                f"{r['error_message']}"
            )

    else:
        lines.append("")
        lines.append(
            "NO GEOMETRY-SWITCH FAILURES FOUND "
            "IN THE TESTED CASES."
        )

    return "\n".join(lines)


def main():
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Temporary environment just to confirm action count.
    scenario = BuildingV2Scenario(
        room_counts=dict(INITIAL_COUNTS)
    )

    check_env = EvacuationEnv(
        scenario=scenario,
        incidents=INCIDENTS,
    )

    num_actions = check_env.action_space.n
    check_env.close()

    if num_actions != 35:
        raise RuntimeError(
            f"Expected 35 guidance actions, "
            f"found {num_actions}"
        )

    cases = [
        (incident, action, seed)
        for incident in INCIDENTS
        for action in range(num_actions)
        for seed in SEEDS
    ]

    print("===== INCIDENT SWITCH SAFETY AUDIT =====")
    print(f"Incidents       : {len(INCIDENTS)}")
    print(f"Actions         : {num_actions}")
    print(f"Seeds           : {len(SEEDS)}")
    print(f"Total jobs      : {len(cases)}")
    print(f"Workers         : {MAX_WORKERS}")
    print(
        f"Max defer time  : "
        f"{MAX_DEFER_SECONDS:.1f}s"
    )
    print()

    start = time.time()

    rows = []
    completed = 0
    immediate_count = 0
    deferred_count = 0
    failure_count = 0

    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = {
            executor.submit(
                run_case,
                incident,
                action,
                seed,
            ): (incident, action, seed)
            for incident, action, seed in cases
        }

        for future in as_completed(futures):
            incident, action, seed = futures[future]

            try:
                result = future.result()

            except Exception as exc:
                result = {
                    "incident": incident,
                    "seed": seed,
                    "action": action,
                    "policy": "",
                    "status": "FAIL",
                    "switch_mode": "ERROR",
                    "initial_blocking_agents": "",
                    "switch_attempts": 0,
                    "geometry_applied_time": "",
                    "scheduled_delay": "",
                    "deferral_wait": "",
                    "final_pending": False,
                    "error_type": type(exc).__name__,
                    "error_message": (
                        "Worker failure: "
                        + str(exc).replace("\n", " | ")
                    ),
                }

            rows.append(result)
            completed += 1

            if result["switch_mode"] == "IMMEDIATE":
                immediate_count += 1

            elif result["switch_mode"] == "DEFERRED":
                deferred_count += 1

                print(
                    "DEFERRED | "
                    f"{result['incident']} | "
                    f"seed={result['seed']} | "
                    f"action={result['action']} | "
                    f"policy={result['policy']} | "
                    f"wait={float(result['deferral_wait']):.4f}s | "
                    f"attempts={result['switch_attempts']}",
                    flush=True,
                )

            if result["status"] == "FAIL":
                failure_count += 1

                print(
                    "FAIL     | "
                    f"{result['incident']} | "
                    f"seed={result['seed']} | "
                    f"action={result['action']} | "
                    f"policy={result['policy']} | "
                    f"mode={result['switch_mode']} | "
                    f"{result['error_type']}",
                    flush=True,
                )

            if (
                completed % 100 == 0
                or completed == len(cases)
            ):
                print(
                    f"Progress {completed}/{len(cases)} | "
                    f"immediate={immediate_count} | "
                    f"deferred={deferred_count} | "
                    f"failures={failure_count}",
                    flush=True,
                )

    rows.sort(
        key=lambda r: (
            INCIDENTS.index(r["incident"]),
            r["action"],
            r["seed"],
        )
    )

    save_csv(rows)

    summary = make_summary(rows)

    SUMMARY_PATH.write_text(summary)

    elapsed = time.time() - start

    print()
    print(summary)
    print()
    print(f"CSV saved to: {CSV_PATH}")
    print(f"Summary saved to: {SUMMARY_PATH}")
    print(f"Wall time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()