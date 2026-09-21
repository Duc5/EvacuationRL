import pathlib
import time
import jupedsim as jps
import numpy as np
from shapely import Point

from simulation.state import AgentState, SimulationState


class JuPedSimBackend:
    """
    JuPedSim implementation of the evacuation simulation.

    Supports two routing styles:

        regional:
            Original v1 guidance-area routing.

        network:
            v2 junction-based adaptive routing.

    This class knows about JuPedSim, but knows nothing about
    Gymnasium, PPO, rewards, observations or action spaces.
    """

    def __init__(
        self,
        scenario,
        record=False,
        trajectory_path="integratedgym_evac.sqlite",
    ):
        self.scenario = scenario
        self.record = record
        self.trajectory_path = pathlib.Path(
            trajectory_path
        )

        self.simulation = None
        self.writer = None
        self.pending_events = []

        # --------------------------------------------------------------
        # JuPedSim targets
        # --------------------------------------------------------------

        # Maps a logical target name to:
        #
        #     (journey_id, stage_id)
        #
        # v1:
        #
        #     "left"
        #     "right"
        #
        # v2:
        #
        #     "A", "B", "C"
        #     "J1", "J2", "J3"
        #
        self.routes = {}

        # --------------------------------------------------------------
        # Per-agent state
        # --------------------------------------------------------------

        # Current target assigned to each pedestrian.
        self.assignments = {}

        # Used only by the original v1 regional guidance.
        self.committed_agents = set()

        # v2:
        # junction currently occupied by each pedestrian.
        #
        # None means the pedestrian is not inside a junction.
        self.active_junctions = {}
        self.routing_check_interval = 0.1
        self._time_since_routing_check = 0.0
        # Useful for debugging / later analysis.
        self.agent_origins = {}

        # --------------------------------------------------------------
        # Current network guidance
        # --------------------------------------------------------------

        # Example:
        #
        # {
        #     "J1": "A",
        #     "J2": "C",
        #     "J3": "B",
        # }
        #
        # This stores what each sign CURRENTLY displays.
        self.current_guidance = {}

        # --------------------------------------------------------------
        # Episode state
        # --------------------------------------------------------------

        self.initial_population = 0

        self.rng = None

    # ==================================================================
    # Simulation creation
    # ==================================================================

    def reset(self, seed=None):

        self._close_writer()
        self._time_since_routing_check = 0.0    
        self.rng = np.random.default_rng(seed)
        self.active_incident = None

        # --------------------------------------------------------------
        # Optional trajectory recording
        # --------------------------------------------------------------

        if self.record:

            if self.trajectory_path.exists():
                self.trajectory_path.unlink()

            self.writer = jps.SqliteTrajectoryWriter(
                output_file=self.trajectory_path,
                every_nth_frame=5,
            )

        else:
            self.writer = None

        # --------------------------------------------------------------
        # Create JuPedSim simulation
        # --------------------------------------------------------------

        self.simulation = jps.Simulation(
            model=jps.CollisionFreeSpeedModel(),
            geometry=self.scenario.geometry,
            trajectory_writer=self.writer,
        )

        # --------------------------------------------------------------
        # Reset episode bookkeeping
        # --------------------------------------------------------------

        self.routes = {}

        self.assignments = {}
        self.committed_agents = set()

        self.active_junctions = {}
        self.agent_origins = {}

        self.current_guidance = {}

        # --------------------------------------------------------------
        # Create target stages / journeys
        # --------------------------------------------------------------

        self._create_exit_routes()

        if self.scenario.routing_mode == "network":
            self._create_junction_routes()

        # --------------------------------------------------------------
        # Spawn pedestrians
        # --------------------------------------------------------------

        if self.scenario.routing_mode == "regional":
            self._spawn_regional_agents()

        elif self.scenario.routing_mode == "network":
            self._spawn_network_agents(seed=seed)

        else:
            raise ValueError(
                f"Unknown routing mode: "
                f"{self.scenario.routing_mode}"
            )

        return self.get_state()

    # ==================================================================
    # Stage / journey creation
    # ==================================================================

    def _create_exit_routes(self):
        """
        Create one exit stage and one simple journey for every exit.
        """

        for exit_name, exit_geometry in self.scenario.exits.items():

            exit_stage_id = self.simulation.add_exit_stage(
                exit_geometry
            )

            journey_id = self.simulation.add_journey(
                jps.JourneyDescription([
                    exit_stage_id
                ])
            )

            self.routes[exit_name] = (
                journey_id,
                exit_stage_id,
            )

    def _create_junction_routes(self):
        """
        Create JuPedSim waypoint stages for every controlled junction.

        A journey containing one waypoint simply causes an agent to
        move toward that waypoint.

        Before the agent actually reaches the waypoint, entering the
        larger junction polygon causes our controller to assign the
        next target.
        """

        for junction_name, waypoint_position in self.scenario.junction_waypoints.items():

            waypoint_stage_id = (
                self.simulation.add_waypoint_stage(
                    waypoint_position,
                    self.scenario.junction_waypoint_radius,
                )
            )

            journey_id = self.simulation.add_journey(
                jps.JourneyDescription([
                    waypoint_stage_id
                ])
            )

            self.routes[junction_name] = (
                journey_id,
                waypoint_stage_id,
            )

    # ==================================================================
    # Agent spawning
    # ==================================================================

    def _spawn_regional_agents(self):
        """
        Original v1 spawning behaviour.
        """

        positions = list(
            self.scenario.start_positions
        )

        self.initial_population = len(positions)

        for position in positions:

            group = self._initial_group(position)

            journey_id, stage_id = self.routes[group]

            parameters = (
                jps.CollisionFreeSpeedModelAgentParameters(
                    position=position,
                    journey_id=journey_id,
                    stage_id=stage_id,
                )
            )

            agent_id = self.simulation.add_agent(
                parameters
            )

            self.assignments[agent_id] = None
    def _spawn_network_agents(self, seed=None):
        positions_by_room = self.scenario.generate_start_positions(seed=seed)

        self.initial_population = sum(
            len(positions) for positions in positions_by_room.values()
        )

        for room_name, positions in positions_by_room.items():
            self._spawn_network_positions(room_name, positions)

        self.pending_events = [
            {
                "time": event["time"],
                "room": event["room"],
                "count": event["count"],
                "triggered": False,
                "incident": event.get("incident")
            }
            for event in self.scenario.dynamic_events
        ]
        initial_count = sum(len(positions) for positions in positions_by_room.values())
        future_count = sum(event["count"] for event in self.pending_events)
        self.initial_population = initial_count + future_count

    # Returns all free spawn position in a room
    def _get_free_spawn_positions(self, room_name, count):
        candidates = self.scenario.generate_room_candidates(room_name)

        occupied = [
            agent.position
            for agent in self.simulation.agents()
        ]

        min_distance = self.scenario.spawn_spacing

        free = []

        for position in candidates:
            x, y = position

            is_free = all(
                (x - ax) ** 2 + (y - ay) ** 2 >= min_distance ** 2
                for ax, ay in occupied
            )

            if is_free:
                free.append(position)

        if len(free) < count:
            raise RuntimeError(
                f"Dynamic event needs {count} free positions in Room {room_name}, "
                f"but only {len(free)} are currently available."
            )

        indices = self.rng.choice(len(free), size=count, replace=False)

        return [free[i] for i in indices]
    def _spawn_network_positions(self, room_name, positions):
        initial_target = self.scenario.initial_targets[room_name]
        journey_id, stage_id = self.routes[initial_target]

        for position in positions:
            parameters = jps.CollisionFreeSpeedModelAgentParameters(
                position=position,
                journey_id=journey_id,
                stage_id=stage_id,
            )

            agent_id = self.simulation.add_agent(parameters)

            self.assignments[agent_id] = initial_target
            self.active_junctions[agent_id] = None
            self.agent_origins[agent_id] = room_name
    # ==================================================================
    # Guidance
    # ==================================================================

    def apply_guidance(self, guidance):
        """
        Apply guidance according to the scenario routing mode.
        """

        self._require_simulation()

        if self.scenario.routing_mode == "regional":

            self._apply_regional_guidance(
                guidance
            )

        elif self.scenario.routing_mode == "network":

            self._set_network_guidance(
                guidance
            )

        else:
            raise ValueError(
                f"Unknown routing mode: "
                f"{self.scenario.routing_mode}"
            )

    # ------------------------------------------------------------------
    # v1 guidance
    # ------------------------------------------------------------------

    def _apply_regional_guidance(
        self,
        guidance,
    ):
        """
        Original v1 behaviour.

        Example:

            {
                "left": "left",
                "right": "right",
            }

        Pedestrians inside the guidance area are immediately
        rerouted according to their spatial region.
        """
        for agent in self.simulation.agents():

            if agent.id in self.committed_agents:
                continue

            position = Point(agent.position)

            if not self.scenario.guidance_area.covers(
                position
            ):
                continue

            x, _ = agent.position

            group = (
                self._current_guidance_group(x)
            )

            target = guidance[group]

            if (
                self.assignments[agent.id]
                == target
            ):
                continue

            self._switch_agent_target(
                agent.id,
                target,
            )

    # ------------------------------------------------------------------
    # v2 guidance
    # ------------------------------------------------------------------

    def _set_network_guidance(
        self,
        guidance,
    ):
        """
        Change what the junction signs currently display.

        This does NOT immediately reroute pedestrians.

        A pedestrian reads the sign only when entering the
        corresponding physical junction.
        
        Guidance is a dictionary of Junction:Next Destination e.g {J1:A, J2:J1, J3:B}
        """

        
        for junction, target in guidance.items():

            # Error handling
            if (junction not in self.scenario.guidance_choices):
                raise ValueError(
                    f"Unknown junction: {junction}"
                )

            # The valid choices that the current junction can lead to
            valid_choices = (self.scenario.guidance_choices[junction])

            # Error handling for invalid routes
            if target not in valid_choices:
                raise ValueError(
                    f"Invalid guidance choice "
                    f"{junction} -> {target}. "
                    f"Valid choices: {valid_choices}"
                )

            self.current_guidance[junction] = target

    @property
    def has_pending_events(self):
        return any(not event["triggered"] for event in self.pending_events)

    # Trigger the dynamic events
    def _trigger_due_events(self):

        for event in self.pending_events:
            if event["triggered"]:
                continue

            if self.elapsed_time < event["time"]:
                continue
            incident = event.get("incident")
            if incident is not None:
                self.simulation.switch_geometry(
                    self.scenario.incident_geometries[incident]
                )
                self.active_incident = incident
            positions = self._get_free_spawn_positions(
                event["room"], event["count"]
            )

            self._spawn_network_positions(event["room"], positions)
            event["triggered"] = True

            # print(
            #     f"[Dynamic event] t={self.elapsed_time:.2f}s | "
            #     f"+{event['count']} agents in Room {event['room']}"
            # )
    # ==================================================================
    # Time advancement
    # ==================================================================
    def advance(self, seconds):
        target_time = self.elapsed_time + seconds

        iterate_time = 0.0
        routing_time = 0.0
        iterations = 0
        routing_checks = 0

        start_total = time.perf_counter()

        while self.elapsed_time < target_time:
            if self.simulation.agent_count() == 0 and not self.has_pending_events:
                break
            time_before = self.elapsed_time

            start = time.perf_counter()
            self.simulation.iterate()
            iterate_time += time.perf_counter() - start

            self._trigger_due_events()

            # Measure actual simulated time advanced by JuPedSim.
            dt = self.elapsed_time - time_before
            self._time_since_routing_check += dt

            # Only check junction entries every 0.1 simulated seconds.
            if self._time_since_routing_check >= self.routing_check_interval:
                start = time.perf_counter()
                self._handle_junction_entries()
                routing_time += time.perf_counter() - start

                routing_checks += 1
                self._time_since_routing_check -= self.routing_check_interval

            iterations += 1

        start = time.perf_counter()
        state = self.get_state()
        state_time = time.perf_counter() - start

        total_time = time.perf_counter() - start_total

        self.last_profile = {
            "iterations": iterations,
            "routing_checks": routing_checks,
            "iterate_time": iterate_time,
            "routing_time": routing_time,
            "state_time": state_time,
            "total_time": total_time,
        }

        return state
    # ==================================================================
    # v2 junction handling
    # ==================================================================

    def _handle_junction_entries(self):
        """
        Detect pedestrians entering controlled junctions.

        An agent reads a sign exactly once per junction encounter.

        Once the agent leaves the junction polygon,
        active_junctions[agent_id] returns to None.

        If they later return, that counts as a new encounter.
        """

        for agent in self.simulation.agents():

            agent_id = agent.id

            position = Point(agent.position)

            current_junction = (self._junction_containing(position))

            previous_junction = (self.active_junctions.get(agent_id))

            # Not currently inside a junction
            if current_junction is None:
                self.active_junctions[agent_id] = None
                continue

            # Still inside the same junction
            if (current_junction== previous_junction):
                continue

            # Newly entered a junction
            if (current_junction not in self.current_guidance):
                raise RuntimeError(
                    f"Agent {agent_id} entered "
                    f"{current_junction}, but no "
                    f"guidance has been configured "
                    f"for that junction."
                )

            # Newly Entered a junction, reset target based on the junction it entered
            target = (self.current_guidance[current_junction])


            self._switch_agent_target(agent_id,target,)

            # Maintain bookkeeping, this agent is now in this newly entered junction
            self.active_junctions[agent_id] = current_junction

    def _junction_containing(self,position,):
        """
        Return the name of the junction containing this position.

        Returns None when the pedestrian is outside all controlled
        junctions.
        """

        for (junction_name,junction_polygon) in self.scenario.junctions.items():

            if junction_polygon.covers(position):
                return junction_name

        return None

    # ==================================================================
    # Route switching
    # ==================================================================

    def _switch_agent_target(self,agent_id,target,):
        """
        Switch an agent toward either:

            - an exit
            - another junction waypoint
        """

        if target not in self.routes:
            raise ValueError(
                f"Unknown routing target: {target}"
            )

        journey_id, stage_id = (self.routes[target])

        self.simulation.switch_agent_journey(
            agent_id,
            journey_id,
            stage_id,
        )

        self.assignments[agent_id] = target

    # ==================================================================
    # State
    # ==================================================================

    def get_state(self):
        """
        Convert JuPedSim's internal state into our simulator-independent
        SimulationState representation.
        """

        self._require_simulation()

        agents = []

        for agent in self.simulation.agents():

            x, y = agent.position

            agents.append(
                AgentState(
                    id=agent.id,
                    position=(float(x),float(y),),
                    assignment=(self.assignments.get(agent.id)),
                    committed=(agent.id in self.committed_agents),
                )
            )

        return SimulationState(
            elapsed_time=(self.simulation.elapsed_time()),
            agents=agents,
            initial_population=(self.initial_population),
            active_incident=(self.active_incident)
        )

    # ==================================================================
    # Convenience information
    # ==================================================================

    @property
    def elapsed_time(self):

        self._require_simulation()

        return (
            self.simulation.elapsed_time()
        )

    @property
    def remaining_agents(self):

        self._require_simulation()

        return (
            self.simulation.agent_count()
        )

    @property
    def is_evacuated(self):

        return (
            self.remaining_agents == 0
        )

    # ==================================================================
    # Original v1 helpers
    # ==================================================================

    def _initial_group(
        self,
        position,
    ):
        """
        Reproduce original v1 reset behaviour.
        """

        x, _ = position

        split = (
            self.scenario.guidance_split_x
        )

        if x < split:
            return "left"

        if x > split:
            return "right"

        return (
            "left"
            if self.rng.random() < 0.5
            else "right"
        )

    def _current_guidance_group(
        self,
        x,
    ):
        """
        Reproduce original v1 action behaviour.
        """

        if (
            x
            <= self.scenario.guidance_split_x
        ):
            return "left"

        return "right"

    def _update_committed_agents(self):
        """
        Original v1 commitment behaviour.
        """

        for agent in self.simulation.agents():

            position = Point(
                agent.position
            )

            if not (
                self.scenario.guidance_area.covers(
                    position
                )
            ):
                self.committed_agents.add(
                    agent.id
                )

    # ==================================================================
    # Validation / cleanup
    # ==================================================================

    def _require_simulation(self):

        if self.simulation is None:
            raise RuntimeError(
                "Simulation has not been created. "
                "Call reset() first."
            )

    def _close_writer(self):

        if self.writer is not None:

            self.writer.close()

            self.writer = None

    def close(self):

        self._close_writer()

        self.simulation = None