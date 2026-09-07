import pathlib

import jupedsim as jps
import numpy as np
from shapely import Point

from simulation.state import AgentState, SimulationState


class JuPedSimBackend:
    """
    JuPedSim implementation of the evacuation simulation.

    This class knows about JuPedSim, but knows nothing about Gymnasium,
    PPO, rewards, observations or action spaces.
    """

    def __init__(
        self,
        scenario,
        record=False,
        trajectory_path="integratedgym_evac.sqlite",
    ):
        self.scenario = scenario

        self.record = record
        self.trajectory_path = pathlib.Path(trajectory_path)

        self.simulation = None
        self.writer = None

        # JuPedSim route information:
        #
        # {
        #     "left": (journey_id, exit_stage_id),
        #     "right": (journey_id, exit_stage_id)
        # }
        self.routes = {}

        # Current route assignment of each pedestrian.
        self.assignments = {}

        # Pedestrians that have left the guidance area
        # and therefore no longer respond to guidance.
        self.committed_agents = set()

        # Used for reproducible randomness.
        self.rng = None

    # Simulation creation
    def reset(self, seed=None):

        self._close_writer()

        self.rng = np.random.default_rng(seed)

        #  Optional trajectory recording 

        if self.record:

            # Avoid accidentally trying to reuse an old JuPedSim database.
            if self.trajectory_path.exists():
                self.trajectory_path.unlink()

            self.writer = jps.SqliteTrajectoryWriter(
                output_file=self.trajectory_path,
                every_nth_frame=5,
            )

        else:
            self.writer = None

        # Create JuPedSim simulation 

        self.simulation = jps.Simulation(
            model=jps.CollisionFreeSpeedModel(),
            geometry=self.scenario.geometry,
            trajectory_writer=self.writer,
        )

        # ---------- Create exits and journeys ----------

        self.routes = {}

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

        # Episode bookkeeping, which pedestrian heading where, who has commited to one exit

        self.assignments = {}
        self.committed_agents = set()

        # ---------- Spawn pedestrians ----------

        for position in self.scenario.start_positions:

            # We only need some valid initial journey.
            #
            # The first Gymnasium action will overwrite it before
            # any simulation movement occurs.
            group = self._initial_group(position)

            journey_id, stage_id = self.routes[group]

            parameters = (
                jps.CollisionFreeSpeedModelAgentParameters(
                    position=position,
                    journey_id=journey_id,
                    stage_id=stage_id,
                )
            )

            agent_id = self.simulation.add_agent(parameters)

            # No RL guidance has actually been applied yet.
            self.assignments[agent_id] = None

        return self.get_state()

    # ------------------------------------------------------------------
    # Guidance
    # ------------------------------------------------------------------

    def apply_guidance(self, guidance):
        """
        Apply the current guidance configuration.

        For the v1 scenario, guidance looks like:

            {
                "left": "left",
                "right": "right"
            }

        Meaning:

            pedestrians in the left guidance region -> left exit
            pedestrians in the right guidance region -> right exit
        """

        self._require_simulation()

        for agent in self.simulation.agents():

            # Once committed, do not reroute.
            if agent.id in self.committed_agents:
                continue

            position = Point(agent.position)

            # Only pedestrians still inside the guidance-controlled
            # region respond to the signs.
            if not self.scenario.guidance_area.covers(position):
                continue

            x, _ = agent.position

            group = self._current_guidance_group(x)

            target = guidance[group]

            # Avoid issuing the same reroute repeatedly.
            if self.assignments[agent.id] == target:
                continue

            journey_id, stage_id = self.routes[target]

            self.simulation.switch_agent_journey(
                agent.id,
                journey_id,
                stage_id,
            )

            self.assignments[agent.id] = target

    # ------------------------------------------------------------------
    # Time advancement
    # ------------------------------------------------------------------

    def advance(self, seconds):
        """
        Advance JuPedSim by approximately `seconds`
        of simulated time.
        """

        self._require_simulation()

        iterations = round(
            seconds / self.simulation.delta_time()
        )

        for _ in range(iterations):

            if self.simulation.agent_count() == 0:
                break

            self.simulation.iterate()

        self._update_committed_agents()

        return self.get_state()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

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
                    position=(float(x), float(y)),
                    assignment=self.assignments.get(agent.id),
                    committed=agent.id in self.committed_agents,
                )
            )

        return SimulationState(
            elapsed_time=self.simulation.elapsed_time(),
            agents=agents,
            initial_population=self.scenario.initial_population,
        )

    # ------------------------------------------------------------------
    # Convenience information
    # ------------------------------------------------------------------

    @property
    def elapsed_time(self):
        self._require_simulation()
        return self.simulation.elapsed_time()

    @property
    def remaining_agents(self):
        self._require_simulation()
        return self.simulation.agent_count()

    @property
    def is_evacuated(self):
        return self.remaining_agents == 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _initial_group(self, position):
        """
        Reproduce the original v1 reset behaviour.

        Pedestrians exactly on the split line are randomly assigned
        an initial placeholder journey.
        """

        x, _ = position
        split = self.scenario.guidance_split_x

        if x < split:
            return "left"

        if x > split:
            return "right"

        return (
            "left"
            if self.rng.random() < 0.5
            else "right"
        )

    def _current_guidance_group(self, x):
        """
        Reproduce the original v1 action behaviour.

        Notice that x == split belongs to the left region here,
        matching the previous _apply_action().
        """

        if x <= self.scenario.guidance_split_x:
            return "left"

        return "right"

    def _update_committed_agents(self):

        for agent in self.simulation.agents():

            position = Point(agent.position)

            if not self.scenario.guidance_area.covers(position):
                self.committed_agents.add(agent.id)

    def _require_simulation(self):

        if self.simulation is None:
            raise RuntimeError(
                "Simulation has not been created. Call reset() first."
            )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _close_writer(self):

        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def close(self):

        self._close_writer()
        self.simulation = None