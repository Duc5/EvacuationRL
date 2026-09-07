from dataclasses import dataclass
from typing import Optional


@dataclass
class AgentState:
    """
    Simulator-independent snapshot of one pedestrian.
    For gymnasium to retrieve data of one pedestrian without exposing the entire Agent object
    """

    id: int
    position: tuple[float, float]
    assignment: Optional[str] = None
    committed: bool = False


@dataclass
class SimulationState:
    """
    Simulator-independent snapshot of the whole evacuation.
    Expose the state of the environment in jupedsim for gymnasium
    """

    elapsed_time: float
    agents: list[AgentState]
    initial_population: int

    @property
    def remaining_agents(self) -> int:
        return len(self.agents)

    @property
    def evacuated_agents(self) -> int:
        return self.initial_population - self.remaining_agents

    @property
    def is_evacuated(self) -> bool:
        return self.remaining_agents == 0