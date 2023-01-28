from dataclasses import dataclass

@dataclass
class AgentConfig:
    num_samples: int = 10
    epsilon: float = 0.1
    gamma: float = .9