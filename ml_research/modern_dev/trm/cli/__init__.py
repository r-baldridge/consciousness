"""TRM CLI modules."""
from .train import main as train
from .infer import main as infer
from .agent import TRMAgent, create_agent, main as agent

__all__ = ["train", "infer", "agent", "TRMAgent", "create_agent"]
