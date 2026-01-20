"""
World Models Research Index

This module contains research index entries for world model architectures
that learn latent dynamics for planning and imagination-based learning.

Methods included:
- World Models (2018): Ha & Schmidhuber - V-M-C architecture
- Dreamer (2019-2023): Hafner et al. - RSSM-based imagination
- Genie (2024): Generative Interactive Environments
"""

from .world_model_basics import (
    get_method_info as get_world_model_info,
    WORLD_MODEL,
    VMCArchitecture,
)
from .dreamer import (
    get_dreamer_v1_info,
    get_dreamer_v2_info,
    get_dreamer_v3_info,
    DREAMER_V1,
    DREAMER_V2,
    DREAMER_V3,
    RSSMArchitecture,
)
from .genie import (
    get_method_info as get_genie_info,
    GENIE,
    GenieArchitecture,
)

__all__ = [
    # World Models
    "get_world_model_info",
    "WORLD_MODEL",
    "VMCArchitecture",
    # Dreamer
    "get_dreamer_v1_info",
    "get_dreamer_v2_info",
    "get_dreamer_v3_info",
    "DREAMER_V1",
    "DREAMER_V2",
    "DREAMER_V3",
    "RSSMArchitecture",
    # Genie
    "get_genie_info",
    "GENIE",
    "GenieArchitecture",
]
