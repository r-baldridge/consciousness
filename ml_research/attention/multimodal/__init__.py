"""
Multimodal Learning Methods

This module contains research index entries for multimodal machine learning
methods that combine multiple data modalities (vision, language, audio).

Methods included:
- CLIP (2021): Contrastive Language-Image Pre-training
- Flamingo (2022): Visual language model for few-shot learning
- LLaVA (2023): Large Language and Vision Assistant
- GPT-4V (2023): Multimodal GPT-4 with vision capabilities
"""

from .clip import get_method_info as get_clip_info
from .flamingo import get_method_info as get_flamingo_info
from .llava import get_method_info as get_llava_info
from .gpt4v import get_method_info as get_gpt4v_info

__all__ = [
    "get_clip_info",
    "get_flamingo_info",
    "get_llava_info",
    "get_gpt4v_info",
]
