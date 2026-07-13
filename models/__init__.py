"""Model-specific implementations."""

from .base import BasePatch, AttentionCapture, ModelConfig

# Model-specific imports
from . import qwen
from . import chatglm

__all__ = [
    "BasePatch",
    "AttentionCapture", 
    "ModelConfig",
    "qwen",
    "chatglm",
]
