"""
Vertex AI Realtime API module.

This module provides support for Vertex AI's Realtime (Live) API through
LiteLLM's unified realtime interface.
"""

from .handler import VertexAIRealtime
from .transformation import VertexAIRealtimeConfig

__all__ = ["VertexAIRealtime", "VertexAIRealtimeConfig"]
