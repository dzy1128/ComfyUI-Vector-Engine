"""
Vector Engine Image Generator Node for ComfyUI

This custom node integrates Vector Engine's Gemini API for image generation
into ComfyUI workflow.
"""

from .vector_engine_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

