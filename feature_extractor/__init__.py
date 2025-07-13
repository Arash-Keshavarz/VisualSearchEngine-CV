"""
Feature Extraction System for Visual Search Engine

This package provides feature extraction capabilities using state-of-the-art
vision models including ViT (Vision Transformer) and CLIP.

Classes:
    BaseFeatureExtractor: Base class for all feature extractors
    ViTFeatureExtractor: Vision Transformer feature extractor
    CLIPFeatureExtractor: CLIP feature extractor
    ModelComparator: Compare different feature extraction models
"""

from .base_extractor import BaseFeatureExtractor
from .vit_extractor import ViTFeatureExtractor
from .clip_extractor import CLIPFeatureExtractor
from .compare_models import ModelComparator

__version__ = "1.0.0"
__author__ = "Visual Search Engine Team"

__all__ = [
    "BaseFeatureExtractor",
    "ViTFeatureExtractor", 
    "CLIPFeatureExtractor",
    "ModelComparator"
] 