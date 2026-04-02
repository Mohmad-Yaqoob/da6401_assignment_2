"""
classification.py — re-exports VGG11 so the autograder can do:
    from models.classification import VGG11
"""

from models.vgg11 import VGG11

__all__ = ["VGG11"]