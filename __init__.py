"""
Feature matching pipeline for image pairs.

This package provides tools for matching features between image pairs
using methods like SuperGlue and KNN with RANSAC filtering.
"""

from data import ImagePair
from evaluation.pose_estimator import PoseEstimator
from filters.ransac_filter import MatchFilter, RANSACFilter
from matchers.feature_matcher import FeatureMatcher
from matchers.knn_matcher import KNNMatcher
from pipeline import MatchingPipeline
from visualization.visualizer import Visualizer

__all__ = [
    'ImagePair',
    'FeatureMatcher',
    'KNNMatcher',
    'MatchFilter',
    'RANSACFilter',
    'PoseEstimator',
    'Visualizer',
    'MatchingPipeline',
]
