import logging
from typing import Dict, Tuple

import cv2
import numpy as np
import torch

from .feature_matcher import FeatureMatcher

logger = logging.getLogger("MatchingPipeline.KNNMatcher")


class KNNMatcher(FeatureMatcher):
    """
    KNN-based feature matcher implementation.
    
    This class implements feature matching using K-Nearest Neighbors with 
    Lowe's ratio test for filtering.
    """

    def __init__(self, ratio_threshold: float = 0.8, distance_type: str = "L2"):
        """
        Initialize a KNN matcher.
        
        Args:
            ratio_threshold: Threshold for Lowe's ratio test (default: 0.8)
            distance_type: Distance metric to use: "L2", "L1", or "Hamming"
        """
        self.ratio_threshold = ratio_threshold

        # Convert distance type string to OpenCV constant
        if distance_type == "L2":
            self.distance_type = cv2.NORM_L2
        elif distance_type == "L1":
            self.distance_type = cv2.NORM_L1
        elif distance_type == "Hamming":
            self.distance_type = cv2.NORM_HAMMING
        else:
            raise ValueError(f"Unknown distance type: {distance_type}")

        logger.info(f"Initialized KNN matcher with ratio threshold: {ratio_threshold}")

    def match(self, data: Dict) -> Dict:
        """
        Match features using KNN with ratio test.
        
        Args:
            data: Dictionary containing descriptors0 and descriptors1
            
        Returns:
            Dictionary with matches0 and matching_scores0
        """
        desc0 = data["descriptors0"]
        desc1 = data["descriptors1"]

        matches, match_confidence = self._match_descriptors_knn(
            desc0, desc1, self.ratio_threshold, self.distance_type
        )

        return {"matches0": matches, "matching_scores0": match_confidence}

    def _match_descriptors_knn(
        self, desc0, desc1, knn_ratio=0.85, distance_type=cv2.NORM_L2
    ):
        """
        Match descriptors using KNN matcher with ratio test.

        Args:
            desc0, desc1: Descriptors from images 0 and 1
            knn_ratio: Ratio test threshold (0.8 is typical)
            distance_type: Distance metric (L2, Hamming, etc.)

        Returns:
            matches: Array of indices for matching points (-1 if no match)
            match_confidence: Confidence scores for each match
        """
        # Convert lists to numpy arrays if needed
        if isinstance(desc0, list):
            desc0 = np.array(desc0)
        if isinstance(desc1, list):
            desc1 = np.array(desc1)

        # Convert descriptors to proper format if needed
        if isinstance(desc0, torch.Tensor):
            desc0 = desc0.detach().cpu().numpy()
        if isinstance(desc1, torch.Tensor):
            desc1 = desc1.detach().cpu().numpy()

        # SuperPoint descriptors are shape (dim, num_keypoints), but OpenCV expects (num_keypoints, dim)
        # if desc0.shape[0] < desc0.shape[1]:
        #     desc0 = desc0.T
        # if desc1.shape[0] < desc1.shape[1]:
        #     desc1 = desc1.T
        desc0 = desc0.T
        desc1 = desc1.T
        
        # Ensure we have the right type - OpenCV needs float32
        desc0 = desc0.astype(np.float32)
        desc1 = desc1.astype(np.float32)

        # Print descriptor shapes for debugging
        logger.info(
            f"KNN Matching: Descriptor shapes - desc0: {desc0.shape}, desc1: {desc1.shape}"
        )

        # Sanity checks
        assert desc0.shape[1] == desc1.shape[1], "Descriptor dimensions don't match"

        # Create matcher and match descriptors
        matcher = cv2.BFMatcher(distance_type)

        try:
            raw_matches = matcher.knnMatch(desc0, desc1, k=2)

            # Apply ratio test
            matches = (
                np.ones(len(desc0), dtype=int) * -1
            )  # Initialize all to -1 (no match)
            match_confidence = np.zeros(len(desc0), dtype=float)

            for i, (m, n) in enumerate(raw_matches):
                if m.distance < knn_ratio * n.distance:
                    matches[m.queryIdx] = m.trainIdx  # Store index of match in desc1
                    # Convert distance to confidence score (higher is better)
                    confidence = 1.0 - m.distance / n.distance
                    match_confidence[m.queryIdx] = confidence

        except Exception as e:
            logger.error(f"Error in knnMatch: {e}")
            logger.error(
                f"Descriptor details - desc0: shape={desc0.shape}, type={desc0.dtype}"
            )
            logger.error(
                f"Descriptor details - desc1: shape={desc1.shape}, type={desc1.dtype}"
            )
            # Return empty matches as fallback
            matches = np.ones(len(desc0), dtype=int) * -1
            match_confidence = np.zeros(len(desc0), dtype=float)

        return matches, match_confidence
