import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

# Import utility functions
from models.utils import (compute_epipolar_error, compute_pose_error,
                          estimate_pose)

logger = logging.getLogger("MatchingPipeline.PoseEstimator")


class PoseEstimator:
    """
    Class for computing and evaluating relative poses.
    
    This class estimates the relative pose between two images based on matched
    keypoints and evaluates the results against ground truth.
    """

    def __init__(self, threshold: float = 1.0):
        """
        Initialize a PoseEstimator.
        
        Args:
            threshold: Threshold in pixels for pose estimation (default: 1.0)
        """
        self.threshold = threshold
        logger.info(f"Initialized PoseEstimator with threshold={threshold}")

    def estimate_and_evaluate(
        self,
        mkpts0: np.ndarray,
        mkpts1: np.ndarray,
        K0: np.ndarray,
        K1: np.ndarray,
        T_0to1: np.ndarray,
        matches: np.ndarray,
    ) -> Dict:
        """
        Estimate pose and compute evaluation metrics.

        Args:
            mkpts0, mkpts1: Matched keypoints
            K0, K1: Camera intrinsic matrices
            T_0to1: Ground truth transformation matrix
            matches: All matches (for statistics)

        Returns:
            Dictionary containing evaluation results:
            - error_t: Translation error
            - error_R: Rotation error
            - precision: Percentage of correct matches
            - matching_score: Percentage of correct matches among all possible matches
            - num_correct: Number of correct matches
            - epipolar_errors: Epipolar errors for each match
        """
        # Compute epipolar errors
        epi_errs = compute_epipolar_error(mkpts0, mkpts1, T_0to1, K0, K1)
        correct = epi_errs < 5e-4
        num_correct = np.sum(correct)
        precision = np.mean(correct) if len(correct) > 0 else 0
        matching_score = num_correct / len(matches) if len(matches) > 0 else 0

        # Estimate pose
        ret = estimate_pose(mkpts0, mkpts1, K0, K1, self.threshold)
        if ret is None:
            err_t, err_R = np.inf, np.inf
        else:
            R, t, inliers = ret
            err_t, err_R = compute_pose_error(T_0to1, R, t)

        # Return evaluation results
        return {
            "error_t": err_t,
            "error_R": err_R,
            "precision": precision,
            "matching_score": matching_score,
            "num_correct": num_correct,
            "epipolar_errors": epi_errs,
        }
