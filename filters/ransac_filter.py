import logging
from abc import ABC, abstractmethod
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger("MatchingPipeline.RANSACFilter")


class MatchFilter(ABC):
    """
    Base class for match filters like RANSAC.
    
    This abstract class defines the interface for algorithms that filter
    matches based on geometric consistency.
    """

    @abstractmethod
    def filter(self, mkpts0, mkpts1, mconf) -> Tuple:
        """
        Filter matches based on geometric consistency.
        
        Args:
            mkpts0: Keypoints from the first image
            mkpts1: Keypoints from the second image
            mconf: Confidence scores for the matches
            
        Returns:
            Tuple containing filtered keypoints and a mask indicating inliers
        """
        pass


class RANSACFilter(MatchFilter):
    """
    RANSAC-based match filter implementation.
    
    This class filters matches using RANSAC to estimate either a fundamental
    matrix or homography.
    """

    def __init__(self, threshold: float = 3.0, method: str = "fundamental"):
        """
        Initialize a RANSAC filter.
        
        Args:
            threshold: RANSAC threshold in pixels (default: 3.0)
            method: Either "fundamental" or "homography" (default: "fundamental")
        """
        self.threshold = threshold
        self.method = method
        logger.info(
            f"Initialized RANSAC filter with threshold={threshold}, method={method}"
        )

    def filter(self, mkpts0, mkpts1, mconf) -> Tuple:
        """
        Filter matches using RANSAC.
        
        Args:
            mkpts0: Keypoints from the first image
            mkpts1: Keypoints from the second image
            mconf: Confidence scores for the matches
            
        Returns:
            Tuple containing:
            - mkpts0_ransac: Filtered keypoints from the first image
            - mkpts1_ransac: Filtered keypoints from the second image
            - mconf_ransac: Filtered confidence scores
            - mask: Boolean mask indicating inliers
        """
        return self.filter_matches_with_ransac(
            mkpts0, mkpts1, mconf, self.threshold, self.method
        )

    def filter_matches_with_ransac(
        self, mkpts0, mkpts1, mconf, ransac_threshold=3.0, method="fundamental"
    ):
        """
        Filter matches using RANSAC with either homography or fundamental matrix estimation.

        Args:
            mkpts0, mkpts1: Nx2 arrays containing matched keypoints from images 0 and 1
            mconf: Confidence scores for the matches
            ransac_threshold: RANSAC threshold for filtering
            method: 'fundamental' or 'homography'

        Returns:
            mkpts0_ransac, mkpts1_ransac: Filtered keypoints (inliers only)
            mconf_ransac: Filtered confidence scores
            mask: Boolean mask indicating inliers
        """
        if len(mkpts0) < 8:
            # Need at least 8 points for fundamental matrix estimation
            # or 4 points for homography
            return mkpts0, mkpts1, mconf, np.ones(len(mkpts0), dtype=bool)

        if method == "fundamental":
            # Find fundamental matrix using RANSAC
            F, mask = cv2.findFundamentalMat(
                mkpts0, mkpts1, cv2.FM_RANSAC, ransac_threshold, 0.999
            )

            # Handle case when no fundamental matrix is found
            if F is None or F.shape == (0, 0):
                return mkpts0, mkpts1, mconf, np.ones(len(mkpts0), dtype=bool)

        elif method == "homography":
            # Find homography using RANSAC
            H, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, ransac_threshold)

            # Handle case when no homography is found
            if H is None:
                return mkpts0, mkpts1, mconf, np.ones(len(mkpts0), dtype=bool)
        else:
            raise ValueError(f"Unknown RANSAC method: {method}")

        # Convert mask to boolean array if it's not already
        if isinstance(mask, np.ndarray) and mask.ndim > 1:
            mask = mask.ravel().astype(bool)
        else:
            mask = mask.astype(bool)

        # Return filtered matches
        mkpts0_ransac = mkpts0[mask]
        mkpts1_ransac = mkpts1[mask]
        mconf_ransac = mconf[mask] if mconf is not None else None

        return mkpts0_ransac, mkpts1_ransac, mconf_ransac, mask
