#! /usr/bin/env python3
#
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#                       Daniel DeTone
#                       Tomasz Malisiewicz
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

import argparse
import logging
# feature_booster_path = Path(__file__).parent / "FeatBooster/FeatureBooster"
# sys.path.append(str(feature_booster_path))
import os
import random
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from loguru import logger

from models.matching import FeatBoostEnhancedMatching, Matching
# Use utils.py functions to create the visualization
from models.utils import (AverageTimer, compute_epipolar_error,
                          compute_pose_error, error_colormap, estimate_pose,
                          make_matching_plot, plot_image_pair, plot_keypoints,
                          plot_matches, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)

# from featurebooster import FeatureBooster

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("MatchingPipeline")

torch.set_grad_enabled(False)


class ImagePair:
    """Class representing an image pair with associated data."""

    def __init__(
        self,
        name0: str,
        name1: str,
        stem0: str,
        stem1: str,
        rot0: int = 0,
        rot1: int = 0,
        intrinsics: Optional[Dict] = None,
        extrinsics: Optional[np.ndarray] = None,
    ):
        self.name0 = name0
        self.name1 = name1
        self.stem0 = stem0
        self.stem1 = stem1
        self.rot0 = rot0
        self.rot1 = rot0
        self.image0 = None
        self.image1 = None
        self.inp0 = None
        self.inp1 = None
        self.image0_prep = None
        self.image1_prep = None
        self.inp0_prep = None
        self.inp1_prep = None
        self.scales0 = None
        self.scales1 = None
        self.intrinsics = intrinsics  # Dictionary containing K0, K1 matrices
        self.extrinsics = extrinsics  # T_0to1 transformation matrix
        self.shape0 = np.zeros(2)
        self.shape1 = np.zeros(2)

        # Paths for outputs
        self.matches_path = None
        self.eval_path = None
        self.viz_path = None
        self.viz_eval_path = None

    def set_paths(self, output_dir: Path, viz_extension: str = "png"):
        """Set output file paths for this image pair"""
        self.matches_path = output_dir / f"{self.stem0}_{self.stem1}_matches.npz"
        self.eval_path = output_dir / f"{self.stem0}_{self.stem1}_evaluation.npz"
        self.viz_path = (
            output_dir / f"{self.stem0}_{self.stem1}_matches.{viz_extension}"
        )
        self.viz_eval_path = (
            output_dir / f"{self.stem0}_{self.stem1}_evaluation.{viz_extension}"
        )

class FeatureMatcher(ABC):
    """Base class for feature matchers like SuperGlue."""

    @abstractmethod
    def match(self, data: Dict) -> Dict:
        """Match features between two images"""
        pass

class KNNMatcher(FeatureMatcher):
    """KNN-based feature matcher implementation."""

    def __init__(self, ratio_threshold: float = 0.8, distance_type: str = "L2"):
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
        """Match features using KNN with ratio test"""
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
        Match descriptors using KNN matcher with ratio test

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
        if desc0.shape[0] < desc0.shape[1]:
            desc0 = desc0.T
        if desc1.shape[0] < desc1.shape[1]:
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


class MatchFilter(ABC):
    """Base class for match filters like RANSAC."""

    @abstractmethod
    def filter(self, mkpts0, mkpts1, mconf) -> Tuple:
        """Filter matches based on geometric consistency"""
        pass


class RANSACFilter(MatchFilter):
    """RANSAC-based match filter implementation."""

    def __init__(self, threshold: float = 3.0, method: str = "fundamental"):
        self.threshold = threshold
        self.method = method
        logger.info(
            f"Initialized RANSAC filter with threshold={threshold}, method={method}"
        )

    def filter(self, mkpts0, mkpts1, mconf) -> Tuple:
        """Filter matches using RANSAC"""
        return self.filter_matches_with_ransac(
            mkpts0, mkpts1, mconf, self.threshold, self.method
        )

    def filter_matches_with_ransac(
        self, mkpts0, mkpts1, mconf, ransac_threshold=3.0, method="fundamental"
    ):
        """
        Filter matches using RANSAC with either homography or fundamental matrix estimation

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


class PoseEstimator:
    """Class for computing and evaluating relative poses."""

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold

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
        Estimate pose and compute evaluation metrics

        Args:
            mkpts0, mkpts1: Matched keypoints
            K0, K1: Camera intrinsic matrices
            T_0to1: Ground truth transformation matrix
            matches: All matches (for statistics)

        Returns:
            Dictionary containing evaluation results
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


class Visualizer:
    """Class for visualizing matching results."""

    def __init__(
        self,
        output_dir: Path,
        viz_extension: str = "png",
        show_keypoints: bool = False,
        fast_viz: bool = False,
        opencv_display: bool = False,
        line_width: float = 0.3,
    ):
        self.output_dir = output_dir
        self.viz_extension = viz_extension
        self.show_keypoints = show_keypoints
        self.fast_viz = fast_viz
        self.opencv_display = opencv_display
        self.line_width = line_width

        # Ensure output directory exists
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def visualize_matches(
        self,
        image_pair: ImagePair,
        kpts0: np.ndarray,
        kpts1: np.ndarray,
        mkpts0: np.ndarray,
        mkpts1: np.ndarray,
        mconf: np.ndarray,
        text: List[str],
        title: str = "Matches",
        small_text: List[str] = [],
    ):
        """Visualize matches between two images"""
        color = cm.jet(mconf)

        make_matching_plot(
            image_pair.image0,
            image_pair.image1,
            kpts0,
            kpts1,
            mkpts0,
            mkpts1,
            color,
            text,
            image_pair.viz_path,
            self.show_keypoints,
            self.fast_viz,
            self.opencv_display,
            title,
            small_text,
            # self.line_width,
        )

    def visualize_ransac_comparison(
        self,
        image_pair: ImagePair,
        kpts0: np.ndarray,
        kpts1: np.ndarray,
        mkpts0: np.ndarray,
        mkpts1: np.ndarray,
        mkpts0_ransac: np.ndarray,
        mkpts1_ransac: np.ndarray,
        mconf: np.ndarray,
        mconf_ransac: np.ndarray,
        text: List[str],
    ):
        """Visualize comparison between original and RANSAC-filtered matches"""
        comparison_path = (
            self.output_dir
            / f"{image_pair.stem0}_{image_pair.stem1}_ransac_comparison.{self.viz_extension}"
        )
        color = cm.jet(mconf)
        ransac_color = cm.winter(mconf_ransac)

        self.make_matching_comparison_plot(
            image_pair.image0,
            image_pair.image1,
            kpts0,
            kpts1,
            mkpts0,
            mkpts1,
            mkpts0_ransac,
            mkpts1_ransac,
            color,
            ransac_color,
            text,
            comparison_path,
            self.show_keypoints,
            self.fast_viz,
            self.line_width,
        )

    def visualize_matcher_comparison(
        self,
        image_pair: ImagePair,
        kpts0: np.ndarray,
        kpts1: np.ndarray,
        mkpts0_sg: np.ndarray,
        mkpts1_sg: np.ndarray,
        mconf_sg: np.ndarray,
        mkpts0_knn: np.ndarray,
        mkpts1_knn: np.ndarray,
        mconf_knn: np.ndarray,
        text_sg: List[str],
        text_knn: List[str],
    ):
        """Visualize comparison between SuperGlue and KNN matches"""
        comparison_path = (
            self.output_dir
            / f"{image_pair.stem0}_{image_pair.stem1}_sg_vs_knn.{self.viz_extension}"
        )

        self.make_superglue_knn_comparison_plot(
            image_pair.image0,
            image_pair.image1,
            kpts0,
            kpts1,
            mkpts0_sg,
            mkpts1_sg,
            mconf_sg,
            mkpts0_knn,
            mkpts1_knn,
            mconf_knn,
            text_sg,
            text_knn,
            comparison_path,
            self.fast_viz,
            self.line_width,
        )

    def visualize_evaluation(
        self,
        image_pair: ImagePair,
        kpts0: np.ndarray,
        kpts1: np.ndarray,
        mkpts0: np.ndarray,
        mkpts1: np.ndarray,
        epi_errs: np.ndarray,
        err_t: float,
        err_R: float,
        matches: np.ndarray,
        num_correct: int,
        text: List[str],
        small_text: List[str] = [],
    ):
        """Visualize evaluation results"""
        # Create color coding based on epipolar error
        color = np.clip((epi_errs - 0) / (1e-3 - 0), 0, 1)
        color = error_colormap(1 - color)

        make_matching_plot(
            image_pair.image0,
            image_pair.image1,
            kpts0,
            kpts1,
            mkpts0,
            mkpts1,
            color,
            text,
            image_pair.viz_eval_path,
            self.show_keypoints,
            self.fast_viz,
            self.opencv_display,
            "Relative Pose",
            small_text,
            # self.line_width,
        )

    def create_videos(self, fps: int = 5):
        """Create videos from visualization images"""
        if not self.viz_extension:
            logger.info("Skipping video creation: viz_extension not specified")
            return

        # Generate videos from the visualization images
        logger.info("Creating videos from match visualizations...")
        self._create_matches_video("*_matches", fps)
        self._create_matches_video("*_ransac_matches", fps)
        self._create_matches_video("*_ransac_comparison", fps)
        self._create_matches_video("*_sg_vs_knn", fps)
        self._create_matches_video("*_sg_vs_knn_ransac", fps)

    def _create_matches_video(self, name_pattern: str = "*_matches", fps: int = 5):
        """
        Create a video from match visualization images in the output directory.

        Args:
            name_pattern: Pattern to match filenames (e.g., "*_matches" or "*_ransac_matches")
            fps: Frames per second for the output video
        """
        match_files = list(self.output_dir.glob(f"{name_pattern}.{self.viz_extension}"))

        if not match_files:
            logger.info(f"No {name_pattern} visualization files found.")
            return

        # Sort files by names to ensure correct order
        match_files.sort()

        # Read the first image to get dimensions
        first_img = cv2.imread(str(match_files[0]))
        if first_img is None:
            logger.error(f"Cannot read image: {match_files[0]}")
            return

        h, w = first_img.shape[:2]

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or 'XVID'
        video_name = f"{name_pattern.replace('*', 'all')}_video.mp4"
        video_path = self.output_dir / video_name
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))

        logger.info(f"Creating video from {len(match_files)} images...")
        for img_path in match_files:
            img = cv2.imread(str(img_path))
            if img is not None:
                out.write(img)

        out.release()
        logger.info(f"Video saved to {video_path}")

    def make_matching_comparison_plot(
        self,
        image0,
        image1,
        kpts0,
        kpts1,
        mkpts0,
        mkpts1,
        mkpts0_ransac,
        mkpts1_ransac,
        color,
        ransac_color,
        text,
        out_path,
        show_keypoints=False,
        fast_viz=False,
        line_width=1,  # Will use this for matplotlib line width
    ):
        """
        Create a side-by-side visualization comparing matches before and after RANSAC using matplotlib
        """
        if fast_viz:
            # Still use OpenCV for fast_viz mode
            # Ensure line_width is an integer for OpenCV
            cv_line_width = max(1, int(round(line_width)))

            # Use OpenCV for faster visualization
            H, W = image0.shape[:2]
            H2, W2 = image1.shape[:2]

            # Create side-by-side images with original matches and RANSAC-filtered matches
            out_img1 = np.ones((max(H, H2), W + W2, 3), np.uint8) * 255
            out_img2 = np.ones((max(H, H2), W + W2, 3), np.uint8) * 255

            # Convert grayscale images to RGB if needed
            if image0.ndim == 2:
                image0_rgb = cv2.cvtColor(image0, cv2.COLOR_GRAY2RGB)
            else:
                image0_rgb = image0

            if image1.ndim == 2:
                image1_rgb = cv2.cvtColor(image1, cv2.COLOR_GRAY2RGB)
            else:
                image1_rgb = image1

            # Place the images on both output canvases
            out_img1[:H, :W] = image0_rgb
            out_img1[:H2, W:] = image1_rgb
            out_img2[:H, :W] = image0_rgb
            out_img2[:H2, W:] = image1_rgb

            # Draw original matches (pre-RANSAC)
            for (x1, y1), (x2, y2), c in zip(mkpts0, mkpts1, color):
                try:
                    # Extract RGB values, ensure they're in valid range, and convert to BGR for OpenCV
                    rgb = np.clip(np.array(c[:3]), 0, 1)
                    # OpenCV uses BGR order
                    c_color = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))

                    cv2.circle(out_img1, (int(x1), int(y1)), 2, c_color, -1)
                    cv2.circle(out_img1, (int(x2) + W, int(y2)), 2, c_color, -1)
                    cv2.line(
                        out_img1,
                        (int(x1), int(y1)),
                        (int(x2) + W, int(y2)),
                        c_color,
                        cv_line_width,  # Use integer line_width
                        cv2.LINE_AA,
                    )
                except Exception as e:
                    # Use a default color (red) if conversion fails
                    print(f"Warning: Color conversion error: {e}")
                    cv2.circle(out_img1, (int(x1), int(y1)), 2, (0, 0, 255), -1)
                    cv2.circle(out_img1, (int(x2) + W, int(y2)), 2, (0, 0, 255), -1)
                    cv2.line(
                        out_img1,
                        (int(x1), int(y1)),
                        (int(x2) + W, int(y2)),
                        (0, 0, 255),
                        cv_line_width,  # Use integer line_width
                        cv2.LINE_AA,
                    )

            # Draw RANSAC-filtered matches
            for (x1, y1), (x2, y2), c in zip(
                mkpts0_ransac, mkpts1_ransac, ransac_color
            ):
                try:
                    # Same color handling as above
                    rgb = np.clip(np.array(c[:3]), 0, 1)
                    c_color = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))

                    cv2.circle(out_img2, (int(x1), int(y1)), 2, c_color, -1)
                    cv2.circle(out_img2, (int(x2) + W, int(y2)), 2, c_color, -1)
                    cv2.line(
                        out_img2,
                        (int(x1), int(y1)),
                        (int(x2) + W, int(y2)),
                        c_color,
                        cv_line_width,  # Use integer line_width
                        cv2.LINE_AA,
                    )
                except Exception as e:
                    # Use a default color (blue) if conversion fails
                    print(f"Warning: Color conversion error: {e}")
                    cv2.circle(out_img2, (int(x1), int(y1)), 2, (255, 0, 0), -1)
                    cv2.circle(out_img2, (int(x2) + W, int(y2)), 2, (255, 0, 0), -1)
                    cv2.line(
                        out_img2,
                        (int(x1), int(y1)),
                        (int(x2) + W, int(y2)),
                        (255, 0, 0),
                        cv_line_width,  # Use integer line_width
                        cv2.LINE_AA,
                    )

            # Add text to images
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(out_img1, "Original Matches", (10, 30), font, 1, (0, 0, 0), 2)
            cv2.putText(
                out_img2, "RANSAC-Filtered Matches", (10, 30), font, 1, (0, 0, 0), 2
            )

            # Stack images vertically for a before/after comparison
            comparison = np.vstack((out_img1, out_img2))

            # Save the comparison image
            cv2.imwrite(str(out_path), comparison)
        else:
            # Use matplotlib to create two separate comparison plots

            # Create output directory if it doesn't exist
            out_dir = Path(out_path).parent
            out_dir.mkdir(exist_ok=True, parents=True)

            # Path for original matches
            orig_path = out_dir / f"{Path(out_path).stem}_orig.{Path(out_path).suffix}"

            # Path for RANSAC matches
            ransac_path = (
                out_dir / f"{Path(out_path).stem}_ransac.{Path(out_path).suffix}"
            )

            # --- Original matches visualization ---
            plt.figure(figsize=(12, 6))

            # Plot image pair side by side
            plot_image_pair([image0, image1])

            # Plot keypoints if requested
            if show_keypoints:
                plot_keypoints(kpts0, kpts1, color="k", ps=4)
                plot_keypoints(kpts0, kpts1, color="w", ps=2)

            # Plot matches
            plot_matches(mkpts0, mkpts1, color, lw=line_width)

            # Add title and text
            fig = plt.gcf()
            txt_color = "k" if image0[:100, :150].mean() > 200 else "w"
            fig.text(
                0.01,
                0.99,
                "\n".join(text + ["Original Matches"]),
                transform=fig.axes[0].transAxes,
                fontsize=15,
                va="top",
                ha="left",
                color=txt_color,
            )

            plt.savefig(str(orig_path), bbox_inches="tight", pad_inches=0)
            plt.close()

            # --- RANSAC-filtered matches visualization ---
            plt.figure(figsize=(12, 6))

            # Plot image pair
            plot_image_pair([image0, image1])

            # Plot keypoints if requested
            if show_keypoints:
                plot_keypoints(kpts0, kpts1, color="k", ps=4)
                plot_keypoints(kpts0, kpts1, color="w", ps=2)

            # Plot RANSAC matches
            plot_matches(mkpts0_ransac, mkpts1_ransac, ransac_color, lw=line_width)

            # Add title and text
            fig = plt.gcf()
            fig.text(
                0.01,
                0.99,
                "\n".join(text + ["RANSAC-Filtered Matches"]),
                transform=fig.axes[0].transAxes,
                fontsize=15,
                va="top",
                ha="left",
                color=txt_color,
            )

            plt.savefig(str(ransac_path), bbox_inches="tight", pad_inches=0)
            plt.close()

            # Create a combined image using OpenCV (for consistent output with fast_viz mode)
            img1 = cv2.imread(str(orig_path))
            img2 = cv2.imread(str(ransac_path))

            if img1 is not None and img2 is not None:
                # Stack images vertically
                combined = np.vstack((img1, img2))
                cv2.imwrite(str(out_path), combined)

                # Remove individual images if desired
                if True:  # Change to False to keep individual images
                    orig_path.unlink(missing_ok=True)
                    ransac_path.unlink(missing_ok=True)

    def make_superglue_knn_comparison_plot(
        self,
        image0,
        image1,
        kpts0,
        kpts1,
        mkpts0_sg,
        mkpts1_sg,
        mconf_sg,
        mkpts0_knn,
        mkpts1_knn,
        mconf_knn,
        text_sg,
        text_knn,
        out_path,
        fast_viz=True,
        line_width=1,  # Will use this for matplotlib line width
    ):
        """Create a visualization comparing SuperGlue and KNN matching results using matplotlib"""

        if fast_viz:
            # Still use OpenCV for fast_viz mode
            # Ensure line_width is an integer for OpenCV
            cv_line_width = max(1, int(round(line_width)))

            # Use OpenCV for faster visualization
            H, W = image0.shape[:2]
            H2, W2 = image1.shape[:2]

            # Create canvases for SuperGlue and KNN
            sg_img = np.ones((max(H, H2), W + W2, 3), np.uint8) * 255
            knn_img = np.ones((max(H, H2), W + W2, 3), np.uint8) * 255

            # Convert grayscale images to RGB if needed
            if image0.ndim == 2:
                image0_rgb = cv2.cvtColor(image0, cv2.COLOR_GRAY2RGB)
            else:
                image0_rgb = image0

            if image1.ndim == 2:
                image1_rgb = cv2.cvtColor(image1, cv2.COLOR_GRAY2RGB)
            else:
                image1_rgb = image1

            # Place images on both canvases
            sg_img[:H, :W] = image0_rgb
            sg_img[:H2, W:] = image1_rgb
            knn_img[:H, :W] = image0_rgb
            knn_img[:H2, W:] = image1_rgb

            # Draw SuperGlue matches
            sg_color = cm.jet(mconf_sg)
            for (x1, y1), (x2, y2), c in zip(mkpts0_sg, mkpts1_sg, sg_color):
                try:
                    rgb = np.clip(np.array(c[:3]), 0, 1)
                    c_color = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))

                    cv2.circle(sg_img, (int(x1), int(y1)), 2, c_color, -1)
                    cv2.circle(sg_img, (int(x2) + W, int(y2)), 2, c_color, -1)
                    cv2.line(
                        sg_img,
                        (int(x1), int(y1)),
                        (int(x2) + W, int(y2)),
                        c_color,
                        cv_line_width,  # Use integer line_width
                        cv2.LINE_AA,
                    )
                except Exception as e:
                    continue

            # Draw KNN matches
            knn_color = cm.cool(mconf_knn)  # Different colormap for KNN
            for (x1, y1), (x2, y2), c in zip(mkpts0_knn, mkpts1_knn, knn_color):
                try:
                    rgb = np.clip(np.array(c[:3]), 0, 1)
                    c_color = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))

                    cv2.circle(knn_img, (int(x1), int(y1)), 2, c_color, -1)
                    cv2.circle(knn_img, (int(x2) + W, int(y2)), 2, c_color, -1)
                    cv2.line(
                        knn_img,
                        (int(x1), int(y1)),
                        (int(x2) + W, int(y2)),
                        c_color,
                        cv_line_width,  # Use integer line_width
                        cv2.LINE_AA,
                    )
                except Exception as e:
                    continue

            # Add titles and text to images
            font = cv2.FONT_HERSHEY_SIMPLEX
            # Add SuperGlue text
            for i, t in enumerate(text_sg):
                cv2.putText(sg_img, t, (10, 30 + i * 30), font, 1, (0, 0, 0), 2)

            # Add KNN text
            for i, t in enumerate(text_knn):
                cv2.putText(knn_img, t, (10, 30 + i * 30), font, 1, (0, 0, 0), 2)

            # Stack images vertically for comparison
            comparison = np.vstack((sg_img, knn_img))

            # Save comparison image
            cv2.imwrite(str(out_path), comparison)
        else:
            # Use matplotlib to create two separate comparison plots

            # Create output directory if it doesn't exist
            out_dir = Path(out_path).parent
            out_dir.mkdir(exist_ok=True, parents=True)

            # Path for SuperGlue matches
            sg_path = out_dir / f"{Path(out_path).stem}_sg.{Path(out_path).suffix}"

            # Path for KNN matches
            knn_path = out_dir / f"{Path(out_path).stem}_knn.{Path(out_path).suffix}"

            # --- SuperGlue matches visualization ---
            plt.figure(figsize=(12, 6))
            # # Use utils.py functions to create the visualization
            # from models.utils import (plot_image_pair, plot_keypoints,
            #                           plot_matches)

            # Plot image pair side by side
            plot_image_pair([image0, image1])

            # Plot matches
            sg_color = cm.jet(mconf_sg)
            plot_matches(mkpts0_sg, mkpts1_sg, sg_color, lw=line_width)

            # Add title and text
            fig = plt.gcf()
            txt_color = "k" if image0[:100, :150].mean() > 200 else "w"
            fig.text(
                0.01,
                0.99,
                "\n".join(text_sg),
                transform=fig.axes[0].transAxes,
                fontsize=15,
                va="top",
                ha="left",
                color=txt_color,
            )

            plt.savefig(str(sg_path), bbox_inches="tight", pad_inches=0)
            plt.close()

            # --- KNN matches visualization ---
            plt.figure(figsize=(12, 6))

            # Plot image pair
            plot_image_pair([image0, image1])

            # Plot KNN matches
            knn_color = cm.cool(mconf_knn)
            plot_matches(mkpts0_knn, mkpts1_knn, knn_color, lw=line_width)

            # Add title and text
            fig = plt.gcf()
            fig.text(
                0.01,
                0.99,
                "\n".join(text_knn),
                transform=fig.axes[0].transAxes,
                fontsize=15,
                va="top",
                ha="left",
                color=txt_color,
            )

            plt.savefig(str(knn_path), bbox_inches="tight", pad_inches=0)
            plt.close()

            # Create a combined image using OpenCV (for consistent output with fast_viz mode)
            img1 = cv2.imread(str(sg_path))
            img2 = cv2.imread(str(knn_path))

            if img1 is not None and img2 is not None:
                # Stack images vertically
                combined = np.vstack((img1, img2))
                cv2.imwrite(str(out_path), combined)

                # Remove individual images if desired
                if True:  # Change to False to keep individual images
                    sg_path.unlink(missing_ok=True)
                    knn_path.unlink(missing_ok=True)

    def visualize_feature_booster_comparison(
        self,
        image_pair: ImagePair,
        kpts0: np.ndarray,
        kpts1: np.ndarray,
        mkpts0_sp: np.ndarray,  # SuperPoint matches
        mkpts1_sp: np.ndarray,
        mconf_sp: np.ndarray,
        mkpts0_fb: np.ndarray,  # FeatureBooster matches
        mkpts1_fb: np.ndarray,
        mconf_fb: np.ndarray,
        text_sp: List[str],
        text_fb: List[str],
    ):
        """Visualize comparison between SuperPoint and FeatureBooster matches"""
        comparison_path = (
            self.output_dir
            / f"{image_pair.stem0}_{image_pair.stem1}_sp_vs_fb.{self.viz_extension}"
        )

        # We can reuse the existing superglue_knn_comparison_plot method
        self.make_superglue_knn_comparison_plot(
            image_pair.image0,
            image_pair.image1,
            kpts0,
            kpts1,
            mkpts0_sp,
            mkpts1_sp,
            mconf_sp,
            mkpts0_fb,
            mkpts1_fb,
            mconf_fb,
            text_sp,
            text_fb,
            comparison_path,
            self.fast_viz,
            self.line_width,
        )


class MatchingPipeline:
    """Class to orchestrate the matching process."""

    def __init__(self, config: Dict):
        self.config = config
        use_cuda = torch.cuda.is_available()

        self.device = (
            "cuda" if torch.cuda.is_available() and not config["force_cpu"] else "cpu"
        )
        logger.info(f'Running inference on device "{self.device}"')

        # Initialize the combined matching model first
        self.matching = (
            Matching(
                {"superpoint": config["superpoint"], "superglue": config["superglue"]}
            )
            .eval()
            .to(self.device)
        )

        self.feat_bst_matching = (
            FeatBoostEnhancedMatching(
                {
                    "superpoint": config["superpoint"],
                    "superglue": config["superglue"],
                    "featurebooster": config["featurebooster"],
                }
            )
            .eval()
            .to(self.device)
        )

        # config_file = Path(__file__).parent / "config.yaml"
        with open(str(config["fb_config"]), "r") as f:
            fb_yaml = yaml.load(f, Loader=yaml.FullLoader)

        # Initialize components using the matching model
        self.feature_extractor = self.matching.superpoint
        self.feature_matcher = self.matching.superglue

        # Initialize other components
        self.knn_matcher = KNNMatcher(config["knn_ratio"], config["knn_distance"])
        self.ransac_filter = RANSACFilter(
            config["ransac_threshold"], config["ransac_method"]
        )
        self.pose_estimator = PoseEstimator(config["pose_threshold"])
        self.visualizer = Visualizer(
            Path(config["output_dir"]),
            config["viz_extension"],
            config["show_keypoints"],
            config["fast_viz"],
            config["opencv_display"],
            config["line_width"],
        )

        # Load image pairs
        self.image_pairs = self._load_image_pairs(
            config["input_pairs"], config["input_dir"]
        )

        # Timer for performance measurement
        self.timer = AverageTimer(newline=True)

    def _load_image_pairs(self, input_pairs: str, input_dir: str) -> List[ImagePair]:
        """Load image pairs from input file."""
        with open(input_pairs, "r") as f:
            pairs = [l.split() for l in f.readlines()]

        if self.config["max_length"] > -1:
            pairs = pairs[: self.config["max_length"]]

        if self.config["shuffle"]:
            random.Random(0).shuffle(pairs)

        image_pairs = []
        for pair in pairs:
            name0, name1 = pair[:2]
            stem0, stem1 = Path(name0).stem, Path(name1).stem
            # If a rotation integer is provided (e.g. from EXIF data), use it:
            if len(pair) >= 5:
                rot0, rot1 = int(pair[2]), int(pair[3])
            else:
                rot0, rot1 = 0, 0
            intrinsics = (
                {
                    "K0": np.array(pair[4:13]).astype(float).reshape(3, 3),
                    "K1": np.array(pair[13:22]).astype(float).reshape(3, 3),
                }
                if len(pair) == 38
                else None
            )
            extrinsics = (
                np.array(pair[22:]).astype(float).reshape(4, 4)
                if len(pair) == 38
                else None
            )
            image_pair = ImagePair(
                name0, name1, stem0, stem1, rot0, rot1, intrinsics, extrinsics
            )
            image_pair.set_paths(
                Path(self.config["output_dir"]), self.config["viz_extension"]
            )
            image_pairs.append(image_pair)

        return image_pairs

    def run(self):
        """Run the matching pipeline."""
        for i, image_pair in enumerate(self.image_pairs):
            self._process_image_pair(image_pair)
            self.timer.print(f"Finished pair {i+1} of {len(self.image_pairs)}")

        if self.config["eval"]:
            self._evaluate()

        # if self.config["viz"]:
        self.visualizer.create_videos()

    def _process_image_pair(self, image_pair: ImagePair):
        """Process a single image pair."""
        # Load images
        img0_path =os.path.join(Path(self.config["input_dir"]) , "day", image_pair.name0)
        img1_path =os.path.join(Path(self.config["input_dir"]) , "night", image_pair.name1)
        logger.info(f"Processing pair {img0_path} {img1_path}")

        # 读取白天的图片
        (
            image_pair.image0,
            image_pair.inp0,
            image_pair.image0_prep,
            image_pair.inp0_prep,
            image_pair.scales0,
            image_pair.shape0[0],
            image_pair.shape0[1],
        ) = read_image(
            img0_path,
            self.device,
            self.config["resize"],
            image_pair.rot0,
            self.config["resize_float"],
            self.config["day_preprocess"]
        )
        
        # 读取晚上的图片
        (
            image_pair.image1,
            image_pair.inp1,
            image_pair.image1_prep,
            image_pair.inp1_prep,
            image_pair.scales1,
            image_pair.shape1[0],
            image_pair.shape1[1],
        ) = read_image(
            img1_path,
            self.device,
            self.config["resize"],
            image_pair.rot1,
            self.config["resize_float"],
            self.config["night_preprocess"]
        )
        
        if image_pair.image0 is None or image_pair.image1 is None:
            logger.error(
                f"Problem reading image pair: {image_pair.name0} {image_pair.name1}"
            )
            return
        if self.config["use_img_postprocess"] and (image_pair.image0_prep is None or image_pair.image1_prep is None):
            logger.error(
                f"Problem reading preprocessed image pair: {image_pair.name0} {image_pair.name1}"
            )
            return

        self.timer.update("load_image")

        # FIXME:
        image_pair.inp0 = image_pair.inp0_prep
        image_pair.image0 = image_pair.image0_prep
        image_pair.inp1 = image_pair.inp1_prep
        image_pair.image1 = image_pair.image1_prep
        # FIXME end
        
        # Extract features through superpoint only
        data0 = {"image": image_pair.inp0}
        data1 = {"image": image_pair.inp1}

        # Use superpoint to extract features from two imgs
        features0 = self.feature_extractor({"image": data0["image"]})
        features1 = self.feature_extractor({"image": data1["image"]})

        keypoints0 = features0["keypoints"]  # tensor shape (1， N, 2)
        scores0 = features0["scores"]  # shape (1, N,)
        descriptors0 = features0["descriptors"]  # shape (1, M, 256)
        keypoints1 = features1["keypoints"]
        scores1 = features1["scores"]
        descriptors1 = features1["descriptors"]
        # Build match_data using the tensor features
        match_data = {
            "image0": image_pair.inp0,
            "image1": image_pair.inp1,
            "keypoints0": keypoints0,
            "keypoints1": keypoints1,
            "scores0": scores0,
            "scores1": scores1,
            "descriptors0": descriptors0,
            "descriptors1": descriptors1,
        }
        
        ## Compute boosted descriptors using FeatureBooster if enabled
        if self.config["use_fb"]:
            match_data_fb = {
                "image0": image_pair.inp0,
                "image1": image_pair.inp1,
                "shape0": image_pair.shape0[0],
                "shape1": image_pair.shape0[1],
            }
            fb_pred = self.feat_bst_matching(match_data_fb)
            fb_pred = {k: v[0].cpu().numpy() for k, v in fb_pred.items()}
            kpts0_fb, kpts1_fb = fb_pred["keypoints0"], fb_pred["keypoints1"]
            matches0_fb_sg, conf_fb_sg = fb_pred["matches0"], fb_pred["matching_scores0"]
            
            valid_fb_sg = matches0_fb_sg > -1
            mkpts0_fb_sg = kpts0_fb[valid_fb_sg]
            mkpts1_fb_sg = kpts1_fb[matches0_fb_sg[valid_fb_sg]]
            mconf_fb_sg = conf_fb_sg[valid_fb_sg]
            
            ## Update match_data with SP+FB+SG results
            match_data_fb.update(
                {
                    "keypoints0": [torch.from_numpy(kpts0_fb).to(self.device).float()],    # list with tensor of shape (N,2)
                    "keypoints1": [torch.from_numpy(kpts1_fb).to(self.device).float()],
                    "scores0": fb_pred['scores0'],          # list with tensor of shape (N,)
                    "scores1": fb_pred['scores1'],
                    "descriptors0": [torch.from_numpy(fb_pred["descriptors0"]).to(self.device).float()],   # list with tensor of shape (256, N)
                    "descriptors1": [torch.from_numpy(fb_pred["descriptors1"]).to(self.device).float()],
                    "matches0": matches0_fb_sg,
                    "matching_scores0": conf_fb_sg,
                }
            )

            # import pdb

            # pdb.set_trace()


        ## superpoint+superglue
        match_data_superglue = {
            "image0": image_pair.inp0,
            "image1": image_pair.inp1,
        }

        # Use superpoint + superglue matching model
        pred = self.matching(match_data_superglue)
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        matches = {
            "matches0": pred["matches0"],
            "matches1": pred["matches1"],
            "matching_scores0": pred["matching_scores0"],
            "matching_scores1": pred["matching_scores1"],
        }
        kpts0, kpts1 = pred["keypoints0"], pred["keypoints1"]
        matches0, conf = pred["matches0"], pred["matching_scores0"]

        # Keep the matching keypoints.
        valid = matches0 > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches0[valid]]
        mconf = conf[valid]

        # Apply RANSAC if requested
        mkpts0_ransac, mkpts1_ransac, mconf_ransac, _ = (
            self.ransac_filter.filter(mkpts0, mkpts1, mconf)
        )

        # Save matches
        out_matches = {
            "keypoints0": features0["keypoints"],
            "keypoints1": features1["keypoints"],
            "matches": matches["matches0"],
            "match_confidence": matches["matching_scores0"],
            "mkpts0_ransac": mkpts0_ransac,
            "mkpts1_ransac": mkpts1_ransac,
            "mconf_ransac": mconf_ransac,
        }
        np.savez(str(image_pair.matches_path), **out_matches)

        # Visualize matches if requested
        if self.config["viz"]:
            text = [
                "SuperGlue",
                f"Keypoints: {len(features0['keypoints'][0])}:{len(features1['keypoints'][0])}",
                f"Matches: {len(mkpts0)}",
            ]
            if self.config["ransac"]:
                text.append(f"RANSAC inliers: {len(mkpts0_ransac)}/{len(mkpts0)}")
            if image_pair.rot0 != 0 or image_pair.rot1 != 0:
                text.append(f"Rotation: {image_pair.rot0}:{image_pair.rot1}")

            small_text = [
                f"Keypoint Threshold: {self.config['superpoint']['keypoint_threshold']:.4f}",
                f"Match Threshold: {self.config['superglue']['match_threshold']:.2f}",
                f"Image Pair: {image_pair.stem0}:{image_pair.stem1}",
            ]

            # 可视化superpoint+superglue matching结果
            self.visualizer.visualize_matches(
                image_pair,
                features0["keypoints"],
                features1["keypoints"],
                mkpts0,
                mkpts1,
                mconf,
                text,
                "Matches",
                small_text,
            )

            # Visualize superpoint+superglue w/o RANSAC 与 RANSAC-filtered 的对比
            if self.config["viz_comparison"] and self.config["ransac"]:
                self.visualizer.visualize_ransac_comparison(
                    image_pair,
                    features0["keypoints"],
                    features1["keypoints"],
                    mkpts0,
                    mkpts1,
                    mkpts0_ransac,
                    mkpts1_ransac,
                    mconf,
                    mconf_ransac,
                    text,
                )

        ## Visualize SP+FB+SG+RANSAC and SP+SG+RANSAC comparison when enabled.
        if self.config["use_fb"] and self.config["compare_fb_sg_ransac"]:
            mkpts0_fb_sg_ransac, mkpts1_fb_sg_ransac, mconf_fb_sg_ransac, _ = (
                self.ransac_filter.filter(mkpts0_fb_sg, mkpts1_fb_sg, mconf_fb_sg)
            )
            self.visualizer.visualize_feature_booster_comparison(
                image_pair,
                features0["keypoints"],
                features1["keypoints"],
                mkpts0_ransac,
                mkpts1_ransac,
                mconf_ransac,  # SP+SG+RANSAC results
                mkpts0_fb_sg_ransac,
                mkpts1_fb_sg_ransac,
                mconf_fb_sg_ransac,  # SP+FB+SG+RANSAC results
                ["SP+SG+RANSAC"],
                ["SP+FB+SG+RANSAC"],
            )

        if self.config["compare_sg_knn_ransac"]:
            knn_matches = self.knn_matcher.match(match_data)
            valid_knn = knn_matches["matches0"] > -1

            # mkpts0_knn = features0["keypoints"][valid_knn]
            # mkpts1_knn = features1["keypoints"][knn_matches["matches0"][valid_knn]]
            # mconf_knn = knn_matches["matching_scores0"][valid_knn]
            mkpts0_knn = kpts0[valid_knn]
            mkpts1_knn = kpts1[knn_matches["matches0"][valid_knn]]
            mconf_knn = conf[valid_knn]

            text_sg_ransac = [
                "SP+SuperGlue+RANSAC ",
                f"Keypts in day|night: {len(features0['keypoints'][0])}:{len(features1['keypoints'][0])}",
                f"Matches: {len(mkpts0_ransac)}",
                f"RANSAC: {len(mkpts0_ransac)}/{len(mkpts0)}",
            ]

            mkpts0_knn_ransac, mkpts1_knn_ransac, mconf_knn_ransac, _ = (
                self.ransac_filter.filter(mkpts0_knn, mkpts1_knn, mconf_knn)
            )
            text_knn_ransac = [
                "SP+KNN+RANSAC",
                f"Keypts in day|night: {len(features0['keypoints'][0])}:{len(features1['keypoints'][0])}",
                f"Matches: {len(mkpts0_knn_ransac)}",
                f"RANSAC: {len(mkpts0_knn_ransac)}/{len(mkpts0_knn)}",
            ]
            self.visualizer.visualize_matcher_comparison(
                image_pair,
                features0["keypoints"],
                features1["keypoints"],
                mkpts0_ransac,
                mkpts1_ransac,
                mconf_ransac,
                mkpts0_knn_ransac,
                mkpts1_knn_ransac,
                mconf_knn_ransac,
                text_sg_ransac,
                text_knn_ransac,
            )
            
        if self.config["use_fb"] and self.config["compare_knn_fb_ransac"]:
            ## superpoint+featurebooster+knn+ransac
            fb_knn_matches = self.knn_matcher.match(match_data_fb)
            valid_fb_knn = fb_knn_matches["matches0"] > -1
            mkpts0_fb_knn = kpts0_fb[valid_fb_knn]
            mkpts1_fb_knn = kpts1_fb[fb_knn_matches["matches0"][valid_fb_knn]]
            mconf_fb_knn = conf[valid_fb_knn]
            
            # Ransac
            mkpts0_fb_knn_ransac, mkpts1_fb_knn_ransac, mconf_fb_knn_ransac, _ = (
                self.ransac_filter.filter(mkpts0_fb_knn, mkpts1_fb_knn, mconf_fb_knn)
            )
            text_fb_knn_ransac = [
                "SP+FB+KNN+RANSAC",
                f"Keypts in day|night: {len(kpts0_fb)}:{len(kpts1_fb)}",
                f"Matches: {len(mkpts0_fb_knn_ransac)}",
                f"RANSAC: {len(mkpts0_fb_knn_ransac)}/{len(mkpts0_fb_knn)}",
            ]
            # small_text_fb_knn = [
            #     f"Keypoint Threshold: {self.config['superpoint']['keypoint_threshold']:.4f}",
            #     # f"Match Threshold: {self.config['superglue']['match_threshold']:.2f}",
            #     f"Image Pair: {image_pair.stem0}:{image_pair.stem1}",
            # ]
            # self.visualizer.visualize_matches(
            #     image_pair,
            #     kpts0_fb,
            #     kpts1_fb,
            #     mkpts0_fb_knn,
            #     mkpts1_fb_knn,
            #     mconf_fb_knn,
            #     text_fb_knn,
            #     "FeatureBooster+KNN",
            #     small_text_fb_knn,
            # )

            # superpoint+knn+ransac
            knn_matches = self.knn_matcher.match(match_data)
            valid_knn = knn_matches["matches0"] > -1

            mkpts0_knn = kpts0[valid_knn]
            mkpts1_knn = kpts1[knn_matches["matches0"][valid_knn]]
            mconf_knn = conf[valid_knn]
            mkpts0_knn_ransac, mkpts1_knn_ransac, mconf_knn_ransac, _ = (
                self.ransac_filter.filter(mkpts0_knn, mkpts1_knn, mconf_knn)
            )
            text_knn_ransac = [
                "SP+KNN+RANSAC",
                f"Keypts in day|night: {len(kpts0)}:{len(kpts1)}",
                f"Matches: {len(mkpts0_knn_ransac)}",
                f"RANSAC: {len(mkpts0_knn_ransac)}/{len(mkpts0_knn)}",
            ]
            self.visualizer.visualize_matcher_comparison(
                image_pair,
                features0["keypoints"],
                features1["keypoints"],
                mkpts0_fb_knn_ransac,
                mkpts1_fb_knn_ransac,
                mconf_fb_knn_ransac,
                mkpts0_knn_ransac,
                mkpts1_knn_ransac,
                mconf_knn_ransac,
                text_fb_knn_ransac,
                text_knn_ransac,
            )
            
        self.timer.update("process_image_pair")

    def _evaluate(self):
        """Evaluate the results."""
        pose_errors = []
        precisions = []
        matching_scores = []
        for image_pair in self.image_pairs:
            eval_path = image_pair.eval_path
            results = np.load(eval_path)
            pose_error = np.maximum(results["error_t"], results["error_R"])
            pose_errors.append(pose_error)
            precisions.append(results["precision"])
            matching_scores.append(results["matching_score"])

        thresholds = [5, 10, 20]
        aucs = pose_auc(pose_errors, thresholds)
        aucs = [100.0 * yy for yy in aucs]
        prec = 100.0 * np.mean(precisions)
        ms = 100.0 * np.mean(matching_scores)
        logger.info(
            "Evaluation Results (mean over {} pairs):".format(len(self.image_pairs))
        )
        logger.info("AUC@5\t AUC@10\t AUC@20\t Prec\t MScore\t")
        logger.info(
            "{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t".format(
                aucs[0], aucs[1], aucs[2], prec, ms
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Image pair matching and pose evaluation with SuperGlue",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input_pairs",
        type=str,
        # default="assets/scannet_sample_pairs_with_gt.txt",
        default="/home/user/data/maploc_data/gen_featpts_gt_dataset/gen_local_feature_dataset/P11_ent1_route1_case2_P11_ent1_route1_case4/scannet_pairs.txt",
        help="Path to the list of image pairs",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        # default="assets/scannet_sample_images/",
        default="/home/user/data/maploc_data/gen_featpts_gt_dataset/gen_local_feature_dataset/P11_ent1_route1_case2_P11_ent1_route1_case4/",
        help="Path to the directory that contains the images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/user/data/maploc_data/gen_featpts_gt_dataset/gen_local_feature_dataset/P11_ent1_route1_case2_P11_ent1_route1_case4/dump_match_pairs/",
        help="Path to the directory in which the .npz results and optionally,"
        "the visualization images are written",
    )

    parser.add_argument(
        "--max_length", type=int, default=-1, help="Maximum number of pairs to evaluate"
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs="+",
        default=[640, 480],
        help="Resize the input image before running inference. If two numbers, "
        "resize to the exact dimensions, if one number, resize the max "
        "dimension, if -1, do not resize",
    )
    parser.add_argument(
        "--resize_float",
        action="store_true",
        help="Resize the image after casting uint8 to float",
    )

    parser.add_argument(
        "--superglue",
        choices={"indoor", "outdoor"},
        default="outdoor",
        help="SuperGlue weights",
    )
    parser.add_argument(
        "--max_keypoints",
        type=int,
        default=1024,
        help="Maximum number of keypoints detected by Superpoint"
        " ('-1' keeps all keypoints)",
    )
    parser.add_argument(
        "--keypoint_threshold",
        type=float,
        default=0.005,
        help="SuperPoint keypoint detector confidence threshold",
    )
    parser.add_argument(
        "--nms_radius",
        type=int,
        default=4,
        help="SuperPoint Non Maximum Suppression (NMS) radius" " (Must be positive)",
    )
    parser.add_argument(
        "--sinkhorn_iterations",
        type=int,
        default=20,
        help="Number of Sinkhorn iterations performed by SuperGlue",
    )
    parser.add_argument(
        "--match_threshold", type=float, default=0.2, help="SuperGlue match threshold"
    )

    parser.add_argument(
        "--viz", action="store_true", help="Visualize the matches and dump the plots"
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Perform the evaluation" " (requires ground truth pose and intrinsics)",
    )
    parser.add_argument(
        "--fast_viz",
        action="store_true",
        help="Use faster image visualization with OpenCV instead of Matplotlib",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Skip the pair if output .npz files are already found",
    )
    parser.add_argument(
        "--show_keypoints",
        action="store_true",
        help="Plot the keypoints in addition to the matches",
    )
    parser.add_argument(
        "--viz_extension",
        type=str,
        default="png",
        choices=["png", "pdf"],
        help="Visualization file extension. Use pdf for highest-quality.",
    )
    parser.add_argument(
        "--opencv_display",
        action="store_true",
        help="Visualize via OpenCV before saving output images",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle ordering of pairs before processing",
    )
    parser.add_argument(
        "--force_cpu", action="store_true", help="Force pytorch to run in CPU mode."
    )

    # Add a RANSAC parameter to the argument parser
    parser.add_argument(
        "--ransac",
        action="store_true",
        help="Apply RANSAC to filter matches",
    )
    parser.add_argument(
        "--ransac_threshold",
        type=float,
        default=3.0,
        help="RANSAC threshold for match filtering",
    )
    parser.add_argument(
        "--ransac_method",
        type=str,
        default="fundamental",
        choices=["fundamental", "homography"],
        help="RANSAC method to use for filtering: fundamental matrix or homography",
    )
    parser.add_argument(
        "--viz_comparison",
        action="store_true",
        help="Visualize comparison between before and after RANSAC filtering",
    )

    # Add KNN comparison parameters to the argument parser
    parser.add_argument(
        "--compare_sg_knn_ransac",
        action="store_true",
        help="Run KNN matching in addition to SuperGlue for comparison",
    )
    parser.add_argument(
        "--knn_ratio",
        type=float,
        default=0.8,
        help="Ratio threshold for Lowe's ratio test in KNN matching",
    )
    parser.add_argument(
        "--knn_distance",
        type=str,
        default="L2",
        choices=["L2", "L1", "Hamming"],
        help="Distance metric for KNN matching",
    )

    # Add a line width parameter to the argument parser
    parser.add_argument(
        "--line_width",
        type=float,
        default=0.3,  # Set default to 0.3 as recommended
        help="Line width for visualizing feature matches",
    )
    # Add new argument for FeatureBooster usage
    parser.add_argument(
        "--use_fb",
        action="store_true",
        help="Use FeatureBooster plugin to boost SuperPoint features",
    )
    parser.add_argument(
        "--fb_config",
        type=str,
        default="fb_config.yaml",
        help="Path to a config file for FeatureBooster",
    )
    parser.add_argument(
        "--compare_knn_fb_ransac",
        action="store_true",
        help="Run KNN matching in addition to FeatureBooster for comparison",
    )
    parser.add_argument(
        "--compare_fb_sg_ransac",
        action="store_true",
        help="Visualize comparison between FeatureBooster and SuperGlue matches",
    )
    parser.add_argument(
        "--use_img_postprocess",
        type=bool,
        default=True,
        help="Apply image post-processing to enhance feature extraction",
    )

    opt = parser.parse_args()
    print(opt)

    assert not (
        opt.opencv_display and not opt.viz
    ), "Must use --viz with --opencv_display"
    assert not (
        opt.opencv_display and not opt.fast_viz
    ), "Cannot use --opencv_display without --fast_viz"
    assert not (opt.fast_viz and not opt.viz), "Must use --viz with --fast_viz"
    assert not (
        opt.fast_viz and opt.viz_extension == "pdf"
    ), "Cannot use pdf extension with --fast_viz"

    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print("Will resize to {}x{} (WxH)".format(opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print("Will resize max dimension to {}".format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print("Will not resize images")
    else:
        raise ValueError("Cannot specify more than two integers for --resize")

    day_img_preprocess = None
    night_img_preprocess = None
    if opt.use_img_postprocess:
        day_img_preprocess = {"name": "clahe", "params": {"clip_limit": 3.0, "tile_grid_size": (8, 8)}, "enabled": True}  # Enhance contrast in day images
        night_img_preprocess = {"name": "denoise", "params": {"h": 10.0}, "enabled": True} # Reduce noise common in night images

    config = {
        "input_pairs": opt.input_pairs,
        "input_dir": opt.input_dir,
        "output_dir": opt.output_dir,
        "max_length": opt.max_length,
        "resize": opt.resize,
        "resize_float": opt.resize_float,
        "superpoint": {
            "nms_radius": opt.nms_radius,
            "keypoint_threshold": opt.keypoint_threshold,
            "max_keypoints": opt.max_keypoints,
        },
        "superglue": {
            "weights": opt.superglue,
            "sinkhorn_iterations": opt.sinkhorn_iterations,
            "match_threshold": opt.match_threshold,
        },
        "featurebooster": {
            "keypoint_dim": 3,
            "keypoint_encoder": [32, 64, 128, 256],
            "descriptor_encoder": [256, 256],
            "descriptor_dim": 256,  # Match SuperPoint's output dimension
            "Attentional_layers": 9,
            # "last_activation": ' ',
            "l2_normalization": True,
            "output_dim": 256,  # Match SuperGlue's expected dimension
        },
        "viz": opt.viz,
        "eval": opt.eval,
        "fast_viz": opt.fast_viz,
        "cache": opt.cache,
        "show_keypoints": opt.show_keypoints,
        "viz_extension": opt.viz_extension,
        "opencv_display": opt.opencv_display,
        "shuffle": opt.shuffle,
        "force_cpu": opt.force_cpu,
        "ransac": opt.ransac,
        "ransac_threshold": opt.ransac_threshold,
        "ransac_method": opt.ransac_method,
        "viz_comparison": opt.viz_comparison,
        "compare_sg_knn_ransac": opt.compare_sg_knn_ransac,
        "knn_ratio": opt.knn_ratio,
        "knn_distance": opt.knn_distance,
        "line_width": opt.line_width,
        "pose_threshold": 1.0,
        "fb_config": opt.fb_config,
        "use_fb": opt.use_fb,  # new flag for feature booster
        "compare_knn_fb_ransac": opt.compare_knn_fb_ransac,
        "compare_fb_sg_ransac": opt.compare_fb_sg_ransac,
        "use_img_postprocess": opt.use_img_postprocess,
        "day_preprocess": [day_img_preprocess],
        "night_preprocess": [night_img_preprocess],
    }

    pipeline = MatchingPipeline(config)
    pipeline.run()
