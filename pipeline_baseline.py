import argparse
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import yaml

from data import ImagePair
from evaluation.pose_estimator import PoseEstimator
from filters.ransac_filter import RANSACFilter
from matchers.knn_matcher import KNNMatcher
from models.matching import FeatBoostEnhancedMatching, Matching
from models.utils import AverageTimer, pose_auc, read_image
from visualization.visualizer import Visualizer

# # Set up feature booster path
# feature_booster_path = Path(__file__).parent / "FeatBooster/FeatureBooster"
# sys.path.append(str(feature_booster_path))
# from featurebooster import FeatureBooster

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("GenBaselinePipeline")

# Disable gradient computation for inference
torch.set_grad_enabled(False)

class GenBaselinePipeline:
    """
    Class to orchestrate the matching process.
    
    This class manages the entire feature matching pipeline, including:
    - Loading image pairs
    - Extracting features
    - Matching features
    - Filtering matches
    - Evaluating results
    - Visualizing matches
    """

    def __init__(self, config: Dict):
        """
        Initialize the matching pipeline.
        
        Args:
            config: Dictionary containing pipeline configuration
        """
        self.config = config
        # self.case_lists = []
        self.case_dir_lists = []
        self.kpts_desc_lists = []
        
        # Set up device for computation
        self.device = (
            "cuda" if torch.cuda.is_available() and not config["force_cpu"] else "cpu"
        )
        logger.info(f'Running inference on device "{self.device}"')

        # Initialize models
        self._init_models()
        
        # Initialize pipeline components
        self._init_components()
        
        # Load image pairs
        for dir in os.listdir(self.config["input_dir"]):
            input_dir = os.path.join(self.config["input_dir"], dir)
            if os.path.isdir(input_dir) and dir.startswith("P") and "case" in dir:
                if os.path.exists(os.path.join(input_dir, "scannet_pairs.txt")):
                    scannet_pair_txt = os.path.join(input_dir, "scannet_pairs.txt")
                else:
                    logger.error(f"Scannet pair file not found for {dir}")
                    continue
                if os.path.exists(os.path.join(input_dir, "day")) and os.path.exists(os.path.join(input_dir, "night")):
                    # image_pairs = self._load_image_pairs(
                    #     scannet_pair_txt, input_dir
                    # )
                    
                    # self.case_lists.append(image_pairs)
                    self.case_dir_lists.append(input_dir)
                else:
                    logger.error(f"Invalid directory structure for {dir}")
                    continue

        # Timer for performance measurement
        self.timer = AverageTimer(newline=True)

    def _init_models(self):
        """Initialize machine learning models for feature extraction and matching."""
        # Initialize the combined matching model
        self.sp_sg_matching = (
            Matching(
                {"superpoint": self.config["superpoint"], "superglue": self.config["superglue"]}
            )
            .eval()
            .to(self.device)
        )
        
        # Extract feature extractor and matcher from the combined model
        # self.sp_feat_extractor = self.sp_sg_matching.superpoint
        self.sg_matcher = self.sp_sg_matching.superglue

    def _init_components(self):
        """Initialize pipeline components for matching, filtering, evaluation, and visualization."""
        # Initialize the KNN matcher    
        self.knn_matcher = KNNMatcher(
            self.config["knn_ratio"], 
            self.config["knn_distance"]
        )
        
        # Initialize the RANSAC filter
        self.ransac_filter = RANSACFilter(
            self.config["ransac_threshold"], 
            self.config["ransac_method"]
        )
        
        # Initialize the visualizer
        self.visualizer = Visualizer(
            Path(self.config["output_dir"]),
            self.config["viz_extension"],
            self.config["show_keypoints"],
            self.config["fast_viz"],
            self.config["opencv_display"],
            self.config["line_width"],
        )

    def _load_image_pairs(self, input_pairs: str, input_dir: str) -> List[ImagePair]:
        """
        Load image pairs from input file.
        
        Args:
            input_pairs: Path to file containing image pair information
            input_dir: Base directory for image files
            
        Returns:
            List of ImagePair objects
        """
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
            
            # Get rotation information if provided
            if len(pair) >= 5:
                rot0, rot1 = int(pair[2]), int(pair[3])
            else:
                rot0, rot1 = 0, 0
                
            # Get intrinsics if provided
            intrinsics = (
                {
                    "K0": np.array(pair[4:13]).astype(float).reshape(3, 3),
                    "K1": np.array(pair[13:22]).astype(float).reshape(3, 3),
                }
                if len(pair) == 38
                else None
            )
            
            # Get extrinsics if provided
            extrinsics = (
                np.array(pair[22:]).astype(float).reshape(4, 4)
                if len(pair) == 38
                else None
            )
            
            # Create ImagePair object
            image_pair = ImagePair(
                name0, name1, stem0, stem1, rot0, rot1, intrinsics, extrinsics
            )
            
            # Set output paths
            image_pair_output_dir = os.path.join(input_dir, "J6_128_scale4-desc_KNN_output")
            if not os.path.exists(image_pair_output_dir):
                os.makedirs(image_pair_output_dir, exist_ok=True)
                
            image_pair.set_paths(
                Path(image_pair_output_dir), self.config["viz_extension"]
            )
            
            image_pair.set_kpts_desc_paths(input_dir)
            
            image_pairs.append(image_pair)

        return image_pairs

    def run(self):
        """Run the matching pipeline on all image pairs."""
        all_inliers = []
        for case_dir in os.listdir(self.config["input_dir"]):
            case_path = os.path.join(self.config["input_dir"], case_dir)
            if os.path.isdir(case_path) and case_dir.startswith("P") and "case" in case_dir:
                logger.info(f"Processing case: {case_dir}")

                self.image_pairs = self._load_image_pairs(
                    os.path.join(case_path, "scannet_pairs.txt"), case_path
                )
                case_inliers = []
                for i, image_pair in enumerate(self.image_pairs):
                    inliers = self._process_image_pair(image_pair, case_path)
                    case_inliers.append(inliers)
                    self.timer.print(f"Finished pair {i+1} of {len(self.image_pairs)}")

                # Record inliers and other metrics into a txt file
                with open(os.path.join(case_path, "j6_128_scale4-desc_KNN_metrics.txt"), "w") as f:
                    for inlier in case_inliers:
                        f.write(f"{inlier}\n")

                all_inliers.extend(case_inliers)

    def plot_violin_diagram(self):
        # After running the pipeline, plot and save the violin diagram from metrics files
        metrics_files = [Path(case_path) / "j6_128_scale4-desc_KNN_metrics.txt" for case_path in self.case_dir_lists]
        logger.info(f"Metrics files: {metrics_files}")
        output_path = Path(self.config["output_dir"]) / "J6_128_scale4-desc_KNN_violin_plot.png"
        self.visualizer.plot_and_save_violin_diagram(metrics_files, output_path)

    def _process_image_pair(self, image_pair: ImagePair, case_path: str):
        """
        Process a single image pair.
        
        Args:
            image_pair: ImagePair object to process

        Returns:
            Number of RANSAC inliers
        """
        # Load images
        img0_path = os.path.join(Path(case_path), "day", image_pair.name0)
        img1_path = os.path.join(Path(case_path), "night", image_pair.name1)
        logger.info(f"Processing pair {img0_path} {img1_path}")

        # Determine preprocess mode based on image pair names or other criteria
        preprocess_mode_day = self.config['preprocess_modes']['day']
        preprocess_mode_night = self.config['preprocess_modes']['night']

        # Read day and night images with appropriate preprocessing
        (
            image_pair.image0,
            image_pair.inp0,
            image_pair.processed_image0,
            image_pair.processed_inp0,
            image_pair.scales0,
            image_pair.shape0[0],
            image_pair.shape0[1],
        ) = read_image(
            img0_path,
            self.device,
            self.config['resize'],
            image_pair.rot0,
            self.config['resize_float'],
            preprocess_mode_day
        )
        
        (
            image_pair.image1,
            image_pair.inp1,
            image_pair.processed_image1,
            image_pair.processed_inp1,
            image_pair.scales1,
            image_pair.shape1[0],
            image_pair.shape1[1],
        ) = read_image(
            img1_path,
            self.device,
            self.config['resize'],
            image_pair.rot1,
            self.config['resize_float'],
            preprocess_mode_night
        )
        
        if image_pair.image0 is None or image_pair.image1 is None:
            logger.error(
                f"Problem reading image pair: {image_pair.name0} {image_pair.name1}"
            )
            return 0

        self.timer.update("load_image")
        
        # Process with SuperGlue
        # inliers = self._process_with_superglue(image_pair)
        
        # Match with knn
        inliers = self._process_with_knn(image_pair)
        
        self.timer.update("process_image_pair")
        return inliers

    def _process_with_superglue(self, image_pair):
        """
        Process image pair with SuperGlue matching.
        
        Args:
            image_pair: ImagePair object
            match_data: Dictionary containing match data
            img0_sp_feat, img1_sp_feat: Feature dictionaries extracted from images
        
        Returns:
            Number of RANSAC inliers
        """
        # Match features using SuperGlue
        match_data_sp_sg = {
            "image0": image_pair.inp0,
            "image1": image_pair.inp1,
        }

        pred = self.sp_sg_matching(match_data_sp_sg)
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        
        matches = {
            "matches0": pred["matches0"],
            "matches1": pred["matches1"],
            "matching_scores0": pred["matching_scores0"],
            "matching_scores1": pred["matching_scores1"],
        }
        
        kpts0, kpts1 = pred["keypoints0"], pred["keypoints1"]
        matches0, conf = pred["matches0"], pred["matching_scores0"]

        # Keep the matching keypoints
        valid = matches0 > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches0[valid]]
        mconf = conf[valid]

        # Apply RANSAC if requested
        if self.config["ransac"]:
            mkpts0_ransac, mkpts1_ransac, mconf_ransac, _ = (
                self.ransac_filter.filter(mkpts0, mkpts1, mconf)
            )
        else:
            mkpts0_ransac, mkpts1_ransac, mconf_ransac = mkpts0, mkpts1, mconf

        # Save matches
        all_data_sp_sg_ransac = {
            "keypoints0": pred["keypoints0"],
            "keypoints1": pred["keypoints1"],
            "scores0": pred["scores0"],
            "scores1": pred["scores1"],
            "descriptors0": pred["descriptors0"],
            "descriptors1": pred["descriptors1"],
            "matches": matches["matches0"],
            "match_confidence": matches["matching_scores0"],
            "mkpts0": mkpts0,
            "mkpts1": mkpts1,
            "mconf": mconf,
            "mkpts0_ransac": mkpts0_ransac,
            "mkpts1_ransac": mkpts1_ransac,
            "mconf_ransac": mconf_ransac,
        }
        np.savez(str(image_pair.matches_path), **all_data_sp_sg_ransac)

        # Visualize matches if requested
        if self.config["viz"]:
            self._visualize_matches(image_pair, kpts0, kpts1, mkpts0, mkpts1, 
                                   mconf, mkpts0_ransac, mkpts1_ransac, mconf_ransac)
            
        # Compare SuperGlue and KNN if requested
        if self.config["compare_sg_knn_ransac"]:
            self._compare_superglue_knn(image_pair, all_data_sp_sg_ransac)

        # Return the number of RANSAC inliers
        return len(mkpts0_ransac)

    def _visualize_matches(self, image_pair, sp_kpts0, sp_kpts1, mkpts0, mkpts1, 
                          mconf, mkpts0_ransac, mkpts1_ransac, mconf_ransac):
        """
        Visualize matches for an image pair.
        
        Args:
            image_pair: ImagePair object
            features0, features1: Feature dictionaries
            mkpts0, mkpts1: Matched keypoints
            mconf: Match confidence scores
            mkpts0_ransac, mkpts1_ransac: RANSAC-filtered keypoints
            mconf_ransac: RANSAC-filtered confidence scores
        """
        # 确保关键点只包含x和y坐标
        sp_kpts0_xy = sp_kpts0[:, :2] if sp_kpts0.shape[1] > 2 else sp_kpts0
        sp_kpts1_xy = sp_kpts1[:, :2] if sp_kpts1.shape[1] > 2 else sp_kpts1
        mkpts0_xy = mkpts0[:, :2] if mkpts0.shape[1] > 2 else mkpts0
        mkpts1_xy = mkpts1[:, :2] if mkpts1.shape[1] > 2 else mkpts1
        mkpts0_ransac_xy = mkpts0_ransac[:, :2] if mkpts0_ransac.shape[1] > 2 else mkpts0_ransac
        mkpts1_ransac_xy = mkpts1_ransac[:, :2] if mkpts1_ransac.shape[1] > 2 else mkpts1_ransac
        
        text = [
            "Baseline:J6_128_scale4-desc + KNN",
            f"Keypoints: {len(sp_kpts0)}:{len(sp_kpts1)}",
            f"Matches: {len(mkpts0)}",
        ]
        
        if self.config["ransac"]:
            text.append(f"RANSAC inliers: {len(mkpts0_ransac)}/{len(mkpts0)}")
            
        if image_pair.rot0 != 0 or image_pair.rot1 != 0:
            text.append(f"Rotation: {image_pair.rot0}:{image_pair.rot1}")

        small_text = [
            f"KNN Match Threshold: {self.config['knn_ratio']:.2f}",
            f"KNN Distance type: {self.config['knn_distance']}",
            f"RANSAC Threshold: {self.config['ransac_threshold']:.2f}",
            f"RANSAC Method: {self.config['ransac_method']}",
            f"Image Pair: {image_pair.stem0}:{image_pair.stem1}",
        ]

        # Visualize j6_128_scale4-desc + KNN + RANSAC matches
        self.visualizer.visualize_matches(
            image_pair,
            sp_kpts0_xy,
            sp_kpts1_xy,
            mkpts0_xy,
            mkpts1_xy,
            mconf,
            text,
            "Matches",
            small_text,
        )

        # Visualize RANSAC filtering comparison if requested
        if self.config["viz_comparison"] and self.config["ransac"]:
            self.visualizer.visualize_ransac_comparison(
                image_pair,
                sp_kpts0_xy,
                sp_kpts1_xy,
                mkpts0_xy,
                mkpts1_xy,
                mkpts0_ransac_xy,
                mkpts1_ransac_xy,
                mconf,
                mconf_ransac,
                text,
            )
            
        self.visualizer.create_videos()

    def _compare_superglue_knn(self, image_pair, all_data_sp_sg_ransac):
        """
        Compare SuperGlue and KNN matching results.
        
        Args:
            image_pair: ImagePair object
            all_data_sp_sg_ransac: Dictionary containing match data from Superpoint+SuperGlue+RANSAC
        """
        # Match using KNN
        knn_match_data = {
            "descriptors0": all_data_sp_sg_ransac["descriptors0"],
            "descriptors1": all_data_sp_sg_ransac["descriptors1"],
        }
        knn_matches = self.knn_matcher.match(knn_match_data)
        valid_knn = knn_matches["matches0"] > -1

        mkpts0_knn = all_data_sp_sg_ransac['keypoints0'][valid_knn]
        mkpts1_knn = all_data_sp_sg_ransac['keypoints1'][knn_matches["matches0"][valid_knn]]
        mconf_knn = knn_matches["matching_scores0"][valid_knn]

        # Apply RANSAC to KNN matches
        mkpts0_knn_ransac, mkpts1_knn_ransac, mconf_knn_ransac, _ = (
            self.ransac_filter.filter(mkpts0_knn, mkpts1_knn, mconf_knn)
        )
        
        # 确保关键点只包含x和y坐标
        keypoints0_xy = all_data_sp_sg_ransac['keypoints0'][:, :2] if all_data_sp_sg_ransac['keypoints0'].shape[1] > 2 else all_data_sp_sg_ransac['keypoints0']
        keypoints1_xy = all_data_sp_sg_ransac['keypoints1'][:, :2] if all_data_sp_sg_ransac['keypoints1'].shape[1] > 2 else all_data_sp_sg_ransac['keypoints1']
        mkpts0_ransac_xy = all_data_sp_sg_ransac['mkpts0_ransac'][:, :2] if all_data_sp_sg_ransac['mkpts0_ransac'].shape[1] > 2 else all_data_sp_sg_ransac['mkpts0_ransac']
        mkpts1_ransac_xy = all_data_sp_sg_ransac['mkpts1_ransac'][:, :2] if all_data_sp_sg_ransac['mkpts1_ransac'].shape[1] > 2 else all_data_sp_sg_ransac['mkpts1_ransac']
        mkpts0_knn_ransac_xy = mkpts0_knn_ransac[:, :2] if mkpts0_knn_ransac.shape[1] > 2 else mkpts0_knn_ransac
        mkpts1_knn_ransac_xy = mkpts1_knn_ransac[:, :2] if mkpts1_knn_ransac.shape[1] > 2 else mkpts1_knn_ransac
        
        # Create text for visualizations
        text_sg_ransac = [
            "SP+SuperGlue+RANSAC ",
            f"Keypts in day|night: {len(all_data_sp_sg_ransac['keypoints0'])}:{len(all_data_sp_sg_ransac['keypoints1'])}",
            f"Matches: {len(all_data_sp_sg_ransac['mkpts0_ransac'])}",
            f"RANSAC: {len(all_data_sp_sg_ransac['mkpts0_ransac'])}/{len(all_data_sp_sg_ransac['mkpts0'])}",
        ]

        text_knn_ransac = [
            "SP+KNN+RANSAC",
            f"Keypts in day|night: {len(all_data_sp_sg_ransac['keypoints0'])}:{len(all_data_sp_sg_ransac['keypoints1'])}",
            f"Matches: {len(mkpts0_knn_ransac)}",
            f"RANSAC: {len(mkpts0_knn_ransac)}/{len(mkpts0_knn)}",
        ]
        
        # Visualize comparison
        self.visualizer.visualize_matcher_comparison(
            image_pair,
            keypoints0_xy,
            keypoints1_xy,
            mkpts0_ransac_xy,
            mkpts1_ransac_xy,
            all_data_sp_sg_ransac['mconf_ransac'],
            mkpts0_knn_ransac_xy,
            mkpts1_knn_ransac_xy,
            mconf_knn_ransac,
            text_sg_ransac,
            text_knn_ransac,
        )

    def _process_with_knn(self, image_pair):
        """
        Process image pair with KNN matching using pre-extracted keypoints and descriptors.
        
        Args:
            image_pair: ImagePair object with paths to keypoints and descriptors
        
        Returns:
            Number of RANSAC inliers
        """
        # Load keypoints and descriptors from files
        image_pair.load_kpts_desc()
        
        # Check if keypoints and descriptors were successfully loaded
        if (image_pair.kpts0 is None or image_pair.kpts1 is None or 
            image_pair.desc0 is None or image_pair.desc1 is None):
            logger.error(f"Failed to load keypoints or descriptors for {image_pair.name0} and {image_pair.name1}")
            return 0
        
        logger.info(f"Loaded keypoints: day={len(image_pair.kpts0)}, night={len(image_pair.kpts1)}")
        
        # Prepare data for KNN matching
        knn_match_data = {
            "descriptors0": image_pair.desc0,
            "descriptors1": image_pair.desc1,
        }
        
        # Match using KNN
        knn_matches = self.knn_matcher.match(knn_match_data)
        valid_knn = knn_matches["matches0"] > -1
        
        # Extract matched keypoints
        mkpts0_knn = image_pair.kpts0[valid_knn]
        mkpts1_knn = image_pair.kpts1[knn_matches["matches0"][valid_knn]]
        mconf_knn = knn_matches["matching_scores0"][valid_knn]
        
        logger.info(f"KNN matches: {len(mkpts0_knn)} out of {len(image_pair.kpts0)} keypoints")
        
        # Apply RANSAC if requested
        if self.config["ransac"]:
            mkpts0_knn_ransac, mkpts1_knn_ransac, mconf_knn_ransac, _ = (
                self.ransac_filter.filter(mkpts0_knn, mkpts1_knn, mconf_knn)
            )
        else:
            mkpts0_knn_ransac, mkpts1_knn_ransac, mconf_knn_ransac = mkpts0_knn, mkpts1_knn, mconf_knn
        
        # Save matches
        all_data_knn_ransac = {
            "keypoints0": image_pair.kpts0,
            "keypoints1": image_pair.kpts1,
            "descriptors0": image_pair.desc0,
            "descriptors1": image_pair.desc1,
            "matches": knn_matches["matches0"],
            "match_confidence": knn_matches["matching_scores0"],
            "mkpts0": mkpts0_knn,
            "mkpts1": mkpts1_knn,
            "mconf": mconf_knn,
            "mkpts0_ransac": mkpts0_knn_ransac,
            "mkpts1_ransac": mkpts1_knn_ransac,
            "mconf_ransac": mconf_knn_ransac,
        }
        np.savez(str(image_pair.matches_path), **all_data_knn_ransac)
        
        # Visualize matches if requested
        if self.config["viz"]:
            self._visualize_matches(image_pair, 
                                   image_pair.kpts0, image_pair.kpts1, 
                                   mkpts0_knn, mkpts1_knn, 
                                   mconf_knn, 
                                   mkpts0_knn_ransac, mkpts1_knn_ransac, 
                                   mconf_knn_ransac)
        
        # Return the number of RANSAC inliers
        num_inliers = len(mkpts0_knn_ransac)
        logger.info(f"KNN+RANSAC: {num_inliers} inliers out of {len(mkpts0_knn)} matches")
        return num_inliers