import argparse
import logging
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# Set up feature booster path
feature_booster_path = Path(__file__).parent / "FeatBooster/FeatureBooster"
sys.path.append(str(feature_booster_path))
from featurebooster import FeatureBooster

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("MatchingPipeline")

# Disable gradient computation for inference
torch.set_grad_enabled(False)

class MatchingPipeline:
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
        self.case_lists = []
        self.case_dir_lists = []
        
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
            if os.path.isdir(input_dir) and dir.startswith("P") and "case" in dir and "P11_ent1_route1_case2-P11_ent1_route1_case4" == dir:
                if os.path.exists(os.path.join(input_dir, "scannet_pairs.txt")):
                    scannet_pair_txt = os.path.join(input_dir, "scannet_pairs.txt")
                else:
                    logger.error(f"Scannet pair file not found for {dir}")
                    continue
                if os.path.exists(os.path.join(input_dir, "day")) and os.path.exists(os.path.join(input_dir, "night")):
                    image_pairs = self._load_image_pairs(
                        scannet_pair_txt, input_dir
                    )
                    self.case_lists.append(image_pairs)
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

        # Initialize FeatureBooster enhanced matching if configured
        if self.config["use_fb"]:
            self.feat_bst_matching = (
                FeatBoostEnhancedMatching(
                    {
                        "superpoint": self.config["superpoint"],
                        "superglue": self.config["superglue"],
                        "featurebooster": self.config["featurebooster"],
                    }
                )
                .eval()
                .to(self.device)
            )

        # Extract feature extractor and matcher from the combined model
        self.sp_feat_extractor = self.sp_sg_matching.superpoint
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
        
        # Initialize the pose estimator
        self.pose_estimator = PoseEstimator(
            self.config["pose_threshold"]
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
            image_pair_output_dir = os.path.join(input_dir, "dump_match_pairs")
            if not os.path.exists(image_pair_output_dir):
                os.makedirs(image_pair_output_dir, exist_ok=True)
                
            image_pair.set_paths(
                Path(image_pair_output_dir), self.config["viz_extension"]
            )
            
            # 设置当前day-night图像对的superpoint特征子路径
            image_pair.set_generated_gt_paths(input_dir=input_dir)
            
            image_pairs.append(image_pair)

        return image_pairs

    def run(self):
        """Run the matching pipeline on all image pairs."""
        all_inliers = []
        candidate_dirs = [
            # "P11_ent1_route1_case1-P11_ent1_route1_case3",
            # "P13_ent1_route1_case1-P13_ent1_route1_case3",
            # "P15_ent1_route1_case3-P15_ent1_route1_case1",
            "P15_ent1_route1_case4-P15_ent1_route1_case2",
            # "P16_ent1_route1_case1-P16_ent1_route1_case3",
            # "P17_ent1_route1_case3-P17_ent1_route1_case1",
            "P18_ent1_route1_case4-P18_ent1_route1_case2",
            # "P19_ent1_route1_case3-P19_ent1_route1_case1",
            "P20_ent1_route1_case3-P20_ent1_route1_case1",
            # "P20_ent1_route1_case4-P20_ent1_route1_case2",
            # "P22_ent1_route1_case2-P22_ent1_route1_case4"
        ]
        
        # 是否使用多线程处理，从配置获取
        use_multithreading = self.config.get("use_multithreading", False)
        num_threads = self.config.get("num_threads", 2)
        
        if use_multithreading:
            logger.info(f"Using multithreading with {num_threads} threads")
            # 使用多线程版本的实现
            all_inliers = self._run_multithreaded(candidate_dirs, num_threads)
        else:
            logger.info("Using single-thread implementation")
            # 原始的单线程实现
            for case_dir in os.listdir(self.config["input_dir"]):
                case_path = os.path.join(self.config["input_dir"], case_dir)
                if os.path.isdir(case_path) and case_dir.startswith("P") and case_dir in candidate_dirs:
                    logger.info(f"Processing case: {case_dir}")
                    # Set the case-specific output directory for the Visualizer
                    case_output_dir = Path(case_path) / "visualizations"
                    self.visualizer.set_case_output_dir(case_output_dir)

                    self.image_pairs = self._load_image_pairs(
                        os.path.join(case_path, "scannet_pairs.txt"), case_path
                    )
                    case_inliers = []
                    for i, image_pair in enumerate(self.image_pairs):
                        if os.path.exists(image_pair.matches_path):
                            logger.info(f"Already processed {image_pair.matches_path}")
                            continue
                        inliers = self._process_image_pair(image_pair, case_path)
                        case_inliers.append(inliers)
                        self.timer.print(f"Finished pair {i+1} of {len(self.image_pairs)}")

                    # Record inliers and other metrics into a txt file
                    with open(os.path.join(case_path, "sp_sg_knn_ransac_metrics.txt"), "w") as f:
                        for inlier in case_inliers:
                            f.write(f"{inlier}\n")

                    all_inliers.extend(case_inliers)

    def _run_multithreaded(self, candidate_dirs, num_threads):
        """使用多线程并行处理多个案例目录
        
        Args:
            candidate_dirs: 要处理的目录列表
            num_threads: 线程数量
            
        Returns:
            所有案例的内点数列表
        """
        all_inliers = []
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # 为每个案例目录创建一个任务
            future_to_case = {}
            for case_dir in os.listdir(self.config["input_dir"]):
                case_path = os.path.join(self.config["input_dir"], case_dir)
                if os.path.isdir(case_path) and case_dir.startswith("P") and case_dir in candidate_dirs:
                    # 提交任务到线程池
                    future = executor.submit(self._process_case_dir, case_dir, case_path)
                    future_to_case[future] = case_dir
            
            # 收集处理结果
            for future in as_completed(future_to_case):
                case_dir = future_to_case[future]
                try:
                    case_inliers = future.result()
                    all_inliers.extend(case_inliers)
                    logger.info(f"Completed processing case: {case_dir}")
                except Exception as e:
                    logger.error(f"Error processing case {case_dir}: {e}")
        
        return all_inliers
    
    def _process_case_dir(self, case_dir, case_path):
        """处理单个案例目录，供多线程并行调用
        
        Args:
            case_dir: 案例目录名
            case_path: 案例目录路径
            
        Returns:
            案例的内点数列表
        """
        logger.info(f"Thread {threading.current_thread().name}: Processing case {case_dir}")
        
        # 设置线程特定的可视化输出目录
        case_output_dir = Path(case_path) / "visualizations"
        case_visualizer = Visualizer(
            Path(self.config["output_dir"]),
            self.config["viz_extension"],
            self.config["show_keypoints"],
            self.config["fast_viz"],
            self.config["opencv_display"],
            self.config["line_width"],
        )
        case_visualizer.set_case_output_dir(case_output_dir)
        
        # 创建线程特定的计时器
        case_timer = AverageTimer(newline=True)
        
        # 加载图像对
        image_pairs = self._load_image_pairs(
            os.path.join(case_path, "scannet_pairs.txt"), case_path
        )
        
        # 处理每个图像对
        case_inliers = []
        for i, image_pair in enumerate(image_pairs):
            if os.path.exists(image_pair.matches_path):
                logger.info(f"Already processed {image_pair.matches_path}")
                continue
                
            # 使用自定义参数调用处理函数
            inliers = self._process_image_pair(image_pair, case_path)
            case_inliers.append(inliers)
            case_timer.print(f"Thread {threading.current_thread().name}: Finished pair {i+1} of {len(image_pairs)}")
        
        # 记录结果到文件
        with open(os.path.join(case_path, "sp_sg_knn_ransac_metrics.txt"), "w") as f:
            for inlier in case_inliers:
                f.write(f"{inlier}\n")
                
        return case_inliers

    def plot_violin_diagram(self):
        # After running the pipeline, plot and save the violin diagram from metrics files
        metrics_files = [Path(case_path) / "sp_sg_knn_ransac_metrics.txt" for case_path in self.case_dir_lists]
        logger.info(f"Metrics files: {metrics_files}")
        output_path = Path(self.config["output_dir"]) / "sp_sg_knn_ransac_violin_plot.png"
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
        preprocess_mode_day = None
        preprocess_mode_night = None

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
        inliers = self._process_with_superglue(image_pair)
        
        self.timer.update("process_image_pair")
        return inliers

    def _process_with_superglue(self, image_pair):
        """
        Process image pair with SuperGlue matching.
        
        Args:
            image_pair: ImagePair object
        
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
        mdesc0 = pred["descriptors0"][:, valid]
        valid_indices = matches0[valid]
        mdesc1 = pred["descriptors1"][:, valid_indices]
        mconf = conf[valid]

        # Apply RANSAC if requested
        if self.config["ransac"]:
            mkpts0_ransac, mkpts1_ransac, mconf_ransac, mdesc0_ransac, mdesc1_ransac, _ = (
                self.ransac_filter.filter_kpts_desc(mkpts0, mkpts1, mconf, mdesc0, mdesc1)
            )
        else:
            mkpts0_ransac, mkpts1_ransac, mconf_ransac, mdesc0_ransac, mdesc1_ransac = mkpts0, mkpts1, mconf, None, None

        # Save matches
        all_data_sp_sg_ransac = {
            "keypoints0": pred["keypoints0"],
            "keypoints1": pred["keypoints1"],
            "scores0": pred["scores0"],
            "scores1": pred["scores1"],
            "descriptors0": pred["descriptors0"],
            "descriptors1": pred["descriptors1"],
            "mkpts0_ransac": mkpts0_ransac,
            "mkpts1_ransac": mkpts1_ransac,
            "mconf_ransac": mconf_ransac,
            "mdesc0_ransac": mdesc0_ransac,
            "mdesc1_ransac": mdesc1_ransac,
        }
        np.savez(str(image_pair.matches_path), **all_data_sp_sg_ransac)
        
        # Save Superpoint keypoints if files don't exist yet
        if not os.path.exists(str(image_pair.sp_kpts0_file)):
            np.save(str(image_pair.sp_kpts0_file), pred["keypoints0"])
            logger.info(f"Saved Superpoint keypoints to {image_pair.sp_kpts0_file}")
        
        if not os.path.exists(str(image_pair.sp_kpts1_file)):
            np.save(str(image_pair.sp_kpts1_file), pred["keypoints1"])
            logger.info(f"Saved Superpoint keypoints to {image_pair.sp_kpts1_file}")

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
            sp_kpts0, sp_kpts1: SuperPoint keypoints
            mkpts0, mkpts1: Matched keypoints
            mconf: Match confidence scores
            mkpts0_ransac, mkpts1_ransac: RANSAC-filtered keypoints
            mconf_ransac: RANSAC-filtered confidence scores
        """
        text = [
            "SuperPoint+SuperGlue+RANSAC",
            f"Keypoints: {len(sp_kpts0)}:{len(sp_kpts1)}",
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

        # Visualize SuperGlue + RANSAC matches
        self.visualizer.visualize_matches(
            image_pair,
            sp_kpts0,
            sp_kpts1,
            mkpts0_ransac,
            mkpts1_ransac,
            mconf_ransac,
            text,
            "Matches",
            small_text,
        )

        # Visualize RANSAC filtering comparison if requested
        if self.config["viz_comparison"] and self.config["ransac"]:
            self.visualizer.visualize_ransac_comparison(
                image_pair,
                sp_kpts0,
                sp_kpts1,
                mkpts0,
                mkpts1,
                mkpts0_ransac,
                mkpts1_ransac,
                mconf,
                mconf_ransac,
                text,
            )

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
            all_data_sp_sg_ransac['keypoints0'],
            all_data_sp_sg_ransac['keypoints1'],
            all_data_sp_sg_ransac['mkpts0_ransac'],
            all_data_sp_sg_ransac['mkpts1_ransac'],
            all_data_sp_sg_ransac['mconf_ransac'],
            mkpts0_knn_ransac,
            mkpts1_knn_ransac,
            mconf_knn_ransac,
            text_sg_ransac,
            text_knn_ransac,
        )

    def _process_with_feature_booster(self, image_pair, match_data, features0, features1):
        """
        Process image pair with FeatureBooster.
        
        Args:
            image_pair: ImagePair object
            match_data: Dictionary containing match data
            features0, features1: Feature dictionaries from images
        """
        # Prepare data for FeatureBooster
        match_data_fb = {
            "image0": image_pair.inp0,
            "image1": image_pair.inp1,
            "shape0": image_pair.shape0[0],
            "shape1": image_pair.shape0[1],
        }
        
        # Process with FeatureBooster
        fb_pred = self.feat_bst_matching(match_data_fb)
        fb_pred = {k: v[0].cpu().numpy() for k, v in fb_pred.items()}
        
        kpts0_fb, kpts1_fb = fb_pred["keypoints0"], fb_pred["keypoints1"]
        matches0_fb_sg, conf_fb_sg = fb_pred["matches0"], fb_pred["matching_scores0"]
        
        # Extract valid matches
        valid_fb_sg = matches0_fb_sg > -1
        mkpts0_fb_sg = kpts0_fb[valid_fb_sg]
        mkpts1_fb_sg = kpts1_fb[matches0_fb_sg[valid_fb_sg]]
        mconf_fb_sg = conf_fb_sg[valid_fb_sg]
        
        # Update match_data with FeatureBooster results
        match_data_fb.update(
            {
                "keypoints0": [torch.from_numpy(kpts0_fb).to(self.device).float()],
                "keypoints1": [torch.from_numpy(kpts1_fb).to(self.device).float()],
                "scores0": fb_pred['scores0'],
                "scores1": fb_pred['scores1'],
                "descriptors0": [torch.from_numpy(fb_pred["descriptors0"]).to(self.device).float()],
                "descriptors1": [torch.from_numpy(fb_pred["descriptors1"]).to(self.device).float()],
                "matches0": matches0_fb_sg,
                "matching_scores0": conf_fb_sg,
            }
        )
        
        # Visualize SP+FB+SG+RANSAC and SP+SG+RANSAC comparison if requested
        if self.config["compare_fb_sg_ransac"]:
            # Apply RANSAC to FeatureBooster matches
            mkpts0_fb_sg_ransac, mkpts1_fb_sg_ransac, mconf_fb_sg_ransac, _ = (
                self.ransac_filter.filter(mkpts0_fb_sg, mkpts1_fb_sg, mconf_fb_sg)
            )
            
            # Get SuperGlue+RANSAC matches for comparison
            pred = self.sp_sg_matching({"image0": image_pair.inp0, "image1": image_pair.inp1})
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
            matches0, conf = pred["matches0"], pred["matching_scores0"]
            valid = matches0 > -1
            mkpts0 = pred["keypoints0"][valid]
            mkpts1 = pred["keypoints1"][matches0[valid]]
            mconf = conf[valid]
            
            # Apply RANSAC to SuperGlue matches
            mkpts0_ransac, mkpts1_ransac, mconf_ransac, _ = (
                self.ransac_filter.filter(mkpts0, mkpts1, mconf)
            )
            
            # Visualize comparison
            self.visualizer.visualize_feature_booster_comparison(
                image_pair,
                features0["keypoints"][0].cpu().numpy(),
                features1["keypoints"][0].cpu().numpy(),
                mkpts0_ransac,
                mkpts1_ransac,
                mconf_ransac,  # SP+SG+RANSAC results
                mkpts0_fb_sg_ransac,
                mkpts1_fb_sg_ransac,
                mconf_fb_sg_ransac,  # SP+FB+SG+RANSAC results
                ["SP+SG+RANSAC"],
                ["SP+FB+SG+RANSAC"],
            )
            
        # Compare KNN+FeatureBooster with regular KNN if requested
        if self.config["compare_knn_fb_ransac"]:
            self._compare_fb_knn(image_pair, match_data, match_data_fb, 
                                features0, features1, kpts0_fb, kpts1_fb)

    def _compare_fb_knn(self, image_pair, match_data, match_data_fb, 
                        features0, features1, kpts0_fb, kpts1_fb):
        """
        Compare KNN matching with and without FeatureBooster.
        
        Args:
            image_pair: ImagePair object
            match_data: Dictionary with original match data
            match_data_fb: Dictionary with FeatureBooster match data
            features0, features1: Feature dictionaries
            kpts0_fb, kpts1_fb: FeatureBooster keypoints
        """
        # SuperPoint+FeatureBooster+KNN+RANSAC
        fb_knn_matches = self.knn_matcher.match(match_data_fb)
        valid_fb_knn = fb_knn_matches["matches0"] > -1
        mkpts0_fb_knn = kpts0_fb[valid_fb_knn]
        mkpts1_fb_knn = kpts1_fb[fb_knn_matches["matches0"][valid_fb_knn]]
        mconf_fb_knn = fb_knn_matches["matching_scores0"][valid_fb_knn]
        
        # Apply RANSAC
        mkpts0_fb_knn_ransac, mkpts1_fb_knn_ransac, mconf_fb_knn_ransac, _ = (
            self.ransac_filter.filter(mkpts0_fb_knn, mkpts1_fb_knn, mconf_fb_knn)
        )
        
        # SuperPoint+KNN+RANSAC (without FeatureBooster)
        knn_matches = self.knn_matcher.match(match_data)
        valid_knn = knn_matches["matches0"] > -1
        
        pred = self.sp_sg_matching({"image0": image_pair.inp0, "image1": image_pair.inp1})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred["keypoints0"], pred["keypoints1"]
        
        mkpts0_knn = kpts0[valid_knn]
        mkpts1_knn = kpts1[knn_matches["matches0"][valid_knn]]
        mconf_knn = knn_matches["matching_scores0"][valid_knn]
        
        # Apply RANSAC
        mkpts0_knn_ransac, mkpts1_knn_ransac, mconf_knn_ransac, _ = (
            self.ransac_filter.filter(mkpts0_knn, mkpts1_knn, mconf_knn)
        )
        
        # Text for visualizations
        text_fb_knn_ransac = [
            "SP+FB+KNN+RANSAC",
            f"Keypts in day|night: {len(kpts0_fb)}:{len(kpts1_fb)}",
            f"Matches: {len(mkpts0_fb_knn_ransac)}",
            f"RANSAC: {len(mkpts0_fb_knn_ransac)}/{len(mkpts0_fb_knn)}",
        ]
        
        text_knn_ransac = [
            "SP+KNN+RANSAC",
            f"Keypts in day|night: {len(kpts0)}:{len(kpts1)}",
            f"Matches: {len(mkpts0_knn_ransac)}",
            f"RANSAC: {len(mkpts0_knn_ransac)}/{len(mkpts0_knn)}",
        ]
        
        # Visualize comparison
        self.visualizer.visualize_matcher_comparison(
            image_pair,
            features0["keypoints"][0].cpu().numpy(),
            features1["keypoints"][0].cpu().numpy(),
            mkpts0_fb_knn_ransac,
            mkpts1_fb_knn_ransac,
            mconf_fb_knn_ransac,
            mkpts0_knn_ransac,
            mkpts1_knn_ransac,
            mconf_knn_ransac,
            text_fb_knn_ransac,
            text_knn_ransac,
        )

    def _evaluate(self):
        """Evaluate the matching results across all image pairs."""
        pose_errors = []
        precisions = []
        matching_scores = []
        
        for image_pair in self.image_pairs:
            eval_path = image_pair.eval_path
            
            # Load evaluation results
            results = np.load(eval_path)
            pose_error = np.maximum(results["error_t"], results["error_R"])
            pose_errors.append(pose_error)
            precisions.append(results["precision"])
            matching_scores.append(results["matching_score"])

        # Calculate metrics
        thresholds = [5, 10, 20]
        aucs = pose_auc(pose_errors, thresholds)
        aucs = [100.0 * yy for yy in aucs]
        prec = 100.0 * np.mean(precisions)
        ms = 100.0 * np.mean(matching_scores)
        
        # Log results
        logger.info(
            "Evaluation Results (mean over {} pairs):".format(len(self.image_pairs))
        )
        logger.info("AUC@5\t AUC@10\t AUC@20\t Prec\t MScore\t")
        logger.info(
            "{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t".format(
                aucs[0], aucs[1], aucs[2], prec, ms
            )
        )
