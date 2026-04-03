"""
特征点匹配与评估管道 (Feature Matching and Evaluation Pipeline)

功能说明:
    本脚本实现了一个完整的图像特征点提取、匹配和评估流程，主要用于分析昼夜图像对的匹配性能。
    整个流程包括：加载图像对，提取特征点和描述子，KNN匹配，RANSAC过滤外点，统计内点数量，
    并生成内点分布的可视化图表。

算法流程:
    1. 初始化与配置
       - 从配置文件加载参数设置
       - 初始化KNN匹配器、RANSAC滤波器和可视化工具
       - 设置设备(CPU/GPU)和时间追踪器
    
    2. 加载图像对
       - 从输入文本文件中读取图像对信息
       - 创建ImagePair对象保存图像的元数据和路径
       - 设置结果输出路径
    
    3. 特征点和描述子处理
       - 从文本文件中加载预计算的特征点坐标和描述子
       - 转换描述子格式以适配匹配算法
       - 对特征点坐标进行缩放以匹配处理后的图像尺寸
    
    4. 特征匹配
       - 使用KNN算法匹配两幅图像的特征描述子
       - 应用Lowe比率测试过滤低质量匹配
       - 提取匹配成功的特征点对
    
    5. 几何验证
       - 使用RANSAC算法估计基础矩阵或单应性矩阵
       - 过滤不符合几何约束的匹配(外点)
       - 保留符合约束的匹配(内点)
    
    6. 统计与评估
       - 统计匹配到的内点数量
       - 保存匹配结果到NPZ文件
       - 记录每个案例的内点数量指标
    
    7. 可视化展示
       - 生成小提琴图显示不同案例的内点分布情况
       - 计算并显示内点数量的基本统计信息(均值、中位数等)

使用说明:
    运行命令: python pipeline_evaluation.py --config config_evaluation.yaml
    配置文件应包含输入输出路径、KNN和RANSAC参数等设置
"""

import argparse
import logging
import os
import random
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import yaml
from matplotlib import cm

from models.utils import error_colormap, make_matching_plot

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("EvaluationPipeline")

# 禁用梯度计算以进行推理
torch.set_grad_enabled(False)


class ImagePair:
    """
    Class representing an image pair with associated data.
    
    This class holds information about a pair of images, including their names,
    rotation information, camera parameters, and the paths for output files.
    """

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
        """
        Initialize an ImagePair instance.
        
        Args:
            name0: Filename of the first image
            name1: Filename of the second image
            stem0: Stem (filename without extension) of the first image
            stem1: Stem of the second image
            rot0: Rotation angle in degrees for the first image (default: 0)
            rot1: Rotation angle in degrees for the second image (default: 0)
            intrinsics: Dictionary containing K0, K1 camera intrinsic matrices
            extrinsics: T_0to1 transformation matrix
        """
        self.name0 = name0
        self.name1 = name1
        self.stem0 = stem0
        self.stem1 = stem1
        self.rot0 = rot0
        self.rot1 = rot1
        self.image0 = None
        self.image1 = None
        self.inp0 = None
        self.inp1 = None
        self.processed_image0 = None
        self.processed_image1 = None
        self.processed_inp0 = None
        self.processed_inp1 = None
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
        
        self.kpts0_path = None
        self.kpts1_path = None
        self.desc0_path = None
        self.desc1_path = None
        
        self.kpts0 = None
        self.kpts1 = None
        self.desc0 = None
        self.desc1 = None

    def set_paths(self, output_dir: Path, viz_extension: str = "png"):
        """
        Set output file paths for this image pair.
        
        Args:
            output_dir: Directory where output files will be stored
            viz_extension: File extension for visualization images
        """
        self.matches_path = output_dir / f"{self.stem0}_{self.stem1}_matches.npz"
        self.eval_path = output_dir / f"{self.stem0}_{self.stem1}_evaluation.npz"
        self.viz_path = (
            output_dir / f"{self.stem0}_{self.stem1}_matches.{viz_extension}"
        )
        self.viz_eval_path = (
            output_dir / f"{self.stem0}_{self.stem1}_evaluation.{viz_extension}"
        )

    def set_kpts_desc_paths(self, input_dir: str) -> bool:
        """
        Set paths for keypoints and descriptors files.
        
        Args:
            input_dir: Base directory for keypoints and descriptors
            
        Returns:
            True if paths were set successfully, False otherwise
        """
        day_folder = os.path.join(input_dir, "day")
        night_folder = os.path.join(input_dir, "night")
        
        # Set paths for keypoints and descriptors
        day_kpts_dir = os.path.join(input_dir, "day_j6-128_locfeat")
        night_kpts_dir = os.path.join(input_dir, "night_j6-128_locfeat")
        
        self.kpts0_path = os.path.join(day_kpts_dir, f"{os.path.basename(self.name0)}.kpts.txt")
        self.kpts1_path = os.path.join(night_kpts_dir, f"{os.path.basename(self.name1)}.kpts.txt")
        self.desc0_path = os.path.join(day_kpts_dir, f"{os.path.basename(self.name0)}.desc.txt")
        self.desc1_path = os.path.join(night_kpts_dir, f"{os.path.basename(self.name1)}.desc.txt")
        
        # Check if paths exist
        if not os.path.exists(self.kpts0_path) or not os.path.exists(self.kpts1_path) or \
           not os.path.exists(self.desc0_path) or not os.path.exists(self.desc1_path):
            logger.error(f"Keypoints or descriptors files not found: {self.kpts0_path} or {self.kpts1_path} or {self.desc0_path} or {self.desc1_path}")
            return False
        else:
            return True

    def load_kpts_desc(self):
        """
        Load keypoints and descriptors for day and night images from text files.
        
        File Format Information:
        -----------------------
        Keypoints file (.kpts):
            First line: Number of keypoints (n)
            Following n lines: x y response
                x, y: Keypoint coordinates (pixels)
                response: Detector response/strength
                
        Descriptors file (.desc):
            First line: Number of descriptors (n) Descriptor_dimension (d)
            Following n lines: d space-separated values for each descriptor
        
        Returns:
        --------
            Tuple containing:
                kpts0: Keypoints for day image, numpy array of shape (n, 3) containing (x, y, response)
                kpts1: Keypoints for night image, numpy array of shape (n, 3) containing (x, y, response)
                desc0: Descriptors for day image, numpy array of shape (d, n)
                       where d is the descriptor dimension (typically 128/256)
                desc1: Descriptors for night image, numpy array of shape (d, n)
        
        Notes:
        ------
        - Keypoints are loaded as (x, y, response) format
        - Descriptors are initially loaded as (n, d) but transposed to (d, n) for matching
        - Keypoints are automatically resized from original image dimensions (3840x2160)
          to target dimensions (910x512) to match resized images used for matching
        """
        # Load keypoints for day image
        # Input: text file with format:
        # <num_keypoints>
        # x1 y1 response1
        # x2 y2 response2
        # ...
        try:
            with open(self.kpts0_path, 'r') as f:
                lines = f.readlines()
                num_kpts = int(lines[0].strip())
                kpts0 = []
                for i in range(1, num_kpts + 1):
                    if i < len(lines):
                        parts = lines[i].strip().split()
                        if len(parts) >= 3:
                            x, y, response = float(parts[0]), float(parts[1]), float(parts[2])
                            kpts0.append((x, y, response))
                self.kpts0 = np.array(kpts0)
                logger.info(f"Loaded {len(kpts0)} keypoints from day image: {self.kpts0_path}")
        except Exception as e:
            logger.error(f"Error loading keypoints from {self.kpts0_path}: {e}")
            self.kpts0 = None
            
        # Load keypoints for night image
        # Same format as day image keypoints
        try:
            with open(self.kpts1_path, 'r') as f:
                lines = f.readlines()
                num_kpts = int(lines[0].strip())
                kpts1 = []
                for i in range(1, num_kpts + 1):
                    if i < len(lines):
                        parts = lines[i].strip().split()
                        if len(parts) >= 3:
                            x, y, response = float(parts[0]), float(parts[1]), float(parts[2])
                            kpts1.append((x, y, response))
                self.kpts1 = np.array(kpts1)
                logger.info(f"Loaded {len(kpts1)} keypoints from night image: {self.kpts1_path}")
        except Exception as e:
            logger.error(f"Error loading keypoints from {self.kpts1_path}: {e}")
            self.kpts1 = None
            
        # Load descriptors for day image
        # Input: text file with format:
        # <num_descriptors> <descriptor_dimension>
        # d11 d12 d13 ... d1n  (values for descriptor 1)
        # d21 d22 d23 ... d2n  (values for descriptor 2)
        # ...
        try:
            with open(self.desc0_path, 'r') as f:
                lines = f.readlines()
                header = lines[0].strip().split()
                if len(header) >= 2:
                    num_desc, dim = int(header[0]), int(header[1])
                    desc0 = []
                    for i in range(1, num_desc + 1):
                        if i < len(lines):
                            values = [float(v) for v in lines[i].strip().split()]
                            if len(values) == dim:
                                desc0.append(values)
                    self.desc0 = np.array(desc0, dtype=np.float32)
                    logger.info(f"Loaded {num_desc} descriptors of dimension {dim} from day image: {self.desc0_path}")
        except Exception as e:
            logger.error(f"Error loading descriptors from {self.desc0_path}: {e}")
            self.desc0 = None
            
        # Load descriptors for night image
        # Same format as day image descriptors
        try:
            with open(self.desc1_path, 'r') as f:
                lines = f.readlines()
                header = lines[0].strip().split()
                if len(header) >= 2:
                    num_desc, dim = int(header[0]), int(header[1])
                    desc1 = []
                    for i in range(1, num_desc + 1):
                        if i < len(lines):
                            values = [float(v) for v in lines[i].strip().split()]
                            if len(values) == dim:
                                desc1.append(values)
                    self.desc1 = np.array(desc1, dtype=np.float32)
                    logger.info(f"Loaded {num_desc} descriptors of dimension {dim} from night image: {self.desc1_path}")
        except Exception as e:
            logger.error(f"Error loading descriptors from {self.desc1_path}: {e}")
            self.desc1 = None
            
        # Transpose descriptors to match expected format for matching
        # Convert from (num_keypoints, dim) to (dim, num_keypoints)
        # This format is required for efficient matching in OpenCV/KNN
        if self.desc0 is not None:
            logger.info(f"Descriptor shape before transpose: {self.desc0.shape}")
            self.desc0 = self.desc0.T
            logger.info(f"Descriptor shape after transpose: {self.desc0.shape}")
        if self.desc1 is not None:
            self.desc1 = self.desc1.T
        
        # Resize keypoints if needed
        # The original images are large (3840x2160), but processing is done on smaller
        # resized versions (910x512). We need to adjust the keypoint coordinates to match.
        if self.kpts0 is not None and self.kpts1 is not None:
            # Define original and target dimensions
            src_width, src_height = 3840, 2160  # Original image dimensions
            target_width, target_height = 910, 512  # Target dimensions after resize
            
            # Calculate resize ratios
            ratio_x = float(target_width) / src_width
            ratio_y = float(target_height) / src_height
            
            # Apply resize to keypoints
            if self.kpts0.shape[1] >= 2:  # Make sure keypoints have at least x, y coordinates
                # Original keypoint coordinates are in full resolution space
                # Scale them to match the resolution used in matching
                self.kpts0[:, 0] *= ratio_x  # Scale x coordinates
                self.kpts0[:, 1] *= ratio_y  # Scale y coordinates
                
            if self.kpts1.shape[1] >= 2:
                self.kpts1[:, 0] *= ratio_x
                self.kpts1[:, 1] *= ratio_y
                
            logger.info(f"Resized keypoints using ratio: x={ratio_x:.4f}, y={ratio_y:.4f}")
            logger.info(f"Keypoint shapes after resize: kpts0={self.kpts0.shape}, kpts1={self.kpts1.shape}")
        
        # Return the loaded keypoints and descriptors
        # - kpts0, kpts1: shape (n, 3) where n is the number of keypoints
        # - desc0, desc1: shape (d, n) where d is the descriptor dimension
        return self.kpts0, self.kpts1, self.desc0, self.desc1        
        
    def get_kpts_desc(self):
        """
        Get keypoints and descriptors for the image pair.
        If not already loaded, this method will load them.
        
        Returns:
            Tuple containing:
                kpts0: Keypoints for day image
                kpts1: Keypoints for night image
                desc0: Descriptors for day image
                desc1: Descriptors for night image
        """
        if self.kpts0 is None or self.kpts1 is None or self.desc0 is None or self.desc1 is None:
            return self.load_kpts_desc()
        return self.kpts0, self.kpts1, self.desc0, self.desc1

class FeatureMatcher(ABC):
    """
    Base class for feature matchers.
    
    This abstract class defines the interface for feature matching algorithms.
    """

    @abstractmethod
    def match(self, data: Dict) -> Dict:
        """
        Match features between two images.
        
        Args:
            data: Dictionary containing descriptors and other data
            
        Returns:
            Dictionary with matching results
        """
        pass


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

        # Make sure descriptors are in the right shape for OpenCV
        # SuperPoint descriptors are shape (dim, num_keypoints), but OpenCV expects (num_keypoints, dim)
        desc0 = desc0.T
        desc1 = desc1.T
        
        # Ensure we have the right type - OpenCV needs float32
        desc0 = desc0.astype(np.float32)
        desc1 = desc1.astype(np.float32)

        # Log descriptor shapes for debugging
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


class AverageTimer:
    """
    Timer for measuring and tracking execution times of different operations.
    """
    def __init__(self, newline=False):
        self.newline = newline
        self.times = {}
        self.reset()

    def reset(self):
        self.times = {}
        self.last_time = {'total': 0}

    def update(self, key):
        import time
        if key not in self.times:
            self.times[key] = 0
        if key not in self.last_time:
            self.last_time[key] = 0
        curr_time = time.time()
        
        # Update the total time
        if key != 'total':
            self.update('total')
            
        self.times[key] += curr_time - self.last_time.get(key, curr_time)
        self.last_time[key] = curr_time

    def print(self, text='Timer'):
        total = self.times.get('total', 0)
        print(f'{text}: {total:.3f}s', end='\n' if self.newline else ' | ')
        for key, val in sorted(self.times.items()):
            if key != 'total' and val > 0:
                print(f'{key}: {val:.3f}s ({val/total:.1%})', end='\n' if self.newline else ' | ')
        if not self.newline:
            print('')


class Visualizer:
    """
    Class for visualizing matching results and statistics.
    
    This class handles the visualization of keypoints, matches,
    and the generation of metrics visualizations.
    """

    def __init__(
        self,
        output_dir: Path,
        viz_extension: str = "png",
        show_keypoints: bool = True,
        fast_viz: bool = True,
        opencv_display: bool = False,
        line_width: int = 1,
    ):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory where visualizations will be saved
            viz_extension: File extension for visualization images
            show_keypoints: Whether to show keypoints in visualizations
            fast_viz: Use faster visualization methods
            opencv_display: Show visualizations using OpenCV windows
            line_width: Width of lines in visualizations
        """
        self.output_dir = output_dir
        self.viz_extension = viz_extension
        self.show_keypoints = show_keypoints
        self.fast_viz = fast_viz
        self.opencv_display = opencv_display
        self.line_width = line_width
        
        # Create the output directory if it doesn't exist
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            
    def plot_and_save_violin_diagram(self, metrics_files, output_path):
        """
        Plot and save violin diagrams showing inlier distributions.
        
        Args:
            metrics_files: List of paths to files containing inlier metrics
            output_path: Path where the visualization will be saved
        """
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
            import seaborn as sns

            # Collect data from metrics files
            all_data = []
            labels = []
            
            for i, file_path in enumerate(metrics_files):
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        inliers = [int(line.strip()) for line in f.readlines() if line.strip()]
                        if inliers:
                            all_data.append(inliers)
                            labels.append(f"Case {i+1}")
            
            if not all_data:
                logger.error("No valid data found for violin plot")
                return
                
            # Create DataFrame for seaborn
            data_dict = {}
            for i, (data, label) in enumerate(zip(all_data, labels)):
                data_dict[label] = data
                
            df = pd.DataFrame(data_dict)
            
            # Create violin plot
            plt.figure(figsize=(12, 8))
            sns.violinplot(data=df)
            plt.title("Inlier Distribution Across Cases")
            plt.ylabel("Number of Inliers")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(output_path)
            logger.info(f"Saved violin plot to {output_path}")
            
            # Display basic statistics
            for label, data in zip(labels, all_data):
                logger.info(f"{label}: Mean={np.mean(data):.1f}, Median={np.median(data)}, Min={min(data)}, Max={max(data)}")
            
        except Exception as e:
            logger.error(f"Error creating violin plot: {e}")

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
        """
        Visualize matches between two images.
        
        Args:
            image_pair: ImagePair object containing the images
            kpts0, kpts1: All keypoints from both images
            mkpts0, mkpts1: Matched keypoints from both images
            mconf: Confidence scores for the matches
            text: List of text lines to display on the visualization
            title: Title for the visualization (default: "Matches")
            small_text: Additional small text to display (default: [])
        """
        color = cm.jet(mconf)

        make_matching_plot(
            image_pair.processed_image0,
            image_pair.processed_image1,
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
        )

def read_image(path, device, resize=None, rotation=0, resize_float=False, preprocess_mode=None):
    """
    Read an image and prepare it for processing.
    
    Args:
        path: Path to the image file
        device: Computation device (CPU or CUDA)
        resize: Image size to resize to (None for no resizing)
        rotation: Rotation angle in degrees
        resize_float: Whether to use floating-point resizing
        preprocess_mode: Mode for preprocessing the image
        
    Returns:
        Tuple containing:
            image: Original image
            inp: Preprocessed image tensor
            scales: Scaling factors
            shape dimensions
    """
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None, None, None, None, None, None

    # Original image dimensions
    h, w = image.shape
    shape = (h, w)
    
    # Store original image for visualization
    image_orig = image.copy()
    
    # Apply preprocessing if specified
    processed_image = None
    processed_inp = None
    
    if preprocess_mode:
        # Apply preprocessing based on mode (day/night)
        if preprocess_mode == 'day':
            # Example preprocessing for day images
            processed_image = cv2.equalizeHist(image)
        elif preprocess_mode == 'night':
            # Example preprocessing for night images
            processed_image = cv2.equalizeHist(image)
            # Additional night-specific processing could be added here
        else:
            processed_image = image.copy()
    
    # Calculate resize dimensions
    w, h = image_orig.shape[1], image_orig.shape[0]
    w_new, h_new = (910, 512)
    scales = (float(w) / float(w_new), float(h) / float(h_new))
    
    # Resize images based on resize_float parameter
    # if resize_float:
    #     processed_image = cv2.resize(image_orig.astype("float32"), (w_new, h_new))

    # else:
    processed_image = cv2.resize(image_orig, (w_new, h_new)).astype("float32")
    
    # # Apply rotation if requested
    # if rotation != 0:
    #     image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE if rotation > 0 else cv2.ROTATE_90_COUNTERCLOCKWISE)
    #     if processed_image is not None:
    #         processed_image = cv2.rotate(processed_image, cv2.ROTATE_90_CLOCKWISE if rotation > 0 else cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    # Convert images to tensors
    inp = torch.from_numpy(image).float().to(device) / 255.0
    inp = inp.unsqueeze(0).unsqueeze(0)
    
    if processed_image is not None:
        processed_inp = torch.from_numpy(processed_image).float().to(device) / 255.0
        processed_inp = processed_inp.unsqueeze(0).unsqueeze(0)
    
    return image_orig, inp, processed_image, processed_inp, scales, h, w


class EvaluationPipeline:
    """
    Class to orchestrate the evaluation process.
    
    This class manages the evaluation pipeline, including:
    - Loading image pairs
    - Extracting features
    - Matching features
    - Filtering matches
    - Evaluating results
    - Visualizing matches
    """

    def __init__(self, config: Dict):
        """
        Initialize the evaluation pipeline.
        
        Args:
            config: Dictionary containing pipeline configuration
        """
        self.config = config
        self.case_dir_lists = []
        
        # Set up device for computation
        self.device = (
            "cuda" if torch.cuda.is_available() and not config["force_cpu"] else "cpu"
        )
        logger.info(f'Running inference on device "{self.device}"')
        
        # Initialize pipeline components
        self._init_components()
        
        # Load case directories
        for dir in os.listdir(self.config["input_dir"]):
            input_dir = os.path.join(self.config["input_dir"], dir)
            if os.path.isdir(input_dir) and dir.startswith("P") and "case" in dir:
                if os.path.exists(os.path.join(input_dir, "scannet_pairs.txt")):
                    if os.path.exists(os.path.join(input_dir, "day")) and os.path.exists(os.path.join(input_dir, "night")):
                        self.case_dir_lists.append(input_dir)
                    else:
                        logger.error(f"Invalid directory structure for {dir}")
                else:
                    logger.error(f"Scannet pair file not found for {dir}")

        # Timer for performance measurement
        self.timer = AverageTimer(newline=True)

    def _init_components(self):
        """Initialize pipeline components for matching, filtering, and visualization."""
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
        """Run the evaluation pipeline on all image pairs."""
        all_inliers = []
        for case_dir in self.case_dir_lists:
            logger.info(f"Processing case: {os.path.basename(case_dir)}")

            self.image_pairs = self._load_image_pairs(
                os.path.join(case_dir, "scannet_pairs.txt"), case_dir
            )
            case_inliers = []
            for i, image_pair in enumerate(self.image_pairs):
                inliers = self._process_image_pair(image_pair, case_dir)
                case_inliers.append(inliers)
                self.timer.print(f"Finished pair {i+1} of {len(self.image_pairs)}")

            # Record inliers metrics into a txt file
            with open(os.path.join(case_dir, "j6_128_scale4-desc_KNN_metrics.txt"), "w") as f:
                for inlier in case_inliers:
                    f.write(f"{inlier}\n")

            all_inliers.extend(case_inliers)

    def plot_violin_diagram(self):
        """Plot and save the violin diagram showing inlier distributions."""
        metrics_files = [os.path.join(case_path, "j6_128_scale4-desc_KNN_metrics.txt") for case_path in self.case_dir_lists]
        logger.info(f"Metrics files: {metrics_files}")
        output_path = os.path.join(self.config["output_dir"], "J6_128_scale4-desc_KNN_violin_plot.png")
        self.visualizer.plot_and_save_violin_diagram(metrics_files, output_path)

    def _process_image_pair(self, image_pair: ImagePair, case_path: str):
        """
        Process a single image pair.
        
        Args:
            image_pair: ImagePair object to process
            case_path: Path to the current case directory

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
        
        # # Visualize matches if requested
        # if self.config["viz"]:
        #     self._visualize_matches(image_pair, image_pair.kpts0, image_pair.kpts1, mkpts0_knn, mkpts1_knn, 
        #                            mconf_knn, mkpts0_knn_ransac, mkpts1_knn_ransac, mconf_knn_ransac)
            
        # Return the number of RANSAC inliers
        num_inliers = len(mkpts0_knn_ransac)
        logger.info(f"KNN+RANSAC: {num_inliers} inliers out of {len(mkpts0_knn)} matches")
        return num_inliers
    
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

        # # Visualize RANSAC filtering comparison if requested
        # if self.config["viz_comparison"] and self.config["ransac"]:
        #     self.visualizer.visualize_ransac_comparison(
        #         image_pair,
        #         sp_kpts0_xy,
        #         sp_kpts1_xy,
        #         mkpts0_xy,
        #         mkpts1_xy,
        #         mkpts0_ransac_xy,
        #         mkpts1_ransac_xy,
        #         mconf,
        #         mconf_ransac,
        #         text,
        #     )
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation Pipeline")
    parser.add_argument(
        "--config", type=str, required=True, default="config_baseline.yaml", help="Path to the YAML configuration file"
    )
    args = parser.parse_args()

    # Load the YAML configuration file
    with open(args.config, "r") as f:
        yaml_config = yaml.safe_load(f)

    # Initialize the pipeline
    pipeline = EvaluationPipeline(yaml_config)
    
    # Run the matching pipeline
    pipeline.run()
    
    # Plot and save the violin diagram
    pipeline.plot_violin_diagram()